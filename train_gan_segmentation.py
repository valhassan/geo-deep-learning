import torch
from tqdm import tqdm
from torch import optim
from pathlib import Path
from typing import Sequence
from omegaconf import DictConfig
from utils.loss import define_loss
from utils.logger import get_logger
from models.model_choice import define_model
from utils.metrics import report_classification, create_metrics_dict, iou
from utils.train_utils import create_dataloader, make_tiles_dir_name, make_dataset_file_name
from hydra.utils import to_absolute_path, instantiate
from utils.utils import get_key_def, get_device_ids, set_device
# Set the logging file
logging = get_logger(__name__)  # import logging


def training(train_loader,
             discriminator,
             segmentor,
             criterion,
             optimizer_s,
             optimizer_d,
             s_scheduler,
             d_scheduler,
             num_classes,
             batch_size,
             device):
    train_metrics = create_metrics_dict(num_classes)
    # Establish convention for real and fake labels during training
    label_is_real = 1.0
    label_is_fake = 0.0
    for batch_index, data in enumerate(tqdm(train_loader,
                                            desc=f'Iterating train batches with {device.type}')):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with real labels batch
        discriminator.zero_grad(set_to_none=True)
        inputs = data['sat_img'].to(device)
        real_labels = data['map_img'].to(device).unsqueeze(1).float()
        label_d = torch.full((batch_size,), label_is_real, dtype=torch.float).to(device)
        # Forward pass real batch through D
        output_d = discriminator(real_labels).view(-1)
        # Calculate loss on all-real batch
        loss_d_real = criterion(output_d, label_d)
        # Calculate gradients for D in backward pass
        loss_d_real.backward()
        d_x = output_d.mean().item()
        ## Train with fake predicted batch
        fake_labels = segmentor(inputs)

        label_d.fill_(label_is_fake)
        # Classify all fake batch with D
        output_d = discriminator(fake_labels.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        loss_d_fake = criterion(output_d, label_d)
        # Calculate the gradients for this batch,
        # accumulated (summed) with previous gradients
        loss_d_fake.backward()
        d_g_z1 = output_d.mean().item()
        # Compute error of D as sum over the fake and the real batches
        loss_d = loss_d_real + loss_d_fake
        # Update D
        optimizer_d.step()
        ############################
        # (2) Update Segmentor network: maximize log(D(S(z)))
        ###########################
        segmentor.zero_grad(set_to_none=True)
        label_d.fill_(label_is_real)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_d = discriminator(fake_labels).view(-1)

        # Calculate G's losses
        loss_g1 = criterion(output_d, label_d)
        # Calculate gradients for G1
        loss_g1.backward(retain_graph=True)
        loss_g2 = criterion(fake_labels, real_labels)
        # Calculate gradients for G2
        loss_g2.backward()
        loss_g = loss_g1 + loss_g2
        # print(lossG)
        # lossG.backward()
        d_g_z2 = output_d.mean().item()
        # Update G
        optimizer_s.step()

        d_scheduler.step()
        s_scheduler.step()
        train_metrics['segmentor_loss'].update(loss_g.item(), batch_size)
        train_metrics['discriminator_loss'].update(loss_d.item(), batch_size)
        train_metrics['real_score_critic'].update(d_x, batch_size)
        train_metrics['fake_score_critic'].update(d_g_z2, batch_size)
    return train_metrics


def train(cfg: DictConfig) -> None:

    # MANDATORY PARAMETERS
    class_keys = len(get_key_def('classes_dict', cfg['dataset']).keys())
    num_classes = class_keys if class_keys == 1 else class_keys + 1  # +1 for background(multiclass mode)
    modalities = get_key_def('modalities', cfg['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    num_bands = len(modalities)
    batch_size = get_key_def('batch_size', cfg['training'], expected_type=int)
    eval_batch_size = get_key_def('eval_batch_size', cfg['training'], expected_type=int, default=batch_size)
    num_epochs = get_key_def('max_epochs', cfg['training'], expected_type=int)

    # OPTIONAL PARAMETERS
    debug = get_key_def('debug', cfg)
    task = get_key_def('task', cfg['general'], default='segmentation')
    dontcare_val = get_key_def("ignore_index", cfg['dataset'], default=-1)
    scale = get_key_def('scale_data', cfg['augmentation'], default=[0, 1])
    batch_metrics = get_key_def('batch_metrics', cfg['training'], default=None)
    crop_size = get_key_def('crop_size', cfg['augmentation'], default=None)

    # overwrite dontcare values in label if loss doens't implement ignore_index
    dontcare2backgr = False if 'ignore_index' in cfg.loss.keys() else True


    # LOGGING PARAMETERS
    experiment_name = get_key_def('project_name', cfg['general'], default='gdl-training')

    # PARAMETERS FOR DATA INPUTS
    samples_size = get_key_def('chip_size', cfg['tiling'], default=256, expected_type=int)
    attr_vals = get_key_def("attribute_values", cfg['dataset'], default=-1)
    overlap = get_key_def('overlap_size', cfg['tiling'], default=0)
    min_annot_perc = get_key_def('min_annot_perc', cfg['tiling'], default=0)

    # GPU PARAMETERS
    num_devices = get_key_def('num_gpus', cfg['training'], default=0)
    if num_devices and not num_devices >= 0:
        raise ValueError("\nMissing mandatory num gpus parameter")
    # Set device(s)
    gpu_devices_dict = get_device_ids(num_devices)
    device = set_device(gpu_devices_dict=gpu_devices_dict)

    # Tiles Directory
    data_path = get_key_def('tiling_data_dir', cfg['tiling'], to_path=True, validate_path_exists=True)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    tiles_dir_name = make_tiles_dir_name(samples_size, num_bands)
    tiles_dir = data_path / experiment_name / tiles_dir_name

    # MODEL PARAMETERS
    segmentor_ckpt_path = get_key_def('state_dict_path', cfg['training'], default=None, expected_type=str)
    discriminator_ckpt_path = get_key_def('discriminator_ckpt', cfg['training'], default=None, expected_type=str)
    state_dict_strict = get_key_def('state_dict_strict_load', cfg['training'], default=True, expected_type=bool)
    class_weights = get_key_def('class_weights', cfg['dataset'], default=None)
    if cfg.loss.is_binary and not num_classes == 1:
        raise ValueError(f"Parameter mismatch: a binary loss was chosen for a {num_classes}-class task")
    elif not cfg.loss.is_binary and num_classes == 1:
        raise ValueError(f"Parameter mismatch: a multiclass loss was chosen for a 1-class (binary) task")
    del cfg.loss.is_binary  # prevent exception at instantiation

    if segmentor_ckpt_path and not Path(segmentor_ckpt_path).is_file():
        raise logging.critical(FileNotFoundError(f'\nCould not locate segmentor checkpoint for training: '
                                                 f'{segmentor_ckpt_path}'))
    if discriminator_ckpt_path and not Path(discriminator_ckpt_path).is_file():
        raise logging.critical(FileNotFoundError(f'\nCould not locate discriminator checkpoint for training: '
                                                 f'{discriminator_ckpt_path}'))


    segmentor = define_model(net_params=cfg.model,
                             in_channels=num_bands,
                             out_classes=num_classes,
                             main_device=device,
                             devices=list(gpu_devices_dict.keys()),
                             state_dict_path=segmentor_ckpt_path,
                             state_dict_strict_load=state_dict_strict)

    discriminator = define_model(net_params=cfg.discriminator,
                                 in_channels=cfg.discriminator.in_channels,
                                 out_classes=num_classes,
                                 main_device=device,
                                 devices=list(gpu_devices_dict.keys()),
                                 state_dict_path=discriminator_ckpt_path,
                                 state_dict_strict_load=state_dict_strict
                                 )



    trn_dataloader, val_dataloader, tst_dataloader = create_dataloader(samples_folder=tiles_dir,
                                                                       batch_size=batch_size,
                                                                       eval_batch_size=eval_batch_size,
                                                                       gpu_devices_dict=gpu_devices_dict,
                                                                       sample_size=samples_size,
                                                                       dontcare_val=dontcare_val,
                                                                       crop_size=crop_size,
                                                                       num_bands=num_bands,
                                                                       min_annot_perc=min_annot_perc,
                                                                       attr_vals=attr_vals,
                                                                       scale=scale,
                                                                       cfg=cfg,
                                                                       dontcare2backgr=dontcare2backgr,
                                                                       debug=debug)
    steps_per_epoch = len(trn_dataloader)


    criterion = define_loss(loss_params=cfg.loss, class_weights=class_weights)
    criterion = criterion.to(device)
    optimizer_s = instantiate(cfg.optimizer, params=segmentor.parameters())
    optimizer_d = instantiate(cfg.optimizer, params=discriminator.parameters())
    s_scheduler = optim.lr_scheduler.OneCycleLR(optimizer_s, max_lr=0.01,
                                                steps_per_epoch=steps_per_epoch, epochs=num_epochs)
    d_scheduler = optim.lr_scheduler.OneCycleLR(optimizer_d, max_lr=0.01,
                                                steps_per_epoch=steps_per_epoch, epochs=num_epochs)

    for epoch in range(num_epochs):
        logging.info(f'\nEpoch {epoch}/{num_epochs - 1}\n' + "-" * len(f'Epoch {epoch}/{num_epochs - 1}'))
        trn_report = training(train_loader=trn_dataloader,
                              discriminator=discriminator,
                              segmentor=segmentor,
                              criterion=criterion,
                              optimizer_s=optimizer_s,
                              optimizer_d=optimizer_d,
                              s_scheduler=s_scheduler,
                              d_scheduler=d_scheduler,
                              num_classes=num_classes,
                              batch_size=batch_size,
                              device=device)



def evaluation(eval_loader,
               segmentor,
               num_classes,
               device,
               dataset='val',
               ):
    segmentor.eval()
    eval_metrics = create_metrics_dict(num_classes=num_classes)

    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} '
                                                                                      f'batches with {device.type}')):
            inputs = data['sat_img'].to(device)
            labels = data['map_img'].to(device).unsqueeze(1).float()
            outputs = segmentor(inputs)









def main(cfg: DictConfig) -> None:
    """
    Function to manage details about the training on segmentation task.

    -------
    1. Pre-processing
    2. Training process

    -------
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    # Preprocessing
    # HERE the code to do for the preprocessing for the segmentation

    # execute the name mode (need to be in this file for now)
    train(cfg)