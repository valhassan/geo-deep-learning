import time
import torch
import shutil
from tqdm import tqdm
from torch import optim
from pathlib import Path
from typing import Sequence
from datetime import datetime
from omegaconf import DictConfig
from collections import OrderedDict
from utils.loss import define_loss
from utils.logger import get_logger, InformationLogger, set_tracker
from models.model_choice import define_model, read_checkpoint, adapt_checkpoint_to_dp_model
from utils.metrics import report_classification, create_metrics_dict, iou
from utils.train_utils import create_dataloader, make_tiles_dir_name, \
    make_dataset_file_name, flatten_outputs, flatten_labels
from hydra.utils import to_absolute_path, instantiate
from utils.utils import get_key_def, get_device_ids, set_device, gpu_stats
from utils.visualization import vis_from_batch, vis_from_dataloader
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
        train_metrics['segmentor-loss'].update(loss_g.item(), batch_size)
        train_metrics['discriminator-loss'].update(loss_d.item(), batch_size)
        train_metrics['real-score-critic'].update(d_x, batch_size)
        train_metrics['fake-score-critic'].update(d_g_z2, batch_size)
    return train_metrics


def evaluation(eval_loader,
               segmentor,
               criterion,
               num_classes,
               batch_size,
               vis_params,
               output_path,
               ep_idx,
               scale,
               batch_metrics=None,
               device=None,
               dataset='val',
               debug=False,
               dontcare=-1,
               ):
    segmentor.eval()
    eval_metrics = create_metrics_dict(num_classes=num_classes)

    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} '
                                                                                      f'batches with {device.type}')):
            inputs = data['sat_img'].to(device)
            labels = data['map_img'].to(device)
            labels_ = labels.unsqueeze(1).float()

            outputs = segmentor(inputs)

            # vis_batch_range: range of batches to perform visualization on.
            # vis_at_eval: (bool) if True, will perform visualization at eval time.
            if vis_params['vis_batch_range'] and vis_params['vis_at_eval']:
                min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                if batch_index in range(min_vis_batch, max_vis_batch, increment):
                    vis_path = output_path.joinpath('visualization')
                    if ep_idx == 0 and batch_index == min_vis_batch:
                        logging.info(f'\nVisualizing on {dataset} outputs for batches in range '
                                     f'{vis_params["vis_batch_range"]} images will be saved to {vis_path}\n')
                    vis_from_batch(vis_params, inputs, outputs,
                                   batch_index=batch_index,
                                   vis_path=vis_path,
                                   labels=labels,
                                   dataset=dataset,
                                   ep_num=ep_idx + 1,
                                   scale=scale)
            outputs_flatten = flatten_outputs(outputs, num_classes)
            labels_flatten = flatten_labels(labels)
            loss = criterion(outputs, labels_)
            eval_metrics['loss'].update(loss.item(), batch_size)
            if (dataset == 'val') and (batch_metrics is not None):
                # Compute metrics every n batches. Time consuming.
                if not batch_metrics <= len(eval_loader):
                    logging.error(f"\nBatch_metrics ({batch_metrics}) is smaller than batch size "
                                  f"{len(eval_loader)}. Metrics in validation loop won't be computed")
                if (batch_index + 1) % batch_metrics == 0:  # +1 to skip val loop at very beginning
                    a, segmentation = torch.max(outputs_flatten, dim=1)
                    eval_metrics = iou(segmentation, labels_flatten, batch_size, num_classes, eval_metrics, dontcare)
                    eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                         ignore_index=dontcare)
            elif dataset == 'tst':
                a, segmentation = torch.max(outputs_flatten, dim=1)
                eval_metrics = iou(segmentation, labels_flatten, batch_size, num_classes, eval_metrics, dontcare)
                eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                     ignore_index=dontcare)
            logging.debug(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))
            if debug and device.type == 'cuda':
                res, mem = gpu_stats(device=device.index)
                logging.debug(OrderedDict(
                    device=device, gpu_perc=f"{res['gpu']} %",
                    gpu_RAM=f"{mem['used'] / (1024 ** 2):.0f}/{mem['total'] / (1024 ** 2):.0f} MiB"))
        if eval_metrics['loss'].avg:
            logging.info(f"\n{dataset} Loss: {eval_metrics['loss'].avg:.4f}")
        if batch_metrics is not None or dataset == 'tst':
            logging.info(f"\n{dataset} precision: {eval_metrics['precision'].avg:.4f}")
            logging.info(f"\n{dataset} recall: {eval_metrics['recall'].avg:.4f}")
            logging.info(f"\n{dataset} fscore: {eval_metrics['fscore'].avg:.4f}")
            logging.info(f"\n{dataset} iou: {eval_metrics['iou'].avg:.4f}")
    return eval_metrics


def train(cfg: DictConfig) -> None:

    # MANDATORY PARAMETERS
    class_keys = len(get_key_def('classes_dict', cfg['dataset']).keys())
    num_classes = class_keys if class_keys == 1 else class_keys + 1  # +1 for background(multiclass mode)
    modalities = get_key_def('bands', cfg['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    num_bands = len(modalities)
    batch_size = get_key_def('batch_size', cfg['training'], expected_type=int)
    eval_batch_size = get_key_def('eval_batch_size', cfg['training'], expected_type=int, default=batch_size)
    num_epochs = get_key_def('max_epochs', cfg['training'], expected_type=int)
    early_stop_epoch = get_key_def('min_epochs', cfg['training'], expected_type=int, default=int(num_epochs * 0.5))

    # OPTIONAL PARAMETERS
    debug = get_key_def('debug', cfg)
    if debug:
        logging.warning(f'\nDebug mode activated')
    task = get_key_def('task', cfg['general'], default='segmentation')
    dontcare_val = get_key_def("ignore_index", cfg['dataset'], default=-1)
    scale = get_key_def('scale_data', cfg['augmentation'], default=[0, 1])
    batch_metrics = get_key_def('batch_metrics', cfg['training'], default=None)
    crop_size = get_key_def('crop_size', cfg['augmentation'], default=None)
    # overwrite dontcare values in label if loss doens't implement ignore_index
    dontcare2backgr = False if 'ignore_index' in cfg.loss.keys() else True

    # LOGGING PARAMETERS
    run_name = get_key_def(['tracker', 'run_name'], cfg, default='gdl')
    tracker_uri = get_key_def(['tracker', 'uri'], cfg, default=None, expected_type=str)
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

    # automatic model naming with unique id for each training
    config_path = None
    for list_path in cfg.general.config_path:
        if list_path['provider'] == 'main':
            config_path = list_path['path']
    output_path = tiles_dir.joinpath('model') / run_name
    if output_path.is_dir():
        last_mod_time_suffix = datetime.fromtimestamp(output_path.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        archive_output_path = output_path.parent / f"{output_path.stem}_{last_mod_time_suffix}"
        shutil.move(output_path, archive_output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    logging.info(f'\n Training artifacts will be saved to: {output_path}')

    # VISUALIZATION PARAMETERS
    colormap_file = get_key_def('colormap_file', cfg['visualization'], None)
    heatmaps = get_key_def('heatmaps', cfg['visualization'], False)
    heatmaps_inf = get_key_def('heatmaps', cfg['inference'], False)
    grid = get_key_def('grid', cfg['visualization'], False)
    mean = get_key_def('mean', cfg['augmentation']['normalization'])
    std = get_key_def('std', cfg['augmentation']['normalization'])
    vis_batch_range = get_key_def('vis_batch_range', cfg['visualization'], default=None)
    vis_at_train = get_key_def('vis_at_train', cfg['visualization'], default=False)
    vis_at_eval = get_key_def('vis_at_evaluation', cfg['visualization'], default=False)
    vis_at_checkpoint = get_key_def('vis_at_checkpoint', cfg['visualization'], default=False)
    ep_vis_min_thresh = get_key_def('vis_at_ckpt_min_ep_diff', cfg['visualization'], default=1)
    vis_at_ckpt_dataset = get_key_def('vis_at_ckpt_dataset', cfg['visualization'], 'val')

    vis_params = {'colormap_file': colormap_file, 'heatmaps': heatmaps, 'heatmaps_inf': heatmaps_inf, 'grid': grid,
                  'mean': mean, 'std': std, 'vis_batch_range': vis_batch_range, 'vis_at_train': vis_at_train,
                  'vis_at_eval': vis_at_eval, 'ignore_index': dontcare_val, 'inference_input_path': None}

    # Save tracking
    set_tracker(mode='train', type='mlflow', task='segmentation', experiment_name=experiment_name, run_name=run_name,
                tracker_uri=tracker_uri, params=cfg,
                keys2log=['general', 'training', 'dataset', 'model', 'optimizer', 'scheduler', 'augmentation'])
    trn_log, val_log, tst_log = [InformationLogger(dataset) for dataset in ['trn', 'val', 'tst']]


    # MODEL PARAMETERS
    segmentor_ckpt_path = get_key_def('state_dict_path', cfg['training'], default=None, expected_type=str)
    discriminator_ckpt_path = get_key_def('discriminator_ckpt', cfg['training'], default=None, expected_type=str)
    state_dict_strict = get_key_def('state_dict_strict_load', cfg['training'], default=True, expected_type=bool)
    class_weights = get_key_def('class_weights', cfg['dataset'], default=None)
    s_checkpoint_stack = [""]
    d_checkpoint_stack = [""]
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

    logging.info(f'\nCreating dataloaders from data in {tiles_dir}...')
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
    since = time.time()
    best_loss = 999
    early_stop_count = 0
    last_vis_epoch = 0
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
        if 'trn_log' in locals():  # only save the value if a tracker is setup
            trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])
        val_report = evaluation(eval_loader=val_dataloader,
                                segmentor=segmentor,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                vis_params=vis_params,
                                output_path=output_path,
                                ep_idx=epoch,
                                scale=scale,
                                batch_metrics=batch_metrics,
                                device=device,
                                dataset='val',
                                debug=debug,
                                dontcare=dontcare_val)
        val_loss = val_report['loss'].avg
        if 'val_log' in locals():  # only save the value if a tracker is setup
            if batch_metrics is not None:
                val_log.add_values(val_report, epoch)
            else:
                val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            logging.info("\nSave checkpoints with a validation loss of {:.4f}".format(val_loss))  # only allow 4 decimals
            # create the checkpoint file
            s_checkpoint_tag = s_checkpoint_stack.pop()
            d_checkpoint_tag = d_checkpoint_stack.pop()
            s_filename = output_path.joinpath(s_checkpoint_tag)
            d_filename = output_path.joinpath(d_checkpoint_tag)
            if s_filename.is_file():
                s_filename.unlink()
            if d_filename.is_file():
                d_filename.unlink()
            s_checkpoint_tag = f'S_{experiment_name}_{num_classes}_{"_".join(map(str, modalities))}_{val_loss:.2f}.pth.tar'
            d_checkpoint_tag = f'D_{experiment_name}_{num_classes}_{"_".join(map(str, modalities))}_{val_loss:.2f}.pth.tar'
            s_filename = output_path.joinpath(s_checkpoint_tag)
            d_filename = output_path.joinpath(d_checkpoint_tag)
            s_checkpoint_stack.append(s_checkpoint_tag)
            d_checkpoint_stack.append(d_checkpoint_tag)
            best_loss = val_loss
            early_stop_count = 0
            segmentor_state_dict = segmentor.module.state_dict() if num_devices > 1 else segmentor.state_dict()
            discriminator_state_dict = discriminator.module.state_dict() if num_devices > 1 else discriminator.state_dict()
            torch.save({'epoch': epoch,
                        'params': cfg,
                        'model_state_dict': segmentor_state_dict,
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer_s.state_dict()}, s_filename)
            torch.save({'epoch': epoch,
                        'params': cfg,
                        'model_state_dict': discriminator_state_dict,
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer_d.state_dict()}, d_filename)

            # VISUALIZATION: generate pngs of img samples, labels and outputs as alternative to follow training
            if vis_batch_range is not None and vis_at_checkpoint and epoch - last_vis_epoch >= ep_vis_min_thresh:
                if last_vis_epoch == 0:
                    logging.info(f'\nVisualizing with {vis_at_ckpt_dataset} dataset samples on checkpointed model for'
                                 f'batches in range {vis_batch_range}')
                vis_from_dataloader(vis_params=vis_params,
                                    eval_loader=val_dataloader if vis_at_ckpt_dataset == 'val' else tst_dataloader,
                                    model=segmentor,
                                    ep_num=epoch + 1,
                                    output_path=output_path,
                                    dataset=vis_at_ckpt_dataset,
                                    scale=scale,
                                    device=device,
                                    vis_batch_range=vis_batch_range)
                last_vis_epoch = epoch
        else:
            early_stop_count += 1
        cur_elapsed = time.time() - since
        if early_stop_count >= early_stop_epoch:
            logging.info(f'Early stopping after patience elapsed!')
            break
    if int(cfg['general']['max_epochs']) > 0:   # if num_epochs is set to 0, model is loaded to evaluate on test set
        s_checkpoint = read_checkpoint(s_filename)
        s_checkpoint = adapt_checkpoint_to_dp_model(s_checkpoint, segmentor)
        segmentor.load_state_dict(state_dict=s_checkpoint['model_state_dict'])

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                segmentor=segmentor,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                vis_params=vis_params,
                                output_path=output_path,
                                ep_idx=num_epochs,
                                scale=scale,
                                batch_metrics=batch_metrics,
                                device=device,
                                dataset='tst',
                                debug=debug,
                                dontcare=dontcare_val)
        if 'tst_log' in locals():  # only save the value if a tracker is setup
            tst_log.add_values(tst_report, num_epochs)


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