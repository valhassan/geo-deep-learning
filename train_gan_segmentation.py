
from torchinfo import summary

from pathlib import Path
from typing import Sequence
from omegaconf import DictConfig

from utils.loss import define_loss
from utils.logger import get_logger

from models.model_choice import define_model
from utils.train_utils import create_dataloader, make_tiles_dir_name, make_dataset_file_name
from hydra.utils import to_absolute_path, instantiate
from utils.utils import get_key_def, get_device_ids, set_device



# Set the logging file
logging = get_logger(__name__)  # import logging




# Instantiate generator model or load from checkpoint
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

    # MODEL PARAMETERS
    segmentor_ckpt_path = get_key_def('state_dict_path', cfg['training'], default=None, expected_type=str)
    discriminator_ckpt_path = get_key_def('discriminator_ckpt', cfg['training'], default=None, expected_type=str)
    state_dict_strict = get_key_def('state_dict_strict_load', cfg['training'], default=True, expected_type=bool)

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
                                 in_channels=num_bands,
                                 out_classes=num_classes,
                                 main_device=device,
                                 devices=list(gpu_devices_dict.keys()),
                                 state_dict_path=discriminator_ckpt_path,
                                 state_dict_strict_load=state_dict_strict
                                 )

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    criterion = define_loss(loss_params=cfg.loss)
    criterion = criterion.to(device)
    optimizer_s = instantiate(cfg.optimizer, params=segmentor.parameters())
    optimizer_d = instantiate(cfg.optimizer, params=discriminator.parameters())

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)




    # summary(discriminator, input_size=(1, 3, 1024, 1024),
    #         col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable"))

    data_path = get_key_def('tiling_data_dir', cfg['tiling'], to_path=True, validate_path_exists=True)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    tiles_dir_name = make_tiles_dir_name(samples_size, num_bands)
    tiles_dir = data_path / experiment_name / tiles_dir_name



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