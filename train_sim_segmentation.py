import torch
import shutil
import multiprocessing as mp
from time import time
from pathlib import Path
from typing import Sequence
from datetime import datetime
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.train_utils import get_num_samples, make_dataset_file_name
from utils import augmentation as aug, create_dataset
from utils.logger import get_logger
from utils.utils import get_key_def, get_device_ids, set_device

logging = get_logger(__name__)  # import logging

def create_dataloader(samples_folder: Path,
                      batch_size: int,
                      dontcare_val: int,
                      crop_size: int,
                      num_bands: int,
                      min_annot_perc: int,
                      attr_vals: Sequence,
                      scale: Sequence,
                      cfg: DictConfig,
                      dontcare2backgr: bool = False,
                      debug: bool = False):
    """
    Function to create dataloader objects for training, validation and test datasets.
    :param samples_folder: path to folder containting .hdf5 files if task is segmentation
    :param batch_size: (int) batch size
    :param dontcare_val: (int) value in label to be ignored during loss calculation
    :param crop_size: (int) size of one side of the square crop performed on original tile during training
    :param num_bands: (int) number of bands in imagery
    :param min_annot_perc: (int) minimum proportion of ground truth containing non-background information
    :param attr_vals: (Sequence)
    :param scale: (List) imagery data will be scaled to this min and max value (ex.: 0 to 1)
    :param cfg: (dict) Parameters found in the yaml config file.
    :param dontcare2backgr: (bool) if True, all dontcare values in label will be replaced with 0 (background value)
                            before training
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    if not samples_folder.is_dir():
        raise FileNotFoundError(f'Could not locate: {samples_folder}')
    experiment_name = samples_folder.stem
    if not len([f for f in samples_folder.glob('*.csv')]) >= 1:
        raise FileNotFoundError(f"Couldn't locate text file containing list of training data in {samples_folder}")

    num_samples, samples_weight = get_num_samples(samples_path=samples_folder,
                                                  params=cfg,
                                                  min_annot_perc=min_annot_perc,
                                                  attr_vals=attr_vals,
                                                  experiment_name=experiment_name
                                                  )
    if not num_samples['trn'] >= batch_size and num_samples['val'] >= batch_size:
        raise ValueError(f"Number of samples in .hd is less than batch size")
    logging.info(f"Number of samples : {num_samples}\n")
    dataset_constr = create_dataset.SegmentationDataset
    datasets = []

    for subset in ["trn"]:
        dataset_file, _ = make_dataset_file_name(experiment_name, min_annot_perc, subset, attr_vals)
        dataset_filepath = samples_folder / dataset_file
        datasets.append(dataset_constr(dataset_filepath, subset, num_bands,
                                       max_sample_count=num_samples[subset],
                                       radiom_transform=aug.compose_transforms(params=cfg,
                                                                               dataset=subset,
                                                                               aug_type='radiometric'),
                                       geom_transform=aug.compose_transforms(params=cfg,
                                                                             dataset=subset,
                                                                             aug_type='geometric',
                                                                             dontcare=dontcare_val,
                                                                             crop_size=crop_size),
                                       totensor_transform=aug.compose_transforms(params=cfg,
                                                                                 dataset=subset,
                                                                                 scale=scale,
                                                                                 dontcare2backgr=dontcare2backgr,
                                                                                 dontcare=dontcare_val,
                                                                                 aug_type='totensor'),
                                       debug=debug))
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                             len(samples_weight))
    trn_dataset = datasets[0]
    return trn_dataset, sampler

def train(cfg: DictConfig) -> None:
    """
    Function to train and validate a model for semantic segmentation.

    -------

    1. Model is instantiated and checkpoint is loaded from path, if provided in
       `your_config.yaml`.
    2. GPUs are requested according to desired amount of `num_gpus` and
       available GPUs.
    3. If more than 1 GPU is requested, model is cast to DataParallel model
    4. Dataloaders are created with `create_dataloader()`
    5. Loss criterion, optimizer and learning rate are set with
       `set_hyperparameters()` as requested in `config.yaml`.
    5. Using these hyperparameters, the application will try to minimize the
       loss on the training data and evaluate every epoch on the validation
       data.
    6. For every epoch, the application shows and logs the loss on "trn" and
       "val" datasets.
    7. For every epoch (if `batch_metrics: 1`), the application shows and logs
       the accuracy, recall and f-score on "val" dataset. Those metrics are
       also computed on each class.
    8. At the end of the training process, the application shows and logs the
       accuracy, recall and f-score on "tst" dataset. Those metrics are also
       computed on each class.

    -------
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")

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
    task = get_key_def('task',  cfg['general'], default='segmentation')
    dontcare_val = get_key_def("ignore_index", cfg['dataset'], default=-1)
    scale = get_key_def('scale_data', cfg['augmentation'], default=[0, 1])
    batch_metrics = get_key_def('batch_metrics', cfg['training'], default=None)
    crop_size = get_key_def('crop_size', cfg['augmentation'], default=None)

    # MODEL PARAMETERS
    checkpoint_stack = [""]
    class_weights = get_key_def('class_weights', cfg['dataset'], default=None)
    if cfg.loss.is_binary and not num_classes == 1:
        raise ValueError(f"Parameter mismatch: a binary loss was chosen for a {num_classes}-class task")
    elif not cfg.loss.is_binary and num_classes == 1:
        raise ValueError(f"Parameter mismatch: a multiclass loss was chosen for a 1-class (binary) task")
    del cfg.loss.is_binary  # prevent exception at instantiation

    # GPU PARAMETERS
    num_devices = get_key_def('num_gpus', cfg['training'], default=0)
    if num_devices and not num_devices >= 0:
        raise ValueError("\nMissing mandatory num gpus parameter")
    max_used_ram = get_key_def('max_used_ram', cfg['training'], default=15)
    max_used_perc = get_key_def('max_used_perc', cfg['training'], default=15)

    # LOGGING PARAMETERS
    run_name = get_key_def(['tracker', 'run_name'], cfg, default='gdl')
    tracker_uri = get_key_def(['tracker', 'uri'], cfg, default=None, expected_type=str)
    experiment_name = get_key_def('project_name', cfg['general'], default='gdl-training')

    # PARAMETERS FOR DATA INPUTS
    samples_size = get_key_def('chip_size', cfg['tiling'], default=256, expected_type=int)
    attr_vals = get_key_def("attribute_values", cfg['dataset'], default=-1)
    overlap = get_key_def('overlap_size', cfg['tiling'], default=0)
    min_annot_perc = get_key_def('min_annot_perc', cfg['tiling'], default=0)

    # Tiles Directory
    data_path = get_key_def('tiling_data_dir', cfg['tiling'], to_path=True, validate_path_exists=True)
    if not data_path.is_dir():
        raise FileNotFoundError(f'Could not locate data path {data_path}')
    # tiles_dir_name = make_tiles_dir_name(samples_size, num_bands)
    # tiles_dir = data_path / experiment_name / tiles_dir_name
    tiles_dir = data_path / experiment_name

    # visualization parameters
    vis_at_train = get_key_def('vis_at_train', cfg['visualization'], default=False)
    vis_at_eval = get_key_def('vis_at_evaluation', cfg['visualization'], default=False)
    vis_batch_range = get_key_def('vis_batch_range', cfg['visualization'], default=None)
    vis_at_checkpoint = get_key_def('vis_at_checkpoint', cfg['visualization'], default=False)
    ep_vis_min_thresh = get_key_def('vis_at_ckpt_min_ep_diff', cfg['visualization'], default=1)
    vis_at_ckpt_dataset = get_key_def('vis_at_ckpt_dataset', cfg['visualization'], 'val')
    colormap_file = get_key_def('colormap_file', cfg['visualization'], None)
    heatmaps = get_key_def('heatmaps', cfg['visualization'], False)
    heatmaps_inf = get_key_def('heatmaps', cfg['inference'], False)
    grid = get_key_def('grid', cfg['visualization'], False)
    mean = get_key_def('mean', cfg['augmentation']['normalization'])
    std = get_key_def('std', cfg['augmentation']['normalization'])
    vis_params = {'colormap_file': colormap_file, 'heatmaps': heatmaps, 'heatmaps_inf': heatmaps_inf, 'grid': grid,
                  'mean': mean, 'std': std, 'vis_batch_range': vis_batch_range, 'vis_at_train': vis_at_train,
                  'vis_at_eval': vis_at_eval, 'ignore_index': dontcare_val, 'inference_input_path': None}

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
    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')
    if dontcare_val < 0 and vis_batch_range:
        logging.warning(f'\nVisualization: expected positive value for ignore_index, got {dontcare_val}.'
                        f'\nWill be overridden to 255 during visualization only. Problems may occur.')

    # overwrite dontcare values in label if loss doens't implement ignore_index
    dontcare2backgr = False if 'ignore_index' in cfg.loss.keys() else True

    # Will check if batch size needs to be a lower value only if cropping samples during training
    calc_eval_bs = True if crop_size else False

    # Set device(s)
    gpu_devices_dict = get_device_ids(num_devices)
    device = set_device(gpu_devices_dict=gpu_devices_dict)

    trn_dataset, sampler = create_dataloader(samples_folder=tiles_dir,
                                       batch_size=batch_size,
                                       dontcare_val=dontcare_val,
                                       crop_size=crop_size,
                                       num_bands=num_bands,
                                       min_annot_perc=min_annot_perc,
                                       attr_vals=attr_vals,
                                       scale=scale,
                                       cfg=cfg,
                                       dontcare2backgr=dontcare2backgr,
                                       debug=debug)
    d = {}
    workers = [x for x in range(2, mp.cpu_count(), 2)]
    logging.info(f'Number of CPUs: {mp.cpu_count()}')
    logging.info(f'List of number of workers to test: {workers}')
    for num_workers in range(2, mp.cpu_count(), 2):
        trn_dataloader = DataLoader(trn_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    drop_last=False,
                                    pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(trn_dataloader, 0):
                pass
        end = time()
        end_test = end - start
        d[num_workers] = end_test
        logging.info(f'num_workers set to:{num_workers}\n'
                     f'train time is: {end_test:.2f} s')

    min_time = min(d.values())
    optimum_workers = [k for k, v in d.items() if v == min_time]
    logging.info(f'minimum time: {min_time:.2f} s, optimum_num_workers: {optimum_workers}')

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