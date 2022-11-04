import shutil
import time
import torch
import numpy as np
from hydra.utils import instantiate
from torch import optim
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Sequence
from collections import OrderedDict
from omegaconf import DictConfig
from utils.logger import InformationLogger, tsv_line, get_logger, set_tracker
from utils.metrics import report_classification, create_metrics_dict, iou
from utils.train_utils import create_dataloader, make_tiles_dir_name, flatten_outputs, flatten_labels
from models.model_choice import read_checkpoint, define_model, adapt_checkpoint_to_dp_model
from utils.loss import verify_weights, define_loss
from utils.utils import gpu_stats, get_key_def, get_device_ids, set_device
from utils.visualization import vis_from_batch, vis_from_dataloader
# Set the logging file
logging = get_logger(__name__)  # import logging

def training(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          num_classes,
          batch_size,
          ep_idx,
          progress_log,
          device,
          scale,
          vis_params,
          debug=False
          ):
    """
    Train the model and return the metrics of the training epoch

    :param train_loader: training data loader
    :param model: model to train
    :param criterion: loss criterion
    :param optimizer: optimizer to use
    :param scheduler: learning rate scheduler
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param device: device used by pytorch (cpu ou cuda)
    :param scale: Scale to which values in sat img have been redefined. Useful during visualization
    :param vis_params: (Dict) Parameters useful during visualization
    :param debug: (bool) Debug mode
    :return: Updated training loss
    """
    model.train()
    train_metrics = create_metrics_dict(num_classes)

    for batch_index, data in enumerate(tqdm(train_loader, desc=f'Iterating train batches with {device.type}')):
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

        inputs = data['sat_img'].to(device)
        labels = data['map_img'].to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(inputs)
        # added for torchvision models that output an OrderedDict with outputs in 'out' key.
        # More info: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        if isinstance(outputs, OrderedDict):
            outputs = outputs['out']

        # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
        # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
        if vis_params['vis_batch_range'] and vis_params['vis_at_train']:
            min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
            if batch_index in range(min_vis_batch, max_vis_batch, increment):
                vis_path = progress_log.parent.joinpath('visualization')
                if ep_idx == 0:
                    logging.info(f'Visualizing on train outputs for batches in range {vis_params["vis_batch_range"]}. '
                                 f'All images will be saved to {vis_path}\n')
                vis_from_batch(vis_params, inputs, outputs,
                               batch_index=batch_index,
                               vis_path=vis_path,
                               labels=labels,
                               dataset='trn',
                               ep_num=ep_idx + 1,
                               scale=scale)

        loss = criterion(outputs, labels) if num_classes > 1 else criterion(outputs, labels.unsqueeze(1).float())

        train_metrics['loss'].update(loss.item(), batch_size)

        if device.type == 'cuda' and debug:
            res, mem = gpu_stats(device=device.index)
            logging.debug(OrderedDict(trn_loss=f"{train_metrics['loss'].val:.2f}",
                                      gpu_perc=f"{res['gpu']} %",
                                      gpu_RAM=f"{mem['used'] / (1024 ** 2):.0f}/{mem['total'] / (1024 ** 2):.0f} MiB",
                                      lr=optimizer.param_groups[0]['lr'],
                                      img=data['sat_img'].numpy().shape,
                                      smpl=data['map_img'].numpy().shape,
                                      bs=batch_size,
                                      out_vals=np.unique(outputs[0].argmax(dim=0).detach().cpu().numpy()),
                                      gt_vals=np.unique(labels[0].detach().cpu().numpy())))

        loss.backward()
        optimizer.step()

    scheduler.step()
    # if train_metrics["loss"].avg is not None:
    #     logging.info(f'Training Loss: {train_metrics["loss"].avg:.4f}')
    return train_metrics


def evaluation(eval_loader,
               model,
               criterion,
               num_classes,
               batch_size,
               ep_idx,
               progress_log,
               scale,
               vis_params,
               batch_metrics=None,
               dataset='val',
               device=None,
               debug=False,
               dontcare=-1):
    """
    Evaluate the model and return the updated metrics
    :param eval_loader: data loader
    :param model: model to evaluate
    :param criterion: loss criterion
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param scale: Scale to which values in sat img have been redefined. Useful during visualization
    :param vis_params: (Dict) Parameters useful during visualization
    :param batch_metrics: (int) Metrics computed every (int) batches. If left blank, will not perform metrics.
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :param debug: if True, debug functions will be performed
    :return: (dict) eval_metrics
    """
    eval_metrics = create_metrics_dict(num_classes)
    model.eval()

    for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} '
                                                                                  f'batches with {device.type}')):
        progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

        with torch.no_grad():
            inputs = data['sat_img'].to(device)
            labels = data['map_img'].to(device)

            labels_flatten = flatten_labels(labels)

            outputs = model(inputs)
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
            # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
            if vis_params['vis_batch_range'] and vis_params['vis_at_eval']:
                min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                if batch_index in range(min_vis_batch, max_vis_batch, increment):
                    vis_path = progress_log.parent.joinpath('visualization')
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

            loss = criterion(outputs, labels.unsqueeze(1).float())

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
            elif (dataset == 'tst'):
                a, segmentation = torch.max(outputs_flatten, dim=1)
                eval_metrics = iou(segmentation, labels_flatten, batch_size, num_classes, eval_metrics, dontcare)
                eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                     ignore_index=dontcare)

            logging.debug(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))

            if debug and device.type == 'cuda':
                res, mem = gpu_stats(device=device.index)
                logging.debug(OrderedDict(
                    device=device, gpu_perc=f"{res['gpu']} %",
                    gpu_RAM=f"{mem['used']/(1024**2):.0f}/{mem['total']/(1024**2):.0f} MiB"
                ))

    if eval_metrics['loss'].avg:
        logging.info(f"\n{dataset} Loss: {eval_metrics['loss'].avg:.4f}")
    if batch_metrics is not None or dataset == 'tst':
        logging.info(f"\n{dataset} precision: {eval_metrics['precision'].avg:.4f}")
        logging.info(f"\n{dataset} recall: {eval_metrics['recall'].avg:.4f}")
        logging.info(f"\n{dataset} fscore: {eval_metrics['fscore'].avg:.4f}")
        logging.info(f"\n{dataset} iou: {eval_metrics['iou'].avg:.4f}")

    return eval_metrics


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
    modalities = get_key_def('modalities', cfg['dataset'], default=("red", "blue", "green"), expected_type=Sequence)
    num_bands = len(modalities)
    batch_size = get_key_def('batch_size', cfg['training'], expected_type=int)
    eval_batch_size = get_key_def('eval_batch_size', cfg['training'], expected_type=int, default=batch_size)
    num_epochs = get_key_def('max_epochs', cfg['training'], expected_type=int)

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
    optimizer = get_key_def('optimizer_name', cfg['optimizer'], default='adam', expected_type=str)  # TODO change something to call the function
    pretrained = get_key_def('pretrained', cfg['model'], default=True, expected_type=(bool, str))
    train_state_dict_path = get_key_def('state_dict_path', cfg['training'], default=None, expected_type=str)
    state_dict_strict = get_key_def('state_dict_strict_load', cfg['training'], default=True, expected_type=bool)
    dropout_prob = get_key_def('factor', cfg['scheduler']['params'], default=None, expected_type=float)
    # if error
    if train_state_dict_path and not Path(train_state_dict_path).is_file():
        raise logging.critical(
            FileNotFoundError(f'\nCould not locate pretrained checkpoint for training: {train_state_dict_path}')
        )
    if class_weights:
        verify_weights(num_classes, class_weights)
    # Read the concatenation point if requested model is deeplabv3 dualhead
    conc_point = get_key_def('conc_point', cfg['model'], None)
    step_size = get_key_def('step_size', cfg['scheduler']['params'], default=4, expected_type=int)
    gamma = get_key_def('gamma', cfg['scheduler']['params'], default=0.9, expected_type=float)

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
    tiles_dir_name = make_tiles_dir_name(samples_size, num_bands)
    tiles_dir = data_path / experiment_name / tiles_dir_name

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

    # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    model = define_model(
        net_params=cfg.model,
        in_channels=num_bands,
        out_classes=num_classes,
        main_device=device,
        devices=list(gpu_devices_dict.keys()),
        state_dict_path=train_state_dict_path,
        state_dict_strict_load=state_dict_strict,
    )
    criterion = define_loss(loss_params=cfg.loss, class_weights=class_weights)
    criterion = criterion.to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    logging.info(f'\nInstantiated {cfg.model._target_} model with {num_bands} input channels and {num_classes} output '
                 f'classes.')

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

    # Save tracking
    set_tracker(mode='train', type='mlflow', task='segmentation', experiment_name=experiment_name, run_name=run_name,
                tracker_uri=tracker_uri, params=cfg,
                keys2log=['general', 'training', 'dataset', 'model', 'optimizer', 'scheduler', 'augmentation'])
    trn_log, val_log, tst_log = [InformationLogger(dataset) for dataset in ['trn', 'val', 'tst']]

    since = time.time()
    best_loss = 999
    last_vis_epoch = 0

    progress_log = output_path / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header

    # VISUALIZATION: generate pngs of inputs, labels and outputs
    if vis_batch_range is not None:
        # Make sure user-provided range is a tuple with 3 integers (start, finish, increment).
        # Check once for all visualization tasks.
        if not len(vis_batch_range) == 3 and all(isinstance(x, int) for x in vis_batch_range):
            raise logging.critical(
                ValueError(f'\nVis_batch_range expects three integers in a list: start batch, end batch, increment.'
                           f'Got {vis_batch_range}')
            )
        vis_at_init_dataset = get_key_def('vis_at_init_dataset', cfg['visualization'], 'val')

        # Visualization at initialization. Visualize batch range before first eopch.
        if get_key_def('vis_at_init', cfg['visualization'], False):
            logging.info(f'\nVisualizing initialized model on batch range {vis_batch_range} '
                         f'from {vis_at_init_dataset} dataset...\n')
            vis_from_dataloader(vis_params=vis_params,
                                eval_loader=val_dataloader if vis_at_init_dataset == 'val' else tst_dataloader,
                                model=model,
                                ep_num=0,
                                output_path=output_path,
                                dataset=vis_at_init_dataset,
                                scale=scale,
                                device=device,
                                vis_batch_range=vis_batch_range)

    for epoch in range(0, num_epochs):
        logging.info(f'\nEpoch {epoch}/{num_epochs - 1}\n' + "-" * len(f'Epoch {epoch}/{num_epochs - 1}'))
        # creating trn_report
        trn_report = training(train_loader=trn_dataloader,
                              model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              num_classes=num_classes,
                              batch_size=batch_size,
                              ep_idx=epoch,
                              progress_log=progress_log,
                              device=device,
                              scale=scale,
                              vis_params=vis_params,
                              debug=debug)
        if 'trn_log' in locals():  # only save the value if a tracker is setup
            trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])
        val_report = evaluation(eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='val',
                                device=device,
                                scale=scale,
                                vis_params=vis_params,
                                debug=debug,
                                dontcare=dontcare_val)
        val_loss = val_report['loss'].avg
        if 'val_log' in locals():  # only save the value if a tracker is setup
            if batch_metrics is not None:
                val_log.add_values(val_report, epoch)
            else:
                val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            logging.info("\nSave checkpoint with a validation loss of {:.4f}".format(val_loss))  # only allow 4 decimals
            # create the checkpoint file
            checkpoint_tag = checkpoint_stack.pop()
            filename = output_path.joinpath(checkpoint_tag)
            if filename.is_file():
                filename.unlink()
            checkpoint_tag = f'{experiment_name}_{num_classes}_{"_".join(modalities)}_{val_loss:.2f}.pth.tar'
            filename = output_path.joinpath(checkpoint_tag)
            checkpoint_stack.append(checkpoint_tag)
            best_loss = val_loss
            # More info:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'params': cfg,
                        'model_state_dict': state_dict,
                        'best_loss': best_loss,
                        'optimizer_state_dict': optimizer.state_dict()}, filename)

            # VISUALIZATION: generate pngs of img samples, labels and outputs as alternative to follow training
            if vis_batch_range is not None and vis_at_checkpoint and epoch - last_vis_epoch >= ep_vis_min_thresh:
                if last_vis_epoch == 0:
                    logging.info(f'\nVisualizing with {vis_at_ckpt_dataset} dataset samples on checkpointed model for'
                                 f'batches in range {vis_batch_range}')
                vis_from_dataloader(vis_params=vis_params,
                                    eval_loader=val_dataloader if vis_at_ckpt_dataset == 'val' else tst_dataloader,
                                    model=model,
                                    ep_num=epoch+1,
                                    output_path=output_path,
                                    dataset=vis_at_ckpt_dataset,
                                    scale=scale,
                                    device=device,
                                    vis_batch_range=vis_batch_range)
                last_vis_epoch = epoch

        cur_elapsed = time.time() - since
        # logging.info(f'\nCurrent elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')

    # load checkpoint model and evaluate it on test dataset.
    if int(cfg['general']['max_epochs']) > 0:   # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = read_checkpoint(filename)
        checkpoint = adapt_checkpoint_to_dp_model(checkpoint, model)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=num_epochs,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='tst',
                                scale=scale,
                                vis_params=vis_params,
                                device=device,
                                dontcare=dontcare_val)
        if 'tst_log' in locals():  # only save the value if a tracker is setup
            tst_log.add_values(tst_report, num_epochs)


def main(cfg: DictConfig) -> None:
    """
    Function to manage details about the training on segmentation task.

    -------
    1. Pre-processing TODO
    2. Training process

    -------
    :param cfg: (dict) Parameters found in the yaml config file.
    """
    # Preprocessing
    # HERE the code to do for the preprocessing for the segmentation

    # execute the name mode (need to be in this file for now)
    train(cfg)
