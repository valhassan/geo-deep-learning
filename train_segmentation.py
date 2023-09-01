import os
import shutil
import time
import random
import numpy as np
import torch

from hydra.utils import instantiate
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Sequence
from collections import OrderedDict
from omegaconf import DictConfig
from lightning.fabric import Fabric
from lightning.fabric.strategies.ddp import DDPStrategy
from utils.augmentation import Transforms
from utils.logger import InformationLogger, tsv_line, get_logger, set_tracker
from utils.metrics import create_metrics_dict, iou
from utils.train_utils import EarlyStopping, prepare_dataset, prepare_dataloader, freeze_model_parts
from models.model_choice import read_checkpoint, define_model, adapt_checkpoint_to_dp_model
from utils.loss import verify_weights, define_loss
from utils.utils import get_key_def
from utils.visualization import vis_from_batch, vis_from_dataloader
# Set the logging file
logging = get_logger(__name__)  # import logging

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    
    
seed_everything(42)
class Trainer:
    def __init__(self,
                 cfg: DictConfig) -> None:
        """ 
        Train and validate a model for semantic segmentation.
        
        :param cfg: (dict) Parameters found in the yaml config file.
        
        """
        self.cfg = cfg
        
        
         # TRAIN ACCELERATOR AND STRATEGY PARAMETERS
        num_devices = get_key_def('num_gpus', cfg['training'], default=0)
        num_nodes = get_key_def('num_nodes', self.cfg['training'], default=1)
        num_tasks = get_key_def('num_tasks', self.cfg['training'], default=0)
        self.strategy = get_key_def("strategy", self.cfg['training'], default="dp")
        precision = get_key_def("precision", self.cfg['training'], default="32-true")
        if num_devices and not num_devices >= 0:
            raise ValueError("\nMissing mandatory num gpus parameter")
        if self.strategy == "dp":
            num_tasks = num_devices
        if self.strategy == "ddp":
            self.strategy = DDPStrategy(find_unused_parameters=True)
        accelerator = get_key_def("accelerator", self.cfg['training'], default="cuda")
        self.fabric = Fabric(accelerator=accelerator, devices=num_tasks, 
                             num_nodes=num_nodes, strategy=self.strategy, precision=precision)
        self.fabric.launch()
        
        # Tiles Directory
        data_path = get_key_def('tiling_data_dir', self.cfg['tiling'], to_path=True, validate_path_exists=True)
        if not data_path.is_dir():
            raise FileNotFoundError(f'Could not locate data path {data_path}')
        self.experiment_name = get_key_def('project_name', self.cfg['general'], default='gdl-training')
        self.tiles_dir = data_path / self.experiment_name
        
        # MLflow LOGGING PARAMETERS
        self.run_name = get_key_def(['tracker', 'run_name'], self.cfg, default='gdl')
        self.tracker_uri = get_key_def(['tracker', 'uri'], self.cfg, default=None, expected_type=str)
        
        # OPTIONAL PARAMETERS
        self.debug = get_key_def('debug', self.cfg)
        self.dontcare_val = get_key_def("ignore_index", self.cfg['dataset'], default=-1)
        self.crop_size = get_key_def('crop_size', self.cfg['augmentation'], default=None)
        self.scale = get_key_def('scale_data', self.cfg['augmentation'], default=[0, 1])
        self.num_workers = get_key_def('num_workers', self.cfg['training'], default=0, expected_type=int)
        # overwrite dontcare values in label if loss doens't implement ignore_index
        self.dontcare2backgr = False if 'ignore_index' in self.cfg.loss.keys() else True
        self.compute_sampler_weights = get_key_def('compute_sampler_weights', self.cfg['training'], 
                                                   default=False, expected_type=bool)
        
        # MANDATORY PARAMETERS
        self.num_devices = num_devices
        self.batch_size = get_key_def('batch_size', self.cfg['training'], expected_type=int)
        self.modalities = get_key_def('bands', self.cfg['dataset'], 
                                 default=("red", "blue", "green"), expected_type=Sequence)
        self.num_bands = len(self.modalities)
        self.num_epochs = get_key_def('max_epochs', self.cfg['training'], expected_type=int)
        class_keys = len(get_key_def('classes_dict', self.cfg['dataset']).keys())
        self.num_classes = class_keys if class_keys == 1 else class_keys + 1  # +1 for background(multiclass mode)
        
        # PARAMETERS FOR DATA INPUTS
        self.min_annot_perc = get_key_def('min_annot_perc', self.cfg['tiling'], default=0)
        self.attr_vals = get_key_def("attribute_values", self.cfg['dataset'], default=-1)
        
        # MODEL PARAMETERS
        self.freeze_model_parts = get_key_def('freeze_parts', self.cfg, default=None)
        self.train_state_dict_path = get_key_def('state_dict_path', self.cfg['training'], 
                                                 default=None, expected_type=str)
        if self.train_state_dict_path and not Path(self.train_state_dict_path).is_file():
            raise (FileNotFoundError(f"Could not locate pretrained checkpoint for training:" 
                                     f"{self.train_state_dict_path}"))
        self.state_dict_strict = get_key_def('state_dict_strict_load', self.cfg['training'], 
                                             default=True, expected_type=bool)
        self.class_weights = get_key_def('class_weights', self.cfg['dataset'], default=None)
        if self.class_weights:
            verify_weights(self.num_classes, self.class_weights)
            
        if self.cfg.loss.is_binary and not self.num_classes == 1:
            raise ValueError(f"Parameter mismatch: a binary loss was chosen for a {self.num_classes}-class task")
        elif not self.cfg.loss.is_binary and self.num_classes == 1:
            raise ValueError(f"Parameter mismatch: a multiclass loss was chosen for a 1-class (binary) task")
        del self.cfg.loss.is_binary  # prevent exception at instantiation
        self.early_stop_epoch = get_key_def('min_epochs', self.cfg['training'], expected_type=int, 
                                            default=int(self.num_epochs * 0.5))
        
        # VISUALIZATION PARAMETERS
        self.vis_batch_range = get_key_def('vis_batch_range', self.cfg['visualization'], default=None)
        self.vis_at_checkpoint = get_key_def('vis_at_checkpoint', self.cfg['visualization'], default=False)
        self.ep_vis_min_thresh = get_key_def('vis_at_ckpt_min_ep_diff', self.cfg['visualization'], default=1)
        self.vis_at_ckpt_dataset = get_key_def('vis_at_ckpt_dataset', self.cfg['visualization'], 'val')
        self.batch_metrics = get_key_def('batch_metrics', cfg['training'], default=None)
        vis_at_train = get_key_def('vis_at_train', self.cfg['visualization'], default=False)
        vis_at_eval = get_key_def('vis_at_evaluation', self.cfg['visualization'], default=False)
         
        colormap_file = get_key_def('colormap_file', cfg['visualization'], None)
        heatmaps = get_key_def('heatmaps', cfg['visualization'], False)
        heatmaps_inf = get_key_def('heatmaps', cfg['inference'], False)
        grid = get_key_def('grid', cfg['visualization'], False)
        
        
        # AUGUMENTATION PARAMETERS
        mean = get_key_def('mean', cfg['augmentation']['normalization'], default=[1.0] * self.num_bands)
        std = get_key_def('std', cfg['augmentation']['normalization'], default=[1.0] * self.num_bands)
        self.transforms = Transforms(mean=mean, std=std)
        
        self.vis_params = {'colormap_file': colormap_file, 'heatmaps': heatmaps, 
                           'heatmaps_inf': heatmaps_inf, 'grid': grid,
                           'mean': mean, 'std': std, 'vis_batch_range': self.vis_batch_range, 
                           'vis_at_train': vis_at_train, 'vis_at_eval': vis_at_eval, 
                           'ignore_index': self.dontcare_val, 'inference_input_path': None}

        if self.debug:
            logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                            f'cause delays in execution.')
        if self.dontcare_val < 0 and self.vis_batch_range:
            logging.warning(f'\nVisualization: expected positive value for ignore_index, got {self.dontcare_val}.'
                            f'\nWill be overridden to 255 during visualization only. Problems may occur.')


    def train_loop(self,
                   model: torch.nn.Module, 
                   train_loader: torch.utils.data.DataLoader, 
                   criterion: torch.nn.modules.loss ,
                   optimizer: torch.optim.Optimizer, 
                   scheduler: dict, 
                   num_classes: int, 
                   batch_size: int, 
                   epoch: int, 
                   vis_output,
                   device: torch.device, 
                   scale: list[int], 
                   vis_params,
                   aux_output: bool = False
                   ):
        """
        Train loop, returns train metrics

        Args:
            model: model to train
            train_loader: train dataloader
            criterion: loss criterion
            optimizer: train optimizer 
            scheduler: learning rate scheduler
            num_classes: number of classes
            batch_size: number of patches (batched)
            epoch: epoch number
            vis_output: output path for artifacts
            device: device type
            scale: ranges of values to scale images
            vis_params: dict containing visualization params

        Returns:
            dict: train metrics
        """
        model.train()
        train_metrics = create_metrics_dict(num_classes)

        lr_scheduler = scheduler["lr_scheduler"]
        onecycle_scheduler = scheduler["onecycle_scheduler"]
        plateau_scheduler = scheduler["plateau_scheduler"]
        
        for batch_index, data in enumerate(tqdm(train_loader, desc=f'Iterating train batches with {device.type}')):
            inputs = data['sat_img']
            labels = data['map_img']
            
            inputs, labels = self.transforms.train_transform(inputs, labels)
            labels = labels.squeeze(1).long()

            # forward
            optimizer.zero_grad()
            if aux_output:
                outputs, outputs_aux = model(inputs)
            else:
                outputs = model(inputs)
           # added for torchvision models that output an OrderedDict with outputs in 'out' key.
             # More info: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
            # vis_batch_train: Always set to false as this might cause problems in distributed training
            if self.fabric.is_global_zero:
                if vis_params['vis_batch_range'] and vis_params['vis_at_train']:
                    min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                    if batch_index in range(min_vis_batch, max_vis_batch, increment):
                        vis_path = vis_output.joinpath('visualization')
                        if epoch == 0:
                            logging.info(f"Visualizing on train outputs for batches in range" 
                                         f"{vis_params['vis_batch_range']}. All images will be saved to {vis_path}\n")
                        vis_from_batch(vis_params, inputs, outputs,
                                       batch_index=batch_index,
                                       vis_path=vis_path,
                                       labels=labels,
                                       dataset='trn',
                                       ep_num=epoch + 1,
                                       scale=scale,
                                       device=device)
            self.fabric.barrier()
            with self.fabric.autocast():
                if aux_output:
                    loss_main = criterion(outputs, labels)
                    loss_aux = criterion(outputs_aux, labels)
                    loss = 0.4 * loss_aux + loss_main
                else:
                    loss = criterion(outputs, labels)
            self.fabric.all_reduce(loss, reduce_op="mean")
            train_metrics['loss'].update(loss.item(), batch_size)
            self.fabric.backward(loss)
            optimizer.step()
            if onecycle_scheduler and not plateau_scheduler:
                lr_scheduler.step()
        if not onecycle_scheduler and not plateau_scheduler:
            lr_scheduler.step()
        if not plateau_scheduler:
            train_metrics['lr'].update(optimizer.param_groups[0]['lr'])
        
        logging.info(f'trn Loss: {train_metrics["loss"].avg:.4f}')
        
        return train_metrics
    
    def evaluation_loop(self,
                        model: torch.nn.Module,
                        eval_loader: torch.utils.data.DataLoader,
                        criterion: torch.nn.modules.loss,
                        epoch: int,
                        num_classes: int,
                        batch_size: int,
                        scale: list[int],
                        vis_params,
                        vis_output,
                        device: torch.device,
                        dontcare: int,
                        dataset: str,
                        batch_metrics):
        """
        Evaluate loop, returns evaluation metrics

        Args:
            model: model to train
            eval_loader: validation/test dataloader
            criterion: loss criterion
            epoch: epoch number
            num_classes: number of classes
            batch_size: number of patches (batched)
            scale: range of values to unscale images
            vis_params: dict containing visualization params
            vis_output: output path for artifacts
            device: device type
            dontcare: class value to ignore. Defaults to -1.
            dataset: dataset type. Defaults to 'val'.
            batch_metrics: metrics computed every batch. Defaults to None.

        Returns:
            dict: evaluation metrics
        """
        
        single_class_mode = True if num_classes == 1 else False
        eval_metrics = create_metrics_dict(num_classes)

        # Evaluate Mode
        model.eval()
        with torch.no_grad():
            for batch_index, data in enumerate(tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} '
                                                                                        f'batches with {device.type}')):
                if dataset == "tst":
                    inputs = data['sat_img'].to(device, non_blocking=True)
                    labels = data['map_img'].to(device, non_blocking=True)
                else:
                    inputs = data['sat_img']
                    labels = data['map_img']
                inputs = self.transforms.normalize_transform(inputs)
                labels = labels.squeeze(1).long()
                outputs = model(inputs)
                if isinstance(outputs, OrderedDict):
                    outputs = outputs['out']

                # vis_batch_range: range of batches to perform visualization on. see README.md for more info.
                # vis_at_eval: (bool) if True, will perform visualization at eval time, as long as vis_batch_range is valid
                if self.fabric.is_global_zero:
                    if vis_params['vis_batch_range'] and vis_params['vis_at_eval']:
                        min_vis_batch, max_vis_batch, increment = vis_params['vis_batch_range']
                        if batch_index in range(min_vis_batch, max_vis_batch, increment):
                            vis_path = vis_output.joinpath('visualization')
                            if epoch == 0 and batch_index == min_vis_batch:
                                logging.info(f'\nVisualizing on {dataset} outputs for batches in range '
                                            f'{vis_params["vis_batch_range"]} images will be saved to {vis_path}\n')
                            vis_from_batch(vis_params, inputs, outputs,
                                           batch_index=batch_index,
                                           vis_path=vis_path,
                                           labels=labels,
                                           dataset=dataset,
                                           ep_num=epoch + 1,
                                           scale=scale,
                                           device=device)
                
                with self.fabric.autocast():
                    loss = criterion(outputs, labels)
                if dataset == "val":
                    self.fabric.barrier()
                    self.fabric.all_reduce(loss, reduce_op="mean")
                eval_metrics['loss'].update(loss.item(), batch_size)
                if dataset == 'tst':
                    if single_class_mode:
                        outputs = torch.sigmoid(outputs)
                        outputs = outputs.squeeze(dim=1)
                    else:
                        outputs = torch.softmax(outputs, dim=1)
                        eval_metrics = iou(outputs, labels, batch_size, num_classes, 
                                        eval_metrics, single_class_mode, dontcare)
            if eval_metrics['loss'].avg:
                logging.info(f"\n{dataset} Loss: {eval_metrics['loss'].avg:.4f}")
            if batch_metrics is not None or dataset == 'tst':
                if single_class_mode:
                    logging.info(f"\n{dataset} iou_0: {eval_metrics['iou_0'].avg:.4f}")
                    logging.info(f"\n{dataset} iou_1: {eval_metrics['iou_1'].avg:.4f}")
                logging.info(f"\n{dataset} iou: {eval_metrics['iou'].avg:.4f}")

        return eval_metrics
        
        
    def run(self):
        since = time.time()
        logging.info(f'\nPreparing datasets (trn, val, tst) from data in {self.tiles_dir}...')
        datasets, num_samples, samples_weight = prepare_dataset(samples_folder=self.tiles_dir,
                                                                batch_size=self.batch_size,
                                                                dontcare_val=self.dontcare_val,
                                                                crop_size=self.crop_size,
                                                                num_bands=self.num_bands,
                                                                min_annot_perc=self.min_annot_perc,
                                                                attr_vals=self.attr_vals,
                                                                scale=self.scale,
                                                                cfg=self.cfg,
                                                                dontcare2backgr=self.dontcare2backgr,
                                                                compute_sampler_weights=self.compute_sampler_weights,
                                                                debug=self.debug
                                                                )
        trn_dataloader, val_dataloader, tst_dataloader = prepare_dataloader(datasets=datasets,
                                                                            samples_weight=samples_weight,
                                                                            num_samples=num_samples,
                                                                            batch_size=self.batch_size,
                                                                            num_workers=self.num_workers)
        self.cfg.training['num_samples']['trn'] = len(trn_dataloader.dataset)
        self.cfg.training['num_samples']['val'] = len(val_dataloader.dataset)
        self.cfg.training['num_samples']['tst'] = len(tst_dataloader.dataset)
        max_iters = self.num_epochs * len(trn_dataloader.dataset)
        trn_dataloader, val_dataloader = self.fabric.setup_dataloaders(trn_dataloader, val_dataloader)
        
        # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
        logging.info(f"\nInstantiate {self.cfg.model._target_} model with {self.num_bands} "
                     f"input channels and {self.num_classes} output classes.")
        
        aux_output = False
        model = define_model(net_params=self.cfg.model,
                            in_channels=self.num_bands,
                            out_classes=self.num_classes,
                            state_dict_path=self.train_state_dict_path,
                            state_dict_strict_load=self.state_dict_strict,
                            )
        
        if self.cfg.model._target_ == "models.hrnet.hrnet_ocr.HRNet":
            from models.hrnet.backbone import model_urls
            from models.hrnet.utils import ModelHelpers
            aux_output = True
            if self.cfg.model.pretrained:
                weights_file = ModelHelpers.load_url(model_urls['hrnetv2'], download=False)
                if self.fabric.is_global_zero:
                    weights_file = ModelHelpers.load_url(model_urls['hrnetv2'], download=True)
                self.fabric.barrier()
                model.load_state_dict(torch.load(weights_file, map_location=None), strict=False)
                
        if self.strategy == "ddp" or self.strategy == "fsdp":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.freeze_model_parts:
            freeze_model_parts(model=model, sub_models=self.freeze_model_parts)
        # if int(torch.__version__[0]) == 2:
        #     model = torch.compile(model=model, mode="reduce-overhead")
        optimizer = instantiate(self.cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
        model = self.fabric.setup_model(model)
        optimizer = self.fabric.setup_optimizer(optimizer)
        # model, optimizer = self.fabric.setup(model, optimizer)
        device = self.fabric.device
        criterion = define_loss(loss_params=self.cfg.loss, class_weights=self.class_weights)
        criterion = criterion.to(device)
    
        onecycle_scheduler = False
        plateau_scheduler = False
        if self.cfg.scheduler._target_ == 'torch.optim.lr_scheduler.OneCycleLR':
            self.cfg.scheduler['total_steps'] = max_iters
            onecycle_scheduler = True
        if self.cfg.scheduler._target_ == 'torch.optim.lr_scheduler.ReduceLROnPlateau':
            plateau_scheduler = True
        lr_scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        scheduler_dict = {"lr_scheduler": lr_scheduler, 
                          "onecycle_scheduler": onecycle_scheduler,
                          "plateau_scheduler": plateau_scheduler}
        early_stopping = EarlyStopping(patience=self.early_stop_epoch)
        
        output_path = self.tiles_dir.joinpath('model') / self.run_name
        if self.fabric.is_global_zero:
            if output_path.is_dir():
                last_mod_time_suffix = datetime.fromtimestamp(output_path.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
                archive_output_path = output_path.parent / f"{output_path.stem}_{last_mod_time_suffix}"
                shutil.move(output_path, archive_output_path)
            output_path.mkdir(parents=True, exist_ok=False)
            logging.info(f'\n Training artifacts will be saved to: {output_path}')

            # Save tracking
            set_tracker(mode='train', type='mlflow', task='segmentation', 
                        experiment_name=self.experiment_name,
                        run_name=self.run_name, tracker_uri=self.tracker_uri, params=self.cfg,
                        keys2log=['training', 'tiling', 'dataset', 'model', 'loss',
                                'optimizer', 'scheduler', 'augmentation'])
            trn_log, val_log, tst_log = [InformationLogger(dataset) for dataset in ['trn', 'val', 'tst']]

            # VISUALIZATION: generate pngs of inputs, labels and outputs
            if self.vis_batch_range is not None:
                # Make sure user-provided range is a tuple with 3 integers (start, finish, increment).
                # Check once for all visualization tasks.
                if not len(self.vis_batch_range) == 3 and all(isinstance(x, int) for x in self.vis_batch_range):
                    raise ValueError(f"\nVis_batch_range expects three integers in a list:"
                                     f"start batch, end batch, increment. Got {self.vis_batch_range}")
                    
                vis_at_init_dataset = get_key_def('vis_at_init_dataset', self.cfg['visualization'], 'val')

                # Visualization at initialization. Visualize batch range before first eopch.
                if get_key_def('vis_at_init', self.cfg['visualization'], False):
                    logging.info(f'\nVisualizing initialized model on batch range {self.vis_batch_range} '
                                f'from {vis_at_init_dataset} dataset...\n')
                    vis_from_dataloader(vis_params=self.vis_params,
                                        eval_loader=val_dataloader if vis_at_init_dataset == 'val' else tst_dataloader,
                                        model=model,
                                        ep_num=0,
                                        output_path=output_path,
                                        dataset=vis_at_init_dataset,
                                        scale=self.scale,
                                        device=device,
                                        vis_batch_range=self.vis_batch_range)
            # Set Counters
            best_loss = 999
            last_vis_epoch = 0
            checkpoint_stack = [""]
        self.fabric.barrier()
        for epoch in range(0, self.num_epochs):
            logging.info(f'\nEpoch {epoch}/{self.num_epochs - 1}\n' + "-" * len(f'Epoch {epoch}/{self.num_epochs - 1}'))
            trn_report = self.train_loop(model=model,
                                         train_loader=trn_dataloader,
                                         optimizer=optimizer,
                                         criterion=criterion,
                                         scheduler=scheduler_dict,
                                         num_classes=self.num_classes,
                                         batch_size=self.batch_size,
                                         epoch=epoch,
                                         vis_output=output_path,
                                         device=device,
                                         scale=self.scale,
                                         vis_params=self.vis_params,
                                         aux_output=aux_output
                                         )
            
            val_report = self.evaluation_loop(model=model,
                                              eval_loader=val_dataloader,
                                              criterion=criterion,
                                              epoch=epoch,
                                              num_classes=self.num_classes,
                                              batch_size=self.batch_size,
                                              scale=self.scale,
                                              vis_params=self.vis_params,
                                              vis_output=output_path,
                                              device=device,
                                              dataset="val",
                                              dontcare=self.dontcare_val,
                                              batch_metrics =self.batch_metrics
                                              )
            val_loss = val_report['loss'].avg
            if plateau_scheduler:
                lr_scheduler.step(val_loss)
                trn_report['lr'].update(optimizer.param_groups[0]['lr'])
                
            if self.fabric.is_global_zero:
                if 'trn_log' in locals():  # only save the value if a tracker is setup
                    trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall',
                                                                  'fscore', 'iou', 'iou-nonbg',
                                                                  'segmentor-loss', 'discriminator-loss',
                                                                  'real-score-critic', 'fake-score-critic'])

                if 'val_log' in locals():  # only save the value if a tracker is setup
                    if self.batch_metrics is not None:
                        val_log.add_values(val_report, epoch)
                    else:
                        val_log.add_values(val_report, epoch, ignore=['precision', 'recall',
                                                                      'fscore', 'lr', 'iou', 'iou-nonbg',
                                                                      'segmentor-loss', 'discriminator-loss',
                                                                      'real-score-critic', 'fake-score-critic'])
                if val_loss < best_loss:
                    logging.info("\nSave checkpoint with a validation loss of {:.4f}".format(val_loss))
                    # create the checkpoint file
                    checkpoint_tag = checkpoint_stack.pop()
                    filename = output_path.joinpath(checkpoint_tag)
                    if filename.is_file():
                        filename.unlink()
                    checkpoint_tag = (f"{self.experiment_name}_{self.num_classes}_" 
                                      f"{'_'.join(map(str, self.modalities))}_{val_loss:.2f}.pth.tar")
                    filename = output_path.joinpath(checkpoint_tag)
                    checkpoint_stack.append(checkpoint_tag)
                    best_loss = val_loss
                # More info:
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
                    state_dict = model.module.state_dict() if self.num_devices > 1 else model.state_dict()
                    torch.save({'epoch': epoch,
                                'params': self.cfg,
                                'model_state_dict': state_dict,
                                'best_loss': best_loss,
                                'optimizer_state_dict': optimizer.state_dict()}, filename)

                    # VISUALIZATION: generate pngs of img samples, labels and outputs as alternative to follow training
                    if (self.vis_batch_range is not None and self.vis_at_checkpoint 
                        and epoch - last_vis_epoch >= self.ep_vis_min_thresh):
                        if last_vis_epoch == 0:
                            logging.info(f"\nVisualizing with {self.vis_at_ckpt_dataset} dataset samples" 
                                         f"on checkpointed model for batches in range {self.vis_batch_range}")
                        vis_from_dataloader(vis_params=self.vis_params,
                                            eval_loader= (val_dataloader 
                                                          if self.vis_at_ckpt_dataset == 'val' else tst_dataloader),
                                            model=model,
                                            ep_num=epoch+1,
                                            output_path=output_path,
                                            dataset=self.vis_at_ckpt_dataset,
                                            scale=self.scale,
                                            device=device,
                                            vis_batch_range=self.vis_batch_range)
                        last_vis_epoch = epoch
                cur_elapsed = time.time() - since
                logging.info(f'\nCurrent elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')
            self.fabric.barrier()
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info(f'Early stopping after patience elapsed!')
                break
        if self.fabric.is_global_zero:
            # load checkpoint model and evaluate it on test dataset.
            # if num_epochs is set to 0, model is loaded to evaluate on test set
            if int(self.cfg['general']['max_epochs']) > 0:
                checkpoint = read_checkpoint(filename)
                checkpoint = adapt_checkpoint_to_dp_model(checkpoint, model)
                model.load_state_dict(state_dict=checkpoint['model_state_dict'])

            if tst_dataloader:
                tst_report = self.evaluation_loop(model=model,
                                                  eval_loader=tst_dataloader,
                                                  criterion=criterion,
                                                  dataset="tst",
                                                  scale=self.scale,
                                                  num_classes=self.num_classes,
                                                  batch_size=self.batch_size,
                                                  epoch=self.num_epochs,
                                                  vis_params=self.vis_params,
                                                  vis_output=output_path,
                                                  batch_metrics=self.batch_metrics,
                                                  dontcare=self.dontcare_val,
                                                  device=device
                                                  )
                
                if 'tst_log' in locals():  # only save the value if a tracker is set up
                    tst_log.add_values(tst_report, self.num_epochs, ignore=['precision', 'iou-nonbg', 
                                                                            'recall', 'lr', 'fscore', 
                                                                            'segmentor-loss', 'discriminator-loss',
                                                                            'real-score-critic', 'fake-score-critic'])


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
    trainer = Trainer(cfg=cfg)
    trainer.run()