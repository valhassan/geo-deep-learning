import os
import rasterio
import numpy as np

import torch
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from typing import Sequence, Iterator, Optional, Dict
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.utils import compute_sample_weight
from models.model_choice import to_dp_model
from utils import augmentation as aug, create_dataset
from utils.utils import get_key_def, get_device_ids, set_device
from utils.logger import get_logger
# Set the logging file
logging = get_logger(__name__)  # import logging


class EarlyStopping:
    """
    Early stops the training if validation loss/metrics doesn't improve after a given patience
    """

    def __init__(self, patience: int = 100, delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 100
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss:
            score = -metrics
        else:
            score = metrics

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class DistributedSamplerWrapper(DistributedSampler):
    """
    Adapted from Catalyst Framework https://github.com/catalyst-team/

    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            create_dataset.DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = create_dataset.DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class TrainEngine:
    def __init__(self, multiproc: DictConfig, engine_type: str = 'cpu'):
        super(TrainEngine, self).__init__()
        self.engine_type = engine_type
        self.ddp_initialized = False
        self.multiproc = multiproc
        self.gpu_devices_dict = {}
        self.gpu_ids = list(self.gpu_devices_dict.keys())

        if self.multiproc.gpus and not self.engine_type == 'cpu':
            self.gpu_devices_dict = get_device_ids(self.multiproc.gpus)
            self.gpu_ids = list(self.gpu_devices_dict.keys())

        if self.engine_type == 'distributed_data_parallel':
            # set up distributed data parallel
            if set(self.multiproc.local_env_var).issubset(os.environ):
                self.multiproc.global_rank = int(os.environ["RANK"])
                self.multiproc.local_rank = int(os.environ["LOCAL_RANK"])
                self.multiproc.ntasks = int(os.environ["LOCAL_WORLD_SIZE"])
                self.multiproc.world_size = int(os.environ["WORLD_SIZE"])

                dist.init_process_group(backend=self.multiproc.dist_backend,
                                        init_method=self.multiproc.dist_url,
                                        rank=self.multiproc.global_rank,
                                        world_size=self.multiproc.world_size)

            elif set(self.multiproc.hpc_env_var).issubset(os.environ):
                self.multiproc.global_rank = int(os.environ["SLURM_PROCID"])
                self.multiproc.local_rank = int(os.environ["SLURM_LOCALID"])
                self.multiproc.ntasks = int(os.environ["SLURM_NTASKS_PER_NODE"])
                self.multiproc.world_size = int(os.environ["WORLD_SIZE"])

                dist.init_process_group(backend=self.multiproc.dist_backend,
                                        init_method=self.multiproc.dist_url,
                                        rank=self.multiproc.global_rank,
                                        world_size=self.multiproc.world_size
                                        )
            if dist.is_initialized():
                self.ddp_initialized = True
            else:
                raise TypeError(f"Distributed Data Parallel is not initialized try CPU/DataParallel engine")

    # Set device(s)

    def get_device(self):

        if self.engine_type == "cpu":
            device = self.engine_type
            return device

        if self.engine_type == "data_parallel":
            device = set_device(gpu_devices_dict=self.gpu_devices_dict)
            return device

        if self.engine_type == "distributed_data_parallel":
            device = torch.device(f"cuda:{self.multiproc.local_rank}")
            return device

    def prepare_model(self, model: torch.nn.Module):
        device = self.get_device()
        model.to(device)

        if self.engine_type == 'cpu':
            return model

        if self.engine_type == 'data_parallel':
            model = to_dp_model(model=model, devices=self.gpu_ids) if len(self.gpu_ids) > 1 else model
            return model

        if self.engine_type == "distributed_data_parallel":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[self.multiproc.local_rank],
                                                              output_device=self.multiproc.local_rank)
            return model

    def prepare_dataloader(self,
                           datasets: Sequence[Dataset],
                           samples_weight: Sequence[float],
                           num_samples: Dict[str, int],
                           batch_size: int, eval_batch_size: int,
                           sample_size: int, num_workers: int = 0):

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        if self.engine_type == 'data_parallel' and num_workers == 0:
            num_workers = len(self.gpu_ids) * 4 if len(self.gpu_ids) > 1 else 4

        samples_weight = torch.from_numpy(samples_weight)
        trn_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                                 len(samples_weight))
        if self.engine_type == "distributed_data_parallel":
            trn_sampler = DistributedSamplerWrapper(trn_sampler)

        if self.gpu_devices_dict and not eval_batch_size:
            max_pix_per_mb_gpu = 280  # TODO: this value may need to be finetuned
            eval_batch_size = calc_eval_batchsize(self.gpu_devices_dict, batch_size, sample_size, max_pix_per_mb_gpu)
        elif not eval_batch_size:
            eval_batch_size = batch_size

        trn_dataset, val_dataset, tst_dataset = datasets
        val_sampler = None
        if self.engine_type == "distributed_data_parallel":
            val_sampler = DistributedSampler(val_dataset)

        trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, sampler=trn_sampler,
                                    drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers,
                                    sampler=val_sampler, drop_last=True)
        tst_dataloader = DataLoader(tst_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
                                    drop_last=True) if num_samples['tst'] > 0 else None

        if len(trn_dataloader) == 0 or len(val_dataloader) == 0:
            raise ValueError(f"\nTrain and validation dataloader should contain at least one data item."
                             f"\nTrain dataloader's length: {len(trn_dataloader)}"
                             f"\nVal dataloader's length: {len(val_dataloader)}")

        return trn_dataloader, val_dataloader, tst_dataloader


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs # this average is necessary!
    return rt

def make_tiles_dir_name(tile_size, num_bands):
    return f'tiles{tile_size}_{num_bands}bands'


def make_dataset_file_name(exp_name: str, min_annot: int, dataset: str, attr_vals: Sequence = None):
    if isinstance(attr_vals, int):
        attr_vals = [attr_vals]
    vals = "_feat" + "-".join([str(val) for val in attr_vals]) if attr_vals else ""
    min_annot_str = f"_min-annot{min_annot}"
    sampling_str = vals + min_annot_str
    dataset_file_name = f'{exp_name}{sampling_str}_{dataset}.csv'
    return dataset_file_name, sampling_str


def flatten_labels(annotations):
    """Flatten labels"""
    flatten = annotations.view(-1)
    return flatten


def flatten_outputs(predictions, number_of_classes):
    """Flatten the prediction batch except the prediction dimensions"""
    logits_permuted = predictions.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    outputs_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return outputs_flatten


def get_num_samples(samples_path,
                    params,
                    min_annot_perc,
                    attr_vals,
                    experiment_name:str,
                    compute_sampler_weights=False):
    """
    Function to retrieve number of samples, either from config file or directly from hdf5 file.
    :param samples_path: (str) Path to samples folder
    :param params: (dict) Parameters found in the yaml config file.
    :param min_annot_perc: (int) minimum annotated percentage
    :param attr_vals: (list) attribute values to keep from source ground truth
    :param experiment_name: (str) experiment name
    :param compute_sampler_weights: (bool)
        if True, weights will be computed from dataset patches to oversample the minority class(es) and undersample
        the majority class(es) during training.
    :return: (dict) number of patches for trn, val and tst.
    """
    num_samples = {'trn': 0, 'val': 0, 'tst': 0}
    weights = []
    samples_weight = None
    for dataset in ['trn', 'val', 'tst']:
        dataset_file, _ = make_dataset_file_name(experiment_name, min_annot_perc, dataset, attr_vals)
        dataset_filepath = samples_path / dataset_file
        if not dataset_filepath.is_file() and dataset == 'tst':
            num_samples[dataset] = 0
            logging.warning(f"No test set. File not found: {dataset_filepath}")
            continue
        if get_key_def(f"num_{dataset}_samples", params['training'], None) is not None:
            num_samples[dataset] = params['training'][f"num_{dataset}_samples"]
            with open(dataset_filepath, 'r') as datafile:
                file_num_samples = len(datafile.readlines())
            if num_samples[dataset] > file_num_samples:
                raise IndexError(f"The number of training samples in the configuration file ({num_samples[dataset]}) "
                                 f"exceeds the number of samples in the hdf5 training dataset ({file_num_samples}).")
        else:
            with open(dataset_filepath, 'r') as datafile:
                num_samples[dataset] = len(datafile.readlines())
        with open(dataset_filepath, 'r') as datafile:
            datalist = datafile.readlines()
            if dataset == 'trn':
                if not compute_sampler_weights:
                    samples_weight = np.ones(num_samples[dataset])
                else:
                    for x in tqdm(range(num_samples[dataset]), desc="Computing sample weights"):
                        label_file = samples_path / datalist[x].split(';')[1]
                        with rasterio.open(label_file, 'r') as label_handle:
                            label = label_handle.read()
                        unique_labels = np.unique(label)
                        weights.append(''.join([str(int(i)) for i in unique_labels]))
                        samples_weight = compute_sample_weight('balanced', weights)
            logging.debug(samples_weight.shape)
            logging.debug(np.unique(samples_weight))

    return num_samples, samples_weight

def calc_eval_batchsize(gpu_devices_dict: dict, batch_size: int, sample_size: int, max_pix_per_mb_gpu: int = 280):
    """
    Calculate maximum batch size that could fit on GPU during evaluation based on thumb rule with harcoded
    "pixels per MB of GPU RAM" as threshold. The batch size often needs to be smaller if crop is applied during training
    @param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids (utils.py)
    @param batch_size: batch size for training
    @param sample_size: size of hdf5 samples
    @return: returns a downgraded evaluation batch size if the original batch size is considered too high compared to
    the GPU's memory
    """
    eval_batch_size_rd = batch_size
    # get max ram for smallest gpu
    smallest_gpu_ram = min(gpu_info['max_ram'] for _, gpu_info in gpu_devices_dict.items())
    # rule of thumb to determine eval batch size based on approximate max pixels a gpu can handle during evaluation
    pix_per_mb_gpu = (batch_size / len(gpu_devices_dict.keys()) * sample_size ** 2) / smallest_gpu_ram
    if pix_per_mb_gpu >= max_pix_per_mb_gpu:
        eval_batch_size = smallest_gpu_ram * max_pix_per_mb_gpu / sample_size ** 2
        eval_batch_size_rd = int(eval_batch_size - eval_batch_size % len(gpu_devices_dict.keys()))
        eval_batch_size_rd = 1 if eval_batch_size_rd < 1 else eval_batch_size_rd
        logging.warning(f'Validation and test batch size downgraded from {batch_size} to {eval_batch_size} '
                        f'based on max ram of smallest GPU available')
    return eval_batch_size_rd


def prepare_dataset(samples_folder: Path,
                    batch_size: int, dontcare_val: int, crop_size: int,
                    num_bands: int, min_annot_perc: int, attr_vals: Sequence,
                    scale: Sequence, cfg: DictConfig, dontcare2backgr: bool = False,
                    compute_sampler_weights: bool = False, debug: bool = False):
    """
        Function to compose segmentation datasets for training, validation and testing.
        :param samples_folder: path to folders containing .tif tiles if task is segmentation
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
        :param compute_sampler_weights: (bool) if True, weights for samples are computed using
                                        sklearn "compute_sample_weight".
        :param debug: (bool) if True, debugging is enabled
        :return: Lists containing SegmentationDataset objects
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
                                                  experiment_name=experiment_name,
                                                  compute_sampler_weights=compute_sampler_weights
                                                  )
    if not num_samples['trn'] >= batch_size and num_samples['val'] >= batch_size:
        raise ValueError(f"Number of samples in .hd is less than batch size")
    logging.info(f"Number of samples : {num_samples}\n")
    dataset_constr = create_dataset.SegmentationDataset
    datasets = []
    for subset in ["trn", "val", "tst"]:
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
    return datasets, num_samples, samples_weight

# def create_dataloader(samples_folder: Path,
#                       batch_size: int,
#                       eval_batch_size: int,
#                       gpu_devices_dict: dict,
#                       sample_size: int,
#                       dontcare_val: int,
#                       crop_size: int,
#                       num_bands: int,
#                       min_annot_perc: int,
#                       attr_vals: Sequence,
#                       scale: Sequence,
#                       cfg: DictConfig,
#                       dontcare2backgr: bool = False,
#                       compute_sampler_weights: bool = False,
#                       debug: bool = False):
#     """
#     Function to create dataloader objects for training, validation and test datasets.
#     :param samples_folder: path to folder containting .hdf5 files if task is segmentation
#     :param batch_size: (int) batch size
#     :param eval_batch_size: (int) Batch size for evaluation (val and test). Optional, calculated automatically if omitted
#     :param gpu_devices_dict: (dict) dictionary where each key contains an available GPU with its ram info stored as value
#     :param sample_size: (int) size of hdf5 samples (used to evaluate eval batch-size)
#     :param dontcare_val: (int) value in label to be ignored during loss calculation
#     :param crop_size: (int) size of one side of the square crop performed on original tile during training
#     :param num_bands: (int) number of bands in imagery
#     :param min_annot_perc: (int) minimum proportion of ground truth containing non-background information
#     :param attr_vals: (Sequence)
#     :param scale: (List) imagery data will be scaled to this min and max value (ex.: 0 to 1)
#     :param cfg: (dict) Parameters found in the yaml config file.
#     :param dontcare2backgr: (bool) if True, all dontcare values in label will be replaced with 0 (background value)
#                             before training
#     :return: trn_dataloader, val_dataloader, tst_dataloader
#     """
#     if not samples_folder.is_dir():
#         raise FileNotFoundError(f'Could not locate: {samples_folder}')
#     experiment_name = samples_folder.stem
#     if not len([f for f in samples_folder.glob('*.csv')]) >= 1:
#         raise FileNotFoundError(f"Couldn't locate text file containing list of training data in {samples_folder}")
#
#     num_samples, samples_weight = get_num_samples(samples_path=samples_folder,
#                                                   params=cfg,
#                                                   min_annot_perc=min_annot_perc,
#                                                   attr_vals=attr_vals,
#                                                   experiment_name=experiment_name,
#                                                   compute_sampler_weights=compute_sampler_weights
#                                                   )
#     if not num_samples['trn'] >= batch_size and num_samples['val'] >= batch_size:
#         raise ValueError(f"Number of samples in .hd is less than batch size")
#     logging.info(f"Number of samples : {num_samples}\n")
#     dataset_constr = create_dataset.SegmentationDataset
#     datasets = []
#
#     for subset in ["trn", "val", "tst"]:
#         dataset_file, _ = make_dataset_file_name(experiment_name, min_annot_perc, subset, attr_vals)
#         dataset_filepath = samples_folder / dataset_file
#         datasets.append(dataset_constr(dataset_filepath, subset, num_bands,
#                                        max_sample_count=num_samples[subset],
#                                        radiom_transform=aug.compose_transforms(params=cfg,
#                                                                                dataset=subset,
#                                                                                aug_type='radiometric'),
#                                        geom_transform=aug.compose_transforms(params=cfg,
#                                                                              dataset=subset,
#                                                                              aug_type='geometric',
#                                                                              dontcare=dontcare_val,
#                                                                              crop_size=crop_size),
#                                        totensor_transform=aug.compose_transforms(params=cfg,
#                                                                                  dataset=subset,
#                                                                                  scale=scale,
#                                                                                  dontcare2backgr=dontcare2backgr,
#                                                                                  dontcare=dontcare_val,
#                                                                                  aug_type='totensor'),
#                                        debug=debug))
#     trn_dataset, val_dataset, tst_dataset = datasets
#
#     # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
#     # Number of workers
#     if cfg.training.num_workers:
#         num_workers = cfg.training.num_workers
#     else:  # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
#         num_workers = len(gpu_devices_dict.keys()) * 4 if len(gpu_devices_dict.keys()) > 1 else 4
#
#     samples_weight = torch.from_numpy(samples_weight)
#     sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
#                                                              len(samples_weight))
#
#     if gpu_devices_dict and not eval_batch_size:
#         max_pix_per_mb_gpu = 280  # TODO: this value may need to be finetuned
#         eval_batch_size = calc_eval_batchsize(gpu_devices_dict, batch_size, sample_size, max_pix_per_mb_gpu)
#     elif not eval_batch_size:
#         eval_batch_size = batch_size
#
#     trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
#                                 drop_last=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
#                                 drop_last=True)
#     tst_dataloader = DataLoader(tst_dataset, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False,
#                                 drop_last=True) if num_samples['tst'] > 0 else None
#
#     if len(trn_dataloader) == 0 or len(val_dataloader) == 0:
#         raise ValueError(f"\nTrain and validation dataloader should contain at least one data item."
#                          f"\nTrain dataloader's length: {len(trn_dataloader)}"
#                          f"\nVal dataloader's length: {len(val_dataloader)}")
#
#     return trn_dataloader, val_dataloader, tst_dataloader