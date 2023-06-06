import rasterio
import numpy as np
import pandas as pd

from pathlib import Path
from rasterio.plot import reshape_as_image
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset, Sampler
from typing import List

from utils.logger import get_logger

# These two import statements prevent exception when using eval(metadata) in SegmentationDataset()'s __init__()
from rasterio.crs import CRS
from affine import Affine

# Set the logging file
logging = get_logger(__name__)  # import logging


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset based on HDF5 parsing."""

    def __init__(self,
                 dataset_list_path,
                 dataset_type,
                 num_bands,
                 max_sample_count=None,
                 dontcare=None,
                 radiom_transform=None,
                 geom_transform=None,
                 totensor_transform=None,
                 debug=False):
        # note: if 'max_sample_count' is None, then it will be read from the dataset at runtime
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.num_bands = num_bands
        self.radiom_transform = radiom_transform
        self.geom_transform = geom_transform
        self.totensor_transform = totensor_transform
        self.debug = debug
        self.dontcare = dontcare
        self.list_path = dataset_list_path
        self.parent_folder = dataset_list_path.parent
        
        if not Path(self.list_path).is_file():
            logging.error(f"Couldn't locate dataset list file: {self.list_path}.\n"
                          f"If purposely omitting test set, this error can be ignored")
        
        self.assets = self._load_data()

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, index):
        
        sat_img, metadata = self._load_image(index)
        map_img = self._load_label(index)

        if isinstance(metadata, np.ndarray) and len(metadata) == 1:
            metadata = metadata[0]
        elif isinstance(metadata, bytes):
            metadata = metadata.decode('UTF-8')
        try:
            metadata = eval(metadata)
        except TypeError:
            pass

        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata, "list_path": self.list_path}
        # radiometric transforms should always precede geometric ones
        if self.radiom_transform:  
            sample = self.radiom_transform(sample)
        # rotation, geometric scaling, flip and crop. 
        # Will also put channels first and convert to torch tensor from numpy.
        if self.geom_transform:
            sample = self.geom_transform(sample)
        sample = self.totensor_transform(sample)

        if self.debug:
            # assert no new class values in map_img
            initial_class_ids = set(np.unique(map_img))
            if self.dontcare is not None:
                initial_class_ids.add(self.dontcare)
            final_class_ids = set(np.unique(sample['map_img'].numpy()))
            if not final_class_ids.issubset(initial_class_ids):
                logging.debug(f"WARNING: Class ids for label before and after augmentations don't match. "
                              f"Ignore if overwritting ignore_index in ToTensorTarget")
                logging.warning(f"\nWARNING: Class values for label before and after augmentations don't match."
                                f"\nUnique values before: {initial_class_ids}"
                                f"\nUnique values after: {final_class_ids}"
                                f"\nIgnore if some augmentations have padded with dontcare value.")
        sample['index'] = index

        return sample
    
    def _load_data(self) -> List[str]:
        """Load the filepaths to images and labels
        
        Returns:
            List[str]: a list of filepaths to train/test data  
        """
        df = pd.read_csv(self.list_path, sep=';', header=None, usecols=[i for i in range(2)])
        assets = [{"image": x, "label": y} for x, y in zip(df[0], df[1])]
        
        return assets
    
    def _load_image(self, index: int):
        """ Load image 

        Args:
            index: poosition of image

        Returns:
            image array and metadata
        """
        image_path = self.parent_folder / self.assets[index]["image"]
        with rasterio.open(image_path, 'r') as image_handle:
                image = reshape_as_image(image_handle.read())
                metadata = image_handle.meta
        assert self.num_bands <= image.shape[-1]
        
        return image, metadata
    
    def _load_label(self, index: int):
        """ Load label 

        Args:
            index: poosition of label

        Returns:
            label array and metadata
        """
        label_path = self.parent_folder / self.assets[index]["label"]
        
        with rasterio.open(label_path, 'r') as label_handle:
                label = reshape_as_image(label_handle.read())
                label = label[..., 0]
        
        return label
        
        

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


