"""GeoDataModule for TorchGeo-based datasets."""

import logging
import random
from collections.abc import Any, Iterator
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset
from torchgeo.samplers import GridGeoSampler

from geo_deep_learning.datasets.geo_dataset import GeoImageDataset

logger = logging.getLogger(__name__)


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that stacks tensors and keeps CRS/BoundingBox as lists."""
    collated = {}
    # Keys that should stay as lists, not stacked
    list_keys = {"mean", "std", "crs", "bbox"}

    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in list_keys:
            # Keep as list for mean/std/metadata
            collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


class GeoDataModule(LightningDataModule):
    """Lightning DataModule for geospatial raster datasets using TorchGeo."""

    def __init__(  # noqa: PLR0913
        self,
        image_root: str | Path,
        label_root: str | Path,
        batch_size: int = 16,
        num_workers: int = 8,
        patch_size: tuple[int, int] = (512, 512),
        image_glob: str = "*.tif",
        label_glob: str = "*_label.tif",
        sampler_type: str = "random",
        train_split: float = 0.8,
        val_split: float = 0.2,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        data_type_max: int = 255,
        **kwargs: object,
    ) -> None:
        """
        Initialize GeoDataModule.

        Args:
            image_root: Root directory containing images
            label_root: Root directory containing labels
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            patch_size: Size of patches to extract (height, width)
            image_glob: Glob pattern for image files
            label_glob: Glob pattern for label files
            sampler_type: Type of sampler ("random" or "grid")
            train_split: Proportion of data for training (e.g., 0.6 for 60%)
            val_split: Proportion of data for validation (e.g., 0.3 for 30%)
            mean: Mean values for normalization (per channel)
            std: Standard deviation values for normalization (per channel)
            data_type_max: Maximum value for data type (e.g., 255 for uint8)
            **kwargs: Additional arguments passed to dataset creation

        """
        super().__init__()
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.image_glob = image_glob
        self.label_glob = label_glob
        self.sampler_type = sampler_type
        self.train_split = train_split
        self.val_split = val_split
        self.data_type_max = data_type_max
        self.kwargs = kwargs
        self.norm_stats = {
            "mean": mean or [0.0, 0.0, 0.0],
            "std": std or [1.0, 1.0, 1.0],
        }

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create train/val/test datasets with random patch-level splitting."""
        self._setup_random_split()

    def _setup_random_split(self) -> None:
        """Split all patches randomly into train/val/test with no overlap."""
        image_dataset = GeoImageDataset(
            paths=self.image_root,
            filename_glob=self.image_glob,
            is_image=True,
            norm_stats=self.norm_stats,
            data_type_max=self.data_type_max,
            **self.kwargs,
        )
        label_dataset = GeoImageDataset(
            paths=self.label_root,
            filename_glob=self.label_glob,
            is_image=False,
            dtype=torch.float32,
            **self.kwargs,
        )
        full_dataset = IntersectionDataset(image_dataset, label_dataset)

        grid_sampler = GridGeoSampler(
            full_dataset,
            size=self.patch_size,
            stride=self.patch_size,
        )
        all_patches = list(grid_sampler)
        random.shuffle(all_patches)

        test_split = 1.0 - self.train_split - self.val_split
        total_patches = len(all_patches)
        train_size = int(total_patches * self.train_split)
        val_size = int(total_patches * self.val_split)

        train_patches = all_patches[:train_size]
        val_patches = all_patches[train_size : train_size + val_size]
        test_patches = all_patches[train_size + val_size :]
        tol = 1e-6

        has_test = len(test_patches) > 0 and abs(test_split) > tol

        if has_test:
            logger.info(
                "Random patch split: train=%d (%.1f%%), val=%d (%.1f%%), "
                "test=%d (%.1f%%) out of %d total patches",
                len(train_patches),
                self.train_split * 100,
                len(val_patches),
                self.val_split * 100,
                len(test_patches),
                test_split * 100,
                total_patches,
            )
        else:
            logger.info(
                "Random patch split: train=%d (%.1f%%), val=%d (%.1f%%) "
                "out of %d total patches (no test set)",
                len(train_patches),
                self.train_split * 100,
                len(val_patches),
                self.val_split * 100,
                total_patches,
            )

        self.train_dataset = self._create_dataset_from_patches(
            full_dataset,
            train_patches,
            self.sampler_type,
        )
        self.val_dataset = self._create_dataset_from_patches(
            full_dataset,
            val_patches,
            self.sampler_type,
        )
        self.test_dataset = (
            self._create_dataset_from_patches(full_dataset, test_patches, "list")
            if has_test
            else None
        )

    def _create_dataset_from_patches(
        self,
        dataset: IntersectionDataset,
        patch_coords: list,
        sampler_type: str,
    ) -> IntersectionDataset:
        """Create dataset wrapper with sampler for specific patch coordinates."""
        if sampler_type == "random":

            class RandomPatchSampler:
                def __init__(self, patches: list, length: int | None = None) -> None:
                    self.patches = patches
                    self.length = length or len(patches)

                def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
                    for _ in range(self.length):
                        yield random.choice(self.patches)  # noqa: S311

                def __len__(self) -> int:
                    return self.length

            sampler = RandomPatchSampler(patch_coords, len(patch_coords))
        else:

            class ListPatchSampler:
                def __init__(self, patches: list) -> None:
                    self.patches = patches

                def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
                    yield from self.patches

                def __len__(self) -> int:
                    return len(self.patches)

            sampler = ListPatchSampler(patch_coords)

        return type(
            "GeoWrapper",
            (),
            {
                "dataset": dataset,
                "sampler": sampler,
                "__len__": lambda _: len(sampler),
            },
        )()

    def train_dataloader(self) -> DataLoader:
        """Dataloader for training."""
        return DataLoader(
            self.train_dataset.dataset,
            batch_size=self.batch_size,
            sampler=self.train_dataset.sampler,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Dataloader for validation."""
        return DataLoader(
            self.val_dataset.dataset,
            batch_size=self.batch_size,
            sampler=self.val_dataset.sampler,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader | None:
        """Dataloader for testing."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset.dataset,
            batch_size=self.batch_size,
            sampler=self.test_dataset.sampler,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    print("Testing GeoDataModule with random bbox split")

    datamodule = GeoDataModule(
        image_root="notebooks/alberta_fire_data",
        label_root="notebooks/alberta_fire_data",
        batch_size=2,
        num_workers=0,
        patch_size=(256, 256),
        image_glob="*rgb.tif",
        label_glob="*label.tif",
        sampler_type="random",
        train_split=0.9,
        val_split=0.1,
    )

    datamodule.setup()

    # Check dataset lengths
    print("\nDataset lengths:")
    print(f"  Train: {len(datamodule.train_dataset)}")
    print(f"  Val: {len(datamodule.val_dataset)}")
    if datamodule.test_dataset:
        print(f"  Test: {len(datamodule.test_dataset)}")
    else:
        print("  Test: None")

    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    print("\nTrain batch:")
    print(f"  Image shape: {train_batch['image'].shape}")
    print(f"  Mask shape: {train_batch['mask'].shape}")
    img_min = train_batch["image"].min()
    img_max = train_batch["image"].max()
    print(f"  Image range: [{img_min:.3f}, {img_max:.3f}]")

    # Test val dataloader
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    print("\nVal batch:")
    print(f"  Image shape: {val_batch['image'].shape}")
    print(f"  Mask shape: {val_batch['mask'].shape}")

    # Test test dataloader if it exists
    if datamodule.test_dataset:
        test_loader = datamodule.test_dataloader()
        test_batch = next(iter(test_loader))
        print("\nTest batch:")
        print(f"  Image shape: {test_batch['image'].shape}")
        print(f"  Mask shape: {test_batch['mask'].shape}")

    print("\n✓ All dataloaders working correctly")
