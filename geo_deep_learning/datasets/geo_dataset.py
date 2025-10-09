"""Stream datasets based on TorchGeo datasets."""

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

logger = logging.getLogger(__name__)


class GeoImageDataset(RasterDataset):
    """Geo Image raster dataset."""

    def __init__(  # noqa: PLR0913
        self,
        paths: str | Path,
        filename_glob: str = "*.tif",
        filename_regex: str = ".*",
        all_bands: list[str] | None = None,
        rgb_bands: list[str] | None = None,
        dtype: torch.dtype | None = None,
        *,
        is_image: bool = True,
        norm_stats: dict[str, list[float]] | None = None,
        data_type_max: int = 255,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize GeoImageDataset."""
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        self._is_image = is_image
        self._dtype = dtype
        self.norm_stats = norm_stats
        self.data_type_max = data_type_max
        if all_bands:
            self.all_bands = tuple(all_bands)
        if rgb_bands:
            self.rgb_bands = tuple(rgb_bands)
        if not rgb_bands and all_bands:
            num_rgb_bands = 3
            self.rgb_bands = (
                tuple(all_bands[:3])
                if len(all_bands) >= num_rgb_bands
                else tuple(all_bands)
            )
        super().__init__(paths=paths, **kwargs)

    @property
    def is_image(self) -> bool:
        """Image or label to return "image" or "mask" keys."""
        return self._is_image

    @property
    def dtype(self) -> torch.dtype:
        """Support custom dtype for images and labels."""
        if self._dtype is not None:
            return self._dtype
        return torch.float32 if self.is_image else torch.long

    def __getitem__(self, query: int) -> dict[str, Any]:
        """Get item with normalization."""
        sample = super().__getitem__(query)

        # Handle -inf in masks
        if not self.is_image:
            mask = sample["mask"]
            mask = torch.where(torch.isinf(mask), torch.tensor(0.0), mask)
            sample["mask"] = mask

        # Normalize images
        if self.is_image and self.norm_stats is not None:
            image = sample["image"]

            # Step 1: Convert to [0, 1] range
            image = image.float() / self.data_type_max

            # Step 2: Normalize with mean/std
            mean = torch.tensor(
                self.norm_stats["mean"],
                dtype=image.dtype,
                device=image.device,
            ).view(-1, 1, 1)
            std = torch.tensor(
                self.norm_stats["std"],
                dtype=image.dtype,
                device=image.device,
            ).view(-1, 1, 1)

            image = (image - mean) / std

            sample["image"] = image

            # Add mean/std to sample for denormalization during visualization
            sample["mean"] = torch.tensor(self.norm_stats["mean"])
            sample["std"] = torch.tensor(self.norm_stats["std"])

        return sample


class GeoIntersectionDataset:
    """
    Wrapper for TorchGeo IntersectionDataset with sampling.

    This class combines image and label datasets and provides samplers.
    """

    def __init__(  # noqa: PLR0913
        self,
        image_dataset: RasterDataset,
        label_dataset: RasterDataset,
        patch_size: tuple[int, int] = (512, 512),
        length: int | None = None,
        sampler_type: str = "random",
        stride: tuple[int, int] | None = None,
    ) -> None:
        """
        Initialize GeoIntersectionDataset.

        Args:
            image_dataset: Image raster dataset
            label_dataset: Label raster dataset
            patch_size: Size of patches to sample (height, width)
            length: Number of samples per epoch (for random sampler)
            sampler_type: Type of sampler - "random" or "grid"
            stride: Stride for grid sampler (defaults to patch_size)

        """
        self.image_dataset = image_dataset
        self.label_dataset = label_dataset
        self.patch_size = patch_size
        self.length = length
        self.sampler_type = sampler_type
        self.stride = stride or patch_size

        self.dataset = IntersectionDataset(image_dataset, label_dataset)

        self.sampler = self._create_sampler()

        logger.info(
            "Created GeoIntersectionDataset with %s sampler, patch_size=%s",
            sampler_type,
            patch_size,
        )

    def _create_sampler(self) -> Sampler:
        """Create appropriate sampler based on type."""
        if self.sampler_type == "random":
            if self.length is None:
                msg = "length must be specified for random sampler"
                raise ValueError(msg)
            return RandomGeoSampler(
                self.dataset,
                size=self.patch_size,
                length=self.length,
            )
        if self.sampler_type == "grid":
            return GridGeoSampler(
                self.dataset,
                size=self.patch_size,
                stride=self.stride,
            )
        msg = f"Unknown sampler type: {self.sampler_type}"
        raise ValueError(msg)

    def __len__(self) -> int:
        """Return length based on sampler."""
        return len(self.sampler)


def create_intersection_dataset(  # noqa: PLR0913
    image_root: str | Path,
    label_root: str | Path,
    split: str = "train",
    patch_size: tuple[int, int] = (512, 512),
    length: int | None = 1000,
    image_glob: str = "*.tif",
    label_glob: str = "*_label.tif",
    sampler_type: str = "random",
    **kwargs: dict[str, Any],
) -> GeoIntersectionDataset:
    """
    Intersect image and label datasets.

    Args:
        image_root (str | Path): Root directory for images
        label_root (str | Path): Root directory for labels
        split (str, optional): Defaults to "train".
        patch_size (tuple[int, int], optional): Defaults to (512, 512).
        length (int, optional): Defaults to 1000.
        image_glob (str, optional): Defaults to "*.tif".
        label_glob (str, optional): Defaults to "*_label.tif".
        sampler_type (str, optional): Defaults to "random".
        **kwargs: Additional arguments

    Returns:
        GeoIntersectionDataset instance

    """
    image_path = Path(image_root) / split
    label_path = Path(label_root) / split

    # Create datasets
    image_dataset = GeoImageDataset(
        paths=image_path,
        filename_glob=image_glob,
        is_image=True,
        **kwargs,
    )

    label_dataset = GeoImageDataset(
        paths=label_path,
        filename_glob=label_glob,
        is_image=False,
        **kwargs,
    )

    # Create intersection dataset with sampler
    return GeoIntersectionDataset(
        image_dataset=image_dataset,
        label_dataset=label_dataset,
        patch_size=patch_size,
        length=length,
        sampler_type=sampler_type,
    )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Example 1: Using specific dataset classes
    image_ds = GeoImageDataset(
        paths="./notebooks/alberta_fire_data",
    )
    label_ds = GeoImageDataset(
        paths="./notebooks/alberta_fire_data",
    )

    dataset = GeoIntersectionDataset(
        image_dataset=image_ds,
        label_dataset=label_ds,
        patch_size=(512, 512),
        length=1000,
        sampler_type="random",
    )

    print(f"Dataset length: {len(dataset)}")
