"""GeoTIFF inference for segmentation models."""

import logging
from pathlib import Path

import numpy as np
import rasterio as rio
import torch
from lightning.pytorch import LightningModule
from rasterio.windows import Window
from torch import nn

logger = logging.getLogger(__name__)


def slide_inference(  # noqa: PLR0913
    inputs: torch.Tensor,
    wavelengths: torch.Tensor,
    segmentation_model: nn.Module,
    n_output_channels: int = 256,
    crop_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] = (341, 341),
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Inference by sliding-window with overlap using batched predictions.

    This is a more efficient version of slide_inference that processes multiple
    tiles in batches for better GPU utilization.

    Args:
        inputs (tensor): the tensor should have a shape 1xCxHxW (single image).
        wavelengths (tensor): the tensor should have a shape 1xN (single image).
        segmentation_model (nn.Module): model with .predict() method.
        n_output_channels (int): number of output channels
        crop_size (tuple): (h_crop, w_crop)
        stride (tuple): (h_stride, w_stride)
        batch_size (int): number of tiles to process simultaneously

    Returns:
        Tensor: The output results from model (1, C, H, W).

    """
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    input_batch_size, _, h_img, w_img = inputs.shape

    if input_batch_size != 1:
        msg = "Currently only supports single image at a time"
        raise ValueError(msg)

    # Handle case where crop is larger than image
    if h_crop > h_img and w_crop > w_img:
        h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)

    # Calculate grid dimensions
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # Initialize accumulators
    preds = inputs.new_zeros((1, n_output_channels, h_img, w_img)).cpu()
    count_mat = inputs.new_zeros((1, 1, h_img, w_img)).to(torch.int8).cpu()

    # Collect all tile positions first
    tiles_info = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            tiles_info.append((y1, x1, y2, x2))

    # Process tiles in batches
    for i in range(0, len(tiles_info), batch_size):
        batch_tiles_info = tiles_info[i : i + batch_size]

        # Extract all tiles in this batch
        crops = torch.cat(
            [inputs[:, :, y1:y2, x1:x2] for y1, x1, y2, x2 in batch_tiles_info],
            dim=0,
        )

        # Predict on batch
        with torch.no_grad():
            # print(f"Crops shape: {crops.shape}")
            # print(f"Wavelengths shape: {wavelengths.shape}")
            crop_preds = segmentation_model.predict(
                crops,
                wavelengths,
                rescale_to=crops.shape[2:],
            )

        # Blend predictions into accumulator
        for crop_pred, (y1, x1, y2, x2) in zip(
            crop_preds,
            batch_tiles_info,
            strict=True,
        ):
            preds[:, :, y1:y2, x1:x2] += crop_pred.unsqueeze(0).cpu()
            count_mat[:, :, y1:y2, x1:x2] += 1

    if (count_mat == 0).sum() != 0:
        msg = "Some pixels were not covered by any tile"
        raise ValueError(msg)
    return preds / count_mat


class GeoTiffSegmentationInference:
    """
    Model-agnostic geotiff inference for segmentation.

    Handles large geotiffs by:
    - Chunked reading with rasterio windows
    - Batched sliding window inference
    - Efficient memory management
    """

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str,
        mean: list[float],
        std: list[float],
        wavelengths: list[float],
        device: str = "cuda",
        tile_size: int = 512,
        overlap: int = 171,
        batch_size: int = 16,
        chunk_size: int = 4096,
        num_classes: int | None = None,
    ) -> None:
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to Lightning checkpoint (.ckpt) or exported (.pt2)
            mean: Mean values for standardization (per channel)
            std: Std values for standardization (per channel)
            wavelengths: Wavelengths for the model
            device: Device to run inference on
            tile_size: Size of tiles for sliding window
            overlap: Overlap between tiles (stride = tile_size - overlap)
            batch_size: Number of tiles to process simultaneously
            chunk_size: Size of chunks to read from geotiff
            num_classes: Required when checkpoint_path is an exported model (.pt2)

        """
        if Path(checkpoint_path).suffix == ".pt2" and num_classes is None:
            msg = "num_classes is required when using an exported model (.pt2)"
            raise ValueError(msg)

        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # Store normalization stats
        self.mean = mean
        self.std = std
        self.wavelengths = wavelengths
        self._exported_module = None
        logger.info("Loading model from %s", checkpoint_path)
        self.model = self._load_model(checkpoint_path, num_classes)

    def _load_model(
        self,
        checkpoint_path: str,
        num_classes: int | None = None,
    ) -> LightningModule | None:
        """Load Lightning checkpoint or torch.export (.pt2) model."""
        path = Path(checkpoint_path)
        if path.suffix == ".pt2":
            exported = torch.export.load(checkpoint_path)
            self._exported_module = exported.module().to(self.device)
            self.num_classes = num_classes
            return None
        model = LightningModule.load_from_checkpoint(
            checkpoint_path,
            weights_from_checkpoint_path=None,
            map_location=self.device,
            strict=False,
            weights_only=True,
        )
        model.eval()
        model.to(self.device)
        self.num_classes = model.num_classes
        if not hasattr(model, "predict"):
            msg = "Model must implement .predict() method"
            raise AttributeError(msg)
        if not hasattr(model, "preprocess"):
            msg = "Model must implement .preprocess() method"
            raise AttributeError(msg)
        return model

    def preprocess(
        self,
        x: torch.Tensor,
        mean: list[float] | torch.Tensor,
        std: list[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess input tensor (used by slide_inference)."""
        if self._exported_module is not None:
            return x  # .pt2 does norm + standardize inside the graph
        return self.model.preprocess(x, mean, std)

    def predict(
        self,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        rescale_to: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Run model forward (used by slide_inference)."""
        if self._exported_module is not None:
            c = x.shape[1]
            mean_t = torch.as_tensor(
                self.mean[:c], dtype=torch.float32, device=x.device,
            )
            std_t = torch.as_tensor(
                self.std[:c], dtype=torch.float32, device=x.device,
            )
            out = self._exported_module(x, mean_t, std_t, wavelengths)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            return self.model.predict(x, wavelengths, rescale_to=rescale_to)
        if rescale_to is not None:
            logits = torch.nn.functional.interpolate(
                logits,
                size=rescale_to,
                mode="bilinear",
                align_corners=False,
            )
        if self.num_classes == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def run(
        self,
        input_path: str,
        output_path: str,
        compress: str = "lzw",
        *,
        output_prob: bool = False,
    ) -> None:
        """
        Run inference on a geotiff file.

        Args:
            input_path: Path to input geotiff
            output_path: Path to output geotiff
            compress: Compression method for output (lzw, deflate, etc.)
            output_prob: If True, output probabilities; if False, output class labels

        """
        logger.info("Starting inference on: %s", input_path)
        logger.info("Output will be saved to: %s", output_path)

        input_path = Path(input_path)
        if not input_path.exists():
            msg = f"Input file not found: {input_path}"
            raise FileNotFoundError(msg)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rio.open(input_path) as src:
            # Get metadata
            profile = src.profile.copy()
            height, width = src.height, src.width

            # if not self.dynamic_encoder and src.count != self.in_channels:
            #     msg = (
            #         f"Input geotiff has {src.count} channels, "
            #         f"but model expects {self.in_channels} channels"
            #     )
            #     raise ValueError(msg)

            # Update profile for output
            if output_prob:
                profile.update(
                    dtype=rio.float32,
                    count=self.num_classes,
                    compress=compress,
                )
            else:
                profile.update(
                    dtype=rio.uint8,
                    count=1,
                    compress=compress,
                )

            # Create output file
            with rio.open(output_path, "w", **profile) as dst:
                # Process in chunks if image is large
                if height <= self.chunk_size and width <= self.chunk_size:
                    # Small image - process in one go
                    logger.info("Processing entire image (small size)")
                    chunk = src.read()
                    pred = self._predict_chunk(chunk)
                    dst.write(pred)
                else:
                    # Large image - process in chunks
                    logger.info(
                        "Processing large image in chunks of size %d",
                        self.chunk_size,
                    )
                    self._predict_chunked(src, dst, output_prob=output_prob)

        logger.info("Inference complete. Output saved to: %s", output_path)

    def _predict_chunked(
        self,
        src: rio.DatasetReader,
        dst: rio.DatasetReader,
        *,
        output_prob: bool,
    ) -> None:
        """Process large geotiff in chunks."""
        height, width = src.height, src.width
        chunk_count = 0
        total_chunks = (
            (height + self.chunk_size - 1)
            // self.chunk_size
            * (width + self.chunk_size - 1)
            // self.chunk_size
        )

        for h_start in range(0, height, self.chunk_size):
            for w_start in range(0, width, self.chunk_size):
                chunk_count += 1
                h_end = min(h_start + self.chunk_size, height)
                w_end = min(w_start + self.chunk_size, width)

                # Create window
                window = Window(w_start, h_start, w_end - w_start, h_end - h_start)

                # Read chunk
                chunk = src.read(window=window)

                # Predict
                pred = self._predict_chunk(chunk, output_prob=output_prob)

                # Write chunk
                dst.write(pred, window=window)

                logger.info("Processed chunk %d/%d", chunk_count, total_chunks)

    def _predict_chunk(
        self,
        chunk_np: np.ndarray,
        *,
        output_prob: bool = False,
    ) -> np.ndarray:
        """
        Predict on a single chunk.

        Args:
            chunk_np: Numpy array (C, H, W)
            output_prob: If True, return probabilities; if False, return class labels

        Returns:
            Predictions as numpy array

        """
        # Convert to tensor
        chunk_tensor = torch.from_numpy(chunk_np).float().unsqueeze(0)  # (1, C, H, W)
        chunk_tensor = chunk_tensor.to(self.device)
        wavelengths_tensor = torch.tensor(
            self.wavelengths,
            dtype=torch.float32,
        ).to(self.device)

        # Preprocess
        chunk_tensor = self.preprocess(chunk_tensor, self.mean, self.std)

        # Calculate stride
        stride = self.tile_size - self.overlap

        # Run slide inference with batching
        with torch.no_grad():
            pred = slide_inference(
                inputs=chunk_tensor,
                wavelengths=wavelengths_tensor,
                segmentation_model=self,
                n_output_channels=self.num_classes,
                crop_size=(self.tile_size, self.tile_size),
                stride=(stride, stride),
                batch_size=self.batch_size,
            )

        # pred shape: (1, C, H, W)
        pred = pred.squeeze(0)  # (C, H, W)

        if output_prob:
            # Return probabilities
            return pred.cpu().numpy().astype(np.float32)
        # Return class labels
        if self.num_classes == 1:
            # Binary segmentation
            threshold = 0.5
            pred = (pred > threshold).long()
        else:
            # Multi-class segmentation
            pred = pred.argmax(dim=0, keepdim=True)

        return pred.cpu().numpy().astype(np.uint8)
