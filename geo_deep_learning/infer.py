#!/usr/bin/env python3
"""CLI script for running segmentation inference on GeoTIFFs."""

import argparse
import logging
from pathlib import Path

from configs import logging_config  # noqa: F401
from geo_deep_learning.tools.geotiff_inference import GeoTiffSegmentationInference

logger = logging.getLogger(__name__)


def main() -> None:
    """Run inference on a geotiff file."""
    parser = argparse.ArgumentParser(
        description="Run segmentation inference on GeoTIFF files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input GeoTIFF file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output GeoTIFF file",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        required=True,
        help="Mean values for standardization (one per channel)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        required=True,
        help="Std values for standardization (one per channel)",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Size of tiles for sliding window inference",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=171,
        help="Overlap between tiles (stride = tile_size - overlap)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of tiles to process simultaneously",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Size of chunks to read from large GeoTIFFs",
    )
    parser.add_argument(
        "--compress",
        type=str,
        default="lzw",
        help="Compression method for output (lzw, deflate, none, etc.)",
    )
    parser.add_argument(
        "--output-probabilities",
        action="store_true",
        help="Output probabilities instead of class labels",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.checkpoint).exists():
        msg = f"Checkpoint file not found: {args.checkpoint}"
        raise FileNotFoundError(msg)

    if not Path(args.input).exists():
        msg = f"Input file not found: {args.input}"
        raise FileNotFoundError(msg)

    # Log configuration
    logger.info("=" * 80)
    logger.info("GeoTIFF Segmentation Inference")
    logger.info("=" * 80)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Device: %s", args.device)
    logger.info("Mean: %s", args.mean)
    logger.info("Std: %s", args.std)
    logger.info("Tile size: %d", args.tile_size)
    logger.info("Overlap: %d", args.overlap)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Chunk size: %d", args.chunk_size)
    logger.info("Compression: %s", args.compress)
    logger.info("Output probabilities: %s", args.output_probabilities)
    logger.info("=" * 80)

    # Create inference engine
    inference = GeoTiffSegmentationInference(
        checkpoint_path=args.checkpoint,
        mean=args.mean,
        std=args.std,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    # Run inference
    inference.predict(
        input_path=args.input,
        output_path=args.output,
        compress=args.compress,
        output_prob=args.output_probabilities,
    )

    logger.info("=" * 80)
    logger.info("Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
