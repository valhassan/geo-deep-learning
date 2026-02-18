"""Patch Lightning checkpoint class_path for geo_deep_learning package prefix."""

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PREFIX_MAP = {
    "tasks_with_models.": "geo_deep_learning.tasks_with_models.",
    "datamodules.": "geo_deep_learning.datamodules.",
    "tools.": "geo_deep_learning.tools.",
}


CLASS_PATH_KEYS = ("class_path", "_class_path")


def _patch_class_paths(obj: object) -> object:
    """Recursively patch class_path and _class_path values in dicts."""
    if isinstance(obj, dict):
        result = {}
        for k, val in obj.items():
            if k in CLASS_PATH_KEYS and isinstance(val, str):
                patched = val
                for short, full in PREFIX_MAP.items():
                    if patched.startswith(short) and not patched.startswith(
                        "geo_deep_learning.",
                    ):
                        patched = full + patched[len(short) :]
                        break
                result[k] = patched
            else:
                result[k] = _patch_class_paths(val)
        return result
    if isinstance(obj, list):
        return [_patch_class_paths(item) for item in obj]
    return obj


def patch_checkpoint(
    checkpoint_path: str,
    output_path: str | None = None,
    *,
    weights_only: bool = True,
) -> None:
    """Patch class_path entries in a Lightning checkpoint to use full paths."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
    ckpt = _patch_class_paths(ckpt)
    out = Path(output_path or checkpoint_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    logger.info("Patched checkpoint saved to %s", out)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Patch Lightning checkpoint class_path for geo_deep_learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint (.ckpt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: overwrite in place)",
    )
    parser.add_argument(
        "--no-weights-only",
        action="store_true",
        help="Allow pickled objects (needed for some older checkpoints)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    patch_checkpoint(
        args.checkpoint,
        args.output,
        weights_only=not args.no_weights_only,
    )


if __name__ == "__main__":
    main()
