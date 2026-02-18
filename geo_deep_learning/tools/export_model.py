"""Export models via torch.export."""

import argparse
import importlib
import logging

logger = logging.getLogger(__name__)

EXPORTERS = {
    "dofa": "geo_deep_learning.tasks_with_models.segmentation_dofa",
}


def run_export(
    model: str,
    checkpoint_path: str,
    output_path: str,
) -> None:
    """Dispatch to task module's export_model."""
    if model not in EXPORTERS:
        msg = f"Unknown model: {model}. Available: {list(EXPORTERS)}"
        raise ValueError(msg)
    mod = importlib.import_module(EXPORTERS[model])
    if not hasattr(mod, "export_model"):
        msg = f"{EXPORTERS[model]}.export_model not implemented"
        raise NotImplementedError(msg)
    mod.export_model(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export model via torch.export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(EXPORTERS.keys()),
        required=True,
        help="Model to export",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output (.pt2)",
    )
    args = parser.parse_args()

    run_export(args.model, args.checkpoint, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
