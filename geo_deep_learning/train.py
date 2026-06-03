"""Train model with Lightning CLI."""

import logging
import warnings

import torch
from lightning.pytorch.cli import ArgsType, LightningCLI

from configs import logging_config  # noqa: F401
from geo_deep_learning.tools.mlflow_logger import LoggerSaveConfigCallback

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Grad strides do not match")
warnings.filterwarnings("ignore", message="AccumulateGrad node's stream")
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


def main(args: ArgsType = None) -> None:
    """Run the main training pipeline."""
    cli = LightningCLI(
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
        args=args,
    )
    if cli.trainer.is_global_zero:
        logger.info(
            "Best model path: %s",
            cli.trainer.checkpoint_callback.best_model_path,
        )
        cli.trainer.logger.log_hyperparams(
            {"best_model_path": cli.trainer.checkpoint_callback.best_model_path},
        )
        logger.info("Done!")


if __name__ == "__main__":
    main()
