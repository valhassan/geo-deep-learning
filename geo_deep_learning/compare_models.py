"""Compare exported model and Lightning model outputs."""

import argparse
import logging

import torch
from lightning.pytorch import LightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WAVELENGTHS = [0.66, 0.55, 0.48, 0.83]
MEAN = [0.1014, 0.1360, 0.1296, 0.2604]
STD = [0.1102, 0.1230, 0.1107, 0.2099]


def _load_models(
    checkpoint_path: str,
    exported_path: str,
) -> tuple[LightningModule, torch.export.ExportedProgram]:
    """Load Lightning and torch.export models."""
    logger.info("Loading Lightning model from: %s", checkpoint_path)
    lightning_model = LightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )
    lightning_model.model.eval()
    logger.info("Loading torch.export model from: %s", exported_path)
    exported_program = torch.export.load(exported_path)
    return lightning_model, exported_program


def _create_test_input(
    num_channels: int,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create and preprocess test input."""
    torch.manual_seed(42)
    h, w = image_size
    logger.info("\nCreating test input: shape (1, %d, %d, %d)", num_channels, h, w)
    raw_input = torch.rand(1, num_channels, h, w) * 255
    mean_t = torch.tensor(MEAN).view(1, -1, 1, 1)
    std_t = torch.tensor(STD).view(1, -1, 1, 1)
    preprocessed = (raw_input / 255.0 - mean_t) / std_t
    wavelengths_t = torch.tensor(WAVELENGTHS, dtype=torch.float32)
    logger.info(
        "Input stats - min: %.4f, max: %.4f",
        preprocessed.min(),
        preprocessed.max(),
    )
    return preprocessed, wavelengths_t, mean_t, std_t


def _run_inference(
    lightning_model: LightningModule,
    exported_program: torch.export.ExportedProgram,
    preprocessed: torch.Tensor,
    wavelengths_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both models and return outputs."""
    logger.info("\nRunning Lightning model...")
    with torch.no_grad():
        lightning_out = lightning_model.model(preprocessed, wavelengths_t)
    logger.info(
        "Lightning output shape: %s, min: %.4f, max: %.4f",
        lightning_out.out.shape,
        lightning_out.out.min().item(),
        lightning_out.out.max().item(),
    )
    logger.info("Running torch.export model...")
    exported_out = exported_program.module()(preprocessed, wavelengths_t)
    exported_out_tensor = (
        exported_out[0] if isinstance(exported_out, tuple) else exported_out
    )
    logger.info(
        "torch.export output shape: %s, min: %.4f, max: %.4f",
        exported_out_tensor.shape,
        exported_out_tensor.min().item(),
        exported_out_tensor.max().item(),
    )
    return lightning_out, exported_out_tensor


def _log_comparison(
    lightning_out: torch.Tensor,
    exported_out_tensor: torch.Tensor,
) -> None:
    """Log comparison results and per-channel stats if needed."""
    diff = torch.abs(lightning_out.out - exported_out_tensor)
    logger.info("\n%s", "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info("Max absolute difference: %.6f", diff.max().item())
    logger.info("Mean absolute difference: %.6f", diff.mean().item())
    logger.info("Std of difference: %.6f", diff.std().item())
    rtol, atol = 1e-4, 1e-4
    is_close = torch.allclose(
        lightning_out.out,
        exported_out_tensor,
        rtol=rtol,
        atol=atol,
    )
    if is_close:
        logger.info("✓ PASS: Outputs are close (rtol=%.0e, atol=%.0e)", rtol, atol)
    else:
        logger.warning("✗ FAIL: Outputs differ significantly!")
        for c in range(lightning_out.out.shape[1]):
            ch_diff = diff[0, c]
            logger.info(
                "  Channel %d: max_diff=%.6f, mean_diff=%.6f",
                c,
                ch_diff.max().item(),
                ch_diff.mean().item(),
            )


def _log_softmax_comparison(
    lightning_out: torch.Tensor,
    exported_out_tensor: torch.Tensor,
) -> None:
    """Log softmax and class prediction comparison."""
    logger.info("\n%s", "=" * 60)
    logger.info("SOFTMAX COMPARISON (actual inference output)")
    logger.info("=" * 60)
    lightning_probs = torch.softmax(lightning_out.out, dim=1)
    exported_probs = torch.softmax(exported_out_tensor, dim=1)
    prob_diff = torch.abs(lightning_probs - exported_probs)
    logger.info("Max probability difference: %.6f", prob_diff.max().item())
    logger.info("Mean probability difference: %.6f", prob_diff.mean().item())
    match_pct = (
        lightning_probs.argmax(dim=1) == exported_probs.argmax(dim=1)
    ).float().mean().item() * 100
    logger.info("Class prediction match: %.2f%%", match_pct)


def _test_dynamic_batch(  # noqa: PLR0913
    exported_program: torch.export.ExportedProgram,
    num_channels: int,
    image_size: tuple[int, int],
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    wavelengths_t: torch.Tensor,
) -> None:
    """Test exported model with batch size 2."""
    logger.info("\n%s", "=" * 60)
    logger.info("Testing dynamic batch dimension...")
    logger.info("=" * 60)
    h, w = image_size
    try:
        batch_2_input = torch.rand(2, num_channels, h, w) * 255
        batch_2_preprocessed = (batch_2_input / 255.0 - mean_t) / std_t
        with torch.no_grad():
            batch_2_out = exported_program.module()(batch_2_preprocessed, wavelengths_t)
        if isinstance(batch_2_out, tuple):
            batch_2_out = batch_2_out[0]
        logger.info("✓ Dynamic batch works! Output shape: %s", batch_2_out.shape)
    except RuntimeError as e:
        logger.warning("✗ Dynamic batch failed: %s", e)


def compare_models(checkpoint_path: str, exported_path: str) -> None:
    """Compare Lightning and exported model outputs."""
    num_channels = len(WAVELENGTHS)
    image_size = (512, 512)
    lightning_model, exported_program = _load_models(checkpoint_path, exported_path)
    preprocessed, wavelengths_t, mean_t, std_t = _create_test_input(
        num_channels,
        image_size,
    )
    lightning_out, exported_out_tensor = _run_inference(
        lightning_model,
        exported_program,
        preprocessed,
        wavelengths_t,
    )
    _log_comparison(lightning_out, exported_out_tensor)
    _log_softmax_comparison(lightning_out, exported_out_tensor)
    _test_dynamic_batch(
        exported_program,
        num_channels,
        image_size,
        mean_t,
        std_t,
        wavelengths_t,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Lightning and torch.export model outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--exported",
        type=str,
        required=True,
        help="Path to torch.export model (.pt2)",
    )
    args = parser.parse_args()
    compare_models(checkpoint_path=args.checkpoint, exported_path=args.exported)


if __name__ == "__main__":
    main()
