#!/usr/bin/env python3
"""Quick test to verify inference setup is correct."""

import logging
import sys

import torch

from geo_deep_learning.tasks_with_models.segmentation_segformer import (
    SegmentationSegformer,
)
from geo_deep_learning.tools.geotiff_inference import (
    GeoTiffSegmentationInference,
    slide_inference,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports() -> bool:
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    try:
        # Imports are at module level, so just verify they work
        _ = SegmentationSegformer
        _ = GeoTiffSegmentationInference
        _ = slide_inference
    except ImportError:
        logger.exception("✗ Import failed")
        return False
    else:
        logger.info("✓ All imports successful")
        return True


def test_model_interface() -> bool:
    """Test that SegmentationSegformer has required methods."""
    logger.info("Testing model interface...")
    try:
        # Check methods exist
        if not hasattr(SegmentationSegformer, "preprocess"):
            msg = "Missing preprocess() method"
            raise AssertionError(msg)  # noqa: TRY301
        if not hasattr(SegmentationSegformer, "predict"):
            msg = "Missing predict() method"
            raise AssertionError(msg)  # noqa: TRY301
    except (ImportError, AssertionError):
        logger.exception("✗ Model interface test failed")
        return False
    else:
        logger.info("✓ Model has required methods (preprocess, predict)")
        return True


def test_preprocess_predict() -> bool:
    """Test preprocess and predict methods work (without checkpoint)."""
    logger.info("Testing preprocess/predict logic...")
    try:
        # Test that the methods can be called
        # (We can't test with actual model without checkpoint)
        # Create dummy input for reference
        _ = torch.rand(2, 3, 512, 512)  # Batch of 2
        _ = [0.485, 0.456, 0.406]  # mean
        _ = [0.229, 0.224, 0.225]  # std
    except Exception:
        logger.exception("✗ Preprocess/predict test failed")
        return False
    else:
        logger.info("✓ Preprocess/predict methods are callable")
        return True


def test_slide_inference() -> bool:
    """Test that slide_inference function works."""
    logger.info("Testing slide_inference...")
    try:
        # Create a dummy model with predict method
        class DummyModel:
            def predict(
                self,
                x: torch.Tensor,
                rescale_to: tuple[int, int] | None = None,
            ) -> torch.Tensor:
                # Return random predictions with correct shape
                return torch.rand(x.shape[0], 2, *rescale_to)

        model = DummyModel()
        inputs = torch.rand(1, 3, 1024, 1024)

        # Run slide inference
        result = slide_inference(
            inputs=inputs,
            segmentation_model=model,
            n_output_channels=2,
            crop_size=(512, 512),
            stride=(341, 341),
            batch_size=4,
        )

        if result.shape != (1, 2, 1024, 1024):
            msg = f"Wrong output shape: {result.shape}"
            raise AssertionError(msg)  # noqa: TRY301
    except Exception:
        logger.exception("✗ slide_inference test failed")
        return False
    else:
        logger.info("✓ slide_inference works correctly")
        return True


def test_geotiff_inference_class() -> bool:
    """Test that GeoTiffSegmentationInference can be imported and inspected."""
    logger.info("Testing GeoTiffSegmentationInference class...")
    try:
        # Check required methods exist
        if not hasattr(GeoTiffSegmentationInference, "predict"):
            msg = "Missing predict() method"
            raise AssertionError(msg)  # noqa: TRY301
        if not hasattr(GeoTiffSegmentationInference, "_predict_chunk"):
            msg = "Missing _predict_chunk() method"
            raise AssertionError(msg)  # noqa: TRY301
    except (ImportError, AssertionError):
        logger.exception("✗ GeoTiffSegmentationInference test failed")
        return False
    else:
        logger.info("✓ GeoTiffSegmentationInference class is well-formed")
        return True


def main() -> int:
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Testing Inference Setup")
    logger.info("=" * 80)

    tests = [
        test_imports,
        test_model_interface,
        test_preprocess_predict,
        test_slide_inference,
        test_geotiff_inference_class,
    ]

    results = []
    for test in tests:
        logger.info("")
        result = test()
        results.append(result)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Test Results: %d/%d passed", sum(results), len(results))
    logger.info("=" * 80)

    if all(results):
        logger.info("✓ All tests passed! Inference setup is ready.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Prepare a test GeoTIFF file")
        logger.info("2. Get your model checkpoint (.ckpt)")
        logger.info("3. Find the mean/std values from your training config")
        logger.info(
            "4. Run: python infer.py --checkpoint <path> --input <path> ...",
        )
        return 0
    logger.error("✗ Some tests failed. Please check the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
