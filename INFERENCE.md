# GeoTIFF Segmentation Inference

This document describes how to run inference on large GeoTIFF files using trained segmentation models.

## Overview

The inference pipeline supports:
- ✅ Large GeoTIFF files (handles arbitrary sizes via chunking)
- ✅ Batched sliding window inference for efficiency
- ✅ Model-agnostic design (works with any Lightning segmentation model)
- ✅ Automatic overlap blending for smooth predictions
- ✅ Output as class labels or probabilities

## Requirements

Models must implement two methods:
1. `preprocess(x, mean, std)` - Apply normalization/standardization
2. `predict(x, rescale_to)` - Forward pass returning probabilities

Currently implemented for:
- ✅ `SegmentationSegformer`

## Usage

### Basic Example

```bash
python infer.py \
    --checkpoint path/to/model.ckpt \
    --input path/to/input.tif \
    --output path/to/output.tif \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225
```

### Full Options

```bash
python infer.py \
    --checkpoint path/to/model.ckpt \
    --input path/to/input.tif \
    --output path/to/output.tif \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225 \
    --device cuda \
    --tile-size 512 \
    --overlap 171 \
    --batch-size 16 \
    --chunk-size 4096 \
    --compress lzw
```

### Output Probabilities

To output probabilities instead of class labels:

```bash
python infer.py \
    --checkpoint path/to/model.ckpt \
    --input path/to/input.tif \
    --output path/to/output_probs.tif \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225 \
    --output-probabilities
```

## Parameters

### Required
- `--checkpoint`: Path to Lightning checkpoint (.ckpt file)
- `--input`: Path to input GeoTIFF file
- `--output`: Path to output GeoTIFF file
- `--mean`: Mean values for standardization (one per channel)
- `--std`: Standard deviation values for standardization (one per channel)

### Optional
- `--device`: Device to run on (default: `cuda`)
- `--tile-size`: Size of tiles for sliding window (default: `512`)
- `--overlap`: Overlap between tiles in pixels (default: `171`)
  - Stride = tile_size - overlap
  - Larger overlap = smoother predictions but slower
- `--batch-size`: Number of tiles processed simultaneously (default: `16`)
  - Increase for faster inference (requires more VRAM)
- `--chunk-size`: Size of chunks read from large files (default: `4096`)
  - For files larger than this, processing is done in chunks
- `--compress`: Output compression method (default: `lzw`)
  - Options: `lzw`, `deflate`, `none`, etc.
- `--output-probabilities`: Output probabilities instead of class labels

## How It Works

### Architecture

```
Input GeoTIFF
    ↓
Chunked Reading (rasterio windows)
    ↓
For each chunk:
    ↓
    Sliding Window Inference (batched)
        ↓
        Tiles → Batch → Model → Predictions
        ↓
    Overlap Blending (count-based averaging)
    ↓
Write to Output GeoTIFF
```

### Memory Management

1. **Large files**: Processed in chunks (default 4096x4096)
2. **Small files**: Processed in one pass
3. **GPU memory**: Controlled by batch_size parameter
4. **Overlap handling**: Accumulator on CPU, inference on GPU

### Performance Tips

1. **Increase batch_size** for faster inference (requires more VRAM)
2. **Decrease overlap** for faster inference (may reduce quality at tile boundaries)
3. **Use COG format** (Cloud Optimized GeoTIFF) for input files when possible
4. **Adjust chunk_size** based on available RAM

## Finding Mean/Std Values

The mean and std values should match those used during training. They are typically:
1. Stored in your training configuration (Lightning CLI config)
2. Passed to the datamodule during training
3. Specific to your dataset/sensor

Example for ImageNet-pretrained models:
- Mean: `0.485 0.456 0.406` (RGB)
- Std: `0.229 0.224 0.225` (RGB)

## Extending to Other Models

To add inference support for other Lightning segmentation models:

1. Add `preprocess()` method:
```python
def preprocess(self, x: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    x = normalization(x, 0, 255, 0.0, 1.0)
    mean = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(std, device=x.device).view(-1, 1, 1)
    x = standardization(x, mean, std)
    return x
```

2. Add `predict()` method:
```python
def predict(self, x: torch.Tensor, rescale_to: tuple[int, int] | None = None) -> torch.Tensor:
    outputs = self(x)
    logits = outputs.out  # Extract main output

    if rescale_to is not None:
        logits = F.interpolate(logits, size=rescale_to, mode='bilinear', align_corners=False)

    # Apply activation
    if self.num_classes == 1:
        return logits.sigmoid()
    else:
        return logits.softmax(dim=1)
```

That's it! The inference engine will work automatically.

## Troubleshooting

### Out of Memory (OOM)
- Decrease `--batch-size`
- Decrease `--chunk-size`
- Use CPU: `--device cpu`

### Slow Inference
- Increase `--batch-size` (if VRAM available)
- Decrease `--overlap`
- Use GPU: `--device cuda`

### Tile Boundary Artifacts
- Increase `--overlap`
- Check that model was trained with similar tile sizes

### Channel Mismatch Error
- Verify input GeoTIFF has same number of channels as model expects
- Check model's `in_channels` parameter

## Example Workflow

1. Train model with Lightning CLI:
```bash
python geo_deep_learning/train.py fit --config configs/segformer.yaml
```

2. Note the mean/std from your config

3. Run inference:
```bash
python infer.py \
    --checkpoint logs/segformer/version_0/checkpoints/best.ckpt \
    --input data/large_geotiff.tif \
    --output predictions/output.tif \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225
```

4. Visualize results with QGIS or other GIS tools
