# GeoTIFF Segmentation Inference

Run inference on large GeoTIFF files using trained Lightning models. This pipeline handles arbitrary file sizes via sliding window chunking and includes overlap blending to minimize boundary artifacts.

## Usage

### Quick Start
```bash
python geo_deep_learning/infer.py \
    --checkpoint path/to/model.ckpt \
    --input path/to/input.tif \
    --output path/to/output.tif \
    --mean 0.485 0.456 0.406 \
    --std 0.229 0.224 0.225
```

### Full Configuration
```bash
python geo_deep_learning/infer.py \
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
    --compress lzw \
    --output-probabilities
```

## Parameters

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--checkpoint` | Path to `.ckpt` file | Required |
| `--input` / `--output` | Input/Output GeoTIFF paths | Required |
| `--mean` / `--std` | Normalization values (one per channel) | Required |
| `--device` | Execution device (`cuda`, `cpu`, `mps`) | `cuda` |
| `--tile-size` | Sliding window tile size | `512` |
| `--overlap` | Pixel overlap between tiles (Stride = tile - overlap) | `171` |
| `--batch-size` | Number of tiles per GPU pass | `16` |
| `--chunk-size` | Internal rasterio window size for large files | `4096` |
| `--output-probabilities` | Save raw probabilities instead of class labels | `False` |

## Model Integration

Models must implement these two methods to be compatible with the inference engine:

1. `preprocess(x, mean, std)`: Handles normalization and standardization.
2. `predict(x, rescale_to)`: Forward pass returning probabilities.

### Implementation Example:
```python
def predict(self, x: torch.Tensor, rescale_to: tuple[int, int] = None) -> torch.Tensor:
    logits = self(self.preprocess(x)).out
    if rescale_to:
        logits = F.interpolate(logits, size=rescale_to, mode='bilinear', align_corners=False)
    
    return logits.sigmoid() if self.num_classes == 1 else logits.softmax(dim=1)
```

## Performance & Memory

- **Large Files**: Automatically processed in chunks (`--chunk-size`) to stay within system RAM.
- **VRAM OOM**: If you hit memory limits, reduce `--batch-size` or `--tile-size`.
- **Artifacts**: Increase `--overlap` if you see visible seams at tile boundaries.
