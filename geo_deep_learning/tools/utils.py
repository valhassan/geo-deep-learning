"""Utility functions."""

import logging

import torch

logger = logging.getLogger(__name__)


def normalization(
    input_tensor: torch.Tensor,
    image_min: int = 0,
    image_max: int = 255,
    norm_min: float = 0.0,
    norm_max: float = 1.0,
) -> torch.Tensor:
    """Normalize the input tensor."""
    input_tensor = input_tensor.to(torch.float32)
    return (norm_max - norm_min) * (input_tensor - image_min) / (
        image_max - image_min
    ) + norm_min


def standardization(
    input_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Standardize the input tensor.

    Args:
        input_tensor: Tensor (B, C, H, W)
        mean: Mean tensor (C, 1, 1)
        std: Std tensor (C, 1, 1)

    Returns:
        Standardized tensor

    """
    input_tensor = input_tensor.to(torch.float32)
    return (input_tensor - mean) / std


def denormalization(
    image: torch.Tensor,
    mean: torch.Tensor | float | None = None,
    std: torch.Tensor | float | None = None,
    data_type_max: int = 255,
) -> torch.Tensor:
    """Denormalize the input tensor."""
    if mean is not None and std is not None:
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, device=image.device)
        if not torch.is_tensor(std):
            std = torch.tensor(std, device=image.device)

        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)

        image = image * std + mean

    return (image * data_type_max).clamp(0, data_type_max).to(torch.uint8)


def manage_bands(
    image: torch.Tensor,
    band_indices: list[int] | None = None,
) -> torch.Tensor:
    """Select specific bands from the image tensor based on band indices."""
    if band_indices is not None:
        bands = image.size(0)
        if max(band_indices) >= bands:
            msg = (
                f"Band index {max(band_indices)} "
                f"is out of range for image with {bands} bands"
            )
            raise ValueError(
                msg,
            )
        band_indices = torch.LongTensor(band_indices).to(image.device)
        return torch.index_select(image, dim=0, index=band_indices)
    return image


def _filter_state_dict_by_parts(
    state_dict: dict[str, torch.Tensor],
    load_parts: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Filter state dict to only include specified parts."""
    filtered_state_dict = {}
    loaded_parts_keys = {part: [] for part in load_parts}
    for k, v in state_dict.items():
        for part in load_parts:
            if k.startswith(f"{part}."):
                filtered_state_dict[k] = v
                loaded_parts_keys[part].append(k)
    return filtered_state_dict, loaded_parts_keys


def _apply_key_mapping(
    state_dict: dict[str, torch.Tensor],
    key_mapping: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Apply key prefix remapping to state dict."""
    remapped = {}
    for k, v in state_dict.items():
        new_key = k
        for src_prefix, dst_prefix in key_mapping.items():
            if k.startswith(f"{src_prefix}."):
                new_key = f"{dst_prefix}.{k[len(src_prefix) + 1:]}"
                break
        remapped[new_key] = v
    return remapped


def load_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    load_parts: str | list[str] | None = None,
    map_location: torch.device | None = None,
    key_mapping: dict[str, str] | None = None,
) -> tuple[list[str], list[str]] | None:
    """
    Load weights from a checkpoint into a model.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        load_parts: List of model parts to load from the source checkpoint
            (e.g., ["encoder", "neck"]). Filtering happens before key_mapping.
        map_location: Optional device mapping for loading the checkpoint
        key_mapping: Optional dict to remap checkpoint key prefixes to model key
            prefixes. E.g., {"encoder": "model.encoder"} will remap
            "encoder.layer1.weight" to "model.encoder.layer1.weight".
            Applied after load_parts filtering.

    Returns:
        Tuple of (missing_keys, unexpected_keys) if selective loading,
        or None if full loading

    """
    logger.info("Loading weights from checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    state_dict = (
        checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    )
    # Remove "model." prefix from Lightning checkpoints
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    # Filter out augmentation modules
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(("geometric_aug.", "radiometric_aug."))
    }

    if isinstance(load_parts, str):
        load_parts = [load_parts]

    # Filter by load_parts FIRST (before remapping)
    if load_parts is not None:
        state_dict, loaded_parts_keys = _filter_state_dict_by_parts(
            state_dict,
            load_parts,
        )
    else:
        loaded_parts_keys = None

    # Apply key remapping AFTER filtering
    if key_mapping:
        state_dict = _apply_key_mapping(state_dict, key_mapping)

    # Load into model
    if load_parts is None:
        model.load_state_dict(state_dict)
        return None

    result = model.load_state_dict(state_dict, strict=False)

    # Log results
    logger.info("Loaded weights for parts: %s", load_parts)
    for part in load_parts:
        num_keys = len(loaded_parts_keys[part])
        if num_keys > 0:
            logger.info("  - %s: %s parameters loaded", part, num_keys)
            examples = loaded_parts_keys[part][:1]
            if examples:
                logger.info("    Examples: %s", ", ".join(examples))
        else:
            logger.info(
                "  - %s: NO PARAMETERS FOUND - check if this part exists",
                part,
            )
    logger.info("Missing keys: %s", len(result.missing_keys))
    examples_missing_keys = result.missing_keys[:2]
    if examples_missing_keys:
        logger.info("    Examples: %s", ", ".join(examples_missing_keys))
    logger.info("Unexpected keys: %s", len(result.unexpected_keys))
    examples_unexpected_keys = result.unexpected_keys[:2]
    if examples_unexpected_keys:
        logger.info("    Examples: %s", ", ".join(examples_unexpected_keys))
    return result
