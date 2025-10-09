"""Visualization tools."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_prediction(  # noqa: PLR0913
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_name: str | None = None,
    num_classes: int = 1,
    class_colors: list[str] | None = None,
    save_samples: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize the input image, ground truth mask, and prediction mask side by side.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        mask (torch.Tensor): Ground truth mask tensor of shape (H, W)
        prediction (torch.Tensor): Predicted mask tensor of shape (H, W)
        sample_name (str, optional): Name of the sample
        num_classes (int): Number of classes in the segmentation
        class_colors (list, optional): List of colors for each class
        save_samples (bool, optional): Whether to save the samples
        save_path (str, optional): Path to save the visualization

    Returns:
        plt.Figure: The figure containing the visualization

    """
    num_classes = num_classes + 1 if num_classes == 1 else num_classes
    image = image.cpu().numpy()
    mask = mask.squeeze(0).long().cpu().numpy()
    prediction = prediction.cpu().numpy()

    image = np.transpose(image, (1, 2, 0))
    num_channels = image.shape[-1]
    rgb_channels = 3
    if num_channels > rgb_channels:
        image = image[..., :rgb_channels]

    # Create a color map for the masks
    if class_colors is None:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = ListedColormap(class_colors)

    sample_name = "sample" if sample_name is None else sample_name

    if save_samples and save_path is not None:
        plt.imsave(save_path / f"{sample_name}_image.png", image)
        plt.imsave(
            save_path / f"{sample_name}_mask.png",
            mask,
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
        )
        plt.imsave(
            save_path / f"{sample_name}_prediction.png",
            prediction,
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
        )

    # Create the visualization
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes = axes.reshape(num_samples, 3) if num_samples > 1 else axes.reshape(1, 3)

    # for i in range(num_samples):
    ax_image, ax_mask, ax_output = axes

    # Plot original image
    ax_image.imshow(image)
    ax_image.set_title("Input Image")
    ax_image.axis("off")
    ax_image.text(
        0.5,
        -0.1,
        f"{sample_name}",
        transform=ax_image.transAxes,
        ha="center",
        va="top",
        wrap=True,
    )

    # Plot ground truth mask
    ax_mask.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax_mask.set_title("Ground Truth Mask")
    ax_mask.axis("off")

    # Plot predicted mask
    ax_output.imshow(prediction, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax_output.set_title("Predicted Mask")
    ax_output.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return fig


def visualize_regression(  # noqa: PLR0913
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_name: str | None = None,
    save_samples: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize the input image, ground truth, and prediction for regression.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        target (torch.Tensor): Ground truth tensor of shape (1, H, W) or (H, W)
        prediction (torch.Tensor): Predicted tensor of shape (1, H, W) or (H, W)
        sample_name (str, optional): Name of the sample
        save_samples (bool, optional): Whether to save the samples
        save_path (str, optional): Path to save the visualization

    Returns:
        plt.Figure: The figure containing the visualization

    """
    image = image.cpu().numpy()
    target = target.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()

    image = np.transpose(image, (1, 2, 0))

    num_channels = image.shape[-1]
    rgb_channels = 3
    if num_channels > rgb_channels:
        image = image[..., :rgb_channels]

    sample_name = "sample" if sample_name is None else sample_name

    # Create the visualization - 1x3 grid
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_image, ax_target, ax_pred = axes

    # Plot original image
    ax_image.imshow(image)
    ax_image.set_title("Input Image", fontsize=14, fontweight="bold")
    ax_image.axis("off")
    ax_image.text(
        0.5,
        -0.05,
        f"{sample_name}",
        transform=ax_image.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # Plot ground truth
    vmin = 0.0
    vmax = 1.0
    im_target = ax_target.imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_target.set_title("Ground Truth", fontsize=14, fontweight="bold")
    ax_target.axis("off")
    plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)

    # Plot prediction
    im_pred = ax_pred.imshow(prediction, cmap="viridis", vmin=vmin, vmax=vmax)
    mae = np.abs(target - prediction).mean()
    ax_pred.set_title(
        f"Prediction (MAE: {mae:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax_pred.axis("off")
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_samples and save_path is not None:
        plt.imsave(save_path / f"{sample_name}_image.png", image)
        plt.imsave(save_path / f"{sample_name}_target.png", target, cmap="viridis")
        plt.imsave(
            save_path / f"{sample_name}_prediction.png",
            prediction,
            cmap="viridis",
        )

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return fig
