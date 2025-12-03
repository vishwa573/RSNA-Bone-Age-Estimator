from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def plot_scatter(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    save_path: Optional[str] = None,
    title: str = "Predicted vs True Bone Age (years)",
) -> None:
    """Scatter plot of predicted vs true ages with y=x reference line."""

    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, edgecolor="k", s=20)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("True Age (years)")
    plt.ylabel("Predicted Age (years)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix heatmap for classification results."""

    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        num_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


def visualize_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """Simple Grad-CAM implementation for a given model and layer.

    Parameters
    ----------
    model : torch.nn.Module
        The CNN model (e.g., BoneAgePredictor).
    input_tensor : torch.Tensor
        Input image tensor of shape (1, C, H, W).
    target_layer : torch.nn.Module
        Convolutional layer to visualize (e.g., model.backbone.layer4[-1].conv2).
    class_idx : int, optional
        Class index for which Grad-CAM is computed. If None, uses the
        predicted class with highest logit from the classification head.

    Returns
    -------
    heatmap : np.ndarray
        Grad-CAM heatmap normalized to [0, 1] with shape (H, W).

    Notes
    -----
    This is a lightweight implementation intended for qualitative analysis.
    """

    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, inp, out):  # type: ignore[override]
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):  # type: ignore[override]
        gradients.append(grad_out[0].detach())

    # Register hooks
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            age_pred, stage_logits = model(input_tensor)

            if class_idx is None:
                class_idx = int(stage_logits.argmax(dim=1).item())

            score = stage_logits[:, class_idx].sum()

        # Backward pass
        model.zero_grad()
        score.backward(retain_graph=True)

        # Get captured activations and gradients
        activation = activations[0]  # (B, C, H, W)
        gradient = gradients[0]  # (B, C, H, W)

        weights = gradient.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activation).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze(0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        heatmap = cam.cpu().numpy()
        return heatmap

    finally:
        fh.remove()
        bh.remove()