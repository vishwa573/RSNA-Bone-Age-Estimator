import os
import random
from typing import List

import cv2  # <--- Using OpenCV for robust resizing
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import Config
from main import create_dataloaders
from src.model import BoneAgePredictor
from src.utils import visualize_gradcam

def overlay_gradcam_on_image(
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original image tensor using OpenCV."""
    
    # 1. De-normalize the image tensor to get back to original RGB
    # ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_np = img_tensor.detach().cpu().numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Convert from (C, H, W) to (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))
    h, w = img_np.shape[:2]

    # 2. Process the Heatmap
    # Ensure heatmap is a numpy array and on CPU
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    heatmap = np.array(heatmap, dtype=np.float32)

    # Squeeze unnecessary dimensions (e.g., (1, 8, 8) -> (8, 8))
    if heatmap.ndim > 2:
        heatmap = np.squeeze(heatmap)

    # 3. Robust Resize using OpenCV
    # We resize the raw heatmap (e.g., 8x8) to match the image (e.g., 256x256)
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # 4. Normalize Heatmap to [0, 1]
    heatmap_resized -= np.min(heatmap_resized)
    if np.max(heatmap_resized) != 0:
        heatmap_resized /= np.max(heatmap_resized)

    # 5. Apply Colormap (Jet)
    # cm.jet returns RGBA (H, W, 4), we only need RGB (H, W, 3)
    colored_heatmap = cm.jet(heatmap_resized)[..., :3]

    # 6. Overlay
    overlay = (1 - alpha) * img_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def generate_gradcam_examples(
    num_examples: int = 5,
    seed: int = 42,
) -> None:
    """Generate Grad-CAM visualizations for a few test images."""
    
    config = Config()
    device = torch.device(config.DEVICE)

    # Re-create loaders to ensure we get the Test split
    # Note: create_dataloaders returns (train, val, test)
    _, _, test_loader = create_dataloaders(config)

    # Load best trained model
    model = BoneAgePredictor(backbone="resnet18")
    best_path = os.path.join(config.MODEL_DIR, "best_model.pt")
    
    if not os.path.exists(best_path):
        print(f"WARNING: Best model not found at {best_path}.")
        print("Please run: python main.py --mode train")
        return

    print(f"Loading model from {best_path}...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect images
    images: List[torch.Tensor] = []
    ages_true: List[float] = []
    stages_true: List[int] = []

    print("Selecting test images...")
    for batch in test_loader:
        imgs = batch["image"]
        ages = batch["age_years"]
        stages = batch["stage"]

        for i in range(imgs.size(0)):
            images.append(imgs[i])
            ages_true.append(float(ages[i].item()))
            stages_true.append(int(stages[i].item()))

            if len(images) >= num_examples:
                break
        if len(images) >= num_examples:
            break

    if not images:
        raise RuntimeError("Test loader returned no images. Check dataset path.")

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Hook into the last convolutional layer of ResNet
    # Usually: model.backbone.layer4[-1]
    target_layer = model.backbone.layer4[-1]

    print(f"Generating {len(images)} Grad-CAM visualizations...")

    for idx, (img_tensor, true_age, true_stage) in enumerate(zip(images, ages_true, stages_true)):
        input_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dim

        # Compute Grad-CAM
        # Note: visualize_gradcam handles the forward/backward pass
        heatmap = visualize_gradcam(model, input_tensor, target_layer, class_idx=None)

        # Overlay
        overlay = overlay_gradcam_on_image(img_tensor, heatmap, alpha=0.4)

        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"True Age: {true_age:.1f}y | Stage: {true_stage}")

        out_path = os.path.join(config.PLOTS_DIR, f"gradcam_example_{idx+1}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved: {out_path}")

    print("Done!")

if __name__ == "__main__":
    generate_gradcam_examples(num_examples=5)