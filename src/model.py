from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class BoneAgePredictor(nn.Module):
    """ResNet-based multi-task model for bone age regression and stage classification.

    The network uses an ImageNet-pretrained ResNet backbone and two heads:
        - reg_head: predicts bone age in years (regression)
        - cls_head: predicts developmental stage (3 classes)
    """

    def __init__(self, backbone: str = "resnet18", num_classes: int = 3) -> None:
        super().__init__()

        if backbone == "resnet18":
            try:
                # Newer torchvision API
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except AttributeError:  # pragma: no cover - for older torchvision
                self.backbone = models.resnet18(pretrained=True)
        elif backbone == "resnet50":
            try:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except AttributeError:  # pragma: no cover
                self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace the final fully connected layer with an identity to get features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Two task-specific heads
        self.reg_head = nn.Linear(in_features, 1)
        self.cls_head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Forward pass.

        Returns
        -------
        age_pred : torch.Tensor
            Shape (batch_size,), predicted age in years.
        stage_logits : torch.Tensor
            Shape (batch_size, num_classes), logits for developmental stage.
        """

        features = self.backbone(x)  # (batch_size, in_features)

        age_pred = self.reg_head(features).squeeze(1)  # (batch_size,)
        stage_logits = self.cls_head(features)  # (batch_size, num_classes)

        return age_pred, stage_logits
