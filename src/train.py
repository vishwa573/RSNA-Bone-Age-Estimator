from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .evaluate import (
    classification_metrics,
    gender_bias_classification,
    gender_bias_regression,
    regression_metrics,
)


class Trainer:
    """Trainer for multi-task bone age prediction (regression + classification)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Config = Config(),
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)

        # Loss functions
        # L1Loss (MAE) for regression and CrossEntropyLoss for classification
        self.reg_criterion = nn.L1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )

        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def _step(
        self, batch: Dict[str, torch.Tensor], train: bool = True
    ) -> Tuple[float, float, float]:
        images = batch["image"].to(self.device)
        ages = batch["age_years"].to(self.device)
        stages = batch["stage"].to(self.device)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        age_pred, stage_logits = self.model(images)

        reg_loss = self.reg_criterion(age_pred, ages)
        cls_loss = self.cls_criterion(stage_logits, stages)
        total_loss = reg_loss + cls_loss

        if train:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

        return total_loss.item(), reg_loss.item(), cls_loss.item()

    def train(self) -> None:
        best_val_loss = float("inf")
        history: Dict[str, List[float]] = {
            "train_total": [],
            "train_reg": [],
            "train_cls": [],
            "val_total": [],
            "val_reg": [],
            "val_cls": [],
        }

        for epoch in range(1, self.config.EPOCHS + 1):
            # Training
            self.model.train()
            train_losses = []
            train_reg_losses = []
            train_cls_losses = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.EPOCHS} [Train]")
            for batch in pbar:
                total_loss, reg_loss, cls_loss = self._step(batch, train=True)

                train_losses.append(total_loss)
                train_reg_losses.append(reg_loss)
                train_cls_losses.append(cls_loss)

                pbar.set_postfix(
                    {
                        "total": f"{np.mean(train_losses):.4f}",
                        "reg": f"{np.mean(train_reg_losses):.4f}",
                        "cls": f"{np.mean(train_cls_losses):.4f}",
                    }
                )

            history["train_total"].append(float(np.mean(train_losses)))
            history["train_reg"].append(float(np.mean(train_reg_losses)))
            history["train_cls"].append(float(np.mean(train_cls_losses)))

            # Validation
            val_total, val_reg, val_cls = self.evaluate(split="val")
            history["val_total"].append(val_total)
            history["val_reg"].append(val_reg)
            history["val_cls"].append(val_cls)

            print(
                f"Epoch {epoch}: Train total={history['train_total'][-1]:.4f}, "
                f"Val total={val_total:.4f}, Val reg={val_reg:.4f}, Val cls={val_cls:.4f}"
            )

            # Save best model based on validation total loss
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_path = os.path.join(self.config.MODEL_DIR, "best_model.pt")
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config.__dict__,
                    },
                    best_path,
                )
                print(f"Saved best model to {best_path}")

        # Optionally evaluate on test set using best model
        if self.test_loader is not None:
            print("Evaluating best model on test set...")
            best_path = os.path.join(self.config.MODEL_DIR, "best_model.pt")
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            self.evaluate(split="test")

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Tuple[float, float, float]:
        """Evaluate the model on the given split.

        Returns total_loss, reg_loss, cls_loss.
        Also prints detailed metrics and gender bias analysis.
        """

        loader = self.val_loader if split == "val" else self.test_loader
        if loader is None:
            raise ValueError(f"No DataLoader provided for split='{split}'")

        self.model.eval()

        total_losses: List[float] = []
        reg_losses: List[float] = []
        cls_losses: List[float] = []

        all_true_ages: List[float] = []
        all_pred_ages: List[float] = []
        all_true_stages: List[int] = []
        all_pred_stages: List[int] = []
        all_genders: List[int] = []

        for batch in tqdm(loader, desc=f"Evaluating [{split}]"):
            images = batch["image"].to(self.device)
            ages = batch["age_years"].to(self.device)
            stages = batch["stage"].to(self.device)
            genders = batch["sex"].cpu().numpy().tolist()

            age_pred, stage_logits = self.model(images)

            reg_loss = self.reg_criterion(age_pred, ages)
            cls_loss = self.cls_criterion(stage_logits, stages)
            total_loss = reg_loss + cls_loss

            total_losses.append(total_loss.item())
            reg_losses.append(reg_loss.item())
            cls_losses.append(cls_loss.item())

            all_true_ages.extend(ages.cpu().numpy().tolist())
            all_pred_ages.extend(age_pred.cpu().numpy().tolist())

            preds_stage = stage_logits.argmax(dim=1)
            all_true_stages.extend(stages.cpu().numpy().tolist())
            all_pred_stages.extend(preds_stage.cpu().numpy().tolist())
            all_genders.extend(genders)

        total_loss_mean = float(np.mean(total_losses))
        reg_loss_mean = float(np.mean(reg_losses))
        cls_loss_mean = float(np.mean(cls_losses))

        # Compute and print metrics
        reg_results = regression_metrics(np.array(all_true_ages), np.array(all_pred_ages))
        cls_results = classification_metrics(
            np.array(all_true_stages), np.array(all_pred_stages)
        )
        reg_bias = gender_bias_regression(
            np.array(all_true_ages), np.array(all_pred_ages), np.array(all_genders)
        )
        cls_bias = gender_bias_classification(
            np.array(all_true_stages), np.array(all_pred_stages), np.array(all_genders)
        )

        print(
            f"[{split}] Losses: total={total_loss_mean:.4f}, reg={reg_loss_mean:.4f}, cls={cls_loss_mean:.4f}"
        )
        print(
            f"[{split}] Regression metrics: MAE={reg_results['mae']:.4f}, "
            f"RMSE={reg_results['rmse']:.4f}, R2={reg_results['r2']:.4f}"
        )
        print(
            f"[{split}] Classification metrics: Acc={cls_results['accuracy']:.4f}, "
            f"F1={cls_results['f1_weighted']:.4f}, QWK={cls_results['qwk']:.4f}"
        )

        print(
            f"[{split}] Gender bias (regression MAE): male={reg_bias['mae_male']:.4f}, "
            f"female={reg_bias['mae_female']:.4f}, diff={reg_bias['mae_diff_male_minus_female']:.4f}"
        )
        print(
            f"[{split}] Gender bias (classification Acc): male={cls_bias['acc_male']:.4f}, "
            f"female={cls_bias['acc_female']:.4f}, diff={cls_bias['acc_diff_male_minus_female']:.4f}"
        )

        return total_loss_mean, reg_loss_mean, cls_loss_mean