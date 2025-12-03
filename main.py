import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.config import Config
from src.dataset import BoneAgeDataset, get_transforms
from src.model import BoneAgePredictor
from src.train import Trainer
from src.utils import plot_confusion_matrix, plot_scatter

# <--- Import the comparison function we just created
from src.evaluate import compare_approaches


def create_dataloaders(
    config: Config,
    use_processed: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with a 70/15/15 split.

    Splits are stratified by developmental stage to keep class balance across sets.

    If ``use_processed`` is True, images are loaded from ``config.PROCESSED_DIR``
    instead of ``config.DATA_DIR``. This is useful when images have been
    pre-resized and saved to disk using ``preprocess.py``.
    """

    csv_path = os.path.join(config.DATA_DIR, config.CSV_FILENAME)

    # Determine where raw images live.
    kaggle_train_dir = os.path.join(
        config.DATA_DIR, "boneage-training-dataset", "boneage-training-dataset"
    )
    raw_img_root = kaggle_train_dir if os.path.isdir(kaggle_train_dir) else config.DATA_DIR

    img_dir = config.PROCESSED_DIR if use_processed else raw_img_root

    # Full dataset (we will split indices)
    full_dataset = BoneAgeDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        transform=None,  # transforms will be set per split
    )

    indices = np.arange(len(full_dataset))
    stages = full_dataset.df["stage"].values

    train_idx, temp_idx, _, temp_stages = train_test_split(
        indices,
        stages,
        test_size=0.30,
        stratify=stages,
        random_state=config.SEED,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_stages,
        random_state=config.SEED,
    )

    # Create subset datasets with appropriate transforms
    train_transform = get_transforms(config.IMG_SIZE, train=True)
    eval_transform = get_transforms(config.IMG_SIZE, train=False)

    # We wrap BoneAgeDataset again for each subset so that each has its own transform
    train_dataset = BoneAgeDataset(csv_path=csv_path, img_dir=img_dir, transform=train_transform)
    val_dataset = BoneAgeDataset(csv_path=csv_path, img_dir=img_dir, transform=eval_transform)
    test_dataset = BoneAgeDataset(csv_path=csv_path, img_dir=img_dir, transform=eval_transform)

    train_dataset = Subset(train_dataset, train_idx.tolist())
    val_dataset = Subset(val_dataset, val_idx.tolist())
    test_dataset = Subset(test_dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader


def run_train(config: Config, use_processed: bool = False) -> None:
    train_loader, val_loader, test_loader = create_dataloaders(config, use_processed=use_processed)

    model = BoneAgePredictor(backbone="resnet18")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )

    trainer.train()


def run_evaluate(config: Config, use_processed: bool = False) -> None:
    """Load best model and run evaluation & plotting on the test set."""

    train_loader, val_loader, test_loader = create_dataloaders(config, use_processed=use_processed)
    if test_loader is None:
        raise RuntimeError("Test loader could not be created.")

    device = torch.device(config.DEVICE)
    model = BoneAgePredictor(backbone="resnet18")
    best_path = os.path.join(config.MODEL_DIR, "best_model.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best model not found at {best_path}. Train the model first.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # -------------------------------------------------------------
    # 1. Standard Metrics Evaluation (using Trainer's evaluate logic)
    # -------------------------------------------------------------
    # We can reuse the Trainer class to run the standard evaluation suite
    # (MAE, RMSE, QWK, Bias Analysis) just like we do at end of training.
    print("Running standard evaluation on Test Set...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader, # Not used here but required by init
        val_loader=val_loader,     # Not used here
        test_loader=test_loader,
        config=config
    )
    trainer.evaluate(split="test")

    # -------------------------------------------------------------
    # 2. Compare Approaches (New Requirement)
    # -------------------------------------------------------------
    # This prints the table comparing Direct Classification vs Regression Thresholding
    compare_approaches(model, test_loader, device)


    # -------------------------------------------------------------
    # 3. Generate Plots
    # -------------------------------------------------------------
    print("\nGenerating final plots...")
    # Collect predictions and targets on test set for plotting
    all_true_ages = []
    all_pred_ages = []
    all_true_stages = []
    all_pred_stages = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            ages = batch["age_years"].to(device)
            stages = batch["stage"].to(device)

            age_pred, stage_logits = model(images)
            preds_stage = stage_logits.argmax(dim=1)

            all_true_ages.extend(ages.cpu().numpy().tolist())
            all_pred_ages.extend(age_pred.cpu().numpy().tolist())
            all_true_stages.extend(stages.cpu().numpy().tolist())
            all_pred_stages.extend(preds_stage.cpu().numpy().tolist())

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Regression scatter plot
    scatter_path = os.path.join(config.PLOTS_DIR, "scatter_pred_vs_true_test.png")
    plot_scatter(all_true_ages, all_pred_ages, save_path=scatter_path)
    print(f"Saved regression scatter plot to {scatter_path}")

    # Classification confusion matrix
    class_names = ["Child (<12)", "Adolescent (12-18)", "Adult (>18)"]
    cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrix_test.png")
    plot_confusion_matrix(all_true_stages, all_pred_stages, class_names=class_names, save_path=cm_path)
    print(f"Saved classification confusion matrix to {cm_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bone Age Prediction Project")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Mode to run: 'train' to train the model, 'evaluate' to evaluate best model on test set.",
    )

    # Optional overrides for some config parameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--use_processed",
        action="store_true",
        help=(
            "If set, load images from the processed directory (Config.PROCESSED_DIR) "
            "instead of the raw directory (Config.DATA_DIR). Use this after running preprocess.py."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config()
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LR = args.lr

    use_processed = bool(getattr(args, "use_processed", False))

    if args.mode == "train":
        run_train(config, use_processed=use_processed)
    elif args.mode == "evaluate":
        run_evaluate(config, use_processed=use_processed)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()