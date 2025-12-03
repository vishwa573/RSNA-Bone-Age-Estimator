import os
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import Config


def _load_raw_image(img_dir: str, img_id: Any) -> Image.Image:
    """Load raw image from the original data directory, trying several extensions.

    This mirrors the logic in BoneAgeDataset._load_image but is local to this script
    so it can run without importing torchvision or torch.
    """

    base_name = str(img_id)
    possible_exts = [".png", ".jpg", ".jpeg"]

    for ext in possible_exts:
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")

    raise FileNotFoundError(
        f"Image for id={img_id} not found with extensions {possible_exts} in {img_dir}"
    )


def preprocess_images(
    config: Config,
    overwrite: bool = False,
) -> None:
    """Preprocess all images once and save resized copies to the processed folder.

    Operations performed:
    - Load each image referenced in the training CSV from `DATA_DIR`.
    - Resize to (IMG_SIZE, IMG_SIZE) as defined in Config.
    - Save as PNG into `PROCESSED_DIR` with the same `<id>.png` naming.

    This allows training to read already-resized images from disk, avoiding
    repeated CPU cost of resizing every epoch.
    """

    # CSV always lives in DATA_DIR root
    csv_dir = config.DATA_DIR
    output_dir = config.PROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, config.CSV_FILENAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        raise ValueError("CSV must contain an 'id' column for image filenames")

    ids = df["id"].tolist()

    # Determine raw image root. For Kaggle RSNA download, images live under
    #   data/raw/boneage-training-dataset/boneage-training-dataset/
    kaggle_train_dir = os.path.join(
        config.DATA_DIR, "boneage-training-dataset", "boneage-training-dataset"
    )
    image_root = kaggle_train_dir if os.path.isdir(kaggle_train_dir) else config.DATA_DIR

    print(
        f"Preprocessing {len(ids)} images from '{image_root}' to '{output_dir}' "
        f"with size {config.IMG_SIZE}x{config.IMG_SIZE}..."
    )

    skipped = 0
    for img_id in tqdm(ids, desc="Preprocessing images"):
        out_path = os.path.join(output_dir, f"{img_id}.png")
        if os.path.exists(out_path) and not overwrite:
            # Already processed
            continue

        try:
            img = _load_raw_image(image_root, img_id)
        except FileNotFoundError:
            skipped += 1
            continue

        img_resized = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)

        # Save as PNG
        img_resized.save(out_path)

    print(
        "Preprocessing complete. "
        f"Skipped {skipped} entries without images. "
        f"Processed images are stored in '{output_dir}'."
    )


if __name__ == "__main__":
    cfg = Config()
    preprocess_images(cfg, overwrite=False)
