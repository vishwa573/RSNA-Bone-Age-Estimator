import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(img_size: int, train: bool = True) -> Callable:
    """Return image transformations for training/validation.

    All images are resized to a fixed size and normalized with ImageNet stats.
    """

    base_transforms = [
        transforms.Resize((img_size, img_size)),
    ]

    if train:
        # Simple augmentations suitable for X-ray images
        base_transforms.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        )

    base_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(base_transforms)


class BoneAgeDataset(Dataset):
    """Dataset for RSNA Bone Age images.

    Expects a CSV with columns:
        - id: image identifier (without extension)
        - boneage: in months
        - sex or male: either 'M'/'F' or 1/0 / True/False

    Additional derived columns:
        - age_years: boneage converted to years
        - stage: discretized developmental stage (0=Child, 1=Adolescent, 2=Adult)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Standardize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "boneage" not in df.columns:
            raise ValueError("CSV must contain a 'boneage' column (in months)")

        if "id" not in df.columns:
            raise ValueError("CSV must contain an 'id' column for image filenames")

        # Handle sex column variations: 'sex' (M/F) or 'male' (0/1 or bool)
        if "sex" in df.columns:
            # Expect 'M'/'F' or similar
            df["sex"] = df["sex"].astype(str).str.upper().str[0]
            df["sex_int"] = (df["sex"] == "M").astype(int)
        elif "male" in df.columns:
            # Already numeric/boolean male indicator
            df["sex_int"] = df["male"].astype(int)
        else:
            raise ValueError("CSV must contain a 'sex' or 'male' column")

        # Convert bone age from months to years for regression target
        df["age_years"] = df["boneage"].astype(float) / 12.0

        # Discretize into developmental stages for classification
        # Child: < 12 years -> 0
        # Adolescent: 12-18 years -> 1
        # Adult: > 18 years -> 2
        def discretize_stage(age_years: float) -> int:
            if age_years < 12.0:
                return 0
            elif age_years <= 18.0:
                return 1
            else:
                return 2

        df["stage"] = df["age_years"].apply(discretize_stage).astype(int)

        # Filter out rows whose corresponding image file is missing to avoid runtime errors
        possible_exts = [".png", ".jpg", ".jpeg"]

        def has_image(img_id: Any) -> bool:
            base_name = str(img_id)
            for ext in possible_exts:
                img_path = os.path.join(self.img_dir, base_name + ext)
                if os.path.exists(img_path):
                    return True
            return False

        before_count = len(df)
        df = df[df["id"].apply(has_image)].reset_index(drop=True)
        after_count = len(df)
        if after_count < before_count:
            print(
                f"[BoneAgeDataset] Filtered out {before_count - after_count} samples without images. "
                f"Remaining: {after_count}."
            )

        self.df = df

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.df)

    def _load_image(self, img_id: Any) -> Image.Image:
        """Load image, trying .png then .jpg extensions.

        Images are loaded as RGB for compatibility with ImageNet-pretrained models.
        """

        base_name = str(img_id)
        possible_exts = [".png", ".jpg", ".jpeg"]

        for ext in possible_exts:
            img_path = os.path.join(self.img_dir, base_name + ext)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                return img

        raise FileNotFoundError(
            f"Image for id={img_id} not found with extensions {possible_exts} in {self.img_dir}"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        row = self.df.iloc[idx]

        img = self._load_image(row["id"])
        if self.transform is not None:
            img = self.transform(img)

        age_years = float(row["age_years"])
        stage = int(row["stage"])
        sex_int = int(row["sex_int"])

        sample = {
            "image": img,
            "age_years": torch.tensor(age_years, dtype=torch.float32),
            "stage": torch.tensor(stage, dtype=torch.long),
            "sex": torch.tensor(sex_int, dtype=torch.long),
        }

        return sample