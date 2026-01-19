"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StrepDataset(Dataset):
    """Dataset for Strep classification with images and symptoms."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        image_size: int = 224,
        transform: Callable = None,
        return_symptoms: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transform = transform
        self.return_symptoms = return_symptoms

        # Symptom columns
        self.symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]

        # Labels: Positive=1, Negative=0
        self.labels = (self.df["label"].astype(str).str.lower() == "positive").astype(int).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        image_name = row["ImageName"]
        label = int(self.labels[idx])

        image_path = self.images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.return_symptoms:
            symptoms = row[self.symptom_cols].values.astype(np.float32)
            return image, symptoms, label
        else:
            return image, label


def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Small-data-friendly transforms:
    - Train: RandomResizedCrop + mild jitter/rotation/flip
    - Val: deterministic resize
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)],
                    p=0.7,
                ),
                transforms.RandomRotation(12),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Light erasing for regularization
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def load_data(csv_path: str, images_dir: str) -> pd.DataFrame:
    """Load CSV and verify images exist."""
    df = pd.read_csv(csv_path)

    missing = []
    images_dir = Path(images_dir)
    for image_name in df["ImageName"]:
        if not (images_dir / image_name).exists():
            missing.append(image_name)

    if missing:
        raise FileNotFoundError(f"Missing images (showing up to 5): {missing[:5]}")

    labels = df["label"].astype(str).str.lower().unique()
    if not all(l in ["positive", "negative"] for l in labels):
        raise ValueError(f"Unexpected labels: {labels}")

    return df
