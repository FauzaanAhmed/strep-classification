"""Data loading and preprocessing utilities."""

import os
from pathlib import Path
from typing import Callable, List, Tuple

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

        # Get symptom columns (exclude ImageName and label)
        self.symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]

        # Convert labels to binary (Positive=1, Negative=0)
        self.labels = (self.df["label"].str.lower() == "positive").astype(int).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        image_name = row["ImageName"]
        label = self.labels[idx]

        # Load image
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
    """Get image transforms for training or validation."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(12),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def load_data(csv_path: str, images_dir: str) -> pd.DataFrame:
    """Load CSV and verify images exist."""
    df = pd.read_csv(csv_path)

    # Verify images exist
    missing_images = []
    for image_name in df["ImageName"]:
        image_path = Path(images_dir) / image_name
        if not image_path.exists():
            missing_images.append(image_name)

    if missing_images:
        raise FileNotFoundError(f"Missing images: {missing_images[:5]}...")

    # Verify labels
    unique_labels = df["label"].str.lower().unique()
    if not all(label in ["positive", "negative"] for label in unique_labels):
        raise ValueError(f"Unexpected labels: {unique_labels}")

    return df
