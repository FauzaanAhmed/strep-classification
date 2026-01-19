"""
Data loading and preprocessing utilities for Strep classification.

This module provides:
- StrepDataset: PyTorch dataset for images and optional symptom features
- get_transforms: Image augmentation pipelines for training/validation
- load_data: CSV loading with validation
"""

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StrepDataset(Dataset):
    """
    PyTorch dataset for Strep classification.
    
    Supports two modes:
    - Image-only: Returns (image, label)
    - With symptoms: Returns (image, symptoms, label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        image_size: int = 224,
        transform: Callable = None,
        return_symptoms: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with ImageName, label, and optional symptom columns
            images_dir: Directory containing image files
            image_size: Target image size (unused but kept for compatibility)
            transform: Image transformation pipeline
            return_symptoms: If True, return symptoms in __getitem__
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transform = transform
        self.return_symptoms = return_symptoms

        # Extract symptom column names (exclude ImageName and label)
        self.symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]

        # Convert labels to binary: Positive=1, Negative=0
        self.labels = (self.df["label"].astype(str).str.lower() == "positive").astype(int).values

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_symptoms=True: (image, symptoms, label)
            If return_symptoms=False: (image, label)
        """
        row = self.df.iloc[idx]
        image_name = row["ImageName"]
        label = int(self.labels[idx])

        # Load and validate image
        image_path = self.images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return with or without symptoms based on mode
        if self.return_symptoms:
            symptoms = row[self.symptom_cols].values.astype(np.float32)
            return image, symptoms, label
        else:
            return image, label


def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Get image transformation pipeline for training or validation.
    
    Training augmentations:
    - Random crop with resize for scale variation
    - Color jitter (brightness, contrast, saturation, hue)
    - Random rotation (±12 degrees)
    - Random horizontal flip (50% probability)
    - Random affine transformations (small translations)
    - Random erasing (10% probability, light erasing)
    
    Validation: Deterministic resize and normalization only.
    
    Args:
        image_size: Target image size (224 for ImageNet-compatible models)
        is_train: If True, apply training augmentations; else validation transforms
        
    Returns:
        Composed transformation pipeline
    """
    # ImageNet normalization statistics
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
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
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
    """
    Load dataset CSV and validate data integrity.
    
    Validates:
    - All referenced images exist in images_dir
    - Labels are valid ("Positive" or "Negative", case-insensitive)
    
    Args:
        csv_path: Path to CSV file with ImageName, label, and symptom columns
        images_dir: Directory containing image files
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If any referenced images are missing
        ValueError: If labels are invalid
    """
    df = pd.read_csv(csv_path)
    images_dir = Path(images_dir)

    # Verify all images exist
    missing = []
    for image_name in df["ImageName"]:
        if not (images_dir / image_name).exists():
            missing.append(image_name)

    if missing:
        raise FileNotFoundError(
            f"Missing images (showing up to 5): {missing[:5]}"
        )

    # Validate labels
    labels = df["label"].astype(str).str.lower().unique()
    valid_labels = {"positive", "negative"}
    if not all(label in valid_labels for label in labels):
        raise ValueError(f"Unexpected labels found: {set(labels) - valid_labels}")

    return df
