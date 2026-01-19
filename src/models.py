"""Model definitions for Strep classification."""

import torch
import torch.nn as nn
from torchvision import models


class ImageOnlyModel(nn.Module):
    """ResNet18-based image-only classifier."""

    def __init__(self, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Classify
        out = self.dropout(features)
        out = self.fc(out)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        features = self.backbone(x)
        return features.view(features.size(0), -1)


class FusionModel(nn.Module):
    """Fusion model combining image and symptom features with attention mechanism."""

    def __init__(self, num_symptoms: int, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        # Image backbone
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.image_backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Image feature projection with normalization
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Symptom MLP with normalization
        self.symptom_mlp = nn.Sequential(
            nn.Linear(num_symptoms, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 features: image and symptom
            nn.Softmax(dim=1),
        )

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, symptoms: torch.Tensor) -> torch.Tensor:
        # Image features
        image_features = self.image_backbone(image)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)

        # Symptom features
        symptom_features = self.symptom_mlp(symptoms)

        # Concatenate for attention
        combined = torch.cat([image_features, symptom_features], dim=1)

        # Attention weights
        attn_weights = self.attention(combined)  # [B, 2]
        attn_image = attn_weights[:, 0:1]  # [B, 1]
        attn_symptom = attn_weights[:, 1:2]  # [B, 1]

        # Weighted features
        weighted_image = image_features * attn_image
        weighted_symptom = symptom_features * attn_symptom

        # Final concatenation
        final_features = torch.cat([weighted_image, weighted_symptom], dim=1)
        out = self.fusion(final_features)
        return out
