"""
Model architectures for Strep classification.

This module defines two model architectures:
- ImageOnlyModel: ResNet18-based classifier using only image features
- FusionModel: Attention-based fusion of image and symptom features
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageOnlyModel(nn.Module):
    """
    Image-only classifier using ResNet18 transfer learning.
    
    Architecture:
    - Backbone: ResNet18 (ImageNet pretrained) without final FC layer
    - Feature extraction: 512-dimensional embeddings
    - Classifier: Dropout → Linear(512 → 1)
    """

    def __init__(self, dropout: float = 0.2, pretrained: bool = True):
        """
        Initialize image-only model.
        
        Args:
            dropout: Dropout probability for classifier head
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Remove final FC layer to extract features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Binary classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features and classify.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Logits tensor [B, 1]
        """
        # Extract 512-dimensional features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Apply dropout and classify
        out = self.dropout(features)
        out = self.fc(out)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without classification.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Feature tensor [B, 512]
        """
        features = self.backbone(x)
        return features.view(features.size(0), -1)


class FusionModel(nn.Module):
    """
    Attention-based fusion model combining image and symptom features.
    
    Architecture:
    - Image branch: ResNet18 → Projection (512 → 256) with BatchNorm
    - Symptom branch: MLP (num_symptoms → 128 → 64) with BatchNorm
    - Attention: Learns dynamic weights for image vs. symptom features
    - Fusion: 3-layer MLP classifier (320 → 128 → 64 → 1)
    """

    def __init__(self, num_symptoms: int, dropout: float = 0.2, pretrained: bool = True):
        """
        Initialize fusion model.
        
        Args:
            num_symptoms: Number of symptom features
            dropout: Dropout probability
            pretrained: Whether to use ImageNet pretrained weights for image backbone
        """
        super().__init__()
        # Image feature extraction backbone
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.image_backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Project image features to 256-dim with normalization
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Process symptom features with normalization
        self.symptom_mlp = nn.Sequential(
            nn.Linear(num_symptoms, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Attention mechanism: learns weights for image vs. symptom features
        self.attention = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output 2 weights: [image_weight, symptom_weight]
            nn.Softmax(dim=1),
        )

        # Final fusion classifier
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
        """
        Forward pass with attention-based feature fusion.
        
        Args:
            image: Input image tensor [B, 3, H, W]
            symptoms: Symptom features tensor [B, num_symptoms]
            
        Returns:
            Logits tensor [B, 1]
        """
        # Extract and project image features
        image_features = self.image_backbone(image)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)  # [B, 256]

        # Process symptom features
        symptom_features = self.symptom_mlp(symptoms)  # [B, 64]

        # Compute attention weights from combined features
        combined = torch.cat([image_features, symptom_features], dim=1)  # [B, 320]
        attn_weights = self.attention(combined)  # [B, 2]
        attn_image = attn_weights[:, 0:1]  # [B, 1]
        attn_symptom = attn_weights[:, 1:2]  # [B, 1]

        # Apply attention weights to features
        weighted_image = image_features * attn_image
        weighted_symptom = symptom_features * attn_symptom

        # Concatenate weighted features and classify
        final_features = torch.cat([weighted_image, weighted_symptom], dim=1)  # [B, 320]
        out = self.fusion(final_features)
        return out
