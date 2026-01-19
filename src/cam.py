"""
Grad-CAM implementation for model interpretability.

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions
of an image are most important for the model's prediction by computing gradients
of the target class score with respect to the final convolutional layer activations.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class GradCAM:
    """
    Grad-CAM implementation for ResNet18-based models.
    
    Generates heatmaps showing which image regions contribute most to predictions
    by analyzing gradients flowing into the final convolutional layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "backbone.7"):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model (must be ImageOnlyModel or compatible)
            target_layer_name: Name of target layer for Grad-CAM (default: layer4)
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Register forward and backward hooks on target layer.
        
        Hooks capture activations during forward pass and gradients during
        backward pass for computing Grad-CAM heatmaps.
        """
        def forward_hook(module, input, output):
            """Capture activations from forward pass."""
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            """Capture gradients from backward pass."""
            self.gradients = grad_output[0]

        # Find and register hooks on target layer (layer4 for ResNet18)
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_cam(
        self, input_tensor: torch.Tensor, class_idx: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            class_idx: Class index to generate CAM for (default: predicted class)
            
        Returns:
            Normalized heatmap array [H, W] with values in [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad_()

        # Forward pass to get predictions and capture activations
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # Backward pass to compute gradients
        self.model.zero_grad()
        output[0, 0].backward()

        # Extract gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # Compute importance weights via global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM as weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        # Apply ReLU and normalize to [0, 1]
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-10)

        return cam

    def overlay_heatmap(
        self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
    ) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            original_image: Original PIL Image
            heatmap: Normalized heatmap array [H, W]
            alpha: Blending factor (0.0 = all original, 1.0 = all heatmap)
            
        Returns:
            PIL Image with heatmap overlay
        """
        import matplotlib.pyplot as plt

        # Resize heatmap to match original image dimensions
        heatmap_resized = np.array(
            Image.fromarray(heatmap).resize(original_image.size, Image.BILINEAR)
        )

        # Convert heatmap to RGB colormap (jet: blue=low, red=high)
        cmap = plt.cm.jet
        heatmap_rgb = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

        # Blend original image with heatmap
        original_array = np.array(original_image.convert("RGB"))
        blended = (alpha * heatmap_rgb + (1 - alpha) * original_array).astype(np.uint8)

        return Image.fromarray(blended)


def export_gradcam_images(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    output_dir: str,
    num_images: int = 6,
    target_layer_name: str = "backbone.7",
) -> None:
    """
    Export Grad-CAM visualizations for selected images from dataset.
    
    Selects a mix of correctly and incorrectly predicted images to provide
    insight into both successful and failed predictions.
    
    Args:
        model: Trained model to visualize
        dataset: Dataset containing images to visualize
        device: Device to run inference on
        output_dir: Directory to save Grad-CAM overlays
        num_images: Number of images to export
        target_layer_name: Layer name for Grad-CAM (default: layer4)
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    gradcam = GradCAM(model, target_layer_name=target_layer_name)

    # Collect predictions for all images
    predictions = []
    labels = []
    images = []
    indices = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            
            # Handle both image-only and fusion model formats
            if len(sample) == 3:
                img, _, label = sample  # Fusion: (img, symptoms, label)
            else:
                img, label = sample  # Image-only: (img, label)

            # Get prediction
            img_tensor = img.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img_tensor)).item()

            # Load original image from disk (before normalization)
            image_name = dataset.df.iloc[idx]["ImageName"]
            original_img_path = Path(dataset.images_dir) / image_name
            original_img = Image.open(original_img_path).convert("RGB").resize((224, 224))

            predictions.append(pred)
            labels.append(label)
            images.append(original_img)
            indices.append(idx)

    predictions = np.array(predictions)
    labels = np.array(labels)
    pred_binary = (predictions >= 0.5).astype(int)

    # Select mix of correct and incorrect predictions
    correct_mask = (pred_binary == labels).astype(bool)
    incorrect_mask = ~correct_mask

    selected_indices = []
    # Prefer incorrect predictions for analysis (up to half)
    if np.sum(incorrect_mask) > 0:
        selected_indices.extend(np.where(incorrect_mask)[0][: num_images // 2])
    # Fill remaining with correct predictions
    if np.sum(correct_mask) > 0:
        selected_indices.extend(
            np.where(correct_mask)[0][: num_images - len(selected_indices)]
        )

    # Fill any remaining slots with any available images
    if len(selected_indices) < num_images:
        remaining = num_images - len(selected_indices)
        all_indices = set(range(len(dataset)))
        available = sorted(all_indices - set(selected_indices))
        selected_indices.extend(available[:remaining])

    # Generate and save Grad-CAM overlays
    transform_for_cam = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for i, idx in enumerate(selected_indices[:num_images]):
        img = images[idx]
        img_tensor = transform_for_cam(img).unsqueeze(0).to(device)

        # Generate CAM and overlay
        cam = gradcam.generate_cam(img_tensor)
        overlay = gradcam.overlay_heatmap(img, cam, alpha=0.45)

        # Save overlay
        image_name = dataset.df.iloc[idx]["ImageName"]
        output_path = Path(output_dir) / f"cam_{i+1}_{image_name}"
        overlay.save(output_path)
