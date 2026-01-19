"""Grad-CAM implementation for model interpretability."""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GradCAM:
    """Grad-CAM for ResNet18."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "backbone.7"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Get the target layer (layer4 for ResNet18)
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate CAM heatmap."""
        self.model.eval()
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        output[0, 0].backward()

        # Compute CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        # Weighted sum
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-10)

        return cam

    def overlay_heatmap(
        self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
    ) -> Image.Image:
        """Overlay heatmap on original image."""
        import matplotlib.pyplot as plt

        # Resize heatmap to match image size
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(original_image.size, Image.BILINEAR))

        # Convert heatmap to RGB
        cmap = plt.cm.jet
        heatmap_rgb = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

        # Blend
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
    """Export Grad-CAM overlays for selected images."""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    gradcam = GradCAM(model, target_layer_name=target_layer_name)

    # Get predictions for all images
    predictions = []
    labels = []
    images = []
    indices = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if len(sample) == 3:
                # Fusion model: (img, symptoms, label)
                img, _, label = sample
            else:
                # Image-only model: (img, label)
                img, label = sample

            img_tensor = img.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img_tensor)).item()

            # Load original image from disk
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
    if np.sum(incorrect_mask) > 0:
        selected_indices.extend(np.where(incorrect_mask)[0][: num_images // 2])
    if np.sum(correct_mask) > 0:
        selected_indices.extend(np.where(correct_mask)[0][: num_images - len(selected_indices)])

    # Fill remaining with any images
    if len(selected_indices) < num_images:
        remaining = num_images - len(selected_indices)
        all_indices = set(range(len(dataset)))
        available = sorted(all_indices - set(selected_indices))
        selected_indices.extend(available[:remaining])

    # Generate CAMs
    transform_no_norm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for i, idx in enumerate(selected_indices[:num_images]):
        img = images[idx]
        img_tensor = transform_no_norm(img).unsqueeze(0).to(device)

        # Generate CAM
        cam = gradcam.generate_cam(img_tensor)
        overlay = gradcam.overlay_heatmap(img, cam, alpha=0.45)

        # Save
        image_name = dataset.df.iloc[idx]["ImageName"]
        output_path = Path(output_dir) / f"cam_{i+1}_{image_name}"
        overlay.save(output_path)
