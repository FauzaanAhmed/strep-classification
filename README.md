# Strep Throat Classification: Deep Learning Take-Home Project

A comprehensive deep learning solution for binary classification of strep-positive vs strep-negative cases using throat images and clinical symptom features. This project implements three complementary approaches: a symptoms-only baseline, an image-only transfer learning model, and a fusion model that combines both modalities.

## Overview

This repository contains a production-ready implementation for medical image classification with the following key features:

- **Three Model Architectures**: Symptoms-only baseline, image-only ResNet18 transfer learning, and attention-based fusion model
- **Robust Evaluation**: 5-fold stratified cross-validation with comprehensive metrics reporting
- **Model Interpretability**: Grad-CAM visualizations for understanding model decisions
- **Production Best Practices**: Early stopping, gradient clipping, focal loss, and advanced data augmentation
- **Reproducibility**: Fixed random seeds, deterministic training, and comprehensive configuration management

## Project Structure

```
strep-classification/
├── src/
│   ├── train_cv.py      # Main training script with 5-fold CV
│   ├── models.py         # Model architectures (ImageOnly, Fusion)
│   ├── data.py           # Dataset loading and augmentation
│   ├── utils.py          # Metrics, loss functions, utilities
│   └── cam.py            # Grad-CAM implementation for interpretability
├── data/
│   ├── sample_dataset_100.csv  # Dataset with labels and symptoms
│   └── images/                  # Throat images (gitignored)
├── outputs/                     # Results directory (gitignored)
│   ├── cv_results.json         # Aggregated metrics across folds
│   ├── checkpoint_*.pt         # Model checkpoints
│   └── gradcam_best_fold/       # Grad-CAM visualizations
├── configs.yaml                 # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Models

### 1. Symptoms-Only Baseline
A logistic regression model trained exclusively on clinical symptom features (Hoarseness, Rhinorrhea, Sore Throat, Congestion, Known Recent Contact, Headache, Fever). This serves as a baseline to assess the predictive power of symptoms alone.

**Architecture**: StandardScaler → LogisticRegression with balanced class weights

### 2. Image-Only Model
A deep learning model using transfer learning from ImageNet-pretrained ResNet18. The model extracts visual features from throat images and learns to classify strep-positive vs strep-negative cases.

**Architecture**:
- Backbone: ResNet18 (ImageNet pretrained)
- Feature extraction: 512-dimensional embeddings
- Classifier: Dropout (0.3) → Linear(512 → 1)
- Training: Staged fine-tuning with discriminative learning rates

### 3. Fusion Model
An attention-based fusion model that combines image embeddings and symptom features. The model learns to dynamically weight the importance of visual vs. clinical features for each prediction.

**Architecture**:
- Image branch: ResNet18 → Projection (512 → 256) with BatchNorm
- Symptom branch: MLP (7 → 128 → 64) with BatchNorm
- Attention mechanism: Learns weights for image vs. symptom features
- Fusion classifier: 3-layer MLP (320 → 128 → 64 → 1)

## Key Features

### Training Strategies
- **Staged Fine-Tuning**: Freeze backbone for initial epochs, then unfreeze only layer4
- **Discriminative Learning Rates**: Lower LR (1.5e-5) for pretrained backbone, higher LR (3e-4) for new layers
- **Focal Loss**: Handles class imbalance with α=0.25, γ=2.0
- **Gradient Clipping**: Max norm 1.0 for training stability (configurable via `max_grad_norm`)
- **Early Stopping**: Patience of 7 epochs based on validation AUC
- **Learning Rate Scheduling**: Linear warmup (2 epochs) → Cosine annealing

### Data Augmentation
- Random crop with resize
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±12°)
- Random horizontal flip (50% probability)
- Random affine transformations (small translations)
- Random erasing (10% probability, light erasing)

### Evaluation
- **5-Fold Stratified Cross-Validation**: Ensures balanced class distribution across folds
- **Comprehensive Metrics**: AUC, Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score
- **Statistical Reporting**: Mean ± standard deviation across folds
- **Test-Time Augmentation**: Horizontal flip averaging for improved robustness

### Interpretability
- **Grad-CAM Visualizations**: Highlights image regions most important for predictions
- **Best Fold Selection**: Visualizations generated from highest-performing fold
- **Mixed Examples**: Includes both correct and incorrect predictions for analysis

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/FauzaanAhmed/strep-classification.git
cd strep-classification
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Place your dataset CSV** in `data/` directory:
   - Required columns: `ImageName`, `label`, plus symptom columns
   - `label` values: "Positive" or "Negative" (case-insensitive)
   - Example: `data/sample_dataset_100.csv`

2. **Place images** in `data/images/` directory:
   - Image filenames must match `ImageName` column in CSV
   - Supported formats: JPG, JPEG, PNG
   - Images will be automatically resized to 224×224 during training

## Usage

### Running the Full Pipeline

Execute the complete training and evaluation pipeline:

```bash
python src/train_cv.py --config configs.yaml
```

This single command will:
1. Run 5-fold stratified cross-validation for all three models
2. Train with early stopping and save best checkpoints
3. Generate comprehensive metrics reports
4. Export Grad-CAM visualizations (if enabled)
5. Save all results to `outputs/` directory

### Configuration

Edit `configs.yaml` to customize:

```yaml
# Model selection
run_symptoms_only: true
run_image_only: true
run_fusion: true

# Training hyperparameters
epochs: 30
batch_size: 16
lr_head: 0.0003
lr_backbone: 0.000015
dropout: 0.3
freeze_backbone_epochs: 3

# Loss function
use_focal_loss: true
focal_alpha: 0.25
focal_gamma: 2.0

# Training stability
max_grad_norm: 1.0  # Gradient clipping threshold

# Interpretability
export_gradcam: true
gradcam_total_images: 6
```

## Outputs

All results are saved to `outputs/` directory:

### `cv_results.json`
Comprehensive metrics aggregated across all folds:
```json
{
  "symptoms_only": {
    "metrics": {
      "auc": {"mean": 0.568, "std": 0.086},
      "accuracy": {"mean": 0.56, "std": 0.049},
      "precision": {"mean": 0.565, "std": 0.054},
      "recall_sensitivity": {"mean": 0.54, "std": 0.080},
      "specificity": {"mean": 0.580, "std": 0.098},
      "f1": {"mean": 0.549, "std": 0.057}
    },
    "confusion_totals": [[29, 21], [23, 27]]
  },
  "image_only": {
    "metrics": {
      "auc": {"mean": 0.648, "std": 0.130},
      "accuracy": {"mean": 0.58, "std": 0.06},
      "precision": {"mean": 0.479, "std": 0.249},
      "recall_sensitivity": {"mean": 0.60, "std": 0.341},
      "specificity": {"mean": 0.56, "std": 0.314},
      "f1": {"mean": 0.516, "std": 0.262}
    },
    "confusion_totals": [[28, 22], [20, 30]]
  },
  "fusion": {
    "metrics": {
      "auc": {"mean": 0.592, "std": 0.060},
      "accuracy": {"mean": 0.51, "std": 0.08},
      "precision": {"mean": 0.409, "std": 0.211},
      "recall_sensitivity": {"mean": 0.620, "std": 0.343},
      "specificity": {"mean": 0.40, "std": 0.352},
      "f1": {"mean": 0.487, "std": 0.254}
    },
    "confusion_totals": [[20, 30], [19, 31]]
  }
}
```

### Model Checkpoints
- `checkpoint_image_fold{1-5}.pt`: Best model weights for each fold
- `checkpoint_fusion_fold{1-5}.pt`: Fusion model checkpoints

### Grad-CAM Visualizations
- `outputs/gradcam_best_fold/`: 6 overlay images showing attention heatmaps
- Filenames: `cam_{1-6}_{ImageName}.png`

## Results Interpretation

The pipeline reports mean ± standard deviation for all metrics across 5 folds:

- **AUC (Area Under ROC Curve)**: Overall discriminative ability (higher is better)
- **Accuracy**: Overall classification correctness
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall

Lower standard deviations indicate more stable performance across folds.

## Technical Details

### Model Architectures
- **Image-Only**: ResNet18 pretrained on ImageNet, fine-tuned on throat images
- **Fusion**: Attention-based late fusion with feature normalization
- **Symptoms-Only**: Logistic regression with balanced class weights

### Training Details
- **Optimizer**: AdamW with weight decay 1e-4
- **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training (enabled by default)
- **Gradient Clipping**: Configurable max norm (default: 1.0) to prevent gradient explosion
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Image Size**: 224×224 pixels
- **Random Seed**: 42 (for reproducibility)

### Computational Requirements
- **GPU**: Recommended (NVIDIA GPU with CUDA support)
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: ~2GB for models and checkpoints
- **Training Time**: ~1-2 hours on GPU, ~4-6 hours on CPU (for 100 images, 5-fold CV)

## Reproducibility

The project ensures reproducibility through:
- Fixed random seed (42) for all random operations
- Deterministic PyTorch operations (`cudnn.deterministic = True`)
- Version-controlled configuration files
- Comprehensive logging of hyperparameters

## Future Improvements

Potential enhancements for production deployment:
- Ensemble methods combining multiple models
- Advanced augmentation techniques (MixUp, CutMix)
- Hyperparameter optimization (Optuna, Ray Tune)
- Model quantization for deployment
- Integration with clinical workflow systems
