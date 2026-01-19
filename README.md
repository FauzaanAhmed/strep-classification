# Strep Classification

Deep learning classifier for strep-positive vs strep-negative classification using throat images and symptom features.

## Models

Three models are implemented:

1. **Symptoms-only baseline**: Logistic Regression on symptom features only
2. **Image-only model**: ResNet18 transfer learning on throat images
3. **Fusion model**: Late fusion combining image embeddings and symptom features

## Setup

### Data Setup

1. Place your CSV file in `data/` directory (e.g., `data/sample_dataset_100.csv`)
2. Place images in `data/images/` directory
   - CSV should have columns: `ImageName`, `label`, plus symptom columns
   - `label` values should be "Positive" or "Negative" (case-insensitive)
   - Image filenames in `ImageName` column must match files in `data/images/`

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run cross-validation training with:

```bash
python src/train_cv.py --config configs.yaml
```

The script will:
- Run 5-fold stratified cross-validation for all enabled models
- Save checkpoints, metrics, and Grad-CAM visualizations
- Output results to `outputs/cv_results.json`

## Outputs

Results are saved to `outputs/`:

- `cv_results.json`: Mean±std metrics across folds (AUC, accuracy, precision, recall, specificity, F1)
- `checkpoint_*.pt`: Model checkpoints for each fold
- `gradcam_best_fold/`: Grad-CAM overlays (6 images) from best-performing fold

## Configuration

Edit `configs.yaml` to adjust:
- Training hyperparameters (learning rate, batch size, epochs)
- Model settings (dropout, freeze backbone epochs)
- Which models to run (`run_symptoms_only`, `run_image_only`, `run_fusion`)
- Data paths and output directory
