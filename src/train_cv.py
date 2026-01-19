"""Main training script with 5-fold cross-validation."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cam import export_gradcam_images
from src.data import StrepDataset, get_transforms, load_data
from src.models import FusionModel, ImageOnlyModel
from src.utils import (
    aggregate_metrics,
    compute_metrics,
    load_config,
    save_results,
    set_seed,
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = True,
    is_fusion: bool = False,
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    for batch in dataloader:
        optimizer.zero_grad()

        if is_fusion:
            images, symptoms, labels = [b.to(device) for b in batch]
        else:
            images, labels = [b.to(device) for b in batch]

        labels = labels.float().unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            if is_fusion:
                outputs = model(images, symptoms)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_fusion: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if is_fusion:
                images, symptoms, labels = [b.to(device) for b in batch]
            else:
                images, labels = [b.to(device) for b in batch]

            labels = labels.float().unsqueeze(1)

            if is_fusion:
                outputs = model(images, symptoms)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()

    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    return total_loss / len(dataloader), all_preds, all_probs, all_labels


def run_symptoms_only_cv(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> Dict[str, any]:
    """Run symptoms-only baseline with Logistic Regression."""
    print("\n" + "=" * 50)
    print("Running Symptoms-Only Baseline")
    print("=" * 50)

    # Get symptom columns
    symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]
    X = df[symptom_cols].values
    y = (df["label"].str.lower() == "positive").astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    confusion_totals = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
        pipeline.fit(X_train, y_train)

        # Predict
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        metrics = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(metrics)

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        confusion_totals += cm

        print(f"  AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

    # Aggregate
    aggregated = aggregate_metrics(fold_metrics)
    return {"metrics": aggregated, "confusion_totals": confusion_totals.tolist()}


def run_image_only_cv(
    df: pd.DataFrame,
    images_dir: str,
    config: Dict,
    device: torch.device,
) -> Tuple[Dict[str, Any], int]:
    """Run image-only model with 5-fold CV. Returns results and best fold index."""
    print("\n" + "=" * 50)
    print("Running Image-Only Model")
    print("=" * 50)

    # Prepare data
    y = (df["label"].str.lower() == "positive").astype(int).values

    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    fold_metrics = []
    confusion_totals = np.zeros((2, 2), dtype=int)
    best_fold_auc = -1
    best_fold_idx = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        print(f"\nFold {fold + 1}/{config['n_splits']}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_dataset = StrepDataset(
            train_df,
            images_dir,
            image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=True),
            return_symptoms=False,
        )
        val_dataset = StrepDataset(
            val_df,
            images_dir,
            image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=False),
            return_symptoms=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        # Model
        model = ImageOnlyModel(dropout=config["dropout"]).to(device)

        # Freeze backbone initially
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Optimizer and criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Cosine LR with warmup (T_max is in epochs)
        total_epochs = config["epochs"] - config["warmup_epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_auc = -1
        patience_counter = 0

        for epoch in range(config["epochs"]):
            # Unfreeze backbone after freeze_backbone_epochs
            if epoch == config["freeze_backbone_epochs"]:
                for param in model.backbone.parameters():
                    param.requires_grad = True

            # Train
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp_enabled=config["amp"],
                is_fusion=False,
            )

            # Validate
            val_loss, val_preds, val_probs, val_labels = validate(
                model, val_loader, criterion, device, is_fusion=False
            )

            # Metrics
            metrics = compute_metrics(val_labels, val_preds, val_probs)
            val_auc = metrics["auc"]

            # Warmup + scheduler
            if epoch < config["warmup_epochs"]:
                # Linear warmup
                warmup_factor = (epoch + 1) / config["warmup_epochs"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config["lr"] * warmup_factor
            else:
                scheduler.step()

            print(f"  Epoch {epoch + 1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save checkpoint
                checkpoint_path = Path(config["output_dir"]) / f"checkpoint_image_fold{fold + 1}.pt"
                os.makedirs(checkpoint_path.parent, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model and evaluate
        checkpoint_path = Path(config["output_dir"]) / f"checkpoint_image_fold{fold + 1}.pt"
        model.load_state_dict(torch.load(checkpoint_path))

        val_loss, val_preds, val_probs, val_labels = validate(model, val_loader, criterion, device, is_fusion=False)
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        fold_metrics.append(metrics)

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        confusion_totals += cm

        print(f"  Final AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

        # Track best fold
        if metrics["auc"] > best_fold_auc:
            best_fold_auc = metrics["auc"]
            best_fold_idx = fold

    # Aggregate
    aggregated = aggregate_metrics(fold_metrics)
    return {"metrics": aggregated, "confusion_totals": confusion_totals.tolist()}, best_fold_idx


def run_fusion_cv(
    df: pd.DataFrame,
    images_dir: str,
    config: Dict,
    device: torch.device,
) -> Dict[str, any]:
    """Run fusion model with 5-fold CV."""
    print("\n" + "=" * 50)
    print("Running Fusion Model")
    print("=" * 50)

    # Prepare data
    symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]
    num_symptoms = len(symptom_cols)
    y = (df["label"].str.lower() == "positive").astype(int).values

    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    fold_metrics = []
    confusion_totals = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        print(f"\nFold {fold + 1}/{config['n_splits']}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_dataset = StrepDataset(
            train_df,
            images_dir,
            image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=True),
            return_symptoms=True,
        )
        val_dataset = StrepDataset(
            val_df,
            images_dir,
            image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=False),
            return_symptoms=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        # Model
        model = FusionModel(num_symptoms=num_symptoms, dropout=config["dropout"]).to(device)

        # Freeze backbone initially
        for param in model.image_backbone.parameters():
            param.requires_grad = False

        # Optimizer and criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Cosine LR with warmup (T_max is in epochs)
        total_epochs = config["epochs"] - config["warmup_epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_auc = -1
        patience_counter = 0

        for epoch in range(config["epochs"]):
            # Unfreeze backbone after freeze_backbone_epochs
            if epoch == config["freeze_backbone_epochs"]:
                for param in model.image_backbone.parameters():
                    param.requires_grad = True

            # Train
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp_enabled=config["amp"],
                is_fusion=True,
            )

            # Validate
            val_loss, val_preds, val_probs, val_labels = validate(
                model, val_loader, criterion, device, is_fusion=True
            )

            # Metrics
            metrics = compute_metrics(val_labels, val_preds, val_probs)
            val_auc = metrics["auc"]

            # Warmup + scheduler
            if epoch < config["warmup_epochs"]:
                # Linear warmup
                warmup_factor = (epoch + 1) / config["warmup_epochs"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config["lr"] * warmup_factor
            else:
                scheduler.step()

            print(f"  Epoch {epoch + 1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Final evaluation
        val_loss, val_preds, val_probs, val_labels = validate(model, val_loader, criterion, device, is_fusion=True)
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        fold_metrics.append(metrics)

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        confusion_totals += cm

        print(f"  Final AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

    # Aggregate
    aggregated = aggregate_metrics(fold_metrics)
    return {"metrics": aggregated, "confusion_totals": confusion_totals.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train Strep classification models with CV")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {config['csv_path']}")
    df = load_data(config["csv_path"], config["images_dir"])

    results = {}

    # Symptoms-only baseline
    if config.get("run_symptoms_only", True):
        results["symptoms_only"] = run_symptoms_only_cv(df, config["n_splits"], config["seed"])

    # Image-only model
    best_fold_idx = None
    if config.get("run_image_only", True):
        results["image_only"], best_fold_idx = run_image_only_cv(df, config["images_dir"], config, device)

        # Export Grad-CAM for best fold
        if config.get("export_gradcam", True) and best_fold_idx is not None:
            print(f"\nExporting Grad-CAM for best fold (fold {best_fold_idx + 1})")
            # Reload best fold data
            y = (df["label"].str.lower() == "positive").astype(int).values
            skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
            train_idx, val_idx = list(skf.split(df, y))[best_fold_idx]
            val_df = df.iloc[val_idx].reset_index(drop=True)

            val_dataset = StrepDataset(
                val_df,
                config["images_dir"],
                image_size=config["image_size"],
                transform=get_transforms(config["image_size"], is_train=False),
                return_symptoms=False,
            )

            # Load best model
            model = ImageOnlyModel(dropout=config["dropout"]).to(device)
            checkpoint_path = Path(config["output_dir"]) / f"checkpoint_image_fold{best_fold_idx + 1}.pt"
            model.load_state_dict(torch.load(checkpoint_path))

            # Export
            gradcam_dir = Path(config["output_dir"]) / "gradcam_best_fold"
            export_gradcam_images(
                model,
                val_dataset,
                device,
                str(gradcam_dir),
                num_images=config["gradcam_total_images"],
                target_layer_name="backbone.7",
            )
            print(f"Grad-CAM images saved to {gradcam_dir}")

    # Fusion model
    if config.get("run_fusion", True):
        results["fusion"] = run_fusion_cv(df, config["images_dir"], config, device)

    # Save results
    output_path = Path(config["output_dir"]) / "cv_results.json"
    save_results(results, str(output_path))
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper().replace('_', '-')}:")
        metrics = model_results["metrics"]
        for metric_name, metric_value in metrics.items():
            mean_val = metric_value["mean"]
            std_val = metric_value["std"]
            print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()
