"""Main training script with 5-fold cross-validation."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

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
    FocalLoss,
    load_config,
    save_results,
    set_seed,
)


def build_warmup_cosine_scheduler(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int):
    """
    One clean schedule: linear warmup -> cosine decay.
    Applies same factor to all param groups (head/backbone).
    """
    import math

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # cosine over remaining epochs
        if total_epochs <= warmup_epochs:
            return 1.0
        t = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def freeze_backbone(model: ImageOnlyModel):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_layer4_in_sequential_backbone(sequential_backbone: nn.Sequential):
    # ResNet children()[:-1] indices:
    # 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2, 6 layer3, 7 layer4, 8 avgpool
    for p in sequential_backbone.parameters():
        p.requires_grad = False
    # Unfreeze only layer4 block
    for p in sequential_backbone[7].parameters():
        p.requires_grad = True


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = True,
    is_fusion: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        if is_fusion:
            images, symptoms, labels = batch
            images = images.to(device)
            symptoms = torch.as_tensor(symptoms, device=device)
            labels = labels.to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

        labels = labels.float().unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda")):
            outputs = model(images, symptoms) if is_fusion else model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        # Gradient clipping for stability
        max_grad_norm = 1.0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_fusion: bool = False,
    use_tta: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validation with optional very light TTA:
    prob = avg(sigmoid(model(img)), sigmoid(model(hflip(img))))
    """
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in dataloader:
        if is_fusion:
            images, symptoms, labels = batch
            images = images.to(device)
            symptoms = torch.as_tensor(symptoms, device=device)
            labels = labels.to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

        labels_f = labels.float().unsqueeze(1)

        # forward pass (no TTA) for loss
        outputs = model(images, symptoms) if is_fusion else model(images)
        loss = criterion(outputs, labels_f)
        total_loss += float(loss.item())

        p1 = torch.sigmoid(outputs)

        if use_tta:
            images_flip = torch.flip(images, dims=[3])
            outputs2 = model(images_flip, symptoms) if is_fusion else model(images_flip)
            p2 = torch.sigmoid(outputs2)
            probs = 0.5 * (p1 + p2)
        else:
            probs = p1

        all_probs.append(probs.squeeze(1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs).astype(np.float32)
    y_true = np.concatenate(all_labels).astype(int)

    y_pred = (y_prob >= 0.5).astype(int)
    return total_loss / max(1, len(dataloader)), y_pred, y_prob, y_true


def run_symptoms_only_cv(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Dict[str, Any]:
    print("\n" + "=" * 50)
    print("Running Symptoms-Only Baseline")
    print("=" * 50)

    symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]
    X = df[symptom_cols].values
    y = (df["label"].astype(str).str.lower() == "positive").astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    confusion_totals = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Improved pipeline with class weights
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                C=1.0,
                solver='lbfgs',
                random_state=seed
            ))
        ])
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(metrics)

        cm = confusion_matrix(y_val, y_pred)
        confusion_totals += cm

        print(f"  AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

    aggregated = aggregate_metrics(fold_metrics)
    return {"metrics": aggregated, "confusion_totals": confusion_totals.tolist()}


def run_image_only_cv(df: pd.DataFrame, images_dir: str, config: Dict, device: torch.device):
    print("\n" + "=" * 50)
    print("Running Image-Only Model")
    print("=" * 50)

    y = (df["label"].astype(str).str.lower() == "positive").astype(int).values
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])

    fold_metrics = []
    confusion_totals = np.zeros((2, 2), dtype=int)
    best_fold_auc = -1
    best_fold_idx = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\nFold {fold}/{config['n_splits']}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = StrepDataset(
            train_df, images_dir, image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=True),
            return_symptoms=False
        )
        val_dataset = StrepDataset(
            val_df, images_dir, image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=False),
            return_symptoms=False
        )

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                  num_workers=config["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=config["num_workers"], pin_memory=True)

        model = ImageOnlyModel(dropout=config["dropout"]).to(device)

        # Freeze everything then later unfreeze only layer4
        freeze_backbone(model)

        # Param groups: head vs backbone
        head_params = list(model.fc.parameters())
        backbone_params = list(model.backbone.parameters())

        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": config["lr_head"]},
                {"params": backbone_params, "lr": config["lr_backbone"]},
            ],
            weight_decay=config["weight_decay"],
        )

        scheduler = build_warmup_cosine_scheduler(optimizer, config["warmup_epochs"], config["epochs"])
        # Use focal loss if enabled, otherwise BCE
        if config.get("use_focal_loss", False):
            criterion = FocalLoss(
                alpha=config.get("focal_alpha", 0.25),
                gamma=config.get("focal_gamma", 2.0)
            ).to(device)
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_auc = -1
        patience_counter = 0
        ckpt_path = Path(config["output_dir"]) / f"checkpoint_image_fold{fold}.pt"
        os.makedirs(ckpt_path.parent, exist_ok=True)

        for epoch in range(config["epochs"]):
            # After freeze epochs, unfreeze only layer4
            if epoch == config["freeze_backbone_epochs"]:
                unfreeze_layer4_in_sequential_backbone(model.backbone)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, amp_enabled=config["amp"], is_fusion=False)

            val_loss, val_preds, val_probs, val_labels = validate(
                model, val_loader, criterion, device, is_fusion=False, use_tta=config.get("use_tta", True)
            )

            metrics = compute_metrics(val_labels, val_preds, val_probs)
            val_auc = metrics["auc"]

            scheduler.step()

            print(f"  Epoch {epoch+1}/{config['epochs']}: Train Loss {train_loss:.4f} | Val AUC {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Reload best and evaluate
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        val_loss, val_preds, val_probs, val_labels = validate(
            model, val_loader, criterion, device, is_fusion=False, use_tta=config.get("use_tta", True)
        )
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        fold_metrics.append(metrics)

        cm = confusion_matrix(val_labels, val_preds)
        confusion_totals += cm

        print(f"  Final AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

        if metrics["auc"] > best_fold_auc:
            best_fold_auc = metrics["auc"]
            best_fold_idx = fold - 1

    aggregated = aggregate_metrics(fold_metrics)
    return {"metrics": aggregated, "confusion_totals": confusion_totals.tolist()}, best_fold_idx


def run_fusion_cv(df: pd.DataFrame, images_dir: str, config: Dict, device: torch.device):
    print("\n" + "=" * 50)
    print("Running Fusion Model")
    print("=" * 50)

    symptom_cols = [col for col in df.columns if col not in ["ImageName", "label"]]
    num_symptoms = len(symptom_cols)
    y = (df["label"].astype(str).str.lower() == "positive").astype(int).values

    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    fold_metrics = []
    confusion_totals = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\nFold {fold}/{config['n_splits']}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = StrepDataset(
            train_df, images_dir, image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=True),
            return_symptoms=True
        )
        val_dataset = StrepDataset(
            val_df, images_dir, image_size=config["image_size"],
            transform=get_transforms(config["image_size"], is_train=False),
            return_symptoms=True
        )

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                  num_workers=config["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=config["num_workers"], pin_memory=True)

        model = FusionModel(num_symptoms=num_symptoms, dropout=config["dropout"]).to(device)

        # Freeze all backbone first, then unfreeze only layer4 later
        for p in model.image_backbone.parameters():
            p.requires_grad = False

        # Include all new layers in head params
        head_params = (
            list(model.symptom_mlp.parameters())
            + list(model.fusion.parameters())
            + list(model.attention.parameters())
            + list(model.image_proj.parameters())
        )
        backbone_params = list(model.image_backbone.parameters())

        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": config["lr_head"]},
                {"params": backbone_params, "lr": config["lr_backbone"]},
            ],
            weight_decay=config["weight_decay"],
        )

        scheduler = build_warmup_cosine_scheduler(optimizer, config["warmup_epochs"], config["epochs"])
        # Use focal loss if enabled, otherwise BCE
        if config.get("use_focal_loss", False):
            criterion = FocalLoss(
                alpha=config.get("focal_alpha", 0.25),
                gamma=config.get("focal_gamma", 2.0)
            ).to(device)
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_auc = -1
        patience_counter = 0
        ckpt_path = Path(config["output_dir"]) / f"checkpoint_fusion_fold{fold}.pt"
        os.makedirs(ckpt_path.parent, exist_ok=True)

        for epoch in range(config["epochs"]):
            if epoch == config["freeze_backbone_epochs"]:
                unfreeze_layer4_in_sequential_backbone(model.image_backbone)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, amp_enabled=config["amp"], is_fusion=True)

            val_loss, val_preds, val_probs, val_labels = validate(
                model, val_loader, criterion, device, is_fusion=True, use_tta=config.get("use_tta", True)
            )

            metrics = compute_metrics(val_labels, val_preds, val_probs)
            val_auc = metrics["auc"]

            scheduler.step()

            print(f"  Epoch {epoch+1}/{config['epochs']}: Train Loss {train_loss:.4f} | Val AUC {val_auc:.4f}")

            # IMPORTANT: save best checkpoint (fixes your fusion underreporting)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Reload best and evaluate
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        val_loss, val_preds, val_probs, val_labels = validate(
            model, val_loader, criterion, device, is_fusion=True, use_tta=config.get("use_tta", True)
        )
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        fold_metrics.append(metrics)

        cm = confusion_matrix(val_labels, val_preds)
        confusion_totals += cm

        print(f"  Final AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

    aggregated = aggregate_metrics(fold_metrics)
    return {"metrics": aggregated, "confusion_totals": confusion_totals.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train Strep classification models with CV")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data from {config['csv_path']}")
    df = load_data(config["csv_path"], config["images_dir"])

    results = {}

    if config.get("run_symptoms_only", True):
        results["symptoms_only"] = run_symptoms_only_cv(df, config["n_splits"], config["seed"])

    best_fold_idx = None
    if config.get("run_image_only", True):
        results["image_only"], best_fold_idx = run_image_only_cv(df, config["images_dir"], config, device)

        if config.get("export_gradcam", True) and best_fold_idx is not None:
            print(f"\nExporting Grad-CAM for best fold (fold {best_fold_idx + 1})")
            y = (df["label"].astype(str).str.lower() == "positive").astype(int).values
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

            model = ImageOnlyModel(dropout=config["dropout"]).to(device)
            checkpoint_path = Path(config["output_dir"]) / f"checkpoint_image_fold{best_fold_idx + 1}.pt"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

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

    if config.get("run_fusion", True):
        results["fusion"] = run_fusion_cv(df, config["images_dir"], config, device)

    output_path = Path(config["output_dir"]) / "cv_results.json"
    save_results(results, str(output_path))
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper().replace('_', '-')}:")
        metrics = model_results["metrics"]
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value['mean']:.4f} ± {metric_value['std']:.4f}")


if __name__ == "__main__":
    main()
