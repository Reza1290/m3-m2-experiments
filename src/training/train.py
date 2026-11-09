"""Unified PyTorch Training Module for CBCT M3-M2 Relationship Classification.

Features:
1. Patient-level leakage-free partitioning (train / val / test).
2. Class-balanced CrossEntropyLoss.
3. AdamW optimizer + CosineAnnealingLR scheduler.
4. Model checkpointing to weights/ with validation Macro-F1 tracking.
5. Evaluation metrics per epoch: Macro-F1, Accuracy, Loss.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.data.dataset import DentalROICropDataset, create_patient_level_splits
from src.models.vit_backbones import create_model


def load_dataset_records(csv_path: Path) -> List[Dict[str, any]]:
    """Load valid samples from the classified dataset CSV."""
    records = []
    crops_dir = Path("data/processed/crops_2d")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            crop_path = r.get("crop_path", "")
            if not crop_path or not Path(crop_path).exists():
                candidate = crops_dir / f"{r['case_id']}_{r['m2_fdi']}_{r['m3_fdi']}.npy"
                if candidate.exists():
                    crop_path = str(candidate.as_posix())

            if crop_path and Path(crop_path).exists():
                records.append({
                    "image": crop_path,
                    "label": int(r["class_id"]),
                    "case_id": r["case_id"],
                    "patient_id": r["case_id"],
                    "quadrant": r["quadrant"],
                    "m2_fdi": int(r["m2_fdi"]),
                    "m3_fdi": int(r["m3_fdi"]),
                })
    return records


def compute_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    """Compute smoothed square-root inverse class frequency weights for CrossEntropyLoss.
    
    Prevents rare classes from causing extreme logit distortion while maintaining balanced gradient pressure.
    """
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Compute validation loss, accuracy, and macro-F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = total_loss / (total_samples + 1e-7)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(np.mean(all_preds == all_targets))

    # Compute Macro-F1 across 3 classes
    f1_scores = []
    for c in range(3):
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores))

    return avg_loss, acc, macro_f1


def train_classifier(
    model_name: str = "resnet18",
    data_csv: Path = Path("data/processed/toothfairy_classified_labels.csv"),
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    pretrained: bool = True,
    device_name: Optional[str] = None,
    weights_dir: Path = Path("weights"),
    reports_dir: Path = Path("reports"),
) -> Dict[str, any]:
    """Train a classifier backbone with leakage-free splits."""
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset_records(data_csv)
    if not records:
        raise ValueError(f"No valid records with existing crop_path found in {data_csv}")

    print(f"\n=======================================================")
    print(f"  Training CDSS Classifier: {model_name.upper()}")
    print(f"  Device: {device} | Total Available Crops: {len(records)}")
    print(f"=======================================================\n")

    # Split dataset at patient level
    train_records, val_records, test_records = create_patient_level_splits(
        samples=records,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=CONFIG.seed,
    )
    print(f"Dataset Partitioning (Leakage-Free):")
    print(f"  Train: {len(train_records)} crops | Val: {len(val_records)} crops | Test: {len(test_records)} crops\n")

    train_dataset = DentalROICropDataset(train_records, is_training=True)
    val_dataset = DentalROICropDataset(val_records, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class weighting
    train_labels = [r["label"] for r in train_records]
    class_weights = compute_class_weights(train_labels, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Instantiate model with transfer learning support
    model = create_model(model_name, num_classes=3, pretrained=pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_weights_path = weights_dir / f"{model_name}_best.pth"
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.1%} | "
            f"Val Macro-F1: {val_f1:.4f} "
            f"({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "time_sec": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
            }, best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path}")

    # Save training history
    history_path = reports_dir / f"training_history_{model_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_weights_path": str(best_weights_path),
        "best_val_f1": best_val_f1,
        "history": history,
        "test_records": test_records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CBCT M3-M2 Classifier")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name (resnet18, swin_small, swin_small_mha)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    args = parser.parse_args()

    train_classifier(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
    )
