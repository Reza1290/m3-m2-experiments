"""
src/training/train_stage2.py
============================
Training Script for Stage 2 Deep Learning Contact Location Classifier (Option B).

Trains on Contact cases strictly from the Train split.
Validates on Contact cases strictly from the Val split.
Applies Square-Root Class Weighting to combat extreme class imbalance:
- Above_CEJ: ~27 cases (minority)
- At_CEJ:    ~15 cases (extreme minority)
- Below_CEJ: ~312 cases (majority)

Saves best checkpoint to weights/stage2_dl_best.pth.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.models.stage2_classifier import (
    Stage2ContactLocationClassifier,
    STAGE2_CLASS_MAP,
    FINAL_CLASS_TO_STAGE2_ID,
)


class ContactLocationDataset(Dataset):
    """Loads 2D crop patches for Contact cases with in-memory caching."""

    def __init__(self, records: List[Dict], is_training: bool = False, preload: bool = True):
        self.records = records
        self.is_training = is_training
        self.cached_images = []

        if is_training:
            self.transform = T.Compose([
                T.ToTensor(),
                T.RandomRotation(degrees=7.0),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

        if preload:
            for r in self.records:
                path = Path(r["crop_path_npy"])
                if path.exists():
                    arr = np.load(path).astype(np.float32)
                else:
                    arr = np.zeros((224, 224), dtype=np.float32)

                if arr.max() > 1.0:
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)

                if arr.ndim == 2:
                    arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)

                self.cached_images.append(arr)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.cached_images:
            arr = self.cached_images[idx]
        else:
            path = Path(self.records[idx]["crop_path_npy"])
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)

        tensor = self.transform(arr)
        target = int(self.records[idx]["stage2_target"])
        cid = str(self.records[idx]["case_id"])
        return tensor, target, cid


def load_contact_records(manifest_path: Path, split_name: str) -> List[Dict]:
    """Load and format records for contact cases only."""
    df = pd.read_csv(manifest_path)
    # Filter by split and exclude Independent cases
    df_split = df[(df["split"] == split_name) & (df["class_name"] != "Independent")].copy()

    # Map class_id (1, 2, 3) to Stage 2 targets (0, 1, 2)
    df_split["stage2_target"] = df_split["class_id"].map(FINAL_CLASS_TO_STAGE2_ID)

    records = df_split.to_dict("records")
    return records


def compute_class_weights(records: List[Dict]) -> torch.Tensor:
    """Compute smoothed inverse square-root class weights for 3 classes."""
    targets = [r["stage2_target"] for r in records]
    counts = np.bincount(targets, minlength=3)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.sum() * 3.0
    return torch.tensor(weights, dtype=torch.float32)


def train_stage2_model(
    backbone_name: str = "swin_small",
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-4,
    device_name: str = "auto",
):
    """Train Stage 2 Deep Learning Classifier."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"=== Stage 2 DL Training ({backbone_name}) on: {device} ===")

    manifest_path = Path("data/manifests/dataset_4class_manifest.csv")
    train_records = load_contact_records(manifest_path, "train")
    val_records = load_contact_records(manifest_path, "val")

    print(f"Train Contact Cases: {len(train_records)}")
    print(f"Val Contact Cases:   {len(val_records)}")

    weights = compute_class_weights(train_records).to(device)
    print(f"Computed Class Weights: {weights.cpu().numpy().round(3)}")

    train_ds = ContactLocationDataset(train_records, is_training=True, preload=True)
    val_ds = ContactLocationDataset(val_records, is_training=False, preload=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = Stage2ContactLocationClassifier(
        backbone_name=backbone_name,
        num_classes=3,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = weights_dir / "stage2_dl_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for tensors, targets, _ in train_loader:
            tensors, targets = tensors.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss /= len(train_records)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for tensors, targets, _ in val_loader:
                tensors, targets = tensors.to(device), targets.to(device)
                logits = model(tensors)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_records)
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_targets, all_preds)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val F1: {val_f1:.4f}) to {best_ckpt_path}")

    # Save training history
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "stage2_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete. Best Validation Macro-F1: {best_val_f1:.4f}")
    return best_ckpt_path


if __name__ == "__main__":
    train_stage2_model(backbone_name="swin_small", epochs=10, batch_size=16)
