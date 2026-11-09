"""Unified Training Script for 2-Class Vision Transformer Ensemble.

Trains the three foundational research models:
1. Swin-Small + MHA(head) (swin_s_mha)
2. Swin-Small (swin_small)
3. ViT-Base/16 (vit_base_16)

Classes:
0: Contact
1: Independent
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.dataset import DentalROICropDataset, create_patient_level_splits
from src.models.vit_backbones import create_model


def load_binary_dataset_records(csv_path: Path) -> List[Dict[str, any]]:
    """Load valid Contact and Independent samples."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = r.get("class_id")
            if cid not in ["0", "1"]:
                continue
            crop_path = r.get("crop_path", "")
            if crop_path and Path(crop_path).exists():
                records.append({
                    "image": crop_path,
                    "label": int(cid),
                    "case_id": r["case_id"],
                    "patient_id": r["case_id"],
                    "quadrant": r.get("quadrant", ""),
                    "m2_fdi": int(r["m2_fdi"]),
                    "m3_fdi": int(r["m3_fdi"]),
                })
    return records


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate accuracy, loss, and macro-F1 on validation set."""
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

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, float(acc), float(macro_f1)


def train_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 2,
    epochs: int = 8,
    lr: float = 3e-5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Train a single architecture with cosine annealing and early checkpointing."""
    print(f"\n=======================================================")
    print(f"  Training Architecture: {model_name} on {device}")
    print(f"=======================================================")

    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    # Compute class weights for binary balance
    labels = [s["label"] for s in train_loader.dataset.samples]
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class distribution: {counts.tolist()} | Class weights: {weights.round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch [{ep:2d}/{epochs:2d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% ({elapsed:.1f}s)")

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "time_sec": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model checkpoint saved to: {save_path}")

    print(f"\n[DONE] Best {model_name}: Epoch {best_epoch} with Val Macro-F1: {best_val_f1*100:.2f}%\n")
    return {"history": history, "best_epoch": best_epoch, "best_val_f1": best_val_f1}


def main():
    parser = argparse.ArgumentParser(description="Train 2-Class Vision Transformer Ensemble")
    parser.add_argument("--csv", default="data/processed/toothfairy_binary_labels.csv")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = load_binary_dataset_records(Path(args.csv))
    print(f"Total valid 2-class samples: {len(records)}")

    # Strict patient-level split (80% train, 10% val, 10% test)
    train_records, val_records, test_records = create_patient_level_splits(
        records, train_ratio=0.80, val_ratio=0.10, seed=42
    )
    print(f"Split sizes: Train={len(train_records)} | Val={len(val_records)} | Test={len(test_records)}")

    train_ds = DentalROICropDataset(train_records, is_training=True, preload=True)
    val_ds = DentalROICropDataset(val_records, is_training=False, preload=True)
    test_ds = DentalROICropDataset(test_records, is_training=False, preload=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3 Arsitektur Sesuai Spesifikasi Paper
    models_to_train = [
        ("swin_s_mha", Path("weights/swin_s_mha_binary_best.pth")),
        ("swin_small", Path("weights/swin_small_binary_best.pth")),
        ("vit_base_16", Path("weights/vit_base_binary_best.pth")),
    ]

    all_histories = {}
    for name, pth in models_to_train:
        res = train_single_model(
            model_name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=2,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=pth,
        )
        all_histories[name] = res

    # Save training logs
    log_path = Path("reports/binary_ensemble_training_history.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)
    print(f"All training logs saved to {log_path}")


if __name__ == "__main__":
    main()
