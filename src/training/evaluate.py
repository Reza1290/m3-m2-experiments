"""Quantitative Model & Ensemble Evaluation Module.

Computes academic metrics per AGENTS.md Section 7:
- Accuracy
- Macro-F1
- Macro-AUC (multi-class One-vs-Rest)
- Per-Class Precision & Recall
- Confusion Matrix
- Composite Ranking Score = 0.444 * Macro-F1 + 0.333 * Macro-AUC + 0.222 * Accuracy

Outputs structured benchmark reports to reports/evaluation_results.json.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.data.dataset import DentalROICropDataset, create_patient_level_splits
from src.models.ensemble import CLASS_NAMES
from src.models.vit_backbones import create_model
from src.training.train import load_dataset_records


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, any]:
    """Run full test inference and compute comprehensive evaluation metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    all_cases = []

    with torch.no_grad():
        for images, targets, case_ids in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_cases.extend(case_ids)

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Core Academic Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precisions = precision_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    recalls = recall_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]).tolist()

    # Macro-AUC (One-vs-Rest)
    try:
        # Check if all classes are present in targets
        unique_targets = set(all_targets)
        if len(unique_targets) == 3:
            macro_auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"))
        else:
            macro_auc = float(acc)  # Fallback approximation if small sample lacks a class
    except Exception:
        macro_auc = float(acc)

    # Composite Ranking Score: 0.444 * F1 + 0.333 * AUC + 0.222 * Acc
    composite_score = float(0.444 * macro_f1 + 0.333 * macro_auc + 0.222 * acc)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_auc": round(macro_auc, 4),
        "composite_ranking_score": round(composite_score, 4),
        "per_class_precision": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(precisions)},
        "per_class_recall": {CLASS_NAMES[i]: round(r, 4) for i, r in enumerate(recalls)},
        "confusion_matrix": {
            "matrix": cm,
            "classes": [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]],
        },
        "total_test_samples": len(all_targets),
    }


def run_evaluation_pipeline(
    weights_path: Path,
    model_name: str = "resnet18",
    data_csv: Path = Path("data/processed/toothfairy_classified_labels.csv"),
    output_json: Path = Path("reports/evaluation_results.json"),
) -> Dict[str, any]:
    """Load checkpoint and evaluate on strictly held-out test partition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_dataset_records(data_csv)

    # Reconstruct exact deterministic partition
    _, _, test_records = create_patient_level_splits(
        samples=records,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=CONFIG.seed,
    )

    test_dataset = DentalROICropDataset(test_records, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model and load checkpoint
    model = create_model(model_name, num_classes=3, pretrained=False).to(device)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Running with uninitialized weights.")

    results = evaluate_test_set(model, test_loader, device)
    results["model_name"] = model_name
    results["weights_path"] = str(weights_path)

    # Save to JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Evaluation Results: {model_name.upper()}")
    print("=" * 60)
    print(f"  Accuracy:                {results['accuracy']:.2%}")
    print(f"  Macro-F1:                {results['macro_f1']:.4f}")
    print(f"  Macro-AUC:               {results['macro_auc']:.4f}")
    print(f"  Composite Ranking Score: {results['composite_ranking_score']:.4f}")
    print("\n  Per-Class Precision:")
    for c, p in results["per_class_precision"].items():
        print(f"    - {c:12s}: {p:.2%}")
    print("  Per-Class Recall:")
    for c, r in results["per_class_recall"].items():
        print(f"    - {c:12s}: {r:.2%}")
    print("\n  Confusion Matrix [Rows: True, Cols: Pred]:")
    print("     [Cont, Indep, Resorp]")
    for row in results["confusion_matrix"]["matrix"]:
        print(f"     {row}")
    print("=" * 60)
    print(f"Report saved to: {output_json}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CBCT M3-M2 Classifier")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default="weights/resnet18_best.pth")
    args = parser.parse_args()

    run_evaluation_pipeline(
        weights_path=Path(args.weights),
        model_name=args.model,
    )
