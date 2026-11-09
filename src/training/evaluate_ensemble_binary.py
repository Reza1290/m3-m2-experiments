"""Evaluation Script for 2-Class Vision Transformer Weighted Soft-Voting Ensemble.

Architecture:
1. Swin-Small + MHA (weight: 1.0)
2. Swin-Small (weight: 0.9)
3. ViT-Base/16 (weight: 0.8)

Applies Clinical Triage Gate (tau = 0.70) on held-out test set (73 cases).
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

from src.data.dataset import DentalROICropDataset, create_patient_level_splits
from src.models.vit_backbones import create_model
from src.training.train_ensemble_binary import load_binary_dataset_records

CLASS_NAMES = {0: "Contact", 1: "Independent"}


def load_trained_models(device: torch.device):
    """Load the three best model checkpoints."""
    m1 = create_model("swin_s_mha", num_classes=2, pretrained=False)
    m1.load_state_dict(torch.load("weights/swin_s_mha_binary_best.pth", map_location=device))
    m1.to(device).eval()

    m2 = create_model("swin_small", num_classes=2, pretrained=False)
    m2.load_state_dict(torch.load("weights/swin_small_binary_best.pth", map_location=device))
    m2.to(device).eval()

    m3 = create_model("vit_base_16", num_classes=2, pretrained=False)
    m3.load_state_dict(torch.load("weights/vit_base_binary_best.pth", map_location=device))
    m3.to(device).eval()

    return [m1, m2, m3]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ensemble Evaluation on: {device}")

    records = load_binary_dataset_records(Path("data/processed/toothfairy_binary_labels.csv"))
    _, _, test_records = create_patient_level_splits(records, train_ratio=0.80, val_ratio=0.10, seed=42)
    print(f"Total Test Samples: {len(test_records)}")

    test_ds = DentalROICropDataset(test_records, is_training=False, preload=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    models = load_trained_models(device)
    weights = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    weights /= weights.sum()

    all_targets = []
    all_case_ids = []
    ensemble_probs = []

    with torch.no_grad():
        for images, targets, case_ids in test_loader:
            images = images.to(device)
            p1 = F.softmax(models[0](images), dim=1).cpu().numpy()
            p2 = F.softmax(models[1](images), dim=1).cpu().numpy()
            p3 = F.softmax(models[2](images), dim=1).cpu().numpy()

            p_ens = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
            ensemble_probs.extend(p_ens)
            all_targets.extend(targets.numpy())
            all_case_ids.extend(case_ids)

    ensemble_probs = np.array(ensemble_probs)
    all_targets = np.array(all_targets)
    all_preds = np.argmax(ensemble_probs, axis=1)
    all_confs = np.max(ensemble_probs, axis=1)

    # Metrics
    overall_acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, labels=[0, 1])
    macro_f1 = np.mean(f1)
    auc = roc_auc_score(all_targets, ensemble_probs[:, 1])
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    # Triage Gate tau = 0.70
    triage_threshold = 0.70
    meets_mask = all_confs >= triage_threshold
    review_mask = ~meets_mask

    meets_acc = accuracy_score(all_targets[meets_mask], all_preds[meets_mask]) if np.sum(meets_mask) > 0 else 0.0

    print("\n=======================================================")
    print("  HASIL EVALUASI ENSEMBLE 2-KELAS (CONTACT vs INDEPENDENT)")
    print("=======================================================")
    print(f"Overall Accuracy        : {overall_acc*100:.2f}%")
    print(f"Macro-F1 Score          : {macro_f1*100:.2f}%")
    print(f"Macro-AUC Score         : {auc*100:.2f}%")
    print(f"Composite Score         : {(0.444*macro_f1 + 0.333*auc + 0.222*overall_acc)*100:.2f}")
    print("\nPer-Class Performance:")
    print(f"  Contact     (0) -> Precision: {prec[0]*100:.2f}% | Recall: {rec[0]*100:.2f}% | F1: {f1[0]*100:.2f}%")
    print(f"  Independent (1) -> Precision: {prec[1]*100:.2f}% | Recall: {rec[1]*100:.2f}% | F1: {f1[1]*100:.2f}%")
    print(f"\nConfusion Matrix:\n{cm}")
    print("\nClinical Triage Gate (tau = 0.70):")
    print(f"  Meets Threshold (Auto-Verified) : {np.sum(meets_mask)} ({np.mean(meets_mask)*100:.1f}%) -> Selective Accuracy: {meets_acc*100:.2f}%")
    print(f"  Review Required (Dokter)        : {np.sum(review_mask)} ({np.mean(review_mask)*100:.1f}%)")
    print("=======================================================\n")

    # Save JSON summary
    res_summary = {
        "overall_accuracy": round(float(overall_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_auc": round(float(auc), 4),
        "composite_score": round(float(0.444*macro_f1 + 0.333*auc + 0.222*overall_acc), 4),
        "selective_accuracy_meets_threshold": round(float(meets_acc), 4),
        "meets_threshold_count": int(np.sum(meets_mask)),
        "review_required_count": int(np.sum(review_mask)),
        "per_class": {
            "Contact": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4), "f1": round(float(f1[0]), 4)},
            "Independent": {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4), "f1": round(float(f1[1]), 4)},
        },
        "confusion_matrix": cm.tolist()
    }
    with open("reports/binary_ensemble_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(res_summary, f, indent=2)
    print("Saved evaluation report to: reports/binary_ensemble_evaluation_results.json")


if __name__ == "__main__":
    main()
