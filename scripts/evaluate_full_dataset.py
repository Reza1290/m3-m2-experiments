"""Full Dataset Evaluation (All 720 Cases) on CBCT M3-M2 Relationship CDSS.

Runs inference across the entire dataset of 720 cases.
Computes:
1. Full dataset Accuracy, Macro-F1, Macro-AUC.
2. Per-class Precision, Recall, and Confusion Matrix.
3. Clinical Triage Gate statistics (threshold tau = 0.70).
4. Saves:
   - reports/full_dataset_evaluation_results.json
   - reports/full_dataset_predictions.csv
"""

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
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
from src.data.dataset import DentalROICropDataset
from src.models.ensemble import CLASS_NAMES
from src.models.vit_backbones import create_model
from src.training.train import load_dataset_records


def run_full_evaluation(
    weights_path: Path = Path("weights/resnet18_best.pth"),
    csv_path: Path = Path("data/processed/toothfairy_classified_labels.csv"),
    triage_threshold: float = 0.70,
):
    print("=================================================================")
    print("  Full Dataset Comprehensive Evaluation (All 720 Cases)")
    print(f"  Model Weights: {weights_path}")
    print(f"  Triage Threshold: {triage_threshold:.0%}")
    print("=================================================================\n")

    # Load all records
    records = load_dataset_records(csv_path)
    total_samples = len(records)
    print(f"Loaded {total_samples} total dataset crops for evaluation.")

    # Create dataset without training augmentations, with in-memory caching
    dataset = DentalROICropDataset(records, is_training=False, preload=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model
    device = torch.device("cpu")
    model = create_model("resnet18", num_classes=3, pretrained=False).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    all_case_ids = []

    t0 = time.time()
    with torch.no_grad():
        for images, targets, case_ids in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_case_ids.extend(case_ids)

    elapsed = time.time() - t0
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Academic metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precisions = precision_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    recalls = recall_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    f1s = f1_score(all_targets, all_preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]).tolist()

    try:
        macro_auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = float(acc)

    composite_score = float(0.444 * macro_f1 + 0.333 * macro_auc + 0.222 * acc)

    # Triage analysis
    confidences = np.max(all_probs, axis=1)
    meets_threshold = confidences >= triage_threshold
    num_meets = int(np.sum(meets_threshold))
    num_review = int(total_samples - num_meets)
    pct_meets = float(num_meets / total_samples * 100)
    pct_review = float(num_review / total_samples * 100)

    # Accuracy on cases that meet threshold (Selective Risk / Accuracy)
    if num_meets > 0:
        acc_meets_threshold = float(accuracy_score(all_targets[meets_threshold], all_preds[meets_threshold]))
    else:
        acc_meets_threshold = 0.0

    print(f"\nEvaluation finished in {elapsed:.2f}s ({elapsed/total_samples*1000:.1f}ms per scan)!\n")
    print("-----------------------------------------------------------------")
    print(f"  Overall Accuracy (Full 720 Cases):   {acc:.2%}")
    print(f"  Accuracy on Meets Threshold Subset:   {acc_meets_threshold:.2%} (Selective Accuracy)")
    print(f"  Macro-F1 Score:                      {macro_f1:.4f}")
    print(f"  Macro-AUC:                           {macro_auc:.4f}")
    print(f"  Composite Ranking Score:             {composite_score:.4f}")
    print("-----------------------------------------------------------------")
    print("\nPer-Class Breakdown:")
    for i, c_name in CLASS_NAMES.items():
        print(f"  Class {i} ({c_name:11s}) | Precision: {precisions[i]:.2%} | Recall: {recalls[i]:.2%} | F1: {f1s[i]:.2%}")

    print("\nConfusion Matrix [Rows: Ground Truth, Cols: Prediction]:")
    print("                 Pred Contact  Pred Indep  Pred Resorp")
    print(f"  True Contact   :  {cm[0][0]:8d}    {cm[0][1]:8d}    {cm[0][2]:8d}")
    print(f"  True Indep     :  {cm[1][0]:8d}    {cm[1][1]:8d}    {cm[1][2]:8d}")
    print(f"  True Resorp    :  {cm[2][0]:8d}    {cm[2][1]:8d}    {cm[2][2]:8d}")

    print("\nClinical Triage Gate Statistics (tau = 0.70):")
    print(f"  - Prediction Meets Threshold (Conf >= 70%): {num_meets:4d} ({pct_meets:5.1f}%)")
    print(f"  - Review Required (Conf < 70%):             {num_review:4d} ({pct_review:5.1f}%)")
    print(f"  - Average Confidence:                       {np.mean(confidences):.2%}")
    print(f"  - Median Confidence:                        {np.median(confidences):.2%}")
    print("-----------------------------------------------------------------\n")

    # Save detailed JSON report
    report = {
        "total_samples": total_samples,
        "overall_accuracy": round(acc, 4),
        "selective_accuracy_meets_threshold": round(acc_meets_threshold, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_auc": round(macro_auc, 4),
        "composite_ranking_score": round(composite_score, 4),
        "triage_gate": {
            "threshold": triage_threshold,
            "meets_threshold_count": num_meets,
            "meets_threshold_percent": round(pct_meets, 2),
            "review_required_count": num_review,
            "review_required_percent": round(pct_review, 2),
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "median_confidence": round(float(np.median(confidences)), 4),
        },
        "per_class": {
            c_name: {
                "precision": round(precisions[i], 4),
                "recall": round(recalls[i], 4),
                "f1_score": round(f1s[i], 4),
            }
            for i, c_name in CLASS_NAMES.items()
        },
        "confusion_matrix": cm,
    }

    report_path = Path("reports/full_dataset_evaluation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full evaluation summary to: {report_path}")

    # Save detailed per-case prediction CSV
    pred_csv_path = Path("reports/full_dataset_predictions.csv")
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "m2_fdi",
            "m3_fdi",
            "ground_truth_id",
            "ground_truth_name",
            "predicted_id",
            "predicted_name",
            "is_correct",
            "confidence",
            "prob_contact",
            "prob_independent",
            "prob_resorption",
            "triage_status",
        ])
        for idx in range(total_samples):
            rec = records[idx]
            gt = all_targets[idx]
            pr = all_preds[idx]
            conf = confidences[idx]
            status = "Prediction Meets Threshold" if conf >= triage_threshold else "Review Required"
            writer.writerow([
                rec["case_id"],
                rec["m2_fdi"],
                rec["m3_fdi"],
                gt,
                CLASS_NAMES[gt],
                pr,
                CLASS_NAMES[pr],
                int(gt == pr),
                round(float(conf), 4),
                round(float(all_probs[idx, 0]), 4),
                round(float(all_probs[idx, 1]), 4),
                round(float(all_probs[idx, 2]), 4),
                status,
            ])
    print(f"Saved per-case prediction CSV to: {pred_csv_path}")


if __name__ == "__main__":
    run_full_evaluation()
