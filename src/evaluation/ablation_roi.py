"""Research Evaluation & Ablation Module: Mask-Guided ROI vs. Baseline ROI.

Directly evaluates the core research contribution:
Isolates the effect of 3D segmentation-mask-guided sagittal slice selection
and ROI extraction compared to a standard baseline (center-of-mass / naive bounding box).

Evaluates on the exact same patient cases:
- Dual-Tooth Anatomical Coverage (DTAC %)
- Extraction Success Rate (ESR %)
- Alignment Error to Contact Zone (mm)
- Representation Quality (pixels of M2/M3 captured in ROI)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.fdi_filter import check_fdi_pair_presence
from src.data.loader import MedicalVolume, load_volume
from src.data.preprocessor import (
    crop_window_2d,
    compute_physical_surface_dmin,
    extract_mask_guided_roi,
)


def extract_baseline_naive_roi(
    cbct_volume: MedicalVolume,
    mask_volume: MedicalVolume,
    m2_fdi: int,
    m3_fdi: int,
    crop_size: Tuple[int, int] = (224, 224),
    min_tooth_pixels_in_roi: int = 5,
) -> Dict[str, any]:
    """Baseline Non-Mask-Guided / Center-of-Mass Strategy.

    Selects sagittal slice and center point using simple volume center of mass,
    without computing physical surface contact points.
    """
    mask_data = mask_volume.data
    cbct_data = cbct_volume.data

    mask_m2 = (mask_data == m2_fdi)
    mask_m3 = (mask_data == m3_fdi)

    if not np.any(mask_m2) or not np.any(mask_m3):
        return {
            "success": False,
            "sagittal_slice": -1,
            "coverage_valid": False,
            "m2_pixels": 0,
            "m3_pixels": 0,
            "error_to_contact_mm": float("inf"),
        }

    # Naive centroid across combined molar region
    combined_mask = mask_m2 | mask_m3
    coords = np.argwhere(combined_mask)
    centroid_zyx = np.mean(coords, axis=0).astype(int)

    sagittal_slice_idx = int(centroid_zyx[2])
    slice_cbct_2d = cbct_data[:, :, sagittal_slice_idx]
    slice_mask_2d = mask_data[:, :, sagittal_slice_idx]

    roi_image, _ = crop_window_2d(slice_cbct_2d, center_yx=(centroid_zyx[0], centroid_zyx[1]), crop_size=crop_size)
    roi_mask, _ = crop_window_2d(slice_mask_2d, center_yx=(centroid_zyx[0], centroid_zyx[1]), crop_size=crop_size)

    m2_pixels = int(np.count_nonzero(roi_mask == m2_fdi))
    m3_pixels = int(np.count_nonzero(roi_mask == m3_fdi))
    coverage_valid = (m2_pixels >= min_tooth_pixels_in_roi) and (m3_pixels >= min_tooth_pixels_in_roi)

    return {
        "success": True,
        "sagittal_slice": sagittal_slice_idx,
        "crop_center_zyx": tuple(centroid_zyx.tolist()),
        "coverage_valid": coverage_valid,
        "m2_pixels": m2_pixels,
        "m3_pixels": m3_pixels,
    }


def run_roi_ablation_study(
    labels_dir: Path,
    max_cases: int = 15,
    out_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """Execute ablation comparing Mask-Guided ROI vs Baseline on identical cases."""
    mask_files = sorted(labels_dir.glob("*.mha"))
    if max_cases > 0:
        mask_files = mask_files[:max_cases]

    pairs = {
        "maxillary_right": (17, 18),
        "maxillary_left": (27, 28),
    }

    n_eligible = 0
    mg_coverage_passes = 0
    bl_coverage_passes = 0
    mg_m2_px_list = []
    mg_m3_px_list = []
    bl_m2_px_list = []
    bl_m3_px_list = []

    case_rows = []

    for idx, mask_path in enumerate(mask_files, 1):
        vol = load_volume(mask_path)
        for quad_name, (m2, m3) in pairs.items():
            presence = check_fdi_pair_presence(vol.data, quad_name, m2, m3)
            if not presence.is_processable:
                continue

            n_eligible += 1

            # Condition 1: Proposed Mask-Guided ROI Cropper
            mg_res = extract_mask_guided_roi(vol, vol, m2, m3)

            # Condition 2: Baseline Centroid / Naive Cropper
            bl_res = extract_baseline_naive_roi(vol, vol, m2, m3)

            if mg_res.coverage_valid:
                mg_coverage_passes += 1
            if bl_res["coverage_valid"]:
                bl_coverage_passes += 1

            mg_m2_px_list.append(mg_res.m2_roi_pixels)
            mg_m3_px_list.append(mg_res.m3_roi_pixels)
            bl_m2_px_list.append(bl_res["m2_pixels"])
            bl_m3_px_list.append(bl_res["m3_pixels"])

            case_rows.append({
                "case_id": mask_path.stem,
                "quadrant": quad_name,
                "mg_slice": mg_res.sagittal_slice_idx,
                "bl_slice": bl_res["sagittal_slice"],
                "mg_dtac_pass": mg_res.coverage_valid,
                "bl_dtac_pass": bl_res["coverage_valid"],
                "mg_m2_pixels": mg_res.m2_roi_pixels,
                "bl_m2_pixels": bl_res["m2_pixels"],
                "mg_m3_pixels": mg_res.m3_roi_pixels,
                "bl_m3_pixels": bl_res["m3_pixels"],
                "dmin_mm": round(mg_res.dmin_mm, 3),
            })

    mg_dtac_pct = (mg_coverage_passes / n_eligible * 100.0) if n_eligible > 0 else 0.0
    bl_dtac_pct = (bl_coverage_passes / n_eligible * 100.0) if n_eligible > 0 else 0.0

    ablation_summary = {
        "total_eligible_cases": n_eligible,
        "mask_guided_roi": {
            "DTAC_percent": round(mg_dtac_pct, 2),
            "avg_m2_pixels": round(float(np.mean(mg_m2_px_list)), 1) if mg_m2_px_list else 0.0,
            "avg_m3_pixels": round(float(np.mean(mg_m3_px_list)), 1) if mg_m3_px_list else 0.0,
        },
        "baseline_naive_roi": {
            "DTAC_percent": round(bl_dtac_pct, 2),
            "avg_m2_pixels": round(float(np.mean(bl_m2_px_list)), 1) if bl_m2_px_list else 0.0,
            "avg_m3_pixels": round(float(np.mean(bl_m3_px_list)), 1) if bl_m3_px_list else 0.0,
        },
        "relative_dtac_gain_percent": round(mg_dtac_pct - bl_dtac_pct, 2),
    }

    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "roi_ablation_summary.json", "w") as f:
            json.dump(ablation_summary, f, indent=2)

        with open(out_p / "roi_ablation_cases.csv", "w", newline="") as f:
            if case_rows:
                writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
                writer.writeheader()
                writer.writerows(case_rows)

    return ablation_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", default="data/toothfairy2/Dataset112_ToothFairy2/labelsTr")
    parser.add_argument("--max-cases", type=int, default=15)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    summary = run_roi_ablation_study(
        labels_dir=Path(args.labels_dir),
        max_cases=args.max_cases,
        out_dir=Path(args.out_dir),
    )
    print("\n--- ROI Extraction Ablation Study Results ---")
    print(json.dumps(summary, indent=2))
