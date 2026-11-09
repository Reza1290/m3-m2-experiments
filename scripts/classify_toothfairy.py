"""Automated 3-Class Classification for Entire ToothFairy2 Dataset (480 Scans).

Classifies all eligible M2-M3 molar pairs across all 4 quadrants into:
- 0: Contact
- 1: Independent
- 2: Resorption

Saves structured outputs to data/processed/toothfairy_classified_labels.csv.
"""

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.fdi_filter import SUPPORTED_FDI_PAIRS, check_fdi_pair_presence
from src.data.loader import load_volume
from src.data.preprocessor import compute_physical_surface_dmin


def run_full_dataset_classification(
    labels_dir: Path,
    output_csv: Path,
    max_scans: Optional[int] = None,
):
    files = sorted(labels_dir.glob("*.mha"))
    if max_scans is not None:
        files = files[:max_scans]

    total_files = len(files)
    results = []

    print(f"===============================================================")
    print(f"  ToothFairy2 Full Dataset 3-Class Classifier")
    print(f"  Total CBCT volumes to process: {total_files}")
    print(f"  Target quadrants: {list(SUPPORTED_FDI_PAIRS.keys())}")
    print(f"===============================================================\n")

    t_start = time.time()

    for idx, fpath in enumerate(files, 1):
        try:
            vol = load_volume(fpath)
            mask = vol.data
            spacing = vol.spacing
        except Exception as e:
            print(f"[{idx:3d}/{total_files}] Error reading {fpath.name}: {e}")
            continue

        for quad_name, (m2, m3) in SUPPORTED_FDI_PAIRS.items():
            pres = check_fdi_pair_presence(mask, quad_name, m2, m3)
            if not pres.is_processable:
                continue

            # Compute physical surface D_min
            dmin_mm, pt_m2, pt_m3 = compute_physical_surface_dmin(
                mask_m2=(mask == m2),
                mask_m3=(mask == m3),
                spacing_zyx=spacing,
            )

            # Sagittal slice index
            sagittal_slice = int(np.round((pt_m2[2] + pt_m3[2]) / 2.0))

            # 3-Class Clinical Assignment:
            # Class 1 (Independent): Intact PDL / bone gap separation (> 0.50 mm)
            # Class 0 (Contact): Abutment without deep root invasion
            # Class 2 (Resorption): Intimate root proximity (<= 0.32 mm) focused at cervical/root elevation
            if dmin_mm > 0.50:
                class_id = 1
                class_name = "Independent"
                conf = min(0.98, 0.70 + float(dmin_mm - 0.50) * 0.25)
            else:
                m2_voxels = np.argwhere(mask == m2)
                m2_z_min, m2_z_max = m2_voxels[:, 0].min(), m2_voxels[:, 0].max()
                z_root_cervical_threshold = m2_z_min + 0.5 * (m2_z_max - m2_z_min)

                if pt_m2[0] <= z_root_cervical_threshold and dmin_mm <= 0.32:
                    class_id = 2
                    class_name = "Resorption"
                    conf = 0.76
                else:
                    class_id = 0
                    class_name = "Contact"
                    conf = 0.82

            results.append({
                "case_id": fpath.stem,
                "quadrant": quad_name,
                "m2_fdi": m2,
                "m3_fdi": m3,
                "dmin_mm": round(dmin_mm, 4),
                "sagittal_slice": sagittal_slice,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(conf, 3),
            })

        if idx % 25 == 0 or idx == total_files:
            elapsed = time.time() - t_start
            rate = idx / elapsed
            est_remaining = (total_files - idx) / rate if rate > 0 else 0
            print(
                f"[{idx:3d}/{total_files}] Processed {idx} scans | "
                f"Valid pairs found so far: {len(results)} | "
                f"Elapsed: {elapsed/60:.1f}m | Est. remaining: {est_remaining/60:.1f}m"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Classification completed successfully!")
    print(f"Total scans scanned: {total_files}")
    print(f"Total valid pairs classified: {len(results)}")
    print(f"Total time: {total_elapsed/60:.2f} minutes")
    print(f"Saved to: {output_csv}")

    classes_count = {}
    for r in results:
        c = r["class_name"]
        classes_count[c] = classes_count.get(c, 0) + 1
    print("Class Distribution:", classes_count)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    labels_path = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr")
    out_path = Path("data/processed/toothfairy_classified_labels.csv")
    run_full_dataset_classification(labels_path, out_path, max_scans=None)
