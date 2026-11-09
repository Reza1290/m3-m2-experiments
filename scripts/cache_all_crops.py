"""Cache 2D Sagittal ROI Crops for Entire ToothFairy2 Dataset (723 Pairs).

Extracts and saves 224x224 cropped patches to data/processed/crops_2d/.
Adds 'crop_path' to data/processed/toothfairy_classified_labels.csv.
"""

import csv
import time
from pathlib import Path
import numpy as np

from src.data.loader import load_volume
from src.data.preprocessor import extract_mask_guided_roi


def cache_crops(
    labels_csv_path: Path = Path("data/processed/toothfairy_classified_labels.csv"),
    labels_dir: Path = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr"),
    raw_dir: Path = Path("data/raw"),
    crops_dir: Path = Path("data/processed/crops_2d"),
    overwrite: bool = True,
):
    crops_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Caching 2D sagittal crops for {len(records)} records (overwrite={overwrite})...")
    t_start = time.time()

    # Group by case_id so each .mha volume is only loaded ONCE from disk
    cases = {}
    for r in records:
        cases.setdefault(r["case_id"], []).append(r)

    total_cases = len(cases)
    print(f"Total unique volumes: {total_cases}")

    updated_records = []
    saved_count = 0

    for idx, (case_id, pair_list) in enumerate(cases.items(), 1):
        mask_path = labels_dir / f"{case_id}.mha"
        raw_path = raw_dir / f"{case_id}_0000.mha"

        if not mask_path.exists():
            print(f"[{idx}/{total_cases}] Missing mask: {mask_path}")
            continue

        try:
            mask_vol = load_volume(mask_path)
            cbct_vol = load_volume(raw_path) if raw_path.exists() else mask_vol
        except Exception as e:
            print(f"[{idx}/{total_cases}] Error loading {case_id}: {e}")
            continue

        for r in pair_list:
            m2 = int(r["m2_fdi"])
            m3 = int(r["m3_fdi"])
            out_filename = f"{case_id}_{m2}_{m3}.npy"
            out_path = crops_dir / out_filename

            # If overwrite requested or not yet extracted, extract via KD-tree & crop
            if overwrite or not out_path.exists():
                crop_res = extract_mask_guided_roi(
                    cbct_volume=cbct_vol,
                    mask_volume=mask_vol,
                    m2_fdi=m2,
                    m3_fdi=m3,
                    crop_size=(224, 224),
                )
                if crop_res.success and crop_res.roi_image is not None:
                    # Normalize to float32 [0, 1]
                    img = crop_res.roi_image.astype(np.float32)
                    if img.max() > img.min():
                        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-7)
                    else:
                        img_norm = np.zeros_like(img)
                    np.save(out_path, img_norm)
                    saved_count += 1
                    r["crop_path"] = str(out_path.as_posix())
                else:
                    r["crop_path"] = ""
            else:
                r["crop_path"] = str(out_path.as_posix())

            updated_records.append(r)

        if idx % 50 == 0 or idx == total_cases:
            elapsed = time.time() - t_start
            print(f"  Processed {idx}/{total_cases} volumes ({saved_count} crops cached) in {elapsed:.1f}s")

    # Update CSV with crop_path
    fieldnames = list(records[0].keys())
    if "crop_path" not in fieldnames:
        fieldnames.append("crop_path")

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_records)

    elapsed = time.time() - t_start
    print(f"\nCompleted! Total {saved_count} crops cached in {elapsed:.1f}s.")
    print(f"Updated CSV saved at: {labels_csv_path}")


if __name__ == "__main__":
    cache_crops()
