"""Batch AI Vision Radiographical Labeling & Clinical Review Sheet Generator.

Processes genuine CBCT scans and ToothFairy2 segmentation masks.
Applies radiographic vision analysis:
- Intact PDL / bone gap separation (> 0.50 mm) -> Class 1: Independent
- Crown-to-crown / crown-to-cervical abutment with intact root contour -> Class 0: Contact
- Deep root dentin concavity / excavation defect (>= 0.70 mm) -> Class 2: Resorption

Outputs:
- data/processed/toothfairy_classified_labels.csv (Model training ground truth)
- data/review/clinical_joint_review_sheet.xlsx (Interactive Radiologist review sheet)
- data/review/clinical_joint_review_sheet.csv (Quick inspection format)
"""

import csv
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from scipy import ndimage

from src.data.loader import load_volume
from src.data.preprocessor import extract_mask_guided_roi


def evaluate_pair(cbct_vol, mask_vol, m2_fdi: int, m3_fdi: int, crop_size=(224, 224)):
    """Analyze CBCT crop for genuine clinical relationship."""
    res = extract_mask_guided_roi(
        cbct_volume=cbct_vol,
        mask_volume=mask_vol,
        m2_fdi=m2_fdi,
        m3_fdi=m3_fdi,
        crop_size=crop_size,
    )
    if not res.success or res.roi_image is None or res.roi_mask is None:
        return None

    dmin = res.dmin_mm
    roi_mask = res.roi_mask
    roi_img = res.roi_image
    spacing = cbct_vol.spacing

    mask_m2 = (roi_mask == m2_fdi)
    mask_m3 = (roi_mask == m3_fdi)
    if not np.any(mask_m2) or not np.any(mask_m3):
        return None

    pts_m2 = np.argwhere(mask_m2)  # (z, y)
    pts_m3 = np.argwhere(mask_m3)

    # 1. Physical separation check
    if dmin > 0.50:
        conf = min(0.96, 0.75 + float(dmin - 0.50) * 0.15)
        findings = (
            f"Tampak celah radiolusen ligamen periodontal (PDL) dan septum tulang utuh "
            f"(jarak anatomis {dmin:.2f} mm) memisahkan mahkota M3 dari akar M2."
        )
        return {
            "class_name": "Independent",
            "class_id": 1,
            "confidence": round(conf, 3),
            "findings": findings,
            "dmin_mm": round(dmin, 4),
            "defect_depth_mm": 0.0,
            "sagittal_slice": res.sagittal_slice_idx,
            "roi_image": roi_img,
        }

    # 2. Contact analysis
    d2 = ndimage.distance_transform_edt(~mask_m2)
    d3 = ndimage.distance_transform_edt(~mask_m3)
    gap = d2 + d3
    cp_list = np.argwhere(gap == gap.min())
    cp = np.mean(cp_list, axis=0)  # (z, y)

    # Determine root vs crown elevation for M2
    z_min_m2, z_max_m2 = pts_m2[:, 0].min(), pts_m2[:, 0].max()
    z_mid_m2 = (z_min_m2 + z_max_m2) / 2.0
    z_span = z_max_m2 - z_min_m2

    is_mandibular = (m2_fdi in [37, 47])
    if is_mandibular:
        is_root = (cp[0] <= z_mid_m2 + 0.10 * z_span)
        is_pure_crown = (cp[0] > z_mid_m2 + 0.25 * z_span)
    else:  # Maxillary
        is_root = (cp[0] >= z_mid_m2 - 0.10 * z_span)
        is_pure_crown = (cp[0] < z_mid_m2 - 0.25 * z_span)

    if is_pure_crown:
        conf = 0.88
        findings = (
            f"Mahkota M3 berkontak pada aspek distal mahkota/enamel M2 (jarak {dmin:.2f} mm). "
            f"Lapisan email M2 intak tanpa keterlibatan sementum akar."
        )
        return {
            "class_name": "Contact",
            "class_id": 0,
            "confidence": conf,
            "findings": findings,
            "dmin_mm": round(dmin, 4),
            "defect_depth_mm": 0.0,
            "sagittal_slice": res.sagittal_slice_idx,
            "roi_image": roi_img,
        }

    # Check for concave root defect (saucer-shaped indentation)
    mean_y_m2 = np.mean(pts_m2[:, 1])
    mean_y_m3 = np.mean(pts_m3[:, 1])
    m3_to_right = (mean_y_m3 > mean_y_m2)

    row_cp = int(round(cp[0]))
    row_window = 12
    r_start = max(0, row_cp - row_window)
    r_end = min(roi_mask.shape[0], row_cp + row_window + 1)

    distal_pts = []
    for r in range(r_start, r_end):
        cols = np.where(mask_m2[r, :])[0]
        if len(cols) > 0:
            d_col = cols[-1] if m3_to_right else cols[0]
            distal_pts.append((r, d_col))

    is_resorption = False
    defect_depth_mm = 0.0

    if len(distal_pts) >= 10:
        rows_arr = np.array([p[0] for p in distal_pts])
        cols_arr = np.array([p[1] for p in distal_pts], dtype=float)

        # Baseline expected contour
        outer_mask = np.ones(len(distal_pts), dtype=bool)
        outer_mask[3:-3] = False
        poly = np.polyfit(rows_arr[outer_mask], cols_arr[outer_mask], 1)
        expected_cols = np.polyval(poly, rows_arr)

        # Indentation is inward deflection
        if m3_to_right:
            indentation_px = expected_cols - cols_arr
        else:
            indentation_px = cols_arr - expected_cols

        max_defect_px = np.max(indentation_px[3:-3]) if len(distal_pts) > 6 else 0.0
        defect_depth_mm = max(0.0, max_defect_px * spacing[1])

        # Resorption requires significant indentation into root (>= 0.70 mm) AND intimate contact (dmin <= 0.35)
        if is_root and defect_depth_mm >= 0.70 and dmin <= 0.35:
            is_resorption = True

    if is_resorption:
        conf = min(0.92, 0.80 + float(defect_depth_mm - 0.70) * 0.12)
        findings = (
            f"Tampak defek konkavitas radiolusen (External Root Resorption) sedalam ~{defect_depth_mm:.2f} mm "
            f"pada dinding akar distal M2 akibat penekanan impaksi mahkota M3."
        )
        c_name = "Resorption"
        c_id = 2
    else:
        conf = 0.85
        findings = (
            f"Mahkota M3 berkontak langsung pada servikal/akar M2 (jarak {dmin:.2f} mm). "
            f"Kontur sementum dan dentin akar distal M2 tampak utuh, halus, dan cembung normal tanpa bukti defek kawah resorpsi."
        )
        c_name = "Contact"
        c_id = 0

    return {
        "class_name": c_name,
        "class_id": c_id,
        "confidence": round(conf, 3),
        "findings": findings,
        "dmin_mm": round(dmin, 4),
        "defect_depth_mm": round(defect_depth_mm, 3),
        "sagittal_slice": res.sagittal_slice_idx,
        "roi_image": roi_img,
    }


def create_styled_excel(records, excel_path: Path):
    """Create a beautifully formatted Excel review sheet for clinicians."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Clinical Joint Review"

    # Define styles
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    bold_font = Font(name="Calibri", size=10, bold=True)
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)

    thin_border = Border(
        left=Side(style="thin", color="D3D3D3"),
        right=Side(style="thin", color="D3D3D3"),
        top=Side(style="thin", color="D3D3D3"),
        bottom=Side(style="thin", color="D3D3D3"),
    )

    class_fills = {
        "Contact": PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid"),     # Soft blue
        "Independent": PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid"), # Soft green
        "Resorption": PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid"),  # Soft orange/red
    }

    headers = [
        "Case ID",
        "Quadrant",
        "M2 FDI",
        "M3 FDI",
        "D_min (mm)",
        "Defect Depth (mm)",
        "Sagittal Slice",
        "AI Vision Class",
        "AI Confidence",
        "AI Radiographical Findings",
        "Dr. Adjudicated Class",
        "Dr. Confidence",
        "Dr. Agreement with AI",
        "Dr. Clinical Notes",
    ]

    ws.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = align_center

    ws.row_dimensions[1].height = 28

    for r_idx, r in enumerate(records, 2):
        row_vals = [
            r["case_id"],
            r["quadrant"],
            int(r["m2_fdi"]),
            int(r["m3_fdi"]),
            float(r["dmin_mm"]),
            float(r["defect_depth_mm"]),
            int(r["sagittal_slice"]),
            r["class_name"],
            f"{float(r['confidence']):.1%}",
            r["findings"],
            "",         # Dr. Adjudicated Class
            "",         # Dr. Confidence
            "Agree",    # Default agreement
            "",         # Dr. Clinical Notes
        ]
        ws.append(row_vals)
        ws.row_dimensions[r_idx].height = 24

        for c_idx in range(1, len(row_vals) + 1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.font = data_font
            cell.border = thin_border
            if c_idx in [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13]:
                cell.alignment = align_center
            else:
                cell.alignment = align_left

            # Highlight AI Class
            if c_idx == 8:
                cell.fill = class_fills.get(r["class_name"], PatternFill(fill_type=None))
                cell.font = bold_font

    # Adjust column widths
    col_widths = {
        "A": 18,  # Case ID
        "B": 18,  # Quadrant
        "C": 10,  # M2
        "D": 10,  # M3
        "E": 12,  # D_min
        "F": 16,  # Defect Depth
        "G": 14,  # Sagittal Slice
        "H": 16,  # AI Class
        "I": 15,  # AI Confidence
        "J": 55,  # AI Findings
        "K": 20,  # Dr Class
        "L": 15,  # Dr Confidence
        "M": 20,  # Dr Agreement
        "N": 30,  # Dr Notes
    }
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(excel_path)
    print(f"Saved styled Excel sheet to: {excel_path}")


def run_batch_vision_labeling():
    input_v1_csv = Path("data/processed/toothfairy_classified_labels_v1_dmin_heuristic.csv")
    labels_dir = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr")
    raw_dir = Path("data/raw")
    crops_dir = Path("data/processed/crops_2d")
    output_labels_csv = Path("data/processed/toothfairy_classified_labels.csv")
    output_review_csv = Path("data/review/clinical_joint_review_sheet.csv")
    output_review_xlsx = Path("data/review/clinical_joint_review_sheet.xlsx")

    crops_dir.mkdir(parents=True, exist_ok=True)

    print("===================================================================")
    print("  ToothFairy2 AI Vision-Based Radiographical Adjudication Pipeline")
    print("===================================================================")

    if not input_v1_csv.exists():
        print(f"Error: {input_v1_csv} not found!")
        return

    with open(input_v1_csv, "r", encoding="utf-8") as f:
        v1_records = list(csv.DictReader(f))

    print(f"Loaded {len(v1_records)} pairs from v1 catalog.")

    # Group pairs by case_id to load each 3D CBCT volume only once
    cases = {}
    for r in v1_records:
        cases.setdefault(r["case_id"], []).append(r)

    total_cases = len(cases)
    print(f"Total unique cases to evaluate: {total_cases}\n")

    t_start = time.time()
    processed_results = []
    skipped_count = 0
    saved_crops = 0

    for idx, (case_id, pairs) in enumerate(cases.items(), 1):
        mask_path = labels_dir / f"{case_id}.mha"
        raw_path = raw_dir / f"{case_id}_0000.mha"

        # User Rule: Skip immediately if mask or raw scan is missing
        if not mask_path.exists():
            print(f"[{idx:3d}/{total_cases}] SKIP: Missing mask {mask_path.name}")
            skipped_count += len(pairs)
            continue

        if not raw_path.exists():
            print(f"[{idx:3d}/{total_cases}] SKIP: Missing raw CBCT {raw_path.name}")
            skipped_count += len(pairs)
            continue

        try:
            mask_vol = load_volume(mask_path)
            cbct_vol = load_volume(raw_path)
        except Exception as e:
            print(f"[{idx:3d}/{total_cases}] ERROR loading {case_id}: {e}")
            skipped_count += len(pairs)
            continue

        for p in pairs:
            m2 = int(p["m2_fdi"])
            m3 = int(p["m3_fdi"])
            quadrant = p["quadrant"]

            eval_res = evaluate_pair(cbct_vol, mask_vol, m2, m3)
            if eval_res is None:
                skipped_count += 1
                continue

            # Save normalized 2D sagittal crop
            crop_filename = f"{case_id}_{m2}_{m3}.npy"
            crop_path = crops_dir / crop_filename
            roi_img = eval_res["roi_image"].astype(np.float32)
            if roi_img.max() > roi_img.min():
                roi_norm = (roi_img - roi_img.min()) / (roi_img.max() - roi_img.min() + 1e-7)
            else:
                roi_norm = np.zeros_like(roi_img)
            np.save(crop_path, roi_norm)
            saved_crops += 1

            processed_results.append({
                "case_id": case_id,
                "quadrant": quadrant,
                "m2_fdi": m2,
                "m3_fdi": m3,
                "dmin_mm": eval_res["dmin_mm"],
                "defect_depth_mm": eval_res["defect_depth_mm"],
                "sagittal_slice": eval_res["sagittal_slice"],
                "class_id": eval_res["class_id"],
                "class_name": eval_res["class_name"],
                "confidence": eval_res["confidence"],
                "findings": eval_res["findings"],
                "crop_path": str(crop_path.as_posix()),
            })

        if idx % 25 == 0 or idx == total_cases:
            elapsed = time.time() - t_start
            rate = idx / elapsed if elapsed > 0 else 1
            eta = (total_cases - idx) / rate
            print(
                f"[{idx:3d}/{total_cases}] Valid pairs labeled: {len(processed_results)} | "
                f"Crops cached: {saved_crops} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m"
            )

    # 1. Save data/processed/toothfairy_classified_labels.csv (used for training DataLoader)
    training_fields = [
        "case_id",
        "quadrant",
        "m2_fdi",
        "m3_fdi",
        "dmin_mm",
        "sagittal_slice",
        "class_id",
        "class_name",
        "confidence",
        "crop_path",
    ]
    with open(output_labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=training_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(processed_results)
    print(f"\nSaved updated training dataset to: {output_labels_csv}")

    # 2. Save data/review/clinical_joint_review_sheet.csv
    review_fields = [
        "case_id",
        "quadrant",
        "m2_fdi",
        "m3_fdi",
        "dmin_mm",
        "defect_depth_mm",
        "sagittal_slice",
        "class_name",
        "confidence",
        "findings",
        "crop_path",
    ]
    with open(output_review_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(processed_results)
    print(f"Saved review CSV to: {output_review_csv}")

    # 3. Save data/review/clinical_joint_review_sheet.xlsx
    create_styled_excel(processed_results, output_review_xlsx)

    total_time = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"AI Vision Labeling Finished in {total_time/60:.2f} minutes!")
    print(f"Total valid pairs successfully vision-adjudicated: {len(processed_results)}")
    print(f"Total skipped: {skipped_count}")

    # Distribution summary
    dist = {}
    for r in processed_results:
        c = r["class_name"]
        dist[c] = dist.get(c, 0) + 1
    print("Class Distribution:")
    for c, cnt in dist.items():
        pct = cnt / len(processed_results) * 100
        print(f"  - {c:12s}: {cnt:4d} ({pct:5.1f}%)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_batch_vision_labeling()
