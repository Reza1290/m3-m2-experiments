import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
from scipy import ndimage
from src.data.loader import load_volume
from src.data.preprocessor import extract_mask_guided_roi

def evaluate_pair_calibrated(roi_img, roi_mask, m2_fdi, m3_fdi, dmin, sagittal_slice, spacing=(0.3, 0.3)):
    mask_m2 = (roi_mask == m2_fdi)
    mask_m3 = (roi_mask == m3_fdi)
    if not np.any(mask_m2) or not np.any(mask_m3):
        return None

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
            "sagittal_slice": sagittal_slice,
        }

    # 2. Contact analysis
    pts_m2 = np.argwhere(mask_m2)  # (z, y)
    pts_m3 = np.argwhere(mask_m3)
    
    d2 = ndimage.distance_transform_edt(~mask_m2)
    d3 = ndimage.distance_transform_edt(~mask_m3)
    gap = d2 + d3
    cp_list = np.argwhere(gap == gap.min())
    cp = np.mean(cp_list, axis=0)  # (z, y)

    # Determine crown vs root elevation empirically from enamel radiodensity
    z_min_m2, z_max_m2 = pts_m2[:, 0].min(), pts_m2[:, 0].max()
    z_mid_m2 = (z_min_m2 + z_max_m2) / 2.0
    z_span = z_max_m2 - z_min_m2

    upper_mask = mask_m2.copy()
    upper_mask[pts_m2[:, 0] <= z_mid_m2, pts_m2[:, 1]] = False
    lower_mask = mask_m2.copy()
    lower_mask[pts_m2[:, 0] > z_mid_m2, pts_m2[:, 1]] = False

    hu_upper = np.mean(roi_img[upper_mask]) if np.any(upper_mask) else 0.0
    hu_lower = np.mean(roi_img[lower_mask]) if np.any(lower_mask) else 0.0

    crown_is_upper = (hu_upper >= hu_lower)
    if crown_is_upper:
        # Upper is crown, lower is root
        is_root = (cp[0] <= z_mid_m2 + 0.15 * z_span)
        is_pure_crown = (cp[0] > z_mid_m2 + 0.30 * z_span)
    else:
        # Lower is crown, upper is root
        is_root = (cp[0] >= z_mid_m2 - 0.15 * z_span)
        is_pure_crown = (cp[0] < z_mid_m2 - 0.30 * z_span)

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
            "sagittal_slice": sagittal_slice,
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

        outer_mask = np.ones(len(distal_pts), dtype=bool)
        outer_mask[3:-3] = False
        poly = np.polyfit(rows_arr[outer_mask], cols_arr[outer_mask], 1)
        expected_cols = np.polyval(poly, rows_arr)

        if m3_to_right:
            indentation_px = expected_cols - cols_arr
        else:
            indentation_px = cols_arr - expected_cols

        max_defect_px = np.max(indentation_px[3:-3]) if len(distal_pts) > 6 else 0.0
        defect_depth_mm = max(0.0, max_defect_px * spacing[1])

        # Resorption criteria:
        # At root/cervical elevation, defect depth >= 0.45 mm (>= 1.5 voxels into root dentin)
        # and close abutment (dmin <= 0.40 mm)
        if is_root and defect_depth_mm >= 0.45 and dmin <= 0.40:
            is_resorption = True

    if is_resorption:
        conf = min(0.92, 0.80 + float(defect_depth_mm - 0.45) * 0.15)
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
        "sagittal_slice": sagittal_slice,
    }

# Test on 30 cases from the dataset
with open('data/review/clinical_joint_review_sheet.csv', 'r', encoding='utf-8') as f:
    records = list(csv.DictReader(f))

classes = {'Independent': 0, 'Contact': 0, 'Resorption': 0}
res_list = []
for r in records:
    dmin = float(r['dmin_mm'])
    defect = float(r['defect_depth_mm'])
    if dmin > 0.50:
        classes['Independent'] += 1
    elif defect >= 0.45 and dmin <= 0.40:
        classes['Resorption'] += 1
        res_list.append(r)
    else:
        classes['Contact'] += 1

print("Class distribution with calibrated defect threshold (0.45 mm):")
for k, v in classes.items():
    print(f"  {k:12s}: {v:4d} ({v/len(records)*100:.1f}%)")
print(f"\nTotal Resorption cases: {len(res_list)}")
for r in res_list[:8]:
    print(f"  {r['case_id']} ({r['m2_fdi']}-{r['m3_fdi']}): Dmin={r['dmin_mm']} mm, Defect={r['defect_depth_mm']} mm")
