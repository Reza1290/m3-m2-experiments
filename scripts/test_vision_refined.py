import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import ndimage
from src.data.loader import load_volume
from src.data.preprocessor import extract_mask_guided_roi


def evaluate_pair(cbct_vol, mask_vol, m2, m3):
    res = extract_mask_guided_roi(cbct_vol, mask_vol, m2, m3)
    if not res.success:
        return None
    dmin = res.dmin_mm
    roi_mask = res.roi_mask
    roi_img = res.roi_image
    spacing = cbct_vol.spacing
    
    mask_m2 = (roi_mask == m2)
    mask_m3 = (roi_mask == m3)
    if not np.any(mask_m2) or not np.any(mask_m3):
        return None
        
    pts_m2 = np.argwhere(mask_m2) # (z, y)
    pts_m3 = np.argwhere(mask_m3)
    
    # 1. Physical separation check
    if dmin > 0.50:
        conf = min(0.96, 0.75 + float(dmin - 0.50) * 0.15)
        findings = f"Tampak celah radiolusen ligamen periodontal (PDL) dan septum tulang utuh (jarak anatomis {dmin:.2f} mm) memisahkan mahkota M3 dari akar M2."
        return {
            "class_name": "Independent",
            "class_id": 1,
            "confidence": round(conf, 3),
            "findings": findings,
            "dmin": round(dmin, 3),
            "defect_depth": 0.0,
            "slice_idx": res.sagittal_slice_idx,
        }
        
    # 2. Contact analysis
    d2 = ndimage.distance_transform_edt(~mask_m2)
    d3 = ndimage.distance_transform_edt(~mask_m3)
    gap = d2 + d3
    cp_list = np.argwhere(gap == gap.min())
    cp = np.mean(cp_list, axis=0) # (z, y)
    
    # Determine root vs crown elevation for M2
    z_min_m2, z_max_m2 = pts_m2[:, 0].min(), pts_m2[:, 0].max()
    z_mid_m2 = (z_min_m2 + z_max_m2) / 2.0
    z_span = z_max_m2 - z_min_m2
    
    is_mandibular = (m2 in [37, 47])
    if is_mandibular:
        is_root = (cp[0] <= z_mid_m2 + 0.10 * z_span)
        is_pure_crown = (cp[0] > z_mid_m2 + 0.25 * z_span)
    else: # Maxillary
        is_root = (cp[0] >= z_mid_m2 - 0.10 * z_span)
        is_pure_crown = (cp[0] < z_mid_m2 - 0.25 * z_span)
        
    if is_pure_crown:
        conf = 0.88
        findings = f"Mahkota M3 berkontak pada aspek distal mahkota/enamel M2 (jarak {dmin:.2f} mm). Lapisan email M2 intak tanpa keterlibatan sementum akar."
        return {
            "class_name": "Contact",
            "class_id": 0,
            "confidence": conf,
            "findings": findings,
            "dmin": round(dmin, 3),
            "defect_depth": 0.0,
            "slice_idx": res.sagittal_slice_idx,
        }
        
    # Check for concave root defect (saucer-shaped indentation)
    mean_y_m2 = np.mean(pts_m2[:, 1])
    mean_y_m3 = np.mean(pts_m3[:, 1])
    m3_to_right = (mean_y_m3 > mean_y_m2)
    
    row_cp = int(round(cp[0]))
    row_window = 12
    r_start = max(0, row_cp - row_window)
    r_end = min(224, row_cp + row_window + 1)
    
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
        
        # Fit a line across endpoints (outer 3 pts on each side) to represent normal root contour
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
        
        # Resorption requires significant indentation into root (> 0.70 mm) AND intimate contact (dmin <= 0.35)
        if is_root and defect_depth_mm >= 0.70 and dmin <= 0.35:
            is_resorption = True
            
    if is_resorption:
        conf = min(0.92, 0.80 + float(defect_depth_mm - 0.70) * 0.12)
        findings = f"Tampak defek konkavitas radiolusen (External Root Resorption) sedalam ~{defect_depth_mm:.2f} mm pada dinding akar distal M2 akibat penekanan impaksi mahkota M3."
        c_name = "Resorption"
        c_id = 2
    else:
        conf = 0.85
        findings = f"Mahkota M3 berkontak langsung pada servikal/akar M2 (jarak {dmin:.2f} mm). Kontur sementum dan dentin akar distal M2 tampak utuh, halus, dan cembung normal tanpa bukti defek kawah resorpsi."
        c_name = "Contact"
        c_id = 0
        
    return {
        "class_name": c_name,
        "class_id": c_id,
        "confidence": round(conf, 3),
        "findings": findings,
        "dmin": round(dmin, 3),
        "defect_depth": round(defect_depth_mm, 3),
        "slice_idx": res.sagittal_slice_idx,
    }


if __name__ == "__main__":
    for cid in ["ToothFairy2F_004", "ToothFairy2F_005", "ToothFairy2F_008", "ToothFairy2F_009"]:
        raw_p = Path(f"data/raw/{cid}_0000.mha")
        mask_p = Path(f"data/toothfairy2/Dataset112_ToothFairy2/labelsTr/{cid}.mha")
        raw_v = load_volume(raw_p)
        mask_v = load_volume(mask_p)
        for m2, m3 in [(17, 18), (27, 28), (37, 38), (47, 48)]:
            out = evaluate_pair(raw_v, mask_v, m2, m3)
            if out:
                print(f"{cid} ({m2}-{m3}): {out['class_name']} (Conf: {out['confidence']:.1%}, Dmin: {out['dmin']} mm, Defect: {out['defect_depth']} mm)")
                print(f"   Findings: {out['findings']}\n")
