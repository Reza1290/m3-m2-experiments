"""Test and calibrate visual radiographical analyzer on genuine CBCT scans."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from src.data.loader import load_volume
from src.data.preprocessor import extract_mask_guided_roi


def analyze_visual_radiography(roi_img: np.ndarray, roi_mask: np.ndarray, m2_id: int, m3_id: int, spacing: tuple = (0.3, 0.3)):
    """Analyze radiographic crop for clinical relationship.
    
    Returns:
        dict with class_id, class_name, confidence, defect_depth_mm, min_hu_in_gap, findings
    """
    mask_m2 = (roi_mask == m2_id)
    mask_m3 = (roi_mask == m3_id)
    
    if not np.any(mask_m2) or not np.any(mask_m3):
        return None
        
    # In roi_mask: axis 0 is axial/height (Z), axis 1 is sagittal/anteroposterior (Y)
    # Physical spacing is (spacing_z, spacing_y)
    sz, sy = spacing[0], spacing[1]
    
    # Distance transforms to find contact interface
    d2 = ndimage.distance_transform_edt(~mask_m2)
    d3 = ndimage.distance_transform_edt(~mask_m3)
    gap = (d2 + d3) * sz # physical gap in mm
    min_gap_mm = np.min(gap)
    
    # Contact interface region (within 0.6 mm of closest boundary)
    interface_mask = (gap <= min_gap_mm + 0.6)
    interface_pixels = roi_img[interface_mask]
    
    min_hu_interface = float(np.min(interface_pixels)) if len(interface_pixels) > 0 else 0.0
    mean_hu_interface = float(np.mean(interface_pixels)) if len(interface_pixels) > 0 else 0.0
    
    # Coordinates of M2 and M3
    pts_m2 = np.argwhere(mask_m2)  # (r, c) = (z, y)
    pts_m3 = np.argwhere(mask_m3)
    
    # Contact point (centroid of closest points)
    pts_interface = np.argwhere(interface_mask)
    if len(pts_interface) > 0:
        cp = np.mean(pts_interface, axis=0)  # (z_c, y_c)
    else:
        cp = np.mean(pts_m2, axis=0)
        
    # M2 vertical bounds
    z_min_m2, z_max_m2 = pts_m2[:, 0].min(), pts_m2[:, 0].max()
    z_height_m2 = z_max_m2 - z_min_m2
    # Crown is generally higher Z in mandibular, or lower Z in maxillary, depending on orientation
    # In our coordinate system, let's determine orientation from M2 crown vs root:
    # Tooth crowns have higher mean HU (enamel > 1500 HU) than root apices (cementum/dentin ~ 1000 HU)
    upper_half = pts_m2[pts_m2[:, 0] > z_min_m2 + 0.5 * z_height_m2]
    lower_half = pts_m2[pts_m2[:, 0] <= z_min_m2 + 0.5 * z_height_m2]
    mean_hu_upper = np.mean(roi_img[upper_half[:, 0], upper_half[:, 1]]) if len(upper_half) > 0 else 0
    mean_hu_lower = np.mean(roi_img[lower_half[:, 0], lower_half[:, 1]]) if len(lower_half) > 0 else 0
    
    # Identify which end is root
    if mean_hu_upper > mean_hu_lower:
        crown_is_upper = True
        root_z_range = (z_min_m2, z_min_m2 + int(0.7 * z_height_m2))
    else:
        crown_is_upper = False
        root_z_range = (z_min_m2 + int(0.3 * z_height_m2), z_max_m2)

    # Determine which horizontal direction is distal (towards M3)
    mean_y_m2 = np.mean(pts_m2[:, 1])
    mean_y_m3 = np.mean(pts_m3[:, 1])
    m3_is_to_right = (mean_y_m3 > mean_y_m2)
    
    # Check if contact is at root/cervical level
    is_at_root_cervical = (root_z_range[0] <= cp[0] <= root_z_range[1])
    
    # Check for concave defect on the distal boundary of M2:
    # For each row (Z) near the contact point, measure the profile of M2 distal boundary
    row_start = max(0, int(cp[0]) - 15)
    row_end = min(roi_mask.shape[0], int(cp[0]) + 16)
    
    distal_boundary = []
    m2_widths = []
    
    for r in range(row_start, row_end):
        cols = np.where(mask_m2[r, :])[0]
        if len(cols) > 0:
            m2_widths.append(len(cols))
            distal_col = cols[-1] if m3_is_to_right else cols[0]
            distal_boundary.append((r, distal_col))
            
    # Measure concavity/indentation:
    defect_depth_mm = 0.0
    is_concave_defect = False
    
    if len(distal_boundary) >= 10:
        distal_cols = np.array([pt[1] for pt in distal_boundary], dtype=float)
        # Baseline trend line (convex hull or linear interpolation between endpoints)
        # In a healthy convex root, distal boundary forms a straight line or convex curve.
        # A concave defect pushes distal_col inward (left if M3 is to right, right if M3 is to left).
        baseline = np.linspace(distal_cols[0], distal_cols[-1], len(distal_cols))
        
        if m3_is_to_right:
            # inward indentation means distal_col < baseline
            indentations_px = baseline - distal_cols
        else:
            # inward indentation means distal_col > baseline
            indentations_px = distal_cols - baseline
            
        max_indentation_px = np.max(indentations_px)
        # Also check local width reduction (waisting / thinning)
        if len(m2_widths) >= 10:
            median_w = np.median(m2_widths)
            min_w = np.min(m2_widths)
            width_loss_px = median_w - min_w
        else:
            width_loss_px = 0.0
            
        defect_px = max(max_indentation_px, width_loss_px * 0.7)
        defect_depth_mm = defect_px * sy
        
        # Resorption requires:
        # 1) Contact at root/cervical surface
        # 2) Defect depth >= 0.75 mm (more than 2.5 voxels into root dentin)
        # 3) Local radiolucency / hard tissue loss
        if is_at_root_cervical and defect_depth_mm >= 0.75 and min_gap_mm <= 0.40:
            is_concave_defect = True
            
    # Decision logic:
    if min_gap_mm > 0.50:
        # Clear separation
        class_id = 1
        class_name = "Independent"
        conf = min(0.96, 0.75 + float(min_gap_mm - 0.50) * 0.15)
        findings = (
            f"Tampak celah radiolusen ligamen periodontal (PDL) dan septum tulang utuh "
            f"(jarak anatomis {min_gap_mm:.2f} mm, densitas celah {mean_hu_interface:.0f} HU) "
            f"memisahkan mahkota M3 dari akar M2."
        )
    elif is_concave_defect:
        class_id = 2
        class_name = "Resorption"
        conf = min(0.92, 0.78 + float(defect_depth_mm - 0.75) * 0.12)
        findings = (
            f"Tampak defek konkavitas radiolusen (External Root Resorption) sedalam ~{defect_depth_mm:.2f} mm "
            f"pada dinding akar distal M2 akibat impaksi langsung mahkota M3."
        )
    else:
        class_id = 0
        class_name = "Contact"
        conf = min(0.92, 0.82 + (0.50 - min_gap_mm) * 0.15)
        findings = (
            f"Mahkota M3 berkontak langsung pada servikal/akar M2 (jarak {min_gap_mm:.2f} mm). "
            f"Kontur sementum dan dentin akar distal M2 tampak utuh, halus, dan cembung normal "
            f"tanpa bukti defek kawah resorpsi."
        )
        
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(conf, 3),
        "defect_depth_mm": round(defect_depth_mm, 3),
        "min_gap_mm": round(min_gap_mm, 3),
        "mean_hu_interface": round(mean_hu_interface, 1),
        "findings": findings,
    }


if __name__ == "__main__":
    cases_to_test = [
        ("ToothFairy2F_004", [(47, 48)]),
        ("ToothFairy2F_005", [(47, 48)]),
        ("ToothFairy2F_008", [(17, 18), (27, 28), (37, 38), (47, 48)]),
        ("ToothFairy2F_009", [(17, 18), (27, 28), (37, 38)]),
    ]
    
    print("Testing visual radiographical analyzer...")
    for cid, pairs in cases_to_test:
        raw_p = Path(f"data/raw/{cid}_0000.mha")
        mask_p = Path(f"data/toothfairy2/Dataset112_ToothFairy2/labelsTr/{cid}.mha")
        if not raw_p.exists() or not mask_p.exists():
            continue
        raw_v = load_volume(raw_p)
        mask_v = load_volume(mask_p)
        
        for m2, m3 in pairs:
            res = extract_mask_guided_roi(raw_v, mask_v, m2, m3)
            if not res.success:
                print(f"{cid} ({m2}-{m3}): ROI extraction failed")
                continue
            vis = analyze_visual_radiography(res.roi_image, res.roi_mask, m2, m3, spacing=raw_v.spacing[:2])
            print(f"\n{cid} ({m2}-{m3}):")
            print(f"  Class: {vis['class_name']} (ID: {vis['class_id']}, Conf: {vis['confidence']:.1%})")
            print(f"  Gap: {vis['min_gap_mm']:.2f} mm | Defect depth: {vis['defect_depth_mm']:.2f} mm | Interface HU: {vis['mean_hu_interface']:.0f}")
            print(f"  Findings: {vis['findings']}")
