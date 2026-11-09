"""
scripts/visualize_cej_cases.py
==============================
Publication-grade visualizer for 3D CEJ geometric relationships on CBCT scans.

Generates dual-view (Sagittal + Coronal) multi-layer diagnostic figures for:
1. Above_CEJ (Coronal contact): ToothFairy2P_540 (27-28) & ToothFairy2P_395 (17-18)
2. At_CEJ (Cervical margin contact): ToothFairy2P_091 (17-18)
3. Below_CEJ (Root / radicular contact): ToothFairy2F_054 (37-38)
4. Independent (Intact periodontal ligament space): ToothFairy2P_025 (37-38)

Displays:
- CBCT grayscale anatomy (bone windowed)
- Tooth segmentation contours/masks (M2 lime green, M3 cyan)
- Detected CEJ reference level (gold dashed line)
- Minimum surface distance vector (c*_M2 to c*_M3)
- Dental arch & tooth orientation metrics (maxillary vs mandibular)
- Comprehensive clinical annotation card
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk

from src.geometry.contact_geometry import analyze_contact_geometry
from src.geometry.cej_detector import detect_m2_cej
from src.geometry.cej_classifier import GeometricCEJClassifier


def get_quadrant_info(m2_fdi: int) -> dict:
    info = {
        17: {"name": "Maxillary Right (Upper Right)", "arch": "Maxillary", "side": "Right", "crown_dir": "Inferior (+Z)", "root_dir": "Superior (-Z)"},
        27: {"name": "Maxillary Left (Upper Left)", "arch": "Maxillary", "side": "Left", "crown_dir": "Inferior (+Z)", "root_dir": "Superior (-Z)"},
        37: {"name": "Mandibular Left (Lower Left)", "arch": "Mandibular", "side": "Left", "crown_dir": "Superior (-Z)", "root_dir": "Inferior (+Z)"},
        47: {"name": "Mandibular Right (Lower Right)", "arch": "Mandibular", "side": "Right", "crown_dir": "Superior (-Z)", "root_dir": "Inferior (+Z)"},
    }
    return info.get(m2_fdi, {"name": "Unknown", "arch": "Unknown", "side": "Unknown", "crown_dir": "Unknown", "root_dir": "Unknown"})


def render_cej_case(
    case_id: str,
    m2_fdi: int,
    m3_fdi: int,
    ground_truth_label: str,
    stage1_pred: str,
    stage1_prob: float,
    opt_b_pred: str,
    out_dir: Path,
):
    print(f"Rendering visualization for {case_id} ({m2_fdi}-{m3_fdi})...")
    raw_path = Path("data/raw") / f"{case_id}_0000.mha"
    lbl_path = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr") / f"{case_id}.mha"

    if not raw_path.exists() or not lbl_path.exists():
        print(f"Error: Missing files for {case_id}")
        return

    # Read images
    itk_raw = sitk.ReadImage(str(raw_path))
    itk_lbl = sitk.ReadImage(str(lbl_path))

    arr_raw = sitk.GetArrayFromImage(itk_raw)  # (Z, Y, X)
    arr_lbl = sitk.GetArrayFromImage(itk_lbl)

    sp = itk_raw.GetSpacing()
    sp_zyx = (sp[2], sp[1], sp[0])

    mask_m2 = (arr_lbl == m2_fdi)
    mask_m3 = (arr_lbl == m3_fdi)

    # Compute contact geometry & CEJ
    cej_res = detect_m2_cej(mask_m2, m2_fdi, sp_zyx)
    geom_res = analyze_contact_geometry(mask_m2, mask_m3, m2_fdi, m3_fdi, sp_zyx)

    # Classify with calibrated classifier
    classifier = GeometricCEJClassifier(tau_cej_mm=1.0)
    opt_c_res = classifier.classify_contact_location(geom_res)

    quad_info = get_quadrant_info(m2_fdi)

    # Determine focal center for visualization
    if geom_res.is_contact and not np.isnan(geom_res.dmin_mm):
        cp_m2 = geom_res.contact_point_m2_zyx  # (z, y, x)
        cp_m3 = geom_res.contact_point_m3_zyx
        focal_z = int((cp_m2[0] + cp_m3[0]) / 2)
        focal_y = int((cp_m2[1] + cp_m3[1]) / 2)
        focal_x = int((cp_m2[2] + cp_m3[2]) / 2)
    else:
        # Independent: use midpoint between tooth centers of mass
        pts_m2 = np.argwhere(mask_m2)
        pts_m3 = np.argwhere(mask_m3)
        c2 = pts_m2.mean(axis=0).astype(int)
        c3 = pts_m3.mean(axis=0).astype(int)
        cp_m2 = tuple(geom_res.contact_point_m2_zyx)
        cp_m3 = tuple(geom_res.contact_point_m3_zyx)
        focal_z = int((c2[0] + c3[0]) / 2)
        focal_y = int((c2[1] + c3[1]) / 2)
        focal_x = int((c2[2] + c3[2]) / 2)

    # Window CBCT intensities: Bone window [-500, 2000] HU -> [0, 1]
    window_min, window_max = -400.0, 1800.0
    arr_windowed = np.clip((arr_raw - window_min) / (window_max - window_min), 0.0, 1.0)

    # Crop bounding box around teeth for zoomed diagnostic display (approx 60mm x 60mm)
    crop_radius_vox = 45  # 45 * 0.3mm = 13.5mm radius = 27mm window, or 60 vox = 36mm
    r_z = 35
    r_y = 55
    r_x = 55

    z_start = max(0, focal_z - r_z)
    z_end = min(arr_raw.shape[0], focal_z + r_z)
    y_start = max(0, focal_y - r_y)
    y_end = min(arr_raw.shape[1], focal_y + r_y)
    x_start = max(0, focal_x - r_x)
    x_end = min(arr_raw.shape[2], focal_x + r_x)

    # Slices to extract:
    # 1. Sagittal plane: fixed X = focal_x (Slice shows Z vertical vs Y anteroposterior)
    sag_cbct = arr_windowed[z_start:z_end, y_start:y_end, focal_x]
    sag_m2 = mask_m2[z_start:z_end, y_start:y_end, focal_x]
    sag_m3 = mask_m3[z_start:z_end, y_start:y_end, focal_x]

    # 2. Coronal plane: fixed Y = focal_y (Slice shows Z vertical vs X buccolingual)
    cor_cbct = arr_windowed[z_start:z_end, focal_y, x_start:x_end]
    cor_m2 = mask_m2[z_start:z_end, focal_y, x_start:x_end]
    cor_m3 = mask_m3[z_start:z_end, focal_y, x_start:x_end]

    # Create figure
    fig = plt.figure(figsize=(16, 9), facecolor="#0f172a")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 8], width_ratios=[1, 1], wspace=0.15, hspace=0.10)

    # Header Card
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    status_color = "#38bdf8" if "Above" in ground_truth_label else ("#facc15" if "At" in ground_truth_label else ("#f87171" if "Below" in ground_truth_label else "#4ade80"))
    header_text = (
        f"Case: {case_id}  |  Tooth {m2_fdi} (M2) & {m3_fdi} (M3)  |  Arch: {quad_info['name']}\n"
        f"Ground Truth: {ground_truth_label.upper()}   •   "
        f"Stage 1 (ViT Ensemble): {stage1_pred} (P={stage1_prob:.2f})   •   "
        f"Option B (DL): {opt_b_pred}   •   "
        f"Option C (3D Geometry): {opt_c_res.class_name}"
    )
    ax_header.text(
        0.5, 0.5, header_text,
        color="white", fontsize=13, fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor=status_color, linewidth=2)
    )

    # Panel 1: Sagittal View
    ax_sag = fig.add_subplot(gs[1, 0])
    ax_sag.imshow(sag_cbct, cmap="gray", origin="upper", aspect=1.0)

    # Mask contours
    if np.any(sag_m2):
        ax_sag.contour(sag_m2, levels=[0.5], colors=["#22c55e"], linewidths=2.0)
    if np.any(sag_m3):
        ax_sag.contour(sag_m3, levels=[0.5], colors=["#06b6d4"], linewidths=2.0)

    # CEJ Reference Line on Sagittal
    if not np.isnan(cej_res.cej_z_index) and (z_start <= cej_res.cej_z_index < z_end):
        cej_rel_z = cej_res.cej_z_index - z_start
        ax_sag.axhline(y=cej_rel_z, color="#eab308", linestyle="--", linewidth=1.8, label=f"CEJ Landmark (z={cej_res.cej_z_index:.1f})")

    # Contact points & line on Sagittal
    if (z_start <= cp_m2[0] < z_end) and (y_start <= cp_m2[1] < y_end):
        p2_z = cp_m2[0] - z_start
        p2_y = cp_m2[1] - y_start
        p3_z = cp_m3[0] - z_start
        p3_y = cp_m3[1] - y_start
        ax_sag.scatter(p2_y, p2_z, color="#ef4444", s=90, marker="*", zorder=5, label="c* M2")
        ax_sag.scatter(p3_y, p3_z, color="#a855f7", s=70, marker="o", zorder=5, label="c* M3")
        ax_sag.plot([p2_y, p3_y], [p2_z, p3_z], color="#f43f5e", linewidth=2.0, linestyle=":", zorder=4)

    ax_sag.set_title(f"Sagittal Plane (X={focal_x})\nAnteroposterior (Y) vs Vertical (Z)", color="white", fontsize=11, pad=8)
    ax_sag.set_xlabel("Y voxel index (Anterior <---> Posterior)", color="#94a3b8", fontsize=9)
    ax_sag.set_ylabel("Z voxel index (Superior <---> Inferior)", color="#94a3b8", fontsize=9)
    ax_sag.tick_params(colors="#94a3b8")

    # Legend & Sagittal metrics box
    ax_sag.legend(loc="lower left", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8)

    # Panel 2: Coronal View
    ax_cor = fig.add_subplot(gs[1, 1])
    ax_cor.imshow(cor_cbct, cmap="gray", origin="upper", aspect=1.0)

    if np.any(cor_m2):
        ax_cor.contour(cor_m2, levels=[0.5], colors=["#22c55e"], linewidths=2.0)
    if np.any(cor_m3):
        ax_cor.contour(cor_m3, levels=[0.5], colors=["#06b6d4"], linewidths=2.0)

    if not np.isnan(cej_res.cej_z_index) and (z_start <= cej_res.cej_z_index < z_end):
        cej_rel_z = cej_res.cej_z_index - z_start
        ax_cor.axhline(y=cej_rel_z, color="#eab308", linestyle="--", linewidth=1.8, label="CEJ Landmark")

    if (z_start <= cp_m2[0] < z_end) and (x_start <= cp_m2[2] < x_end):
        p2_z = cp_m2[0] - z_start
        p2_x = cp_m2[2] - x_start
        p3_z = cp_m3[0] - z_start
        p3_x = cp_m3[2] - x_start
        ax_cor.scatter(p2_x, p2_z, color="#ef4444", s=90, marker="*", zorder=5)
        ax_cor.scatter(p3_x, p3_z, color="#a855f7", s=70, marker="o", zorder=5)
        ax_cor.plot([p2_x, p3_x], [p2_z, p3_z], color="#f43f5e", linewidth=2.0, linestyle=":", zorder=4)

    ax_cor.set_title(f"Coronal Plane (Y={focal_y})\nBuccolingual (X) vs Vertical (Z)", color="white", fontsize=11, pad=8)
    ax_cor.set_xlabel("X voxel index (Medial <---> Lateral)", color="#94a3b8", fontsize=9)
    ax_cor.set_ylabel("Z voxel index (Superior <---> Inferior)", color="#94a3b8", fontsize=9)
    ax_cor.tick_params(colors="#94a3b8")

    # Annotations card on Coronal panel
    dz_str = f"{geom_res.delta_z_cej_mm:+.2f} mm" if not np.isnan(geom_res.delta_z_cej_mm) else "N/A"
    dmin_str = f"{geom_res.dmin_mm:.2f} mm" if not np.isnan(geom_res.dmin_mm) else "N/A"
    metric_text = (
        f"• 3D Physical D_min: {dmin_str}\n"
        f"• Elevation Delta_z: {dz_str}\n"
        f"• CEJ Status: {cej_res.status}\n"
        f"• Crown Axis: {quad_info['crown_dir']}\n"
        f"• M2 Mask: Lime Green\n"
        f"• M3 Mask: Cyan"
    )
    ax_cor.text(
        0.96, 0.04, metric_text,
        transform=ax_cor.transAxes,
        color="#e2e8f0", fontsize=8.5,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#475569", alpha=0.92)
    )

    out_file = out_dir / f"cej_case_{case_id}_{m2_fdi}_{m3_fdi}_{ground_truth_label}.png"
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved figure to: {out_file}")


def main():
    out_dir = Path("reports/figures/cej_case_visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_to_render = [
        # 1. Above CEJ (Coronal crown contact)
        ("ToothFairy2P_540", 27, 28, "Above_CEJ", "Contact", 0.779, "Below_CEJ"),
        ("ToothFairy2P_395", 17, 18, "Above_CEJ", "Contact", 0.537, "Below_CEJ"),
        # 2. At CEJ (Cervical margin contact)
        ("ToothFairy2P_091", 17, 18, "At_CEJ", "Contact", 0.616, "Below_CEJ"),
        # 3. Below CEJ (Root dentin contact)
        ("ToothFairy2F_054", 37, 38, "Below_CEJ", "Contact", 0.815, "Below_CEJ"),
        # 4. Independent (Intact periodontal ligament gap)
        ("ToothFairy2P_025", 37, 38, "Independent", "Contact", 0.718, "Below_CEJ"),
    ]

    for cid, m2, m3, gt, s1_pred, s1_prob, opt_b in cases_to_render:
        render_cej_case(
            case_id=cid,
            m2_fdi=m2,
            m3_fdi=m3,
            ground_truth_label=gt,
            stage1_pred=s1_pred,
            stage1_prob=s1_prob,
            opt_b_pred=opt_b,
            out_dir=out_dir
        )


if __name__ == "__main__":
    main()
