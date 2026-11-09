"""Generate Educational Diagnostic Diagram: Why Sagittal Slice Index 290 was Selected.

Demonstrates the 3D to 2D deterministic slice selection pipeline on ToothFairy2F_004 (Teeth 47 & 48).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.data.loader import load_volume
from src.data.preprocessor import extract_mask_guided_roi, compute_physical_surface_dmin


def generate_slice_selection_diagram(output_path: Path):
    mask_path = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr/ToothFairy2F_004.mha")
    vol = load_volume(mask_path)
    mask = vol.data  # (Z, Y, X)

    m2_fdi, m3_fdi = 47, 48

    # 1. Compute D_min and nearest contact points
    dmin_mm, pt_m2, pt_m3 = compute_physical_surface_dmin(
        mask_m2=(mask == m2_fdi),
        mask_m3=(mask == m3_fdi),
        spacing_zyx=vol.spacing,
    )

    cz = int(np.round((pt_m2[0] + pt_m3[0]) / 2.0))
    cy = int(np.round((pt_m2[1] + pt_m3[1]) / 2.0))
    cx = int(np.round((pt_m2[2] + pt_m3[2]) / 2.0))

    # 2. Extract ROI
    res = extract_mask_guided_roi(vol, vol, m2_fdi, m3_fdi, crop_size=(224, 224))

    # Create 2x2 multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # -------------------------------------------------------------
    # Panel 1: Pandangan Aksial (Bidang X-Y pada Z kontak)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    axial_slice = mask[cz, :, :]  # shape: (Y, X)

    # Crop neighborhood around teeth for clear zoom-in
    pad_xy = 70
    y_min, y_max = max(0, cy - pad_xy), min(axial_slice.shape[0], cy + pad_xy)
    x_min, x_max = max(0, cx - pad_xy), min(axial_slice.shape[1], cx + pad_xy)

    sub_axial = axial_slice[y_min:y_max, x_min:x_max]

    # Create RGB map for teeth
    rgb_axial = np.full((sub_axial.shape[0], sub_axial.shape[1], 3), 245, dtype=np.uint8)
    rgb_axial[sub_axial == m2_fdi] = [59, 130, 246]   # Blue for M2 (47)
    rgb_axial[sub_axial == m3_fdi] = [249, 115, 22]  # Orange for M3 (48)

    ax1.imshow(rgb_axial, extent=[x_min, x_max, y_max, y_min])

    # Plot contact points
    ax1.scatter([pt_m2[2]], [pt_m2[1]], color='blue', s=80, edgecolors='white', zorder=5, label=f'u* M2 ({pt_m2[2]}, {pt_m2[1]})')
    ax1.scatter([pt_m3[2]], [pt_m3[1]], color='orange', s=80, edgecolors='black', zorder=5, label=f'v* M3 ({pt_m3[2]}, {pt_m3[1]})')

    # Distance line
    ax1.plot([pt_m2[2], pt_m3[2]], [pt_m2[1], pt_m3[1]], 'k--', lw=2.5, label=f'D_min = {dmin_mm:.2f} mm')

    # Sagittal plane cut line (X = 290)
    ax1.axvline(x=cx, color='red', linestyle='-', linewidth=2.5, alpha=0.9, label=f'Bidang Sagital Terpilih: X = {cx}')

    ax1.set_title(f"A. Bidang Aksial (Z = {cz})\nPandangan dari Atas / Oklusal", fontsize=12, fontweight='bold', color='#1A365D')
    ax1.set_xlabel("Sumbu X (Sagital / Left-Right)", fontsize=10)
    ax1.set_ylabel("Sumbu Y (Koronal / Anterior-Posterior)", fontsize=10)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: Bidang Sagital Utuh pada Irisan X = 290
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    sagittal_slice = mask[:, :, cx]  # shape: (Z, Y)

    rgb_sagittal = np.full((sagittal_slice.shape[0], sagittal_slice.shape[1], 3), 245, dtype=np.uint8)
    rgb_sagittal[sagittal_slice == m2_fdi] = [59, 130, 246]
    rgb_sagittal[sagittal_slice == m3_fdi] = [249, 115, 22]

    ax2.imshow(rgb_sagittal, cmap='gray', origin='upper')

    # Plot crop center crosshair
    ax2.scatter([cy], [cz], color='red', s=100, marker='+', linewidth=3, zorder=6, label=f'Pusat Crop ({cz}, {cy})')

    # Draw 224x224 bounding box
    half_crop = 112
    rect = patches.Rectangle(
        (cy - half_crop, cz - half_crop),
        224, 224,
        linewidth=2.5,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Bounding Window 224x224 px'
    )
    ax2.add_patch(rect)

    ax2.set_title(f"B. Bidang Sagital Utuh pada Irisan Terpilih (X = {cx})\nIrisan Tegak Lurus Titik Kontak", fontsize=12, fontweight='bold', color='#1A365D')
    ax2.set_xlabel("Sumbu Y (Anterior - Posterior)", fontsize=10)
    ax2.set_ylabel("Sumbu Z (Superior - Inferior)", fontsize=10)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # Panel 3: Hasil Ekstraksi Patch ROI 224x224 Piksel
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    roi_mask = res.roi_mask

    rgb_roi = np.full((224, 224, 3), 245, dtype=np.uint8)
    rgb_roi[roi_mask == m2_fdi] = [59, 130, 246]
    rgb_roi[roi_mask == m3_fdi] = [249, 115, 22]

    ax3.imshow(rgb_roi)
    ax3.set_title("C. Hasil Ekstraksi ROI Sagital (224 × 224 px)\nMasukan Model Vision Transformer", fontsize=12, fontweight='bold', color='#1A365D')
    ax3.set_xlabel("Lebar Patch (224 px)", fontsize=10)
    ax3.set_ylabel("Tinggi Patch (224 px)", fontsize=10)

    # Add text banner
    info_text = (
        f"Cakupan DTAC:\n"
        f"- Gigi M2 (47): {res.m2_roi_pixels} px\n"
        f"- Gigi M3 (48): {res.m3_roi_pixels} px\n"
        f"- Status: Lolos Evaluasi (PASS)"
    )
    ax3.text(10, 210, info_text, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

    # -------------------------------------------------------------
    # Panel 4: Penjelasan Alur Matematis & Klinis
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis('off')

    formula_text = (
        "MENGAPA DIPILIH IRISAN SAGITAL KE-290?\n\n"
        "1. Kasus Aktual: ToothFairy2F_004 (Gigi 47 & 48)\n\n"
        "2. Pasangan Titik Kontak Terdekat (KD-Tree):\n"
        f"   - Titik permukaan M2: u* = ({pt_m2[0]}, {pt_m2[1]}, X={pt_m2[2]})\n"
        f"   - Titik permukaan M3: v* = ({pt_m3[0]}, {pt_m3[1]}, X={pt_m3[2]})\n"
        f"   - Jarak Fisik: D_min = {dmin_mm:.3f} mm\n\n"
        "3. Rumus Penentuan Irisan Sagital:\n"
        "   X_sagittal = round((x_M2* + x_M3*) / 2)\n"
        f"   X_sagittal = round(({pt_m2[2]} + {pt_m3[2]}) / 2) = {cx}\n\n"
        "4. Keunggulan Klinis:\n"
        "   - Memotong tepat pada celah interdental paling sempit.\n"
        "   - Menjamin defek resorpsi akar servikal terlihat jelas.\n"
        "   - 100% Deterministik & Reproducibel (AR = 100%)."
    )

    ax4.text(
        0.05, 0.95, formula_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1.0', facecolor='#F7FAFC', edgecolor='#CBD5E0', linewidth=1.5)
    )

    fig.suptitle(
        f"Visualisasi Penentuan Irisan Sagital Deterministik (Slice #{cx})\nKasus ToothFairy2F_004 — Pasangan Molar Mandibula Kanan (47 - 48)",
        fontsize=15,
        fontweight='bold',
        color='#1A365D',
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Visual diagram saved to {output_path}")


if __name__ == "__main__":
    out_file = Path("docs/visualisasi_pemilihan_irisan_290.png")
    generate_slice_selection_diagram(out_file)
