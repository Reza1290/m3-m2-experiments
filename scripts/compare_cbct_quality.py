"""
Bandingkan windowing dan aspect ratio yang benar:
1. Sebelumnya (gepeng dan terlalu kontras)
2. Dental Windowing yang Benar (Proporsi anatomi 1:1, windowing tajam enamel-dentin-pulpa)
"""
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ZIP_PATH = Path(r"d:\Workspace\TA3\data\toothfairy2\ToothFairy2_Dataset.zip")
OUT_IMG = Path(r"d:\Workspace\TA3\docs\perbandingan_kualitas_cbct.png")

zf = zipfile.ZipFile(ZIP_PATH)
with zf.open("Dataset112_ToothFairy2/imagesTr/ToothFairy2F_008_0000.mha") as f:
    while True:
        line = f.readline()
        if b"ElementDataFile" in line: break
    vol = np.frombuffer(f.read(), dtype=np.float64).reshape((270, 410, 410))

# Ambil irisan sagital X=291
sl = vol[:, :, 291] # (Z=270, Y=410)

# Windowing Lama: W=2200, L=600 -> [-500, 1700]
w_old = np.clip((sl - (-500)) / (1700 - (-500)), 0, 1)

# Dental Windowing Akurat (Dental Standard Window: L=1000, W=3000 -> membedakan pulpa, dentin, enamel, dan tulang alveolar)
vmin, vmax = -200, 2200
w_dental = np.clip((sl - vmin) / (vmax - vmin), 0, 1)

# Dental High-Contrast (khusus batas gigi & tulang)
w_sharp = np.clip((sl - 100) / (2000 - 100), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0a0c16")

# 1. Tampilan Awal
axes[0].imshow(w_old, cmap="gray", origin="upper")
axes[0].set_title("1. Tampilan Sebelumnya\n(Terlalu Terang/Overexposed, Detail Gigi Hilang)", color="#f87171", fontsize=10, fontweight="bold")
axes[0].axis("off")

# 2. Dental Windowing Standar (Benar)
axes[1].imshow(w_dental, cmap="gray", origin="upper")
axes[1].set_title("2. Dental Windowing Standar\n(Terlihat Saluran Pulpa, Email, & Tulang Alveolar)", color="#38bdf8", fontsize=10, fontweight="bold")
axes[1].axis("off")

# 3. Zoom Tajam Area M2-M3
cz, cy = 32, 154 # pusat kontak
z_min, z_max = max(0, cz - 60), min(270, cz + 60)
y_min, y_max = max(0, cy - 80), min(410, cy + 80)

crop = w_dental[z_min:z_max, y_min:y_max]
axes[2].imshow(crop, cmap="gray", origin="upper", interpolation="bicubic")
axes[2].set_title(f"3. Zoom Area M2-M3 (Interpolasi Halus)\n(Batas Akar & Mahkota Sangat Jelas)", color="#34d399", fontsize=10, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=160, facecolor="#0a0c16")
print(f"Perbandingan tersimpan di: {OUT_IMG}")
