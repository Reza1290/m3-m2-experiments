"""
Render REAL CBCT slice (radiological grayscale) + segmentation mask contour overlay
directly from ToothFairy2_Dataset.zip without extracting all 160 GB!
"""
import zipfile
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree

ZIP_PATH   = Path(r"d:\Workspace\TA3\data\toothfairy2\ToothFairy2_Dataset.zip")
LABELS_DIR = Path(r"d:\Workspace\TA3\data\toothfairy2\Dataset112_ToothFairy2\labelsTr")
CSV_PATH   = Path(r"d:\Workspace\TA3\data\processed\dmin_results.csv")
OUT_DIR    = Path(r"d:\Workspace\TA3\data\processed\slideshow")
THRESHOLD  = 0.5

# Test case 1 dulu: ToothFairy2F_008 (pair 17-18)
case_id = "ToothFairy2F_008"
pair = (17, 18)

print(f"Reading mask {case_id}...")
with open(LABELS_DIR / f"{case_id}.mha", "rb") as f:
    meta = {}
    while True:
        line = f.readline()
        if not line: break
        l_str = line.decode("ascii", errors="ignore").strip()
        if "=" in l_str:
            k, v = l_str.split("=", 1)
            meta[k.strip()] = v.strip()
        if b"ElementDataFile" in line: break
    dim = [int(x) for x in meta["DimSize"].split()]
    mask_arr = np.fromfile(f, dtype=np.uint8).reshape((dim[2], dim[1], dim[0]))

spacing = np.array([float(x) for x in meta["ElementSpacing"].split()])
spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])

m2_mask = (mask_arr == pair[0])
m3_mask = (mask_arr == pair[1])

# Surface voxels
def get_surface(m):
    return m ^ ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 3))

s2 = get_surface(m2_mask)
s3 = get_surface(m3_mask)
sv2 = np.argwhere(s2)
sv3 = np.argwhere(s3)

tree = cKDTree(sv2 * spacing_zyx)
dists, idx2 = tree.query(sv3 * spacing_zyx)
best = int(np.argmin(dists))
dmin_mm = float(dists[best])
c2 = sv2[idx2[best]]
c3 = sv3[best]

x_mid = int((c2[2] + c3[2]) // 2)

print(f"Reading real CBCT from zip for {case_id} at slice X={x_mid}...")
zf = zipfile.ZipFile(ZIP_PATH)
with zf.open(f"Dataset112_ToothFairy2/imagesTr/{case_id}_0000.mha") as f:
    while True:
        line = f.readline()
        if b"ElementDataFile" in line: break
    raw_data = f.read()
    cbct_vol = np.frombuffer(raw_data, dtype=np.float64).reshape((dim[2], dim[1], dim[0]))

cbct_slice = cbct_vol[:, :, x_mid]  # (Z, Y)
mask_slice = mask_arr[:, :, x_mid]

# Windowing HU (Bone window: W=2000, L=500 -> [-500, 1500])
W, L = 2200, 600
vmin = L - W / 2
vmax = L + W / 2
cbct_norm = np.clip((cbct_slice - vmin) / (vmax - vmin), 0.0, 1.0)

# Build visualization: 3 panels
# Panel 1: Pure CBCT slice
# Panel 2: CBCT + Mask Overlay (Semi-transparent)
# Panel 3: Zoomed 224x224 ROI around contact
fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.1]})

# 1. Raw CBCT
axes[0].imshow(cbct_norm, cmap="gray", origin="upper")
axes[0].set_title(f"1. Raw CBCT (Sagittal X={x_mid})", fontsize=10, color="white", fontweight="bold")
axes[0].axis("off")

# 2. CBCT + Semi-transparent Mask Overlay
axes[1].imshow(cbct_norm, cmap="gray", origin="upper")
# Overlay M2 & M3
overlay = np.zeros((*cbct_norm.shape, 4), dtype=np.float32)
overlay[mask_slice == pair[0]] = [0.15, 0.55, 1.0, 0.45] # Blue M2
overlay[mask_slice == pair[1]] = [1.0, 0.45, 0.1, 0.45]  # Orange M3
# other teeth
other = (mask_slice > 0) & (mask_slice != pair[0]) & (mask_slice != pair[1])
overlay[other] = [0.7, 0.7, 0.7, 0.25]
axes[1].imshow(overlay, origin="upper")

# Titik kontak
axes[1].plot([c2[1], c3[1]], [c2[0], c3[0]], "r--", linewidth=1.5)
axes[1].plot(c2[1], c2[0], "o", color="#38bdf8", markersize=6, markeredgecolor="white")
axes[1].plot(c3[1], c3[0], "o", color="#fb923c", markersize=6, markeredgecolor="white")
axes[1].set_title(f"2. CBCT + Mask Annotation", fontsize=10, color="white", fontweight="bold")
axes[1].axis("off")

# 3. Zoomed 224x224 ROI around contact
center_z = int((c2[0] + c3[0]) // 2)
center_y = int((c2[1] + c3[1]) // 2)
half_box = 112

z_min = max(0, center_z - half_box)
z_max = min(cbct_vol.shape[0], center_z + half_box)
y_min = max(0, center_y - half_box)
y_max = min(cbct_vol.shape[1], center_y + half_box)

crop_cbct = cbct_norm[z_min:z_max, y_min:y_max]
crop_overlay = overlay[z_min:z_max, y_min:y_max]

axes[2].imshow(crop_cbct, cmap="gray", origin="upper")
axes[2].imshow(crop_overlay, origin="upper")
# Adjusted points
axes[2].plot([c2[1] - y_min, c3[1] - y_min], [c2[0] - z_min, c3[0] - z_min], "y--", linewidth=2)
axes[2].plot(c2[1] - y_min, c2[0] - z_min, "o", color="#38bdf8", markersize=8, markeredgecolor="white")
axes[2].plot(c3[1] - y_min, c3[0] - z_min, "o", color="#fb923c", markersize=8, markeredgecolor="white")
axes[2].set_title(f"3. Zoomed ROI Patch ({crop_cbct.shape[1]}x{crop_cbct.shape[0]})\nD_min = {dmin_mm:.4f} mm",
                  fontsize=10, color="#38bdf8", fontweight="bold")
axes[2].axis("off")

fig.patch.set_facecolor("#0a0c14")
plt.tight_layout()
test_out = OUT_DIR / "sample_real_cbct_008.png"
plt.savefig(test_out, dpi=140, facecolor="#0a0c14")
print(f"Saved real CBCT preview: {test_out}")
