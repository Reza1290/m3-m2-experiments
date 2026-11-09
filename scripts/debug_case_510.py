import sys
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from src.data.loader import load_volume
from src.data.preprocessor import crop_window_2d
from scipy import ndimage

cid = 'ToothFairy2P_510'
m2, m3 = 47, 48
s_idx = 280
dmin = 0.3

raw_v = load_volume(Path(f'data/raw/{cid}_0000.mha'))
mask_v = load_volume(Path(f'data/toothfairy2/Dataset112_ToothFairy2/labelsTr/{cid}.mha'))

slice_cbct = raw_v.data[:, :, s_idx]
slice_mask = mask_v.data[:, :, s_idx]

mask_m2 = (slice_mask == m2)
mask_m3 = (slice_mask == m3)
d2 = ndimage.distance_transform_edt(~mask_m2)
d3 = ndimage.distance_transform_edt(~mask_m3)
gap = d2 + d3
cp = np.mean(np.argwhere(gap == gap.min()), axis=0)
center_zy = (int(round(cp[0])), int(round(cp[1])))

roi_img, _ = crop_window_2d(slice_cbct, center_zy, (224, 224), pad_value=0.0)
roi_mask, _ = crop_window_2d(slice_mask, center_zy, (224, 224), pad_value=0)

roi_m2 = (roi_mask == m2)
roi_m3 = (roi_mask == m3)
pts_m2 = np.argwhere(roi_m2)
pts_m3 = np.argwhere(roi_m3)

roi_d2 = ndimage.distance_transform_edt(~roi_m2)
roi_d3 = ndimage.distance_transform_edt(~roi_m3)
roi_gap = roi_d2 + roi_d3
roi_cp = np.mean(np.argwhere(roi_gap == roi_gap.min()), axis=0)

z_min, z_max = pts_m2[:, 0].min(), pts_m2[:, 0].max()
z_mid = (z_min + z_max) / 2.0
z_span = z_max - z_min

upper_pts = pts_m2[pts_m2[:, 0] > z_mid]
lower_pts = pts_m2[pts_m2[:, 0] <= z_mid]
hu_upper = float(np.mean(roi_img[upper_pts[:, 0], upper_pts[:, 1]]))
hu_lower = float(np.mean(roi_img[lower_pts[:, 0], lower_pts[:, 1]]))

crown_is_upper = (hu_upper >= hu_lower)
if crown_is_upper:
    is_root = (roi_cp[0] <= z_mid + 0.15 * z_span)
    is_pure_crown = (roi_cp[0] > z_mid + 0.30 * z_span)
else:
    is_root = (roi_cp[0] >= z_mid - 0.15 * z_span)
    is_pure_crown = (roi_cp[0] < z_mid - 0.30 * z_span)

print(f"z_min: {z_min}, z_max: {z_max}, z_mid: {z_mid:.1f}, roi_cp[0]: {roi_cp[0]:.1f}")
print(f"hu_upper: {hu_upper:.1f}, hu_lower: {hu_lower:.1f}, crown_is_upper: {crown_is_upper}")
print(f"is_root: {is_root}, is_pure_crown: {is_pure_crown}")
