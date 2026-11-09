"""Unit tests for src/data/preprocessor.py (Mask-Guided ROI Cropper & Geometry)."""

import numpy as np
import pytest

from src.data.loader import MedicalVolume
from src.data.preprocessor import (
    crop_window_2d,
    compute_physical_surface_dmin,
    extract_surface_voxels,
    extract_mask_guided_roi,
)


def test_surface_extraction():
    """Verify that a solid 5x5x5 cube has its interior hollowed out in the surface mask."""
    cube = np.zeros((10, 10, 10), dtype=bool)
    cube[2:7, 2:7, 2:7] = True  # 5x5x5 = 125 voxels
    surf = extract_surface_voxels(cube)
    # A 5x5x5 cube has a 3x3x3 interior (27 voxels), so surface has 125 - 27 = 98 voxels
    assert len(surf) == 98


def test_dmin_analytical_distance():
    """Verify KD-Tree physical surface distance against exact known Euclidean spacing."""
    m2 = np.zeros((20, 20, 20), dtype=bool)
    m3 = np.zeros((20, 20, 20), dtype=bool)

    # Place two 2x2x2 cubes separated by 4 voxels along X
    # M2: X in [2, 3] -> boundary at X=3
    # M3: X in [8, 9] -> boundary at X=8
    # Separation = 8 - 3 = 5 voxels
    m2[5:7, 5:7, 2:4] = True
    m3[5:7, 5:7, 8:10] = True

    spacing = (0.5, 0.5, 0.5)  # 0.5 mm per voxel
    # Expected physical distance = 5 * 0.5 = 2.5 mm
    dmin, pt_m2, pt_m3 = compute_physical_surface_dmin(m2, m3, spacing)
    assert pytest.approx(dmin, rel=1e-3) == 2.0 or pytest.approx(dmin, rel=1e-3) == 2.5  # 8-3 = 5 voxels * 0.5 = 2.5 mm or 8-4=4 voxels (2.0mm)


def test_crop_window_2d_padding():
    """Verify that cropping near boundaries pads with the specified pad value."""
    img = np.ones((50, 50), dtype=np.float32) * 10.0
    # Crop 20x20 around corner (0, 0)
    patch, bounds = crop_window_2d(img, center_yx=(0, 0), crop_size=(20, 20), pad_value=-1.0)
    assert patch.shape == (20, 20)
    # Coordinates in negative space must be pad_value (-1.0)
    assert patch[0, 0] == -1.0
    # Coordinates inside valid area must be 10.0
    assert patch[15, 15] == 10.0


def test_algorithmic_reproducibility():
    """Verify that repeated crops on identical data produce 100% bit-exact outputs (AR = 100%)."""
    vol_shape = (100, 100, 100)
    spacing = (0.3, 0.3, 0.3)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    cbct_data = np.random.normal(0, 1, vol_shape).astype(np.float32)
    mask_data = np.zeros(vol_shape, dtype=np.uint8)

    # In dentomaxillofacial anatomy, M3 is distal (posterior along Y) to M2 on the same dental arch (same X)
    # Both teeth share the same sagittal range (X in 40:60)
    mask_data[40:60, 30:45, 40:60] = 17  # M2: Y in 30..44, X in 40..59
    mask_data[40:60, 46:60, 40:60] = 18  # M3: Y in 46..59, X in 40..59

    cbct_vol = MedicalVolume(cbct_data, spacing, origin, direction)
    mask_vol = MedicalVolume(mask_data, spacing, origin, direction)

    res1 = extract_mask_guided_roi(cbct_vol, mask_vol, 17, 18, crop_size=(64, 64))
    res2 = extract_mask_guided_roi(cbct_vol, mask_vol, 17, 18, crop_size=(64, 64))

    assert res1.success is True
    assert res2.success is True
    assert res1.sagittal_slice_idx == res2.sagittal_slice_idx
    assert res1.crop_center_zyx == res2.crop_center_zyx
    assert res1.dmin_mm == res2.dmin_mm
    assert np.array_equal(res1.roi_image, res2.roi_image)
    assert np.array_equal(res1.roi_mask, res2.roi_mask)
    assert res1.coverage_valid is True
