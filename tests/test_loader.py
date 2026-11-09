"""Unit tests for src/data/loader.py (MedicalVolume I/O and spatial validation)."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import SimpleITK as sitk

from src.data.loader import MedicalVolume, load_volume, validate_spatial_compatibility


def test_voxel_physical_roundtrip():
    """Verify that voxel_to_physical and physical_to_voxel are exact inverses."""
    data = np.zeros((10, 20, 30), dtype=np.int16)
    spacing = (1.5, 0.5, 0.5)  # (sz, sy, sx)
    origin = (100.0, 50.0, -25.0)  # (oz, oy, ox)
    direction = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    vol = MedicalVolume(
        data=data,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )

    test_voxels = [
        (0, 0, 0),
        (5, 10, 15),
        (9, 19, 29),
    ]

    for vox in test_voxels:
        phys = vol.voxel_to_physical(vox)
        reconstructed_vox = vol.physical_to_voxel(phys)
        np.testing.assert_allclose(reconstructed_vox, vox, atol=1e-5)


def test_spatial_compatibility_check():
    """Verify matching and mismatching spatial metadata detection."""
    shape = (20, 30, 40)
    spacing = (0.3, 0.3, 0.3)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    vol1 = MedicalVolume(np.zeros(shape), spacing, origin, direction)
    vol2 = MedicalVolume(np.ones(shape), spacing, origin, direction)
    is_valid, msg = validate_spatial_compatibility(vol1, vol2)
    assert is_valid is True
    assert "aligned" in msg.lower()

    # Shape mismatch
    vol_wrong_shape = MedicalVolume(np.zeros((15, 30, 40)), spacing, origin, direction)
    is_valid, msg = validate_spatial_compatibility(vol1, vol_wrong_shape)
    assert is_valid is False
    assert "shape mismatch" in msg.lower()

    # Spacing mismatch
    vol_wrong_spacing = MedicalVolume(np.zeros(shape), (0.5, 0.3, 0.3), origin, direction)
    is_valid, msg = validate_spatial_compatibility(vol1, vol_wrong_spacing)
    assert is_valid is False
    assert "spacing mismatch" in msg.lower()


def test_load_volume_mha():
    """Verify saving a synthetic SimpleITK volume and loading it via load_volume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_synth.mha"
        arr = np.random.randint(0, 50, size=(16, 32, 48), dtype=np.uint8)

        # Convert to SimpleITK image and save
        itk_img = sitk.GetImageFromArray(arr)
        itk_img.SetSpacing((0.3, 0.3, 0.5))  # (sx, sy, sz)
        itk_img.SetOrigin((10.0, 20.0, 30.0))  # (ox, oy, oz)
        sitk.WriteImage(itk_img, str(tmp_file))

        loaded = load_volume(tmp_file)
        assert loaded.shape == (16, 32, 48)
        assert np.array_equal(loaded.data, arr)
        assert np.allclose(loaded.spacing, (0.5, 0.3, 0.3))
        assert np.allclose(loaded.origin, (30.0, 20.0, 10.0))
