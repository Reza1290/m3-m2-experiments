"""Unit tests for src/data/fdi_filter.py (FDI molar presence checking)."""

from pathlib import Path
import numpy as np
import pytest

from src.data.fdi_filter import (
    PresenceStatus,
    check_fdi_pair_presence,
    scan_volume_molar_pairs,
    SUPPORTED_FDI_PAIRS,
)
from src.data.loader import MedicalVolume, load_volume


def test_empty_mask_presence():
    mask = np.zeros((30, 40, 50), dtype=np.uint8)
    res = check_fdi_pair_presence(mask, "maxillary_right", 17, 18)
    assert res.status == PresenceStatus.EMPTY_MASK
    assert res.is_processable is False


def test_both_present():
    mask = np.zeros((30, 40, 50), dtype=np.uint8)
    mask[5:15, 10:20, 10:20] = 17  # M2
    mask[5:15, 10:20, 25:35] = 18  # M3
    res = check_fdi_pair_presence(mask, "maxillary_right", 17, 18)
    assert res.status == PresenceStatus.BOTH_PRESENT
    assert res.is_processable is True
    assert res.m2_voxel_count == 1000
    assert res.m3_voxel_count == 1000


def test_m3_absent():
    mask = np.zeros((30, 40, 50), dtype=np.uint8)
    mask[5:15, 10:20, 10:20] = 17  # M2 only
    res = check_fdi_pair_presence(mask, "maxillary_right", 17, 18)
    assert res.status == PresenceStatus.ABSENT_M3
    assert res.is_processable is False
    assert res.m2_voxel_count == 1000
    assert res.m3_voxel_count == 0


def test_both_absent():
    mask = np.zeros((30, 40, 50), dtype=np.uint8)
    mask[0:5, 0:5, 0:5] = 11  # Central incisor only
    res = check_fdi_pair_presence(mask, "maxillary_right", 17, 18)
    assert res.status == PresenceStatus.ABSENT_BOTH
    assert res.is_processable is False


def test_real_toothfairy_mask():
    """Test with actual ToothFairy2 label mask ToothFairy2F_008.mha if available."""
    label_path = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr/ToothFairy2F_008.mha")
    if not label_path.exists():
        pytest.skip("ToothFairy2F_008.mha not present in labelsTr")

    vol = load_volume(label_path)
    results = scan_volume_molar_pairs(vol)

    assert "maxillary_right" in results
    assert "maxillary_left" in results
    # As audited in dmin_results.csv, ToothFairy2F_008 has both 17-18 and 27-28 present!
    assert results["maxillary_right"].status == PresenceStatus.BOTH_PRESENT
    assert results["maxillary_left"].status == PresenceStatus.BOTH_PRESENT
    assert results["maxillary_right"].is_processable is True
    assert results["maxillary_left"].is_processable is True
