"""
tests/test_geometry_cej.py
==========================
Unit tests for 3D CEJ Detection, Contact Geometry, and Classification.
Tests:
- Physical coordinate conversion and voxel spacing.
- KD-Tree surface minimum distance.
- Crown vs root orientation (Mandibular vs Maxillary).
- Threshold boundary conditions.
- Independent / No-contact scenarios.
- Empty masks, truncated volumes, and CEJ_UNCERTAIN detection.
- Stage 2 mapping to final 4 classes.
"""

import numpy as np
import pytest

from src.geometry.cej_detector import detect_m2_cej, CEJDetectionResult
from src.geometry.contact_geometry import (
    analyze_contact_geometry,
    extract_surface_voxels,
    ContactGeometryResult,
)
from src.geometry.cej_classifier import (
    GeometricCEJClassifier,
    CEJClassificationResult,
    CLASS_NAMES_4,
)


def create_synthetic_tooth(
    shape=(60, 40, 40),
    z_range=(10, 50),
    radius_crown=12,
    radius_neck=8,
    radius_root=10,
    is_mandibular=True,
):
    """Create a synthetic 3D tooth mask with an anatomical cervical constriction (waist)."""
    mask = np.zeros(shape, dtype=bool)
    z_min, z_max = z_range
    z_span = z_max - z_min

    cy, cx = shape[1] // 2, shape[2] // 2

    # In mandibular: z_min is root apex, z_max is occlusal crown. Neck is at ~60% from z_min
    neck_z = int(z_min + 0.60 * z_span) if is_mandibular else int(z_min + 0.40 * z_span)

    for z in range(z_min, z_max):
        if is_mandibular:
            # Below neck = root, above neck = crown
            if z < neck_z:
                r = radius_neck + (radius_root - radius_neck) * (1 - (z - z_min) / (neck_z - z_min + 1e-5))
            else:
                r = radius_neck + (radius_crown - radius_neck) * ((z - neck_z) / (z_max - neck_z + 1e-5))
        else:
            # Maxillary: z_min is crown, z_max is root
            if z < neck_z:
                r = radius_neck + (radius_crown - radius_neck) * (1 - (z - z_min) / (neck_z - z_min + 1e-5))
            else:
                r = radius_neck + (radius_root - radius_neck) * ((z - neck_z) / (z_max - neck_z + 1e-5))

        for y in range(shape[1]):
            for x in range(shape[2]):
                if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                    mask[z, y, x] = True

    return mask, neck_z


class TestCEJDetector:
    def test_empty_mask(self):
        empty = np.zeros((40, 40, 40), dtype=bool)
        res = detect_m2_cej(empty, fdi_label=47)
        assert res.is_uncertain is True
        assert res.status == "INVALID_MASK"
        assert np.isnan(res.cej_z_index)

    def test_truncated_mask(self):
        small = np.zeros((40, 40, 40), dtype=bool)
        small[10:14, 15:25, 15:25] = True  # z-span = 4
        res = detect_m2_cej(small, fdi_label=47)
        assert res.is_uncertain is True
        assert res.status == "CEJ_UNCERTAIN"

    def test_mandibular_cej_detection(self):
        mask, true_neck_z = create_synthetic_tooth(is_mandibular=True)
        res = detect_m2_cej(mask, fdi_label=47, spacing_zyx=(0.3, 0.3, 0.3))
        assert res.status == "CEJ_DETECTED"
        assert res.is_uncertain is False
        assert abs(res.cej_z_index - true_neck_z) <= 3  # within 3 slices
        assert res.cervical_constriction_ratio < 0.95

    def test_maxillary_cej_detection(self):
        mask, true_neck_z = create_synthetic_tooth(is_mandibular=False)
        res = detect_m2_cej(mask, fdi_label=17, spacing_zyx=(0.3, 0.3, 0.3))
        assert res.status == "CEJ_DETECTED"
        assert res.is_uncertain is False
        assert abs(res.cej_z_index - true_neck_z) <= 3


class TestContactGeometry:
    def test_extract_surface_voxels(self):
        solid = np.zeros((20, 20, 20), dtype=bool)
        solid[5:15, 5:15, 5:15] = True
        surf = extract_surface_voxels(solid)
        assert len(surf) > 0
        # A 10x10x10 cube has 1000 voxels, interior 8x8x8=512, surface = 1000 - 512 = 488 voxels
        assert len(surf) == 488

    def test_independent_no_contact(self):
        mask_m2, _ = create_synthetic_tooth(shape=(60, 60, 60))
        # Place M3 far away along X axis
        mask_m3 = np.zeros((60, 60, 60), dtype=bool)
        mask_m3[20:40, 20:40, 45:55] = True

        res = analyze_contact_geometry(
            mask_m2, mask_m3, m2_fdi=47, m3_fdi=48,
            spacing_zyx=(0.5, 0.5, 0.5),
            dmin_contact_threshold_mm=0.50
        )
        assert res.is_contact is False
        assert res.dmin_mm > 0.50

    def test_contact_above_cej(self):
        mask_m2, neck_z = create_synthetic_tooth(shape=(60, 60, 60), is_mandibular=True)
        # In mandibular, crown is at higher Z than neck_z (e.g. neck_z=34, crown up to 50)
        # Place M3 touching M2 directly at crown (z=45, Above CEJ)
        mask_m3 = np.zeros((60, 60, 60), dtype=bool)
        # M2 center is at y=30, x=30. Crown radius ~12, so M2 surface extends to x~42.
        mask_m3[44:48, 28:32, 42:48] = True

        res = analyze_contact_geometry(
            mask_m2, mask_m3, m2_fdi=47, m3_fdi=48,
            spacing_zyx=(0.3, 0.3, 0.3),
            dmin_contact_threshold_mm=0.50
        )
        assert res.is_contact is True
        assert res.dmin_mm <= 0.50
        assert res.delta_z_cej_mm > 1.0  # Well above CEJ

    def test_contact_below_cej(self):
        mask_m2, neck_z = create_synthetic_tooth(shape=(60, 60, 60), is_mandibular=True)
        # Place M3 touching M2 at root (z=20, below neck_z=34)
        mask_m3 = np.zeros((60, 60, 60), dtype=bool)
        # Root radius ~10, so surface extends to x~40.
        mask_m3[18:22, 28:32, 40:46] = True

        res = analyze_contact_geometry(
            mask_m2, mask_m3, m2_fdi=47, m3_fdi=48,
            spacing_zyx=(0.3, 0.3, 0.3),
            dmin_contact_threshold_mm=0.50
        )
        assert res.is_contact is True
        assert res.dmin_mm <= 0.50
        assert res.delta_z_cej_mm < -1.0  # Well below CEJ


class TestCEJClassifier:
    def test_independent_mapping(self):
        clf = GeometricCEJClassifier(tau_cej_mm=1.0, dmin_contact_threshold_mm=0.50)
        dummy_cej = CEJDetectionResult(
            cej_z_index=30.0, cej_z_phys_mm=9.0, confidence=0.9, is_uncertain=False,
            status="CEJ_DETECTED", diagnostic_reason="OK", coronal_apex_z=50.0,
            apical_apex_z=10.0, cervical_constriction_ratio=0.8
        )
        geom = ContactGeometryResult(
            dmin_mm=1.20, contact_point_m2_zyx=(30, 20, 20),
            contact_point_m3_zyx=(30, 20, 25), delta_z_cej_mm=0.0,
            cej_result=dummy_cej, status="OK", is_contact=False, diagnostic_reason="No contact"
        )
        out = clf.classify_contact_location(geom)
        assert out.class_id == 0
        assert out.class_name == "Independent"

    def test_above_cej_mapping(self):
        clf = GeometricCEJClassifier(tau_cej_mm=1.0)
        dummy_cej = CEJDetectionResult(
            cej_z_index=30.0, cej_z_phys_mm=9.0, confidence=0.9, is_uncertain=False,
            status="CEJ_DETECTED", diagnostic_reason="OK", coronal_apex_z=50.0,
            apical_apex_z=10.0, cervical_constriction_ratio=0.8
        )
        geom = ContactGeometryResult(
            dmin_mm=0.20, contact_point_m2_zyx=(40, 20, 20),
            contact_point_m3_zyx=(40, 20, 21), delta_z_cej_mm=2.5,  # > +1.0
            cej_result=dummy_cej, status="OK", is_contact=True, diagnostic_reason="Contact"
        )
        out = clf.classify_contact_location(geom)
        assert out.class_id == 1
        assert out.class_name == "Above_CEJ"

    def test_at_cej_mapping(self):
        clf = GeometricCEJClassifier(tau_cej_mm=1.0)
        dummy_cej = CEJDetectionResult(
            cej_z_index=30.0, cej_z_phys_mm=9.0, confidence=0.9, is_uncertain=False,
            status="CEJ_DETECTED", diagnostic_reason="OK", coronal_apex_z=50.0,
            apical_apex_z=10.0, cervical_constriction_ratio=0.8
        )
        geom = ContactGeometryResult(
            dmin_mm=0.20, contact_point_m2_zyx=(31, 20, 20),
            contact_point_m3_zyx=(31, 20, 21), delta_z_cej_mm=0.3,  # within [-1.0, +1.0]
            cej_result=dummy_cej, status="OK", is_contact=True, diagnostic_reason="Contact"
        )
        out = clf.classify_contact_location(geom)
        assert out.class_id == 2
        assert out.class_name == "At_CEJ"

    def test_below_cej_mapping(self):
        clf = GeometricCEJClassifier(tau_cej_mm=1.0)
        dummy_cej = CEJDetectionResult(
            cej_z_index=30.0, cej_z_phys_mm=9.0, confidence=0.9, is_uncertain=False,
            status="CEJ_DETECTED", diagnostic_reason="OK", coronal_apex_z=50.0,
            apical_apex_z=10.0, cervical_constriction_ratio=0.8
        )
        geom = ContactGeometryResult(
            dmin_mm=0.20, contact_point_m2_zyx=(20, 20, 20),
            contact_point_m3_zyx=(20, 20, 21), delta_z_cej_mm=-3.0,  # < -1.0
            cej_result=dummy_cej, status="OK", is_contact=True, diagnostic_reason="Contact"
        )
        out = clf.classify_contact_location(geom)
        assert out.class_id == 3
        assert out.class_name == "Below_CEJ"

    def test_uncertain_cej_fallback(self):
        clf = GeometricCEJClassifier(uncertain_fallback="At_CEJ")
        dummy_cej = CEJDetectionResult(
            cej_z_index=np.nan, cej_z_phys_mm=np.nan, confidence=0.2, is_uncertain=True,
            status="CEJ_UNCERTAIN", diagnostic_reason="Ambiguous", coronal_apex_z=50.0,
            apical_apex_z=10.0, cervical_constriction_ratio=0.98
        )
        geom = ContactGeometryResult(
            dmin_mm=0.20, contact_point_m2_zyx=(20, 20, 20),
            contact_point_m3_zyx=(20, 20, 21), delta_z_cej_mm=np.nan,
            cej_result=dummy_cej, status="OK_CEJ_UNCERTAIN", is_contact=True, diagnostic_reason="Contact"
        )
        out = clf.classify_contact_location(geom)
        assert out.status == "CEJ_UNCERTAIN"
        assert out.class_name == "At_CEJ"
        assert out.class_id == 2
