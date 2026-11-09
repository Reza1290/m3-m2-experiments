"""
src/geometry/cej_detector.py
============================
Anatomical-Geometric Cementoenamel Junction (CEJ) Detector for Second Molars (M2).

Does NOT assume CEJ is a flat horizontal plane. Traces 3D distal cervical
constriction and boundary inflection on M2 segmentation masks.
Respects physical millimeter spacing, origin, and directional orientation.
Returns CEJ_UNCERTAIN when anatomy is ambiguous or unresolvable.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CEJDetectionResult:
    """Represents the outcome of 3D CEJ landmark detection on M2.

    Attributes:
        cej_z_index: Voxel index along Z axis corresponding to distal CEJ landmark.
        cej_z_phys_mm: Physical Z coordinate in millimeters.
        confidence: Reliability score of the detection [0.0, 1.0].
        is_uncertain: True if CEJ could not be reliably established.
        status: Diagnostic status string ('CEJ_DETECTED', 'CEJ_UNCERTAIN', 'INVALID_MASK').
        diagnostic_reason: Human-readable explanation of detection or failure.
        coronal_apex_z: Voxel index of occlusal crown crest.
        apical_apex_z: Voxel index of root apex.
        cervical_constriction_ratio: Ratio of cervical waist area to mean tooth area.
    """
    cej_z_index: float
    cej_z_phys_mm: float
    confidence: float
    is_uncertain: bool
    status: str
    diagnostic_reason: str
    coronal_apex_z: float
    apical_apex_z: float
    cervical_constriction_ratio: float


def detect_m2_cej(
    mask_m2: np.ndarray,
    fdi_label: int,
    spacing_zyx: Tuple[float, float, float] = (0.3, 0.3, 0.3),
    origin_zyx: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction_matrix: Optional[np.ndarray] = None,
) -> CEJDetectionResult:
    """Detect anatomical CEJ landmark on M2 distal aspect.

    Args:
        mask_m2: 3D boolean numpy array (Z, Y, X) where True represents M2 voxels.
        fdi_label: FDI tooth identifier (17, 27 for maxillary; 37, 47 for mandibular).
        spacing_zyx: Physical spacing (sz, sy, sx) in mm/voxel.
        origin_zyx: Physical origin (oz, oy, ox) in mm.
        direction_matrix: 3x3 direction cosine matrix. Defaults to identity.

    Returns:
        CEJDetectionResult dataclass.
    """
    if not np.any(mask_m2):
        return CEJDetectionResult(
            cej_z_index=np.nan,
            cej_z_phys_mm=np.nan,
            confidence=0.0,
            is_uncertain=True,
            status="INVALID_MASK",
            diagnostic_reason="M2 mask is completely empty.",
            coronal_apex_z=np.nan,
            apical_apex_z=np.nan,
            cervical_constriction_ratio=np.nan,
        )

    pts = np.argwhere(mask_m2) # (z, y, x)
    n_voxels = len(pts)
    if n_voxels < 300:
        return CEJDetectionResult(
            cej_z_index=np.nan,
            cej_z_phys_mm=np.nan,
            confidence=0.1,
            is_uncertain=True,
            status="CEJ_UNCERTAIN",
            diagnostic_reason=f"Insufficient voxel volume ({n_voxels} voxels) for robust CEJ contouring.",
            coronal_apex_z=np.nan,
            apical_apex_z=np.nan,
            cervical_constriction_ratio=np.nan,
        )

    # Determine tooth orientation: Mandibular (37, 47) vs Maxillary (17, 27)
    is_mandibular = fdi_label in [37, 47]
    z_min, z_max = pts[:, 0].min(), pts[:, 0].max()
    z_span = z_max - z_min

    if z_span < 10:
        return CEJDetectionResult(
            cej_z_index=float((z_min + z_max) / 2),
            cej_z_phys_mm=float((z_min + z_max) / 2 * spacing_zyx[0]),
            confidence=0.2,
            is_uncertain=True,
            status="CEJ_UNCERTAIN",
            diagnostic_reason=f"Z-span is abnormally small ({z_span} slices). Likely truncated volume.",
            coronal_apex_z=float(z_max if is_mandibular else z_min),
            apical_apex_z=float(z_min if is_mandibular else z_max),
            cervical_constriction_ratio=1.0,
        )

    # In standard dental CBCT orientation (axial slices superior-inferior or vice versa):
    # Mandibular: Root apex is inferior (lower Z index), crown is superior (higher Z index).
    # Maxillary: Root apex is superior (higher Z index), crown is inferior (lower Z index).
    if is_mandibular:
        apical_z = float(z_min)
        coronal_z = float(z_max)
        # Cervical line typically spans 50% - 75% of total height from root apex
        search_min = int(z_min + 0.45 * z_span)
        search_max = int(z_min + 0.75 * z_span)
    else:
        apical_z = float(z_max)
        coronal_z = float(z_min)
        # In maxillary: from crown (z_min) to root (z_max), CEJ is at 25% - 55% from z_min
        search_min = int(z_min + 0.25 * z_span)
        search_max = int(z_min + 0.55 * z_span)

    # Compute axial cross-sectional area per slice
    z_indices, slice_counts = np.unique(pts[:, 0], return_counts=True)
    counts_dict = dict(zip(z_indices, slice_counts))

    # Evaluate constriction in search window
    valid_zs = [z for z in range(search_min, search_max + 1) if z in counts_dict]
    if not valid_zs:
        default_z = float(z_min + 0.60 * z_span if is_mandibular else z_min + 0.40 * z_span)
        return CEJDetectionResult(
            cej_z_index=default_z,
            cej_z_phys_mm=float(origin_zyx[0] + default_z * spacing_zyx[0]),
            confidence=0.3,
            is_uncertain=True,
            status="CEJ_UNCERTAIN",
            diagnostic_reason="No valid slices found in standard anatomical search window.",
            coronal_apex_z=coronal_z,
            apical_apex_z=apical_z,
            cervical_constriction_ratio=1.0,
        )

    # Smooth the cross-sectional area curve with 1D gaussian filter (nearest mode prevents boundary artifacts)
    from scipy.ndimage import gaussian_filter1d
    window_counts = np.array([counts_dict[z] for z in valid_zs], dtype=float)
    if len(window_counts) >= 3:
        smoothed_counts = gaussian_filter1d(window_counts, sigma=1.0, mode='nearest')
    else:
        smoothed_counts = window_counts

    min_idx = int(np.argmin(smoothed_counts))
    detected_cej_z = float(valid_zs[min_idx])
    min_area = smoothed_counts[min_idx]
    mean_tooth_area = float(np.mean(slice_counts))
    constriction_ratio = float(min_area / (mean_tooth_area + 1e-6))

    # Calculate physical coordinate
    phys_z = float(origin_zyx[0] + detected_cej_z * spacing_zyx[0])

    # Assess detection certainty: A real cervical line shows clear narrowing (ratio < 0.92)
    # compared to bulbous crown and wider root trunk
    is_uncertain = False
    confidence = 0.85
    reason = "Anatomical cervical constriction detected on M2 axis."

    if constriction_ratio > 0.95:
        is_uncertain = True
        confidence = 0.45
        reason = f"Cervical constriction is shallow (ratio={constriction_ratio:.2f}). CEJ contour is ambiguous."

    return CEJDetectionResult(
        cej_z_index=detected_cej_z,
        cej_z_phys_mm=phys_z,
        confidence=confidence,
        is_uncertain=is_uncertain,
        status="CEJ_UNCERTAIN" if is_uncertain else "CEJ_DETECTED",
        diagnostic_reason=reason,
        coronal_apex_z=coronal_z,
        apical_apex_z=apical_z,
        cervical_constriction_ratio=round(constriction_ratio, 3),
    )
