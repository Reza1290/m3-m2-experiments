"""
src/geometry/contact_geometry.py
================================
3D KD-Tree Surface Proximity & Signed CEJ Elevation Analyzer.

Calculates:
1. Physical Euclidean surface-to-surface minimum distance (D_min) via KD-Tree.
2. Exact 3D closest contact coordinate pair on M2 and M3.
3. Signed vertical elevation distance (Delta_z_mm) from the M2 contact point
   to the anatomical CEJ reference (positive = coronal/Above CEJ, negative = radicular/Below CEJ).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from src.geometry.cej_detector import CEJDetectionResult, detect_m2_cej


@dataclass
class ContactGeometryResult:
    """Detailed geometric analysis result between M2 and M3."""
    dmin_mm: float
    contact_point_m2_zyx: Tuple[int, int, int]
    contact_point_m3_zyx: Tuple[int, int, int]
    delta_z_cej_mm: float
    cej_result: CEJDetectionResult
    status: str
    is_contact: bool
    diagnostic_reason: str


def extract_surface_voxels(mask_3d: np.ndarray) -> np.ndarray:
    """Extract exterior boundary voxels of a 3D binary mask using 6-connectivity."""
    if not np.any(mask_3d):
        return np.empty((0, 3), dtype=int)
    eroded = ndimage.binary_erosion(mask_3d, structure=ndimage.generate_binary_structure(3, 1))
    boundary = mask_3d & (~eroded)
    return np.argwhere(boundary)


def analyze_contact_geometry(
    mask_m2: np.ndarray,
    mask_m3: np.ndarray,
    m2_fdi: int,
    m3_fdi: int,
    spacing_zyx: Tuple[float, float, float] = (0.3, 0.3, 0.3),
    origin_zyx: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    dmin_contact_threshold_mm: float = 0.50,
    cej_result_override: Optional[CEJDetectionResult] = None,
) -> ContactGeometryResult:
    """Compute physical 3D contact geometry and CEJ-relative elevation.

    Args:
        mask_m2: 3D boolean mask for tooth M2.
        mask_m3: 3D boolean mask for tooth M3.
        m2_fdi: FDI identifier for M2 (17, 27, 37, 47).
        m3_fdi: FDI identifier for M3 (18, 28, 38, 48).
        spacing_zyx: Physical millimeter spacing per voxel (sz, sy, sx).
        origin_zyx: Physical millimeter origin (oz, oy, ox).
        dmin_contact_threshold_mm: Operational distance threshold for contact (default 0.5 mm).
        cej_result_override: Optional precomputed CEJDetectionResult.

    Returns:
        ContactGeometryResult.
    """
    # Validate masks
    if not np.any(mask_m2) or not np.any(mask_m3):
        dummy_cej = CEJDetectionResult(
            cej_z_index=np.nan, cej_z_phys_mm=np.nan, confidence=0.0,
            is_uncertain=True, status="EMPTY_MASK", diagnostic_reason="One or both tooth masks are empty.",
            coronal_apex_z=np.nan, apical_apex_z=np.nan, cervical_constriction_ratio=np.nan
        )
        return ContactGeometryResult(
            dmin_mm=np.nan,
            contact_point_m2_zyx=(0, 0, 0),
            contact_point_m3_zyx=(0, 0, 0),
            delta_z_cej_mm=np.nan,
            cej_result=dummy_cej,
            status="EMPTY_MASK",
            is_contact=False,
            diagnostic_reason="Missing anatomical mask for M2 or M3."
        )

    # 1. CEJ Detection on M2
    if cej_result_override is not None:
        cej_res = cej_result_override
    else:
        cej_res = detect_m2_cej(mask_m2, m2_fdi, spacing_zyx, origin_zyx)

    # 2. Extract surface voxels
    surf_m2 = extract_surface_voxels(mask_m2)
    surf_m3 = extract_surface_voxels(mask_m3)

    if len(surf_m2) == 0 or len(surf_m3) == 0:
        return ContactGeometryResult(
            dmin_mm=np.nan,
            contact_point_m2_zyx=(0, 0, 0),
            contact_point_m3_zyx=(0, 0, 0),
            delta_z_cej_mm=np.nan,
            cej_result=cej_res,
            status="EMPTY_SURFACE",
            is_contact=False,
            diagnostic_reason="Zero boundary surface voxels found."
        )

    # Convert surface coordinates to physical millimeter space for KD-Tree
    scale_zyx = np.array(spacing_zyx, dtype=np.float64)
    surf_m2_phys = surf_m2 * scale_zyx
    surf_m3_phys = surf_m3 * scale_zyx

    tree_m3 = cKDTree(surf_m3_phys)
    dists, indices = tree_m3.query(surf_m2_phys, k=1)

    min_idx_m2 = int(np.argmin(dists))
    dmin_phys = float(dists[min_idx_m2])
    min_idx_m3 = int(indices[min_idx_m2])

    cp_m2 = tuple(surf_m2[min_idx_m2])
    cp_m3 = tuple(surf_m3[min_idx_m3])

    is_contact = (dmin_phys <= dmin_contact_threshold_mm)

    # 3. Compute signed vertical elevation relative to CEJ
    # Mandibular (37, 47): Crown is at higher Z, root at lower Z.
    # Maxillary (17, 27): Crown is at lower Z, root at higher Z.
    is_mandibular = m2_fdi in [37, 47]
    sz = spacing_zyx[0]

    if np.isnan(cej_res.cej_z_index):
        delta_z = np.nan
    else:
        if is_mandibular:
            delta_z = float((cp_m2[0] - cej_res.cej_z_index) * sz)
        else:
            delta_z = float((cej_res.cej_z_index - cp_m2[0]) * sz)

    reason = (
        f"D_min={dmin_phys:.2f}mm. "
        f"Contact point M2 at Z={cp_m2[0]}. "
        f"CEJ at Z={cej_res.cej_z_index:.1f}. "
        f"Delta_z={delta_z:+.2f}mm."
    )

    return ContactGeometryResult(
        dmin_mm=round(dmin_phys, 3),
        contact_point_m2_zyx=(int(cp_m2[0]), int(cp_m2[1]), int(cp_m2[2])),
        contact_point_m3_zyx=(int(cp_m3[0]), int(cp_m3[1]), int(cp_m3[2])),
        delta_z_cej_mm=round(delta_z, 3) if not np.isnan(delta_z) else np.nan,
        cej_result=cej_res,
        status="OK" if not cej_res.is_uncertain else "OK_CEJ_UNCERTAIN",
        is_contact=is_contact,
        diagnostic_reason=reason
    )
