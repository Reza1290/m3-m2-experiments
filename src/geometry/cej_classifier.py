"""
src/geometry/cej_classifier.py
==============================
Deterministic 3D Geometric Classifier for CEJ-Referenced Contact Location (Option C Stage 2).

Maps physical signed elevation (Delta_z_mm) to:
- Class 0: Independent (D_min > contact_threshold)
- Class 1: Above_CEJ (Delta_z > +tau_cej)
- Class 2: At_CEJ (-tau_cej <= Delta_z <= +tau_cej)
- Class 3: Below_CEJ (Delta_z < -tau_cej)

Applies configurable threshold tau_cej and handles CEJ_UNCERTAIN gracefully.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from src.geometry.contact_geometry import ContactGeometryResult


@dataclass
class CEJClassificationResult:
    """Final 4-Class outcome of 3D Geometric CEJ Classification."""
    class_id: int
    class_name: str
    confidence: float
    status: str
    diagnostic_reason: str
    delta_z_cej_mm: float
    dmin_mm: float
    tau_cej_mm: float
    details: Dict[str, Any]


CLASS_NAMES_4 = {
    0: "Independent",
    1: "Above_CEJ",
    2: "At_CEJ",
    3: "Below_CEJ"
}

LOCATION_CLASSES_3 = {
    0: "Above_CEJ",
    1: "At_CEJ",
    2: "Below_CEJ"
}


class GeometricCEJClassifier:
    """Stage 2 Geometric Decision Engine."""

    def __init__(
        self,
        tau_cej_mm: float = 1.0,
        dmin_contact_threshold_mm: float = 0.50,
        uncertain_fallback: str = "At_CEJ",
    ):
        """
        Args:
            tau_cej_mm: Half-width of the cervical transition margin (default 1.0 mm).
            dmin_contact_threshold_mm: Threshold for contact presence (default 0.50 mm).
            uncertain_fallback: Fallback class name if CEJ is uncertain ('At_CEJ', 'Below_CEJ').
        """
        self.tau_cej_mm = tau_cej_mm
        self.dmin_contact_threshold_mm = dmin_contact_threshold_mm
        self.uncertain_fallback = uncertain_fallback

    def classify_contact_location(self, geom_result: ContactGeometryResult) -> CEJClassificationResult:
        """Classify contact location from precomputed 3D contact geometry.

        Args:
            geom_result: ContactGeometryResult from analyze_contact_geometry.

        Returns:
            CEJClassificationResult.
        """
        # Case 1: Mask or computation failure
        if np.isnan(geom_result.dmin_mm):
            return CEJClassificationResult(
                class_id=-1,
                class_name="UNKNOWN",
                confidence=0.0,
                status="FAILED",
                diagnostic_reason=geom_result.diagnostic_reason,
                delta_z_cej_mm=np.nan,
                dmin_mm=np.nan,
                tau_cej_mm=self.tau_cej_mm,
                details={"error": geom_result.status}
            )

        # Case 2: No Contact (Independent)
        if not geom_result.is_contact:
            conf = min(0.98, 0.75 + float(geom_result.dmin_mm - self.dmin_contact_threshold_mm) * 0.15)
            return CEJClassificationResult(
                class_id=0,
                class_name="Independent",
                confidence=round(conf, 3),
                status="INDEPENDENT_NO_CONTACT",
                diagnostic_reason=f"No physical contact. D_min = {geom_result.dmin_mm:.2f} mm > {self.dmin_contact_threshold_mm:.2f} mm.",
                delta_z_cej_mm=geom_result.delta_z_cej_mm,
                dmin_mm=geom_result.dmin_mm,
                tau_cej_mm=self.tau_cej_mm,
                details={"is_contact": False}
            )

        # Case 3: Physical Contact Confirmed -> Evaluate CEJ location
        dz = geom_result.delta_z_cej_mm

        # If elevation difference is completely unresolvable (NaN)
        if np.isnan(dz):
            fallback_id = 2 if self.uncertain_fallback == "At_CEJ" else 3
            return CEJClassificationResult(
                class_id=fallback_id,
                class_name=self.uncertain_fallback,
                confidence=0.35,
                status="CEJ_UNCERTAIN",
                diagnostic_reason=f"CEJ landmark could not be determined. {geom_result.cej_result.diagnostic_reason}",
                delta_z_cej_mm=dz,
                dmin_mm=geom_result.dmin_mm,
                tau_cej_mm=self.tau_cej_mm,
                details={"fallback_applied": True, "cej_status": geom_result.cej_result.status}
            )

        # Objective geometric thresholding (respects delta_z with confidence penalty if uncertain)
        is_uncertain = geom_result.cej_result.is_uncertain
        base_penalty = 0.20 if is_uncertain else 0.0

        if dz > self.tau_cej_mm:
            c_id = 1
            c_name = "Above_CEJ"
            conf = max(0.40, min(0.95, 0.80 + (dz - self.tau_cej_mm) * 0.05) - base_penalty)
            reason = f"Contact point is +{dz:.2f} mm coronal/above CEJ (tau={self.tau_cej_mm:.1f} mm). Enamel contact."
            if is_uncertain:
                reason += f" (Note: Cervical constriction shallow, ratio={geom_result.cej_result.cervical_constriction_ratio:.2f})."
        elif dz < -self.tau_cej_mm:
            c_id = 3
            c_name = "Below_CEJ"
            conf = max(0.40, min(0.95, 0.80 + (-dz - self.tau_cej_mm) * 0.03) - base_penalty)
            reason = f"Contact point is {dz:.2f} mm radicular/below CEJ (tau={self.tau_cej_mm:.1f} mm). Root dentin contact."
            if is_uncertain:
                reason += f" (Note: Cervical constriction shallow, ratio={geom_result.cej_result.cervical_constriction_ratio:.2f})."
        else:
            c_id = 2
            c_name = "At_CEJ"
            conf = max(0.40, 0.85 - base_penalty)
            reason = f"Contact point is {dz:+.2f} mm within cervical transition band [-{self.tau_cej_mm:.1f}, +{self.tau_cej_mm:.1f}] mm."

        return CEJClassificationResult(
            class_id=c_id,
            class_name=c_name,
            confidence=round(conf, 3),
            status="CEJ_UNCERTAIN" if is_uncertain else "CLASSIFIED",
            diagnostic_reason=reason,
            delta_z_cej_mm=dz,
            dmin_mm=geom_result.dmin_mm,
            tau_cej_mm=self.tau_cej_mm,
            details={
                "cej_z_slice": geom_result.cej_result.cej_z_index,
                "cp_m2_z": geom_result.contact_point_m2_zyx[0],
                "constriction_ratio": geom_result.cej_result.cervical_constriction_ratio,
                "is_uncertain": is_uncertain
            }
        )
