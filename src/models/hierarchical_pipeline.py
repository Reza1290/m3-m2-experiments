"""
src/models/hierarchical_pipeline.py
===================================
Unified Hierarchical Pipeline Engine for Option B & Option C.

Enforces:
1. Stage 1: Identical Mask-Guided Vision Transformer Weighted Ensemble (Swin-S+MHA, Swin-S, ViT-B/16).
   If Independent -> outputs Class 0 (Independent).
2. If Contact:
   - Option B Branch: Passes 2D ROI to Stage 2 DL Classifier -> Class 1, 2, or 3.
   - Option C Branch: Passes 3D Masks to Stage 2 3D Geometry Analyzer -> Class 1, 2, or 3.
3. Produces unified HierarchicalPredictionResult dataclass for side-by-side comparison.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.models.vit_backbones import create_model
from src.models.stage2_classifier import (
    Stage2ContactLocationClassifier,
    STAGE2_CLASS_MAP,
    STAGE2_TO_FINAL_CLASS_ID,
)
from src.geometry.contact_geometry import analyze_contact_geometry, ContactGeometryResult
from src.geometry.cej_classifier import GeometricCEJClassifier, CEJClassificationResult


FINAL_CLASS_NAMES = {
    0: "Independent",
    1: "Above_CEJ",
    2: "At_CEJ",
    3: "Below_CEJ",
}


@dataclass
class HierarchicalPredictionResult:
    """Unified result structure for Option B and Option C."""
    approach: str                  # 'Option_B_Hierarchical_DL' or 'Option_C_Hybrid_Geometry'
    final_class_id: int            # 0: Independent, 1: Above_CEJ, 2: At_CEJ, 3: Below_CEJ
    final_class_name: str
    stage1_binary_pred: str        # 'Contact' or 'Independent'
    stage1_confidence: float
    stage1_contact_prob: float
    stage2_location_pred: Optional[str] # Above_CEJ, At_CEJ, Below_CEJ, or None if Independent
    stage2_confidence: Optional[float]
    diagnostic_reason: str
    metadata: Dict[str, Any]


class HierarchicalCDSSPipeline:
    """Master CDSS Pipeline comparing Option B and Option C."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        stage1_weights_paths: Optional[Dict[str, str]] = None,
        stage2_dl_path: Optional[str] = None,
        stage2_backbone: str = "resnet18",
        tau_cej_mm: float = 1.0,
        dmin_contact_threshold_mm: float = 0.50,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tau_cej_mm = tau_cej_mm
        self.dmin_contact_threshold_mm = dmin_contact_threshold_mm

        # 1. Load Stage 1 Binary Ensemble Models
        self.stage1_models = []
        default_s1_paths = {
            "swin_s_mha": "weights/swin_s_mha_binary_best.pth",
            "swin_small": "weights/swin_small_binary_best.pth",
            "vit_base_16": "weights/vit_base_binary_best.pth",
        }
        paths_s1 = stage1_weights_paths or default_s1_paths

        for name, ckpt_path in paths_s1.items():
            p = Path(ckpt_path)
            if p.exists():
                m = create_model(name, num_classes=2, pretrained=False)
                m.load_state_dict(torch.load(p, map_location=self.device))
                m.to(self.device).eval()
                self.stage1_models.append(m)
            else:
                print(f"Warning: Stage 1 checkpoint {ckpt_path} not found. Skipping {name}.")

        # Soft voting weights (1.0, 0.9, 0.8) normalized
        self.ensemble_weights = np.array([1.0, 0.9, 0.8][:len(self.stage1_models)], dtype=np.float32)
        if len(self.ensemble_weights) > 0:
            self.ensemble_weights /= self.ensemble_weights.sum()

        # 2. Load Option B Stage 2 DL Classifier
        self.stage2_dl_model = None
        s2_path = Path(stage2_dl_path or "weights/stage2_dl_best.pth")
        if s2_path.exists():
            self.stage2_dl_model = Stage2ContactLocationClassifier(
                backbone_name=stage2_backbone,
                num_classes=3,
                pretrained=False
            )
            self.stage2_dl_model.load_state_dict(torch.load(s2_path, map_location=self.device))
            self.stage2_dl_model.to(self.device).eval()
        else:
            print(f"Note: Stage 2 DL checkpoint {s2_path} not yet available.")

        # 3. Initialize Option C Geometric Classifier
        self.stage2_geom_classifier = GeometricCEJClassifier(
            tau_cej_mm=tau_cej_mm,
            dmin_contact_threshold_mm=dmin_contact_threshold_mm,
            uncertain_fallback="At_CEJ"
        )

    def predict_stage1_binary(self, crop_tensor: torch.Tensor) -> Tuple[str, float, float]:
        """Run Stage 1 Binary Ensemble.

        Args:
            crop_tensor: (1, 3, 224, 224) or (3, 224, 224) torch.Tensor.

        Returns:
            Tuple of (binary_label, confidence, prob_contact).
            Binary label is 'Contact' or 'Independent'.
        """
        if crop_tensor.ndim == 3:
            crop_tensor = crop_tensor.unsqueeze(0)
        crop_tensor = crop_tensor.to(self.device)

        if not self.stage1_models:
            # Fallback heuristic if checkpoints missing
            return "Contact", 0.50, 0.50

        ensemble_probs = np.zeros(2, dtype=np.float32)
        with torch.no_grad():
            for m, w in zip(self.stage1_models, self.ensemble_weights):
                logits = m(crop_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                ensemble_probs += w * probs

        pred_id = int(np.argmax(ensemble_probs))
        # 0: Contact, 1: Independent
        pred_label = "Contact" if pred_id == 0 else "Independent"
        conf = float(ensemble_probs[pred_id])
        prob_contact = float(ensemble_probs[0])
        return pred_label, conf, prob_contact

    def predict_option_b(self, crop_tensor: torch.Tensor) -> HierarchicalPredictionResult:
        """Run Option B: Hierarchical Deep Learning."""
        s1_pred, s1_conf, s1_p_contact = self.predict_stage1_binary(crop_tensor)

        if s1_pred == "Independent":
            return HierarchicalPredictionResult(
                approach="Option_B_Hierarchical_DL",
                final_class_id=0,
                final_class_name="Independent",
                stage1_binary_pred=s1_pred,
                stage1_confidence=s1_conf,
                stage1_contact_prob=s1_p_contact,
                stage2_location_pred=None,
                stage2_confidence=None,
                diagnostic_reason=f"Stage 1 detected no contact (P_indep={1.0 - s1_p_contact:.2f}).",
                metadata={"stage1_only": True}
            )

        # Stage 2: Deep Learning Classifier
        if crop_tensor.ndim == 3:
            crop_tensor = crop_tensor.unsqueeze(0)
        crop_tensor = crop_tensor.to(self.device)

        if self.stage2_dl_model is not None:
            stage2_id, final_id, s2_conf = self.stage2_dl_model.predict_location(crop_tensor)
            s2_id = int(stage2_id.cpu().numpy()[0])
            f_id = int(final_id.cpu().numpy()[0])
            s2_confidence = float(s2_conf.cpu().numpy()[0])
            s2_name = STAGE2_CLASS_MAP[s2_id]
        else:
            # Fallback if checkpoint missing
            s2_id, f_id, s2_confidence = 2, 3, 0.50
            s2_name = "Below_CEJ"

        return HierarchicalPredictionResult(
            approach="Option_B_Hierarchical_DL",
            final_class_id=f_id,
            final_class_name=FINAL_CLASS_NAMES[f_id],
            stage1_binary_pred=s1_pred,
            stage1_confidence=s1_conf,
            stage1_contact_prob=s1_p_contact,
            stage2_location_pred=s2_name,
            stage2_confidence=s2_confidence,
            diagnostic_reason=f"Stage 1 Contact ({s1_conf:.2f}) -> Stage 2 DL predicted {s2_name} ({s2_confidence:.2f}).",
            metadata={"stage2_id": s2_id}
        )

    def predict_option_c(
        self,
        crop_tensor: torch.Tensor,
        mask_m2: np.ndarray,
        mask_m3: np.ndarray,
        m2_fdi: int,
        m3_fdi: int,
        spacing_zyx: Tuple[float, float, float] = (0.3, 0.3, 0.3),
        origin_zyx: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        allow_geometric_binary_override: bool = False,
    ) -> HierarchicalPredictionResult:
        """Run Option C: Hybrid Deep Learning + 3D Geometry.
        
        Args:
            allow_geometric_binary_override: If False (default), Stage 1 gating is strictly 
                preserved (Contact stays Contact; Stage 2 determines location only).
                If True, 3D geometry is allowed to override Stage 1 false positives back 
                to Independent when D_min > 0.50 mm.
        """
        s1_pred, s1_conf, s1_p_contact = self.predict_stage1_binary(crop_tensor)

        if s1_pred == "Independent":
            return HierarchicalPredictionResult(
                approach="Option_C_Hybrid_Geometry",
                final_class_id=0,
                final_class_name="Independent",
                stage1_binary_pred=s1_pred,
                stage1_confidence=s1_conf,
                stage1_contact_prob=s1_p_contact,
                stage2_location_pred=None,
                stage2_confidence=None,
                diagnostic_reason=f"Stage 1 detected no contact (P_indep={1.0 - s1_p_contact:.2f}).",
                metadata={"stage1_only": True}
            )

        # Stage 2: 3D Geometry CEJ Analysis
        geom_res = analyze_contact_geometry(
            mask_m2, mask_m3, m2_fdi, m3_fdi, spacing_zyx, origin_zyx,
            dmin_contact_threshold_mm=self.dmin_contact_threshold_mm
        )
        geom_class_res = self.stage2_geom_classifier.classify_contact_location(geom_res)

        f_id = geom_class_res.class_id
        f_name = geom_class_res.class_name
        override_applied = False

        # If geometric classifier returned Independent (e.g. D_min > threshold in 3D)
        if f_id == 0:
            if allow_geometric_binary_override:
                # Permitted override: Convert Stage 1 false positive back to Independent
                s2_name = None
                override_applied = True
            else:
                # Strict Stage 1 preservation: Determine location relative to CEJ from closest point
                dz = geom_res.delta_z_cej_mm
                if not np.isnan(dz):
                    if dz > self.tau_cej_mm:
                        f_id, f_name = 1, "Above_CEJ"
                    elif dz < -self.tau_cej_mm:
                        f_id, f_name = 3, "Below_CEJ"
                    else:
                        f_id, f_name = 2, "At_CEJ"
                else:
                    f_id, f_name = 2, "At_CEJ"
                s2_name = f_name
        else:
            s2_name = f_name

        return HierarchicalPredictionResult(
            approach="Option_C_Hybrid_Geometry",
            final_class_id=f_id,
            final_class_name=f_name,
            stage1_binary_pred=s1_pred,
            stage1_confidence=s1_conf,
            stage1_contact_prob=s1_p_contact,
            stage2_location_pred=s2_name,
            stage2_confidence=geom_class_res.confidence,
            diagnostic_reason=f"Stage 1 Contact ({s1_conf:.2f}) -> Stage 2 3D Geometry: {geom_class_res.diagnostic_reason}",
            metadata={
                "dmin_mm": geom_class_res.dmin_mm,
                "delta_z_cej_mm": geom_class_res.delta_z_cej_mm,
                "cej_status": geom_res.cej_result.status,
                "geom_details": geom_class_res.details,
                "override_applied": override_applied,
            }
        )

    def predict_both(
        self,
        crop_tensor: torch.Tensor,
        mask_m2: np.ndarray,
        mask_m3: np.ndarray,
        m2_fdi: int,
        m3_fdi: int,
        spacing_zyx: Tuple[float, float, float] = (0.3, 0.3, 0.3),
        origin_zyx: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        allow_geometric_binary_override: bool = False,
    ) -> Dict[str, HierarchicalPredictionResult]:
        """Run both Option B and Option C side-by-side on identical input."""
        res_b = self.predict_option_b(crop_tensor)
        res_c = self.predict_option_c(
            crop_tensor, mask_m2, mask_m3, m2_fdi, m3_fdi, spacing_zyx, origin_zyx,
            allow_geometric_binary_override=allow_geometric_binary_override
        )
        return {"Option_B": res_b, "Option_C": res_c}
