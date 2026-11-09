"""Weighted Soft-Voting Vision Transformer Ensemble & Clinical Triage Gate.

Combines predictions from three architectures:
1. Swin-Small + MHA (weight: 1.0)
2. Swin-Small (weight: 0.9)
3. ViT-Base/16 (weight: 0.8)

Applies confidence-based selective classification triage (threshold tau = 0.70).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG


CLASS_NAMES = {
    0: "Contact",
    1: "Independent",
    2: "Resorption",
}

TRIAGE_THRESHOLD_DEFAULT = 0.70
ENSEMBLE_WEIGHTS_DEFAULT = (1.0, 0.9, 0.8)


@dataclass
class EnsemblePredictionResult:
    """Structured result of the ensemble classification and triage gate."""
    predicted_class: int
    class_name: str
    confidence: float
    class_probabilities: Dict[str, float]  # e.g. {"Contact": 0.82, "Independent": 0.05, "Resorption": 0.13}
    individual_probabilities: Dict[str, Dict[str, float]]
    triage_status: str  # "Prediction Meets Threshold" or "Review Required"
    review_required: bool
    diagnostic_reason: str


class VisionTransformerEnsemble(nn.Module):
    """Production Weighted Soft-Voting Ensemble of Swin-Small+MHA, Swin-Small, and ViT-B/16."""

    def __init__(
        self,
        models: List[nn.Module],
        weights: Tuple[float, float, float] = ENSEMBLE_WEIGHTS_DEFAULT,
        model_names: Tuple[str, str, str] = ("Swin-Small+MHA", "Swin-Small", "ViT-B/16"),
        triage_threshold: float = TRIAGE_THRESHOLD_DEFAULT,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if len(models) != len(weights):
            raise ValueError(f"Models count ({len(models)}) must match weights count ({len(weights)})")

        self.models = nn.ModuleList(models)
        self.weights = list(weights)
        self.weight_sum = float(sum(weights))
        self.model_names = list(model_names)
        self.triage_threshold = float(triage_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        for m in self.models:
            m.to(self.device)
            m.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute aggregated ensemble probabilities.

        Args:
            x: Input image tensor of shape (B, 3, 224, 224) or (B, 1, 224, 224).

        Returns:
            (p_final, [p_model_1, p_model_2, ...])
        """
        # If single channel (grayscale), replicate to 3 channels for ImageNet backbones
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = x.to(self.device)

        # Standardize with ImageNet mean and std if input values are in [0, 1] range
        if x.min() >= -0.01 and x.max() <= 1.01:
            mean = torch.tensor(CONFIG.imagenet_mean, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(CONFIG.imagenet_std, device=self.device).view(1, 3, 1, 1)
            x = (x - mean) / std

        b_size = x.shape[0]

        individual_probs = []
        weighted_sum = torch.zeros((b_size, len(CLASS_NAMES)), device=self.device)

        for model, weight in zip(self.models, self.weights):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            individual_probs.append(probs)
            weighted_sum += weight * probs

        p_final = weighted_sum / self.weight_sum
        return p_final, individual_probs

    def predict_single(
        self,
        image_tensor: torch.Tensor,
        upstream_success: bool = True,
        upstream_failure_reason: str = "",
    ) -> EnsemblePredictionResult:
        """Evaluate a single 2D ROI crop through the ensemble and clinical triage gate.

        Args:
            image_tensor: Tensor of shape (1, 3, 224, 224) or (1, 1, 224, 224) or (224, 224).
            upstream_success: Whether volumetric ingestion and ROI cropping succeeded.
            upstream_failure_reason: Diagnostic error message if upstream processing failed.

        Returns:
            EnsemblePredictionResult with class distribution and triage status.
        """
        # If upstream processing failed, immediately return Review Required
        if not upstream_success:
            return EnsemblePredictionResult(
                predicted_class=-1,
                class_name="UNAVAILABLE",
                confidence=0.0,
                class_probabilities={name: 0.0 for name in CLASS_NAMES.values()},
                individual_probabilities={},
                triage_status="Review Required",
                review_required=True,
                diagnostic_reason=f"Upstream preprocessing failure: {upstream_failure_reason}",
            )

        if image_tensor.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)

        p_final, ind_probs = self.forward(image_tensor)

        p_final_np = p_final.cpu().numpy()[0]
        pred_class = int(np.argmax(p_final_np))
        confidence = float(p_final_np[pred_class])

        class_probs = {CLASS_NAMES[i]: float(p_final_np[i]) for i in range(len(CLASS_NAMES))}

        ind_probs_dict = {}
        for name, prob in zip(self.model_names, ind_probs):
            prob_np = prob.cpu().numpy()[0]
            ind_probs_dict[name] = {CLASS_NAMES[i]: float(prob_np[i]) for i in range(len(CLASS_NAMES))}

        # Triage gate logic
        if confidence >= self.triage_threshold:
            triage_status = "Prediction Meets Threshold"
            review_required = False
            diag = (
                f"Confidence ({confidence:.1%}) meets operational threshold ({self.triage_threshold:.0%}). "
                f"Prediction: {CLASS_NAMES[pred_class]}."
            )
        else:
            triage_status = "Review Required"
            review_required = True
            diag = (
                f"Confidence ({confidence:.1%}) is below operational threshold ({self.triage_threshold:.0%}). "
                f"Ambiguous boundary / low certainty. Case routed to manual radiological review."
            )

        return EnsemblePredictionResult(
            predicted_class=pred_class,
            class_name=CLASS_NAMES[pred_class],
            confidence=confidence,
            class_probabilities=class_probs,
            individual_probabilities=ind_probs_dict,
            triage_status=triage_status,
            review_required=review_required,
            diagnostic_reason=diag,
        )
