"""Unit tests for Vision Transformer Ensemble and Clinical Triage Gate."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.ensemble import (
    CLASS_NAMES,
    VisionTransformerEnsemble,
    EnsemblePredictionResult,
)
from src.models.vit_backbones import SwinSmallMHA, create_model


class DummyClassifier(nn.Module):
    """Deterministic dummy network for exact probability verification."""

    def __init__(self, fixed_probs: torch.Tensor):
        super().__init__()
        # Store unnormalized logits that produce fixed_probs via softmax
        self.logits = nn.Parameter(torch.log(fixed_probs + 1e-9), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size = x.shape[0]
        return self.logits.repeat(b_size, 1)


def test_swin_mha_forward_shape():
    """Verify SwinSmallMHA forward output tensor shape."""
    model = SwinSmallMHA(num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 3)


def test_ensemble_soft_voting_math():
    """Verify that ensemble calculates exact weighted average probabilities."""
    # Model 1 probs: [0.8, 0.1, 0.1]
    # Model 2 probs: [0.6, 0.2, 0.2]
    # Model 3 probs: [0.7, 0.2, 0.1]
    p1 = torch.tensor([0.8, 0.1, 0.1])
    p2 = torch.tensor([0.6, 0.2, 0.2])
    p3 = torch.tensor([0.7, 0.2, 0.1])

    m1 = DummyClassifier(p1)
    m2 = DummyClassifier(p2)
    m3 = DummyClassifier(p3)

    weights = (1.0, 0.9, 0.8)  # sum = 2.7
    ensemble = VisionTransformerEnsemble(
        models=[m1, m2, m3],
        weights=weights,
        triage_threshold=0.70,
        device="cpu",
    )

    x = torch.zeros(1, 3, 224, 224)
    p_final, ind_probs = ensemble(x)

    # Analytical expectation:
    # Class 0: (1.0*0.8 + 0.9*0.6 + 0.8*0.7) / 2.7 = (0.8 + 0.54 + 0.56) / 2.7 = 1.9 / 2.7 = 0.7037037
    # Class 1: (1.0*0.1 + 0.9*0.2 + 0.8*0.2) / 2.7 = (0.1 + 0.18 + 0.16) / 2.7 = 0.44 / 2.7 = 0.1629629
    # Class 2: (1.0*0.1 + 0.9*0.2 + 0.8*0.1) / 2.7 = (0.1 + 0.18 + 0.08) / 2.7 = 0.36 / 2.7 = 0.1333333
    expected_c0 = (1.0 * 0.8 + 0.9 * 0.6 + 0.8 * 0.7) / 2.7
    expected_c1 = (1.0 * 0.1 + 0.9 * 0.2 + 0.8 * 0.2) / 2.7
    expected_c2 = (1.0 * 0.1 + 0.9 * 0.2 + 0.8 * 0.1) / 2.7

    p_np = p_final.cpu().numpy()[0]
    assert np.isclose(p_np[0], expected_c0, atol=1e-4)
    assert np.isclose(p_np[1], expected_c1, atol=1e-4)
    assert np.isclose(p_np[2], expected_c2, atol=1e-4)
    assert np.isclose(np.sum(p_np), 1.0, atol=1e-5)


def test_triage_gate_threshold_logic():
    """Verify high-confidence pass and low-confidence route to Review Required."""
    # High confidence case (> 0.70)
    p_high = torch.tensor([0.85, 0.10, 0.05])
    m_high = DummyClassifier(p_high)
    ens_high = VisionTransformerEnsemble(models=[m_high], weights=(1.0,), triage_threshold=0.70, device="cpu")

    res_high = ens_high.predict_single(torch.zeros(224, 224))
    assert res_high.triage_status == "Prediction Meets Threshold"
    assert res_high.review_required is False
    assert res_high.predicted_class == 0

    # Low confidence case (< 0.70)
    p_low = torch.tensor([0.45, 0.35, 0.20])
    m_low = DummyClassifier(p_low)
    ens_low = VisionTransformerEnsemble(models=[m_low], weights=(1.0,), triage_threshold=0.70, device="cpu")

    res_low = ens_low.predict_single(torch.zeros(224, 224))
    assert res_low.triage_status == "Review Required"
    assert res_low.review_required is True
    assert "below operational threshold" in res_low.diagnostic_reason.lower()


def test_upstream_failure_triage_handling():
    """Verify that any upstream failure forces Review Required status."""
    p = torch.tensor([0.99, 0.01, 0.00])
    m = DummyClassifier(p)
    ens = VisionTransformerEnsemble(models=[m], weights=(1.0,), device="cpu")

    res = ens.predict_single(
        torch.zeros(224, 224),
        upstream_success=False,
        upstream_failure_reason="Tooth 18 absent in mask",
    )
    assert res.triage_status == "Review Required"
    assert res.review_required is True
    assert res.predicted_class == -1
    assert "Tooth 18 absent" in res.diagnostic_reason
