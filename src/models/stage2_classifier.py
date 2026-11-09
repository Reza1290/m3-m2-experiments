"""
src/models/stage2_classifier.py
===============================
Stage 2 Deep Learning Contact Location Classifier (Option B).

Classifies confirmed Contact cases into:
- 0: Above_CEJ
- 1: At_CEJ
- 2: Below_CEJ

Provides seamless mapping to final 4-class indices:
- 0 (Above_CEJ) -> Class 1
- 1 (At_CEJ)    -> Class 2
- 2 (Below_CEJ) -> Class 3
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.vit_backbones import create_model


STAGE2_CLASS_MAP = {
    0: "Above_CEJ",
    1: "At_CEJ",
    2: "Below_CEJ"
}

# Mapping from Stage 2 (3-class) to Final System (4-class)
STAGE2_TO_FINAL_CLASS_ID = {
    0: 1,  # Above_CEJ
    1: 2,  # At_CEJ
    2: 3,  # Below_CEJ
}

FINAL_CLASS_TO_STAGE2_ID = {
    1: 0,
    2: 1,
    3: 2
}


class Stage2ContactLocationClassifier(nn.Module):
    """Deep Learning Classifier for 3-class CEJ Contact Location."""

    def __init__(
        self,
        backbone_name: str = "swin_small",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Args:
            backbone_name: 'swin_small', 'swin_s_mha', 'vit_base_16', or 'resnet18'.
            num_classes: 3 for Above/At/Below CEJ.
            pretrained: Whether to load ImageNet pre-trained backbone.
            dropout: Dropout rate before classification head.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Re-use standard backbone builder from repository
        self.model = create_model(backbone_name, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking (B, 3, 224, 224) tensor and returning (B, 3) logits."""
        return self.model(x)

    def predict_location(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference helper returning (stage2_class_id, final_4class_id, confidence).

        Args:
            x: Input image tensor (B, 3, 224, 224).

        Returns:
            Tuple of:
            - stage2_class_id: Tensor of shape (B,) with values in {0, 1, 2}.
            - final_4class_id: Tensor of shape (B,) with values in {1, 2, 3}.
            - confidence: Tensor of shape (B,) with softmax probability.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            conf, stage2_pred = torch.max(probs, dim=1)

            # Map to final 4-class id
            final_pred = stage2_pred + 1  # 0->1, 1->2, 2->3

        return stage2_pred, final_pred, conf
