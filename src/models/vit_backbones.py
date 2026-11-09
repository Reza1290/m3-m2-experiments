"""Vision Transformer Model Architectures.

Implements the three primary backbones from the research specification:
1. Swin-Small with extra Multi-Head Attention (MHA) block before classification head.
2. Standard Swin-Small (swin_small_patch4_window7_224).
3. Standard ViT-Base/16 (vit_base_patch16_224).
4. DummyClassifier for deterministic testing and interface simulation.
"""

from typing import Optional

import torch
import torch.nn as nn
import timm


class DummyClassifier(nn.Module):
    """Deterministic reference classifier for testing and interface simulation."""

    def __init__(self, fixed_probs: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(torch.log(fixed_probs + 1e-9), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size = x.shape[0]
        return self.logits.repeat(b_size, 1)


class SwinSmallMHA(nn.Module):
    """Swin-Small backbone enhanced with a Multi-Head Attention (MHA) block.

    Architecture:
    - Backbone: Swin-Small feature extractor (output dim: 768)
    - Additional Block: LayerNorm -> Multi-Head Attention (d_model=768, heads=8) -> Residual Add -> LayerNorm
    - Head: Linear(768, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = False,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Load Swin-Small without final classifier head (num_classes=0 returns pooled feature vector)
        self.backbone = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, 224, 224)
        feats = self.backbone(x)  # (B, 768)

        # Reshape to sequence format (B, 1, 768) for MultiheadAttention
        seq = feats.unsqueeze(1)
        norm_seq = self.norm1(seq)
        attn_out, _ = self.mha(norm_seq, norm_seq, norm_seq)
        seq = seq + self.dropout(attn_out)
        out_feat = self.norm2(seq).squeeze(1)  # (B, 768)

        logits = self.head(out_feat)  # (B, num_classes)
        return logits


def create_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    """Factory method to instantiate research classifier backbones."""
    name_clean = model_name.lower().strip()
    if name_clean in ("swin_small_mha", "swin_s_mha"):
        return SwinSmallMHA(num_classes=num_classes, pretrained=pretrained)
    elif name_clean in ("swin_small", "swin_s"):
        return timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif name_clean in ("vit_base_16", "vit_b_16", "vit_base_patch16_224"):
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif name_clean in ("resnet18", "resnet34", "mobilenetv3_small_100"):
        return timm.create_model(
            name_clean,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported: 'swin_small_mha', 'swin_small', 'vit_base_16', 'resnet18'."
        )
