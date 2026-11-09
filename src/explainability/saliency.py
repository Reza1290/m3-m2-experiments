"""Explainability Module: Saliency Maps & Attention Heatmap Overlays.

Generates gradient-based attention overlays on 2D CBCT sagittal crops
to visualize anatomical regions driving model predictions.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn


def compute_input_gradient_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute normalized input-gradient saliency map.

    Saliency = max_c |d(Score_target) / dx|
    """
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)

    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(dim=-1).item())

    score = logits[0, target_class]
    score.backward()

    # Gradients shape: (1, C, H, W)
    grads = x.grad.data.abs()
    # Take max across color channels
    saliency, _ = torch.max(grads, dim=1)
    saliency_np = saliency.squeeze(0).cpu().numpy()

    # Normalize to [0, 1]
    s_min, s_max = saliency_np.min(), saliency_np.max()
    if s_max > s_min:
        saliency_norm = (saliency_np - s_min) / (s_max - s_min)
    else:
        saliency_norm = np.zeros_like(saliency_np)

    return saliency_norm


def overlay_heatmap_on_image(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a 2D float heatmap with a 2D grayscale base image.

    Args:
        base_image: (H, W) or (H, W, 3) image in [0, 1] or uint8 [0, 255].
        heatmap: (H, W) float array in [0, 1].
        alpha: Weight of the heatmap overlay (0.0 to 1.0).

    Returns:
        RGB image of shape (H, W, 3) as uint8.
    """
    # Normalize base_image to uint8 [0, 255]
    if base_image.max() <= 1.0:
        base_u8 = (base_image * 255.0).astype(np.uint8)
    else:
        base_u8 = np.clip(base_image, 0, 255).astype(np.uint8)

    if base_u8.ndim == 2:
        base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_u8

    # Resize heatmap to match base image dimensions if needed
    h_src, w_src = heatmap.shape[:2]
    h_dst, w_dst = base_rgb.shape[:2]
    if (h_src, w_src) != (h_dst, w_dst):
        heatmap = cv2.resize(heatmap, (w_dst, h_dst), interpolation=cv2.INTER_LINEAR)

    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_u8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
    return blended
