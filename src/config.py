"""Configuration parameters for Maxillary Molar Relationship CBCT AI System.

Based on Karslıoğlu et al. (Diagnostics 2026, 16, 2123).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Base Project Paths (Supports modular multi-repo & shared workspace)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent if PROJECT_ROOT.parent.name == "projects" else PROJECT_ROOT

DATA_DIR = Path(os.getenv("TA3_DATA_DIR", PROJECT_ROOT / "data"))
if not DATA_DIR.exists() and (WORKSPACE_ROOT / "data").exists():
    DATA_DIR = WORKSPACE_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
MASKS_DIR = DATA_DIR / "masks"
TOOTHFAIRY2_DIR = DATA_DIR / "toothfairy2"
PROCESSED_DIR = DATA_DIR / "processed"
CROPS_2D_DIR = PROCESSED_DIR / "crops_2d"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

WEIGHTS_DIR = Path(os.getenv("TA3_WEIGHTS_DIR", PROJECT_ROOT / "weights"))
if not WEIGHTS_DIR.exists() and (WORKSPACE_ROOT / "weights").exists():
    WEIGHTS_DIR = WORKSPACE_ROOT / "weights"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = WORKSPACE_ROOT / "documents"

# Ensure runtime directories exist
for p in [DATA_DIR, RAW_DATA_DIR, MASKS_DIR, TOOTHFAIRY2_DIR, PROCESSED_DIR, CROPS_2D_DIR, WEIGHTS_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SystemConfig:
    # Random Seed for Reproducibility
    seed: int = 42

    # Image Resolution & Normalization
    image_size: Tuple[int, int] = (224, 224)
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Class Mapping (Diagnostics 2026, 16, 2123)
    num_classes: int = 3
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Contact",
        1: "Independent",
        2: "Resorption",
    })

    # Anatomical FDI Target Notation (All 4 Quadrants)
    # Maxillary (Rahang Atas)
    fdi_maxillary_right_m2: int = 17
    fdi_maxillary_right_m3: int = 18
    fdi_maxillary_left_m2: int = 27
    fdi_maxillary_left_m3: int = 28
    # Mandibular (Rahang Bawah)
    fdi_mandibular_left_m2: int = 37
    fdi_mandibular_left_m3: int = 38
    fdi_mandibular_right_m2: int = 47
    fdi_mandibular_right_m3: int = 48
    # Backward compatibility aliases
    fdi_right_m2: int = 17
    fdi_right_m3: int = 18
    fdi_left_m2: int = 27
    fdi_left_m3: int = 28
    fdi_left_sinus: int = 5    # ToothFairy2 label for Left Maxillary Sinus
    fdi_right_sinus: int = 6   # ToothFairy2 label for Right Maxillary Sinus
    fdi_sinus_label: int = 5   # Default / fallback sinus label

    # 3D Deterministic Geometric Cutoffs
    d_min_independent_threshold_mm: float = 0.5   # > 0.5 mm = Independent
    d_min_geometric_gate_threshold_mm: float = 0.8  # > 0.8 mm = Penalize Contact/Resorption
    sinus_proximity_contact_threshold_mm: float = 0.5  # < 0.5 mm = Contact with Sinus

    # Clinical Triage Gate
    triage_confidence_threshold: float = 0.70  # >= 0.70: Auto-Verified, < 0.70: Review Required

    # Ensemble Architecture & Weights
    ensemble_weights: Tuple[float, float, float] = (1.0, 0.9, 0.8)
    ensemble_model_names: Tuple[str, str, str] = (
        "swin_small_mha",
        "swin_small",
        "vit_base_16",
    )

    # Composite Evaluation Formula: 0.444*Macro-F1 + 0.333*Macro-AUC + 0.222*Accuracy
    composite_weights: Tuple[float, float, float] = (0.444, 0.333, 0.222)

    # Training Hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-2
    train_split_ratio: float = 0.80
    test_split_ratio: float = 0.20

    # Augmentation Parameters
    flip_prob: float = 0.5
    rotation_deg: float = 15.0
    affine_translation: float = 0.1
    color_jitter: float = 0.2


# Default Global Configuration Instance
CONFIG = SystemConfig()
