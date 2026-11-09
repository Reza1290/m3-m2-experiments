"""Unit tests for src/data/dataset.py (Patient-level splitting & dataset pipeline)."""

import numpy as np
import pytest
import torch

from src.data.dataset import DentalROICropDataset, create_patient_level_splits


def test_patient_level_split_leakage_free():
    """Verify that all samples with the same patient_id remain in the same split partition."""
    samples = []
    # 10 patients, 2 molar pairs each = 20 samples
    for p in range(10):
        for side in ["right", "left"]:
            samples.append({
                "patient_id": f"PATIENT_{p:02d}",
                "case_id": f"PATIENT_{p:02d}_{side}",
                "image": np.zeros((224, 224), dtype=np.float32),
                "label": p % 3,
            })

    train_set, val_set, test_set = create_patient_level_splits(
        samples, train_ratio=0.70, val_ratio=0.10, seed=42
    )

    train_pids = {s["patient_id"] for s in train_set}
    val_pids = {s["patient_id"] for s in val_set}
    test_pids = {s["patient_id"] for s in test_set}

    # Verify zero overlap between partitions
    assert len(train_pids.intersection(val_pids)) == 0
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(val_pids.intersection(test_pids)) == 0

    # Total samples conserved
    assert len(train_set) + len(val_set) + len(test_set) == 20


def test_dental_dataset_tensor_shapes():
    """Verify that DentalROICropDataset outputs normalized (3, 224, 224) tensors."""
    sample_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    samples = [
        {"image": sample_img, "label": 0, "case_id": "case_0"},
        {"image": sample_img, "label": 2, "case_id": "case_1"},
    ]

    ds = DentalROICropDataset(samples, is_training=False)
    assert len(ds) == 2

    tensor, label, case_id = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert label == 0
    assert case_id == "case_0"
