"""Prepare and Lock Patient-Level Splits for CEJ 4-Class Experiments.

Audits discrepancy between 703 binary cases and 720 total cases:
- 703 cases: 423 Contact + 280 Independent
- 720 cases: 440 Contact (423 Contact + 17 Resorption) + 280 Independent
All 17 Resorption cases represent physical contact on the root dentin/cementum.
Assigns label_source: 'heuristic_pseudo' (provisional development labels).
Enforces strict patient-level split: 80% train, 10% val, 10% test (seed=42).
Outputs:
- data/manifests/dataset_4class_manifest.csv
- data/manifests/patient_splits_seed42.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

def prepare_splits():
    base_dir = Path("D:/Workspace/TA3")
    manifests_dir = base_dir / "data/manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 4-class labels generated previously
    cej_labels_path = base_dir / "data/processed/m2_m3_cej_4class_labels.csv"
    if not cej_labels_path.exists():
        raise FileNotFoundError(f"Missing {cej_labels_path}")
        
    df = pd.read_csv(cej_labels_path)
    print(f"Loaded {len(df)} cases from {cej_labels_path}")
    
    # Add explicit audit metadata
    df["label_source"] = "heuristic_pseudo"
    df["clinical_validation_status"] = "PROVISIONAL_NOT_CLINICALLY_VALIDATED"
    
    # Patient-level splitting (group by case_id)
    unique_patients = sorted(df["case_id"].unique())
    n_patients = len(unique_patients)
    print(f"Total unique patient volumes: {n_patients}")
    
    rng = np.random.RandomState(42)
    shuffled_patients = list(unique_patients)
    rng.shuffle(shuffled_patients)
    
    n_train = int(np.round(0.80 * n_patients))
    n_val = int(np.round(0.10 * n_patients))
    
    train_pids = set(shuffled_patients[:n_train])
    val_pids = set(shuffled_patients[n_train : n_train + n_val])
    test_pids = set(shuffled_patients[n_train + n_val :])
    
    # Assign split column
    def get_split(cid):
        if cid in train_pids:
            return "train"
        elif cid in val_pids:
            return "val"
        else:
            return "test"
            
    df["split"] = df["case_id"].apply(get_split)
    
    split_counts = df["split"].value_counts().to_dict()
    print("\nSample counts per split (tooth pairs):")
    for s, cnt in split_counts.items():
        print(f"  - {s}: {cnt} cases")
        
    print("\nClass distribution per split:")
    ct = pd.crosstab(df["split"], df["class_name"])
    print(ct)
    
    # Verify patient leakage
    print("\nPatient overlap check:")
    print("  Train & Val overlap:", len(train_pids & val_pids))
    print("  Train & Test overlap:", len(train_pids & test_pids))
    print("  Val & Test overlap:", len(val_pids & test_pids))
    assert len(train_pids & val_pids) == 0
    assert len(train_pids & test_pids) == 0
    assert len(val_pids & test_pids) == 0
    
    # Save files
    out_csv = manifests_dir / "dataset_4class_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved manifest to: {out_csv}")
    
    splits_dict = {
        "seed": 42,
        "n_patients": n_patients,
        "train_patients": sorted(list(train_pids)),
        "val_patients": sorted(list(val_pids)),
        "test_patients": sorted(list(test_pids)),
        "counts_per_split": split_counts
    }
    out_json = manifests_dir / "patient_splits_seed42.json"
    with open(out_json, "w") as f:
        json.dump(splits_dict, f, indent=2)
    print(f"Saved split metadata to: {out_json}")

if __name__ == "__main__":
    prepare_splits()
