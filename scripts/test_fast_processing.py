import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
from src.data.loader import load_volume
from src.data.preprocessor import crop_window_2d

with open('data/processed/toothfairy_classified_labels_v1_dmin_heuristic.csv', 'r') as f:
    v1_rows = list(csv.DictReader(f))

# Take first 10 cases
cases = {}
for r in v1_rows:
    cases.setdefault(r['case_id'], []).append(r)

sample_cases = list(cases.items())[:10]

t0 = time.time()
count = 0
for cid, pairs in sample_cases:
    raw_p = Path(f'data/raw/{cid}_0000.mha')
    mask_p = Path(f'data/toothfairy2/Dataset112_ToothFairy2/labelsTr/{cid}.mha')
    if not raw_p.exists() or not mask_p.exists():
        continue
    raw_v = load_volume(raw_p)
    mask_v = load_volume(mask_p)
    for p in pairs:
        s_idx = int(p['sagittal_slice'])
        dmin = float(p['dmin_mm'])
        slice_cbct = raw_v.data[:, :, s_idx]
        slice_mask = mask_v.data[:, :, s_idx]
        count += 1

elapsed = time.time() - t0
print(f"Processed 10 volumes ({count} pairs) in {elapsed:.2f} seconds! ({elapsed/10:.2f}s per volume)")
