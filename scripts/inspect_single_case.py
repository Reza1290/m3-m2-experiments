import sys
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
import csv
from src.data.loader import load_volume
from scripts.vision_labeling_fast import analyze_slice_radiography

with open('data/review/clinical_joint_review_sheet.csv', 'r') as f:
    rows = {r['case_id'] + '_' + r['m2_fdi']: r for r in csv.DictReader(f)}

for cid, m2, m3 in [('ToothFairy2P_510', 47, 48), ('ToothFairy2P_452', 47, 48), ('ToothFairy2F_051', 17, 18), ('ToothFairy2F_026', 27, 28)]:
    k = f"{cid}_{m2}"
    if k not in rows:
        continue
    r = rows[k]
    s_idx = int(r['sagittal_slice'])
    dmin = float(r['dmin_mm'])
    
    raw_p = Path(f'data/raw/{cid}_0000.mha')
    mask_p = Path(f'data/toothfairy2/Dataset112_ToothFairy2/labelsTr/{cid}.mha')
    if not raw_p.exists() or not mask_p.exists():
        continue
    raw_v = load_volume(raw_p)
    mask_v = load_volume(mask_p)
    res = analyze_slice_radiography(raw_v.data[:, :, s_idx], mask_v.data[:, :, s_idx], m2, m3, dmin, raw_v.spacing[:2])
    print(f"{cid} ({m2}-{m3}): Class={res['class_name']}, Defect={res['defect_depth_mm']} mm, Dmin={dmin} mm")
    print(f"   Findings: {res['findings']}\n")
