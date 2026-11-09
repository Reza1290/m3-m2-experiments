import sys
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
import csv
import numpy as np

# Load the audit data
with open('data/review/clinical_joint_review_sheet.csv', 'r') as f:
    rows = list(csv.DictReader(f))

# Let's count cases by defect depth for touching cases (Dmin <= 0.50 mm)
touching_cases = [r for r in rows if float(r['dmin_mm']) <= 0.50]

print(f"Total touching cases: {len(touching_cases)}")
res_candidates_40 = [r for r in touching_cases if float(r['defect_depth_mm']) >= 0.40]
res_candidates_35 = [r for r in touching_cases if float(r['defect_depth_mm']) >= 0.35]
res_candidates_30 = [r for r in touching_cases if float(r['defect_depth_mm']) >= 0.30]

print(f"Cases with Defect >= 0.40 mm: {len(res_candidates_40)} ({len(res_candidates_40)/720*100:.1f}%)")
print(f"Cases with Defect >= 0.35 mm: {len(res_candidates_35)} ({len(res_candidates_35)/720*100:.1f}%)")
print(f"Cases with Defect >= 0.30 mm: {len(res_candidates_30)} ({len(res_candidates_30)/720*100:.1f}%)")

print("\nTop 20 Defect Cases currently labeled Contact:")
missed = [r for r in res_candidates_35 if r['class_name'] == 'Contact']
for idx, r in enumerate(missed[:20], 1):
    print(f"{idx:2d}. {r['case_id']} ({r['m2_fdi']}-{r['m3_fdi']}) [{r['quadrant']}]: "
          f"Defect={r['defect_depth_mm']} mm, Dmin={r['dmin_mm']} mm")
