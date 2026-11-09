import csv
import numpy as np

with open('data/review/clinical_joint_review_sheet.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in review sheet: {len(rows)}")

# Analyze all touching cases (Dmin <= 0.50 mm)
touching = [r for r in rows if float(r['dmin_mm']) <= 0.50]
print(f"Total touching/contacting cases (Dmin <= 0.50 mm): {len(touching)}")

defects = [float(r['defect_depth_mm']) for r in touching]
dmins = [float(r['dmin_mm']) for r in touching]

print("\n--- Defect Depth Breakdown (Touching Cases) ---")
for t in [0.70, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    cnt = sum(d >= t for d in defects)
    pct = cnt / len(touching) * 100
    pct_total = cnt / len(rows) * 100
    print(f"  Defect >= {t:.2f} mm: {cnt:3d} cases ({pct:5.1f}% of touching, {pct_total:4.1f}% of all 720 pairs)")

# Find all cases with defect >= 0.20 mm
print("\n--- Candidates with Defect Depth >= 0.20 mm ---")
candidates = [r for r in touching if float(r['defect_depth_mm']) >= 0.20]
candidates.sort(key=lambda x: float(x['defect_depth_mm']), reverse=True)

for idx, r in enumerate(candidates, 1):
    print(f"{idx:2d}. {r['case_id']} ({r['m2_fdi']}-{r['m3_fdi']}) [{r['quadrant']}]: "
          f"Defect={float(r['defect_depth_mm']):.3f} mm, Dmin={float(r['dmin_mm']):.3f} mm, "
          f"Current={r['class_name']}")
