import csv

with open('data/review/clinical_joint_review_sheet.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

candidates = [r for r in rows if float(r['defect_depth_mm']) >= 0.30 and float(r['dmin_mm']) <= 0.40]
print(f"Candidates with defect >= 0.30 mm and dmin <= 0.40 mm: {len(candidates)}")

print("\nSample candidates:")
for r in candidates[:15]:
    print(f"  {r['case_id']} ({r['m2_fdi']}-{r['m3_fdi']}): Dmin={r['dmin_mm']} mm, Defect={r['defect_depth_mm']} mm, Current={r['class_name']}")
