import csv

with open('data/review/clinical_joint_review_sheet.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

defects = [float(r['defect_depth_mm']) for r in rows]
dmins = [float(r['dmin_mm']) for r in rows if float(r['dmin_mm']) <= 0.5]
print(f"Total rows: {len(rows)}")
print(f"Rows with dmin <= 0.5: {len(dmins)}")
print(f"Defect depth max: {max(defects):.3f}")
print(f"Defect depth > 0.5 mm: {sum(d >= 0.5 for d in defects)}")
print(f"Defect depth > 0.3 mm: {sum(d >= 0.3 for d in defects)}")
print(f"Defect depth > 0.2 mm: {sum(d >= 0.2 for d in defects)}")
print(f"Defect depth > 0.1 mm: {sum(d >= 0.1 for d in defects)}")

res_cases = [r for r in rows if r['class_name'] == 'Resorption']
print(f"\nResorption cases found ({len(res_cases)}):")
for r in res_cases:
    print(f"  {r['case_id']} ({r['m2_fdi']}-{r['m3_fdi']}): Dmin={r['dmin_mm']}, Defect={r['defect_depth_mm']}")
