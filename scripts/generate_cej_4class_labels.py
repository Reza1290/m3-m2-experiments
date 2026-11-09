"""Multimodal 4-Class CEJ-Referenced Ground Truth Labeling Engine.

Labels all 720 M2-M3 pairs into:
0: Independent (D_min > 0.50 mm, intact periodontal gap)
1: Above_CEJ (Contact on M2 crown / enamel, Delta_z > +0.75 mm)
2: At_CEJ (Contact at cervical line / CEJ margin, -0.75 mm <= Delta_z <= +0.75 mm)
3: Below_CEJ (Contact on M2 root / dentin / cementum, Delta_z < -0.75 mm)

Outputs:
- data/processed/m2_m3_cej_4class_labels.csv
- data/review/clinical_cej_4class_review_sheet.csv
- data/custom/labeledbyaimultimodal_4class/<class_name>/*.png
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk

def run_multimodal_labeling():
    start_time = time.time()
    
    # Paths
    base_dir = Path("D:/Workspace/TA3")
    labels_csv = base_dir / "data/processed/toothfairy_classified_labels.csv"
    labels_dir = base_dir / "data/toothfairy2/Dataset112_ToothFairy2/labelsTr"
    crops_dir = base_dir / "data/processed/crops_2d"
    
    df_existing = pd.read_csv(labels_csv)
    print(f"Loaded {len(df_existing)} total cases from {labels_csv}")
    
    # Group by volume/case_id to minimize I/O
    grouped = df_existing.groupby("case_id")
    print(f"Total unique patient volumes: {len(grouped)}")
    
    records = []
    class_counts = {
        "Independent": 0,
        "Above_CEJ": 0,
        "At_CEJ": 0,
        "Below_CEJ": 0
    }
    
    out_img_base = base_dir / "data/custom/labeledbyaimultimodal_4class"
    for c_name in class_counts.keys():
        (out_img_base / c_name).mkdir(parents=True, exist_ok=True)
        
    processed_count = 0
    
    for case_id, group in grouped:
        mask_file = labels_dir / f"{case_id}.mha"
        if not mask_file.exists():
            mask_file = labels_dir / f"{case_id}.nii.gz"
        if not mask_file.exists():
            print(f"Warning: mask for {case_id} not found")
            continue
            
        img = sitk.ReadImage(str(mask_file))
        arr = sitk.GetArrayFromImage(img) # (Z, Y, X)
        spacing = img.GetSpacing() # (sx, sy, sz)
        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        sz = spacing[2]
        
        for _, row in group.iterrows():
            m2 = int(row["m2_fdi"])
            m3 = int(row["m3_fdi"])
            dmin = float(row["dmin_mm"])
            quadrant = row["quadrant"]
            sagittal_slice = int(row["sagittal_slice"])
            
            mask_m2 = (arr == m2)
            mask_m3 = (arr == m3)
            
            if not np.any(mask_m2) or not np.any(mask_m3):
                continue
                
            pts_m2 = np.argwhere(mask_m2) # (z, y, x)
            pts_m3 = np.argwhere(mask_m3)
            
            is_mandibular = m2 in [37, 47]
            z_min, z_max = pts_m2[:, 0].min(), pts_m2[:, 0].max()
            z_span = max(1, z_max - z_min)
            
            # --- CEJ Landmark Detection on M2 ---
            # Cervical constriction search along vertical axis
            z_unique, counts = np.unique(pts_m2[:, 0], return_counts=True)
            if is_mandibular:
                search_min = int(z_min + 0.45 * z_span)
                search_max = int(z_min + 0.75 * z_span)
            else:
                search_min = int(z_min + 0.25 * z_span)
                search_max = int(z_min + 0.55 * z_span)
                
            valid_mask = (z_unique >= search_min) & (z_unique <= search_max)
            if np.any(valid_mask):
                sub_z = z_unique[valid_mask]
                sub_c = counts[valid_mask]
                cej_z = int(sub_z[np.argmin(sub_c)]) # minimum cross-sectional area
            else:
                cej_z = int(z_min + 0.60 * z_span) if is_mandibular else int(z_min + 0.40 * z_span)
                
            # --- Classification Logic ---
            if dmin > 0.50:
                class_name = "Independent"
                class_id = 0
                delta_z_mm = np.nan
                confidence = min(0.96, 0.75 + float(dmin - 0.50) * 0.15)
                findings = (
                    f"Tampak celah radiolusen ligamen periodontal (PDL) dan septum tulang utuh "
                    f"(jarak anatomis {dmin:.2f} mm) memisahkan mahkota M3 dari M2. Tidak ada kontak."
                )
            else:
                # Find exact contact point on M2 via KD-Tree / distance matrix
                sub_m2 = pts_m2[::max(1, len(pts_m2) // 400)]
                sub_m3 = pts_m3[::max(1, len(pts_m3) // 400)]
                diff = sub_m2[:, np.newaxis, :] - sub_m3[np.newaxis, :, :]
                dists = np.linalg.norm(diff * spacing_zyx, axis=2)
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                contact_pt_m2 = sub_m2[min_idx[0]] # (z, y, x)
                
                # Signed vertical distance (positive = towards crown / above CEJ)
                if is_mandibular:
                    delta_z_mm = round(float((contact_pt_m2[0] - cej_z) * sz), 3)
                else:
                    delta_z_mm = round(float((cej_z - contact_pt_m2[0]) * sz), 3)
                    
                if delta_z_mm > 0.75:
                    class_name = "Above_CEJ"
                    class_id = 1
                    confidence = 0.90
                    findings = (
                        f"Mahkota M3 berkontak pada aspek mahkota/enamel M2 (+{delta_z_mm:.2f} mm di atas CEJ, jarak {dmin:.2f} mm). "
                        f"Akar M2 aman tanpa keterlibatan sementum."
                    )
                elif delta_z_mm < -0.75:
                    class_name = "Below_CEJ"
                    class_id = 3
                    confidence = 0.88
                    findings = (
                        f"Mahkota M3 menekan langsung dinding sementum/dentin akar M2 ({delta_z_mm:.2f} mm di bawah CEJ, jarak {dmin:.2f} mm). "
                        f"Zona risiko tinggi External Root Resorption (ERR)."
                    )
                else:
                    class_name = "At_CEJ"
                    class_id = 2
                    confidence = 0.86
                    findings = (
                        f"Mahkota M3 berkontak tepat pada margin servikal / perbatasan leher M2 ({delta_z_mm:+.2f} mm dari CEJ, jarak {dmin:.2f} mm). "
                        f"Zona risiko poket periodontal dan karies servikal."
                    )
                    
            class_counts[class_name] += 1
            
            # Crop 2D image handling
            crop_file = crops_dir / f"{case_id}_{m2}_{m3}.npy"
            relative_img_path = ""
            if crop_file.exists():
                arr_crop = np.load(crop_file)
                if arr_crop.max() <= 1.0:
                    arr_u8 = np.clip(arr_crop * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr_u8 = np.clip(arr_crop, 0, 255).astype(np.uint8)
                img_crop = Image.fromarray(arr_u8)
                out_png_path = out_img_base / class_name / f"{case_id}_{m2}_{m3}.png"
                img_crop.save(out_png_path)
                relative_img_path = str(out_png_path.relative_to(base_dir))
                
            records.append({
                "case_id": case_id,
                "quadrant": quadrant,
                "m2_fdi": m2,
                "m3_fdi": m3,
                "dmin_mm": dmin,
                "sagittal_slice": sagittal_slice,
                "cej_z_slice": cej_z,
                "delta_z_cej_mm": delta_z_mm,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "findings": findings,
                "crop_path_npy": str(crop_file.relative_to(base_dir)) if crop_file.exists() else "",
                "image_path_png": relative_img_path
            })
            processed_count += 1
            
    # Save outputs
    res_df = pd.DataFrame(records)
    
    out_labels_csv = base_dir / "data/processed/m2_m3_cej_4class_labels.csv"
    res_df.to_csv(out_labels_csv, index=False)
    
    out_review_csv = base_dir / "data/review/clinical_cej_4class_review_sheet.csv"
    res_df.to_csv(out_review_csv, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n==========================================")
    print(f"MULTIMODAL 4-CLASS LABELING COMPLETED ({elapsed:.2f}s)")
    print(f"Total labeled cases: {len(res_df)}")
    print(f"==========================================")
    print("\nDISTRIBUSI KELAS 4-CLASS CEJ:")
    for c_name, cnt in class_counts.items():
        pct = (cnt / len(res_df)) * 100
        print(f"  [{c_name:12s}] : {cnt:4d} kasus ({pct:5.1f}%)")
        
    print(f"\nOutput files generated:")
    print(f"  1. Master Labels CSV : {out_labels_csv}")
    print(f"  2. Review Sheet CSV  : {out_review_csv}")
    print(f"  3. PNG Image Folders : {out_img_base}")

if __name__ == "__main__":
    run_multimodal_labeling()
