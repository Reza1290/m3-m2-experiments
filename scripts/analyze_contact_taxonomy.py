"""Analyze Contact Categories Across ToothFairy2 Dataset.

Categorizes the 423 Contact Cases by:
1. Anatomical Elevation: Coronal vs Cervical vs Radicular (Wang et al., 2017)
2. Contact Extent / Area: Point Contact vs Broad Surface Contact (Smail-Faugeron et al., 2016)
3. Winter's Angulation: Vertical vs Mesioangular vs Distoangular vs Horizontal (Shiller 1979)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.decomposition import PCA
import SimpleITK as sitk

def run_analysis():
    start_time = time.time()
    csv_path = Path("data/review/clinical_joint_review_sheet.csv")
    df = pd.read_csv(csv_path)
    
    # Filter only Contact cases (423 cases)
    contacts_df = df[df["class_name"] == "Contact"].copy()
    print(f"Total Contact cases to analyze: {len(contacts_df)}")
    
    labels_dir = Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr")
    
    # Group by case_id to load mask only once
    grouped = contacts_df.groupby("case_id")
    print(f"Unique patient volumes with contact: {len(grouped)}")
    
    results = []
    processed_count = 0
    
    for case_id, group in grouped:
        mask_file = labels_dir / f"{case_id}.mha"
        if not mask_file.exists():
            mask_file = labels_dir / f"{case_id}.nii.gz"
        if not mask_file.exists():
            continue
            
        # Load mask volume
        img = sitk.ReadImage(str(mask_file))
        arr = sitk.GetArrayFromImage(img) # (Z, Y, X)
        spacing = img.GetSpacing() # (sx, sy, sz)
        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        
        for _, row in group.iterrows():
            m2_fdi = int(row["m2_fdi"])
            m3_fdi = int(row["m3_fdi"])
            
            mask_m2 = (arr == m2_fdi)
            mask_m3 = (arr == m3_fdi)
            
            if not np.any(mask_m2) or not np.any(mask_m3):
                continue
                
            pts_m2 = np.argwhere(mask_m2) # (z, y, x)
            pts_m3 = np.argwhere(mask_m3)
            
            # --- 1. WINTER'S ANGULATION (3D PCA) ---
            # Max component for M2 and M3
            pca_m2 = PCA(n_components=1).fit(pts_m2 * spacing_zyx)
            pca_m3 = PCA(n_components=1).fit(pts_m3 * spacing_zyx)
            
            v2 = pca_m2.components_[0]
            v3 = pca_m3.components_[0]
            
            # Orient vectors towards crown (apico-coronal)
            # Mandibular (37-38, 47-48): root is inferior (low z), crown is superior (high z)
            # Maxillary (17-18, 27-28): root is superior (high z), crown is inferior (low z)
            is_mandibular = m2_fdi in [37, 47]
            if is_mandibular:
                if v2[0] < 0: v2 = -v2
                if v3[0] < 0: v3 = -v3
            else:
                if v2[0] > 0: v2 = -v2
                if v3[0] > 0: v3 = -v3
                
            # Angle between tooth axes
            cos_theta = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))
            theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
            # Tilt direction: check if M3 crown leans towards M2 (mesial) or away (distal)
            # M3 is distal to M2. In arch, mesial means towards M2.
            m2_center = np.mean(pts_m2, axis=0)
            m3_center = np.mean(pts_m3, axis=0)
            
            # Vector from M3 center to M2 center
            vec_to_m2 = (m2_center - m3_center) * spacing_zyx
            # Project v3 along vec_to_m2
            proj = np.dot(v3, vec_to_m2)
            is_mesial = (proj > 0)
            
            # Winter Classification (Shiller 1979)
            if theta_deg <= 10.0:
                winter_cat = "Vertical"
            elif 10.0 < theta_deg <= 70.0:
                winter_cat = "Mesioangular" if is_mesial else "Distoangular"
            elif 70.0 < theta_deg <= 110.0:
                winter_cat = "Horizontal"
            else:
                winter_cat = "Inverted"
                
            # --- 2. ANATOMICAL ELEVATION ZONE (Wang et al., 2017) ---
            # Find contact point (min distance between surfaces)
            # Subsample for speed
            sub_m2 = pts_m2[::max(1, len(pts_m2) // 400)]
            sub_m3 = pts_m3[::max(1, len(pts_m3) // 400)]
            
            # Compute distance matrix between subsets
            diff = sub_m2[:, np.newaxis, :] - sub_m3[np.newaxis, :, :]
            dists = np.linalg.norm(diff * spacing_zyx, axis=2)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            contact_pt_m2 = sub_m2[min_idx[0]] # (z, y, x)
            
            # M2 vertical span
            z_m2_min = pts_m2[:, 0].min()
            z_m2_max = pts_m2[:, 0].max()
            z_span = max(1, z_m2_max - z_m2_min)
            
            # Relative height on M2 from apex (0.0) to occlusal crown (1.0)
            if is_mandibular:
                rel_height = (contact_pt_m2[0] - z_m2_min) / z_span
            else:
                rel_height = (z_m2_max - contact_pt_m2[0]) / z_span
                
            if rel_height >= 0.67:
                elevation_zone = "Coronal (Crown)"
            elif 0.35 <= rel_height < 0.67:
                elevation_zone = "Cervical (CEJ)"
            else:
                elevation_zone = "Radicular (Root)"
                
            # --- 3. CONTACT SURFACE EXTENT (Point vs Surface / Broad) ---
            # Contact area approximation: count voxels of M2 within 0.8 mm of M3
            close_pts = np.sum(dists < 0.8) # proxy for interface size
            # In 2D/3D interface: point contact has very few close points, broad has many
            if close_pts <= 6:
                extent_cat = "Point Contact"
            else:
                extent_cat = "Broad Surface Contact"
                
            results.append({
                "case_id": case_id,
                "m2_fdi": m2_fdi,
                "m3_fdi": m3_fdi,
                "theta_deg": round(theta_deg, 1),
                "winter_cat": winter_cat,
                "rel_height": round(rel_height, 3),
                "elevation_zone": elevation_zone,
                "close_pts": int(close_pts),
                "extent_cat": extent_cat,
            })
            processed_count += 1
            
    res_df = pd.DataFrame(results)
    res_df.to_csv("data/review/contact_categories_analysis.csv", index=False)
    
    elapsed = time.time() - start_time
    print(f"\n--- ANALYSIS COMPLETE ({elapsed:.2f}s) ---")
    print(f"Total analyzed: {len(res_df)}")
    
    print("\n1. DISTRIBUSI LOKASI ELEVASI ANATOMIS (Wang et al., 2017):")
    print(res_df["elevation_zone"].value_counts())
    print((res_df["elevation_zone"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")
    
    print("\n2. DISTRIBUSI LUAS PERMUKAAN KONTAK:")
    print(res_df["extent_cat"].value_counts())
    print((res_df["extent_cat"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")
    
    print("\n3. DISTRIBUSI ARAH KEMIRINGAN WINTER (Shiller 1979):")
    print(res_df["winter_cat"].value_counts())
    print((res_df["winter_cat"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

if __name__ == "__main__":
    run_analysis()
