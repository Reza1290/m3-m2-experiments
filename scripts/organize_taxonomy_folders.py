"""Organize crops into clinical taxonomy folders:
1. 1_ketinggian_anatomi (Coronal, Cervical, Radicular, Independent)
2. 2_kemiringan_winter (Mesioangular, Horizontal, Vertical, Distoangular, Inverted)
3. 3_luas_permukaan (Point_Contact, Broad_Surface)
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def organize_taxonomy_folders():
    base_dir = Path("data/custom/labeledbyaimultimodal")
    
    # Load analysis and labels
    contact_analysis_path = Path("data/review/contact_categories_analysis.csv")
    df_contact = pd.read_csv(contact_analysis_path)
    
    labels_csv = Path("data/processed/toothfairy_classified_labels.csv")
    df_labels = pd.read_csv(labels_csv)
    
    # Create target directories
    dir_elevation = base_dir / "1_ketinggian_anatomi"
    dir_winter = base_dir / "2_kemiringan_winter"
    dir_extent = base_dir / "3_luas_permukaan"
    
    # Clean up or recreate target directories
    for d in [dir_elevation, dir_winter, dir_extent]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Organizing {len(df_contact)} contact cases into taxonomy folders...")
    
    manifest_records = []
    
    # Process contact cases
    for _, row in df_contact.iterrows():
        cid = row["case_id"]
        m2 = int(row["m2_fdi"])
        m3 = int(row["m3_fdi"])
        
        crop_path = Path(f"data/processed/crops_2d/{cid}_{m2}_{m3}.npy")
        if not crop_path.exists():
            continue
            
        arr = np.load(crop_path)
        arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_u8)
        
        filename = f"{cid}_{m2}_{m3}.png"
        
        # 1. Ketinggian
        el_raw = str(row["elevation_zone"])
        if "Coronal" in el_raw:
            el_folder = "Coronal_Contact"
        elif "Cervical" in el_raw:
            el_folder = "Cervical_Contact"
        else:
            el_folder = "Radicular_Contact"
            
        (dir_elevation / el_folder).mkdir(parents=True, exist_ok=True)
        img.save(dir_elevation / el_folder / filename)
        
        # 2. Kemiringan Winter
        w_folder = str(row["winter_cat"]).replace(" ", "_")
        (dir_winter / w_folder).mkdir(parents=True, exist_ok=True)
        img.save(dir_winter / w_folder / filename)
        
        # 3. Luas Permukaan
        ext_folder = str(row["extent_cat"]).replace(" ", "_")
        (dir_extent / ext_folder).mkdir(parents=True, exist_ok=True)
        img.save(dir_extent / ext_folder / filename)
        
        manifest_records.append({
            "case_id": cid,
            "m2_fdi": m2,
            "m3_fdi": m3,
            "status": "Contact",
            "ketinggian_anatomi": el_folder,
            "rel_height": row["rel_height"],
            "kemiringan_winter": w_folder,
            "theta_deg": row["theta_deg"],
            "luas_permukaan": ext_folder,
            "filename": filename
        })

    # Also add Independent cases to 1_ketinggian_anatomi for completeness
    df_indep = df_labels[df_labels["class_name"] == "Independent"]
    indep_dir = dir_elevation / "Independent_NoContact"
    indep_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Adding {len(df_indep)} Independent cases to 1_ketinggian_anatomi/Independent_NoContact...")
    for _, row in df_indep.iterrows():
        cid = row["case_id"]
        m2 = int(row["m2_fdi"])
        m3 = int(row["m3_fdi"])
        crop_path = Path(f"data/processed/crops_2d/{cid}_{m2}_{m3}.npy")
        if not crop_path.exists():
            continue
            
        arr = np.load(crop_path)
        arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_u8)
        filename = f"{cid}_{m2}_{m3}.png"
        img.save(indep_dir / filename)
        
        manifest_records.append({
            "case_id": cid,
            "m2_fdi": m2,
            "m3_fdi": m3,
            "status": "Independent",
            "ketinggian_anatomi": "Independent_NoContact",
            "rel_height": np.nan,
            "kemiringan_winter": "N/A",
            "theta_deg": np.nan,
            "luas_permukaan": "N/A",
            "filename": filename
        })
        
    out_manifest = pd.DataFrame(manifest_records)
    manifest_path = base_dir / "taxonomy_manifest.csv"
    out_manifest.to_csv(manifest_path, index=False)
    
    print("\n=== SUMMARY FOLDER STRUKTUR BARU ===")
    print("\n1. 1_ketinggian_anatomi:")
    for sub in sorted(dir_elevation.iterdir()):
        print(f"   - {sub.name}: {len(list(sub.glob('*.png')))} images")
        
    print("\n2. 2_kemiringan_winter:")
    for sub in sorted(dir_winter.iterdir()):
        print(f"   - {sub.name}: {len(list(sub.glob('*.png')))} images")
        
    print("\n3. 3_luas_permukaan:")
    for sub in sorted(dir_extent.iterdir()):
        print(f"   - {sub.name}: {len(list(sub.glob('*.png')))} images")
        
    print(f"\nManifest: {manifest_path.resolve()}")

if __name__ == "__main__":
    organize_taxonomy_folders()
