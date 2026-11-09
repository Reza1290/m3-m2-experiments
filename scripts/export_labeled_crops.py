"""Export labeled crops to data/custom/labeledbyaimultimodal.

Exports the 720 cases into standard class subdirectories:
- data/custom/labeledbyaimultimodal/Contact/*.png
- data/custom/labeledbyaimultimodal/Independent/*.png
- data/custom/labeledbyaimultimodal/Resorption/*.png

Also exports a dataset_manifest.csv with full clinical & geometric metadata.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def export_crops():
    base_out = Path("data/custom/labeledbyaimultimodal")
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels_csv = Path("data/processed/toothfairy_classified_labels.csv")
    df_labels = pd.read_csv(labels_csv)
    print(f"Loaded {len(df_labels)} labeled cases from {labels_csv}")
    
    # Try loading contact analysis for elevation & winter
    contact_analysis_path = Path("data/review/contact_categories_analysis.csv")
    if contact_analysis_path.exists():
        df_contact = pd.read_csv(contact_analysis_path)
        # merge on case_id, m2_fdi, m3_fdi
        df_merged = pd.merge(
            df_labels,
            df_contact[["case_id", "m2_fdi", "m3_fdi", "theta_deg", "winter_cat", "elevation_zone"]],
            on=["case_id", "m2_fdi", "m3_fdi"],
            how="left"
        )
    else:
        df_merged = df_labels.copy()
        df_merged["winter_cat"] = "N/A"
        df_merged["elevation_zone"] = "N/A"
        
    df_merged["elevation_zone"] = df_merged["elevation_zone"].fillna("N/A")
    df_merged["winter_cat"] = df_merged["winter_cat"].fillna("N/A")
    
    manifest_rows = []
    class_counts = {}
    
    for _, row in df_merged.iterrows():
        c_name = str(row["class_name"])
        class_dir = base_out / c_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        crop_path = Path(row["crop_path"])
        if not crop_path.exists():
            print(f"Warning: {crop_path} not found")
            continue
            
        arr = np.load(crop_path)
        # Normalize to uint8 0-255
        if arr.max() <= 1.0:
            arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
            
        img = Image.fromarray(arr_u8)
        
        out_filename = f"{row['case_id']}_{row['m2_fdi']}_{row['m3_fdi']}.png"
        out_filepath = class_dir / out_filename
        img.save(out_filepath)
        
        class_counts[c_name] = class_counts.get(c_name, 0) + 1
        
        manifest_rows.append({
            "case_id": row["case_id"],
            "quadrant": row["quadrant"],
            "m2_fdi": row["m2_fdi"],
            "m3_fdi": row["m3_fdi"],
            "class_name": c_name,
            "confidence": row["confidence"],
            "dmin_mm": row["dmin_mm"],
            "elevation_zone": row["elevation_zone"],
            "winter_cat": row["winter_cat"],
            "relative_image_path": str(out_filepath.relative_to(base_out)),
            "absolute_image_path": str(out_filepath.resolve())
        })
        
    out_manifest = pd.DataFrame(manifest_rows)
    manifest_path = base_out / "dataset_manifest.csv"
    out_manifest.to_csv(manifest_path, index=False)
    
    print("\n=== EXPORT COMPLETE ===")
    print(f"Total exported images: {len(out_manifest)}")
    print(f"Output directory: {base_out.resolve()}")
    print("Class distribution:")
    for k, v in class_counts.items():
        print(f"  - {k}: {v} images")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    export_crops()
