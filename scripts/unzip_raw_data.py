"""Fast Unzipper for Raw CBCT Volumes (ToothFairy2).

Extracts 480 raw CBCT volumes from data/toothfairy2/ToothFairy2_Dataset.zip
directly to data/raw/<case_id>_0000.mha.

Monitors remaining SSD space to guarantee safe execution.
"""

import os
import shutil
import time
import zipfile
from pathlib import Path


def get_free_space_gb(path: Path) -> float:
    """Return free disk space in Gigabytes."""
    stat = shutil.disk_usage(path.anchor if path.anchor else ".")
    return stat.free / (1024 ** 3)


def unzip_raw_cbct_dataset(
    zip_path: Path = Path("data/toothfairy2/ToothFairy2_Dataset.zip"),
    out_dir: Path = Path("data/raw"),
    min_headroom_gb: float = 15.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    free_gb = get_free_space_gb(out_dir)
    print("=" * 65)
    print("  ToothFairy2 Raw CBCT Extractor (High Speed NVMe)")
    print(f"  Source Archive: {zip_path}")
    print(f"  Target Folder:  {out_dir}")
    print(f"  Current Free SSD Space: {free_gb:.2f} GB")
    print(f"  Safety Headroom Required: {min_headroom_gb:.1f} GB")
    print("=" * 65)

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    t_start = time.time()

    with zipfile.ZipFile(zip_path, "r") as z:
        infolist = z.infolist()
        raw_entries = [
            f for f in infolist
            if "imagesTr" in f.filename and f.filename.endswith("_0000.mha")
        ]
        total_files = len(raw_entries)
        total_uncompressed_gb = sum(f.file_size for f in raw_entries) / (1024 ** 3)

        print(f"\nFound {total_files} raw CBCT volumes ({total_uncompressed_gb:.2f} GB total).")

        extracted_count = 0
        skipped_count = 0
        bytes_written = 0

        for idx, entry in enumerate(raw_entries, 1):
            filename = Path(entry.filename).name
            target_path = out_dir / filename

            # Check if file already exists with identical size
            if target_path.exists() and target_path.stat().st_size == entry.file_size:
                skipped_count += 1
                continue

            # Check disk space safety check
            cur_free = get_free_space_gb(out_dir)
            if cur_free < min_headroom_gb:
                print(f"\n[ALERT] Free space reached safety limit ({cur_free:.2f} GB). Halting extraction.")
                break

            # Fast 4MB chunked stream extraction
            with z.open(entry) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 4)

            bytes_written += entry.file_size
            extracted_count += 1

            if extracted_count % 25 == 0 or idx == total_files:
                elapsed = time.time() - t_start
                rate_mb = (bytes_written / (1024 ** 2)) / (elapsed + 1e-5)
                rem_files = total_files - (extracted_count + skipped_count)
                eta_sec = (rem_files * (elapsed / max(1, extracted_count)))
                print(
                    f"[{idx:3d}/{total_files:3d}] "
                    f"Extracted: {extracted_count:3d} | "
                    f"Skipped: {skipped_count:3d} | "
                    f"Rate: {rate_mb:5.1f} MB/s | "
                    f"Free SSD: {cur_free:5.1f} GB | "
                    f"ETA: {eta_sec / 60:.1f}m"
                )

    elapsed_total = time.time() - t_start
    final_free = get_free_space_gb(out_dir)
    print("\n" + "=" * 65)
    print(f"Extraction Completed in {elapsed_total:.1f}s ({elapsed_total / 60:.2f} minutes)!")
    print(f"  - Extracted: {extracted_count} volumes")
    print(f"  - Already Present: {skipped_count} volumes")
    print(f"  - Total CBCT Scans in {out_dir}: {len(list(out_dir.glob('*_0000.mha')))}")
    print(f"  - Final SSD Free Space: {final_free:.2f} GB")
    print("=" * 65)


if __name__ == "__main__":
    unzip_raw_cbct_dataset()
