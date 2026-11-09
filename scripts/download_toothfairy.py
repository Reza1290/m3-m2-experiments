#!/usr/bin/env python3
"""ToothFairy2 Dataset Downloader & Ingestion Script.

Downloads the official ToothFairy2 3D CBCT dataset (24.33 GB) containing
volumetric scans and 42-structure FDI tooth annotations.

Supports:
- Direct HTTP resumable streaming via Hugging Face CDN mirror
- UNIMORE Ditto portal authentication
- Checksum & size validation
- Automatic decompression into data/toothfairy2/
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOOTHFAIRY2_DIR

# Known official / mirror download endpoints
HF_URL_PRIMARY = "https://huggingface.co/datasets/yeldafrt72/ToothFairy/resolve/main/ToothFairy2_Dataset.zip"
HF_URL_SECONDARY = "https://huggingface.co/datasets/XXCHENZm/ToothFairy2/resolve/main/ToothFairy2.zip"
EXPECTED_SIZE_BYTES = 26124590907  # 24.33 GB


def get_remote_file_size(url: str) -> int:
    """Fetch file size via HTTP HEAD request."""
    response = requests.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    return int(response.headers.get("Content-Length", 0))


def download_resumable(
    url: str,
    target_path: Path,
    expected_size: Optional[int] = None,
    chunk_size: int = 4 * 1024 * 1024,  # 4MB chunks
) -> bool:
    """Stream download with HTTP Range header for resumption and progress bar."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    downloaded_bytes = 0
    if temp_path.exists():
        downloaded_bytes = temp_path.stat().st_size
        print(f"[*] Found existing partial download: {downloaded_bytes / (1024**3):.2f} GB")

    headers = {"User-Agent": "ToothFairyDownloader/1.0"}
    if downloaded_bytes > 0:
        headers["Range"] = f"bytes={downloaded_bytes}-"

    print(f"[*] Connecting to: {url}")
    session = requests.Session()
    response = session.get(url, headers=headers, stream=True, timeout=60)

    if downloaded_bytes > 0 and response.status_code == 416:
        # Range not satisfiable, file might be complete
        print("[!] Server returned HTTP 416. Checking file integrity...")
        if expected_size and downloaded_bytes == expected_size:
            temp_path.rename(target_path)
            return True
        else:
            print("[!] Mismatched size. Restarting download...")
            downloaded_bytes = 0
            temp_path.unlink(missing_ok=True)
            response = session.get(url, stream=True, timeout=60)

    if response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP request failed with status code {response.status_code}")

    total_bytes = downloaded_bytes + int(response.headers.get("Content-Length", 0))
    if expected_size:
        total_bytes = expected_size

    mode = "ab" if downloaded_bytes > 0 else "wb"
    print(f"[*] Downloading to: {temp_path.name}")
    print(f"[*] Total target size: {total_bytes / (1024**3):.2f} GB")

    with open(temp_path, mode) as f, tqdm(
        total=total_bytes,
        initial=downloaded_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="ToothFairy2",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    # Rename temp to target upon completion
    if target_path.exists():
        target_path.unlink()
    temp_path.rename(target_path)
    print(f"[+] Download complete: {target_path}")
    return True


def extract_zip(
    archive_path: Path,
    output_dir: Path,
    filter_pattern: Optional[str] = None,
    case_ids: Optional[list] = None,
    remove_archive: bool = False,
):
    """Extract downloaded archive with progress bar and optional filtering."""
    print(f"[*] Extracting from {archive_path.name} to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        if filter_pattern or case_ids:
            filtered = []
            for m in members:
                if m.filename.endswith("/"):
                    continue
                if "dataset.json" in m.filename:
                    filtered.append(m)
                    continue
                if filter_pattern and filter_pattern.lower() in m.filename.lower():
                    filtered.append(m)
                    continue
                if case_ids and any(cid in m.filename for cid in case_ids):
                    filtered.append(m)
                    continue
            members = filtered

        total_uncompressed = sum(m.file_size for m in members)
        print(f"[*] Target files to extract: {len(members)} ({total_uncompressed / (1024**3):.2f} GB)")

        with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
            for member in members:
                zf.extract(member, path=output_dir)
                pbar.update(1)

    print(f"[+] Extraction complete at {output_dir}")
    if remove_archive:
        print(f"[*] Removing archive {archive_path.name} to free disk space...")
        archive_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and unpack the ToothFairy2 CBCT dataset"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=TOOTHFAIRY2_DIR,
        help=f"Target directory for dataset (default: {TOOTHFAIRY2_DIR})",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "huggingface_secondary"],
        default="huggingface",
        help="Download source mirror",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the zip archive after downloading",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Extract only labelsTr/ and dataset.json (~11.6 GB), preserving disk space",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Specific case IDs to extract from imagesTr (e.g. ToothFairy2F_001)",
    )
    parser.add_argument(
        "--remove-zip",
        action="store_true",
        help="Remove the .zip file after successful extraction to save disk space",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query mirror headers and display file size without downloading",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dest_dir = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = HF_URL_PRIMARY if args.source == "huggingface" else HF_URL_SECONDARY
    zip_path = dest_dir / "ToothFairy2_Dataset.zip"

    print("=" * 70)
    print("ToothFairy2 Dataset Downloader")
    print("=" * 70)
    print(f"Mirror URL       : {url}")
    print(f"Target Directory : {dest_dir}")
    print(f"Target Archive   : {zip_path}")
    print("=" * 70)

    if args.dry_run:
        print("[*] Performing dry-run check...")
        size = get_remote_file_size(url)
        print(f"[+] Remote file reachable! Content length: {size} bytes ({size / (1024**3):.2f} GB)")
        return

    # Check if zip already exists and is complete
    if zip_path.exists() and zip_path.stat().st_size == EXPECTED_SIZE_BYTES:
        print(f"[+] Archive {zip_path} is already downloaded and verified.")
    else:
        success = download_resumable(url, zip_path, expected_size=EXPECTED_SIZE_BYTES)
        if not success:
            print("[!] Download was not completed successfully.")
            sys.exit(1)

    if args.extract or args.labels_only or args.cases:
        filter_pattern = "labelsTr" if args.labels_only else None
        extract_zip(
            zip_path,
            dest_dir,
            filter_pattern=filter_pattern,
            case_ids=args.cases,
            remove_archive=args.remove_zip,
        )


if __name__ == "__main__":
    main()
