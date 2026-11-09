"""
src/geometry/spatial_dist.py
============================
Modul komputasi jarak minimum permukaan (D_min) antara pasangan
gigi M2-M3 dari mask segmentasi ToothFairy2 (.mha format).

Satuan output: milimeter (mm) — koordinat fisik, bukan indeks voxel.

Penggunaan CLI:
    python -m src.geometry.spatial_dist --n 50 --pairs maxillary
    python -m src.geometry.spatial_dist --n all --out data/processed/dmin_results.csv
"""

import numpy as np
import time
import csv
import argparse
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree


# ── Konstanta FDI ────────────────────────────────────────────────────────────
PAIR_MAXILLARY  = [(17, 18), (27, 28)]          # Fokus utama proyek
PAIR_MANDIBULAR = [(37, 38), (47, 48)]          # Opsional
PAIR_ALL        = PAIR_MAXILLARY + PAIR_MANDIBULAR

LABELS_DIR = Path(r"d:\Workspace\TA3\data\toothfairy2\Dataset112_ToothFairy2\labelsTr")
OUTPUT_DIR = Path(r"d:\Workspace\TA3\data\processed")
DMIN_THRESHOLD_MM = 0.5   # Parameter desain operasional (lihat catatan di README)


# ── I/O ──────────────────────────────────────────────────────────────────────
def read_mha(filepath: Path) -> tuple[dict, np.ndarray]:
    """
    Baca file .mha MetaImage tanpa dependensi SimpleITK/nibabel.
    Returns:
        meta  : dict header (termasuk 'ElementSpacing', 'DimSize')
        arr   : np.ndarray uint8 shape (Z, Y, X)
    """
    meta = {}
    with open(filepath, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_str = line.decode("ascii", errors="ignore").strip()
            if "=" in line_str:
                k, v = line_str.split("=", 1)
                meta[k.strip()] = v.strip()
            if b"ElementDataFile" in line:
                break
        dim = [int(x) for x in meta["DimSize"].split()]   # [X, Y, Z]
        data = np.fromfile(f, dtype=np.uint8)
        arr  = data.reshape((dim[2], dim[1], dim[0]))      # → (Z, Y, X)
    return meta, arr


def get_spacing_zyx(meta: dict) -> np.ndarray:
    """
    Ambil spacing dari header .mha.
    Header menyimpan dalam urutan X Y Z → kita balik ke Z Y X.
    Returns: array [sz, sy, sx] dalam mm/voxel
    """
    sp = [float(x) for x in meta["ElementSpacing"].split()]
    return np.array([sp[2], sp[1], sp[0]])


# ── Geometri ─────────────────────────────────────────────────────────────────
def get_surface_voxels(mask_3d: np.ndarray) -> np.ndarray:
    """
    Ekstrak voxel permukaan mask via erosion 3D (26-connectivity).
    Surface = mask XOR eroded_mask.
    Returns: indeks voxel permukaan, shape (N, 3) dalam urutan (Z, Y, X)
    """
    struct = ndimage.generate_binary_structure(3, 3)   # 26-connectivity
    eroded = ndimage.binary_erosion(mask_3d, structure=struct)
    surface = mask_3d ^ eroded
    return np.argwhere(surface)


def compute_dmin(
    mask_m2: np.ndarray,
    mask_m3: np.ndarray,
    spacing_zyx: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Hitung D_min (mm) antara permukaan mask M2 dan M3.

    Rumus:
        D_min = min_{u ∈ ∂M2, v ∈ ∂M3} || (u - v) * spacing ||_2

    Args:
        mask_m2      : boolean array (Z,Y,X) untuk gigi M2
        mask_m3      : boolean array (Z,Y,X) untuk gigi M3
        spacing_zyx  : [sz, sy, sx] dalam mm/voxel

    Returns:
        dist_mm  : jarak minimum dalam mm (koordinat fisik)
        coord_m2 : indeks voxel titik terdekat di M2 [z,y,x]
        coord_m3 : indeks voxel titik terdekat di M3 [z,y,x]
    """
    # Ekstrak voxel permukaan
    sv_m2 = get_surface_voxels(mask_m2)   # (N2, 3) indeks
    sv_m3 = get_surface_voxels(mask_m3)   # (N3, 3) indeks

    # Konversi indeks → koordinat fisik (mm)
    # PENTING: inilah yang menjamin satuan output adalah mm
    phys_m2 = sv_m2 * spacing_zyx         # (N2, 3) mm
    phys_m3 = sv_m3 * spacing_zyx         # (N3, 3) mm

    # KDTree query: untuk setiap titik M3, cari tetangga terdekat di M2
    tree = cKDTree(phys_m2)
    dists, idx_m2 = tree.query(phys_m3)   # dists dalam mm ✅

    # Titik pasangan dengan jarak minimum
    best = int(np.argmin(dists))
    dist_mm  = float(dists[best])
    coord_m3 = sv_m3[best]
    coord_m2 = sv_m2[idx_m2[best]]

    return dist_mm, coord_m2, coord_m3


def classify_dmin(dist_mm: float, threshold: float = DMIN_THRESHOLD_MM) -> str:
    """
    Klasifikasi berdasarkan D_min.
    > threshold  → INDEPENDENT (terpisah, ada septum tulang)
    <= threshold → CONTACT_OR_RESORPTION (perlu validasi visual dokter)
    """
    return "INDEPENDENT" if dist_mm > threshold else "CONTACT_OR_RESORPTION"


# ── Pipeline Utama ────────────────────────────────────────────────────────────
def scan_dataset(
    labels_dir: Path,
    n_files: int | None = None,
    pairs: list[tuple] = PAIR_MAXILLARY,
    verbose: bool = True
) -> list[dict]:
    """
    Jalankan komputasi D_min untuk semua file .mha di labels_dir.

    Args:
        labels_dir : folder berisi file *.mha
        n_files    : batasi jumlah file (None = semua 480)
        pairs      : list pasangan FDI, default maxillary [(17,18),(27,28)]
        verbose    : cetak progress ke terminal

    Returns:
        results : list of dict per pasangan gigi
    """
    files = sorted(labels_dir.glob("*.mha"))
    if n_files is not None:
        files = files[:n_files]

    results = []
    total_files = len(files)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  ToothFairy2 D_min Scanner")
        print(f"  Files    : {total_files}")
        print(f"  Pairs    : {pairs}")
        print(f"  Threshold: {DMIN_THRESHOLD_MM} mm")
        print(f"{'='*70}\n")

    for i, fp in enumerate(files, 1):
        t_file = time.time()
        try:
            meta, arr = read_mha(fp)
            spacing_zyx = get_spacing_zyx(meta)
        except Exception as e:
            if verbose:
                print(f"[{i:3d}/{total_files}] ERROR baca {fp.name}: {e}")
            continue

        for m2_id, m3_id in pairs:
            mask_m2 = (arr == m2_id)
            mask_m3 = (arr == m3_id)

            # Cek keberadaan gigi
            m2_present = bool(np.sum(mask_m2) > 0)
            m3_present = bool(np.sum(mask_m3) > 0)

            if not m2_present or not m3_present:
                status = "ABSENT_M2" if not m2_present else "ABSENT_M3"
                if not m2_present and not m3_present:
                    status = "ABSENT_BOTH"
                results.append({
                    "file"       : fp.name,
                    "pair"       : f"{m2_id}-{m3_id}",
                    "m2_voxels"  : int(np.sum(mask_m2)),
                    "m3_voxels"  : int(np.sum(mask_m3)),
                    "dmin_mm"    : None,
                    "class"      : status,
                    "time_s"     : round(time.time() - t_file, 3),
                })
                continue

            # Hitung D_min
            t0 = time.time()
            try:
                dist_mm, coord_m2, coord_m3 = compute_dmin(mask_m2, mask_m3, spacing_zyx)
                cls = classify_dmin(dist_mm)
                elapsed = round(time.time() - t0, 3)

                results.append({
                    "file"       : fp.name,
                    "pair"       : f"{m2_id}-{m3_id}",
                    "m2_voxels"  : int(np.sum(mask_m2)),
                    "m3_voxels"  : int(np.sum(mask_m3)),
                    "dmin_mm"    : round(dist_mm, 4),
                    "class"      : cls,
                    "coord_m2"   : coord_m2.tolist(),
                    "coord_m3"   : coord_m3.tolist(),
                    "time_s"     : elapsed,
                })

                if verbose:
                    tag = "INDEP " if cls == "INDEPENDENT" else "KONTAK"
                    print(f"[{i:3d}/{total_files}] {fp.name} [{m2_id}-{m3_id}]  "
                          f"D_min={dist_mm:6.3f} mm  [{tag}]  ({elapsed:.2f}s)")

            except Exception as e:
                if verbose:
                    print(f"[{i:3d}/{total_files}] ERROR D_min {fp.name} [{m2_id}-{m3_id}]: {e}")
                results.append({
                    "file": fp.name, "pair": f"{m2_id}-{m3_id}",
                    "dmin_mm": None, "class": f"ERROR: {e}", "time_s": 0
                })

    return results


def print_summary(results: list[dict]) -> None:
    """Cetak ringkasan statistik ke terminal."""
    valid = [r for r in results if r.get("dmin_mm") is not None]
    absent = [r for r in results if r.get("class", "").startswith("ABSENT")]
    independent = [r for r in valid if r["class"] == "INDEPENDENT"]
    contact_zone = [r for r in valid if r["class"] == "CONTACT_OR_RESORPTION"]

    dmin_vals = [r["dmin_mm"] for r in valid]

    print(f"\n{'='*70}")
    print(f"  RINGKASAN HASIL")
    print(f"{'='*70}")
    print(f"  Total pasangan dievaluasi : {len(results)}")
    print(f"  Gigi tidak ditemukan      : {len(absent)}")
    print(f"  Pasangan valid (dihitung) : {len(valid)}")
    print(f"")
    print(f"  D_min > {DMIN_THRESHOLD_MM} mm  -> INDEPENDENT        : "
          f"{len(independent):4d} ({100*len(independent)/len(valid):.1f}%)")
    print(f"  D_min <= {DMIN_THRESHOLD_MM} mm -> CONTACT/RESORPTION : "
          f"{len(contact_zone):4d} ({100*len(contact_zone)/len(valid):.1f}%)")
    print(f"")
    if dmin_vals:
        print(f"  Statistik D_min (mm):")
        print(f"    Min    : {min(dmin_vals):.4f} mm")
        print(f"    Max    : {max(dmin_vals):.4f} mm")
        print(f"    Mean   : {np.mean(dmin_vals):.4f} mm")
        print(f"    Median : {np.median(dmin_vals):.4f} mm")
        print(f"    Std    : {np.std(dmin_vals):.4f} mm")
    print(f"{'='*70}\n")


def save_csv(results: list[dict], output_path: Path) -> None:
    """Simpan hasil ke file CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "pair", "m2_voxels", "m3_voxels", "dmin_mm", "class", "time_s"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✓ Hasil disimpan ke: {output_path}")


def plot_histogram(results: list[dict], output_path: Path | None = None) -> None:
    """Plot distribusi D_min dalam mm."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib tidak tersedia, skip plot.")
        return

    valid = [r for r in results if r.get("dmin_mm") is not None]
    dmin_vals = [r["dmin_mm"] for r in valid]
    if not dmin_vals:
        return

    colors = ["#2ecc71" if v > DMIN_THRESHOLD_MM else "#e74c3c" for v in dmin_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(dmin_vals)), sorted(dmin_vals), color=sorted(colors), width=1.0)
    ax.axhline(DMIN_THRESHOLD_MM, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold {DMIN_THRESHOLD_MM} mm")
    ax.set_xlabel("Pasangan Gigi (diurutkan berdasar D_min)")
    ax.set_ylabel("D_min (mm)")
    ax.set_title("Distribusi D_min Pasangan M2-M3 — ToothFairy2")

    indep_patch = mpatches.Patch(color="#2ecc71", label="INDEPENDENT (>0.5 mm)")
    cont_patch  = mpatches.Patch(color="#e74c3c", label="CONTACT/RESORPTION (≤0.5 mm)")
    ax.legend(handles=[indep_patch, cont_patch,
                        plt.Line2D([0],[0], color="black", linestyle="--", label=f"Threshold {DMIN_THRESHOLD_MM} mm")])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  ✓ Plot disimpan ke: {output_path}")
    else:
        plt.show()


# ── Entry Point CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hitung D_min pasangan M2-M3 dari ToothFairy2")
    parser.add_argument("--n",     default="50",        help="Jumlah file (angka atau 'all')")
    parser.add_argument("--pairs", default="maxillary", choices=["maxillary","mandibular","all"],
                        help="Pasangan FDI yang dihitung")
    parser.add_argument("--out",   default="data/processed/dmin_results.csv",
                        help="Path output CSV")
    parser.add_argument("--plot",  action="store_true", help="Tampilkan histogram")
    parser.add_argument("--no-verbose", action="store_true", help="Sembunyikan progress per file")
    args = parser.parse_args()

    # Resolve jumlah file
    n = None if args.n.lower() == "all" else int(args.n)

    # Resolve pairs
    pair_map = {"maxillary": PAIR_MAXILLARY, "mandibular": PAIR_MANDIBULAR, "all": PAIR_ALL}
    pairs = pair_map[args.pairs]

    # Jalankan scan
    results = scan_dataset(
        labels_dir=LABELS_DIR,
        n_files=n,
        pairs=pairs,
        verbose=not args.no_verbose
    )

    # Ringkasan
    print_summary(results)

    # Simpan CSV
    out_path = Path(r"d:\Workspace\TA3") / args.out
    save_csv(results, out_path)

    # Plot opsional
    if args.plot:
        plot_path = out_path.parent / "dmin_histogram.png"
        plot_histogram(results, plot_path)
