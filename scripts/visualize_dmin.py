"""
scripts/visualize_dmin.py
Visualisasi distribusi D_min dari hasil CSV scanner.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, tidak butuh display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

CSV_PATH  = Path(r"d:\Workspace\TA3\data\processed\dmin_results.csv")
OUT_PATH  = Path(r"d:\Workspace\TA3\data\processed\dmin_visual.png")
THRESHOLD = 0.5

# ── Baca CSV ──────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

valid  = [r for r in rows if r["dmin_mm"]]
absent = [r for r in rows if not r["dmin_mm"]]

dmin_vals  = [float(r["dmin_mm"]) for r in valid]
labels     = [r["class"] for r in valid]
files      = [r["file"].replace("ToothFairy2F_","#").replace(".mha","") + " [" + r["pair"] + "]" for r in valid]

indep  = [v for v in dmin_vals if v >  THRESHOLD]
kontak = [v for v in dmin_vals if v <= THRESHOLD]

# ── Warna per titik ───────────────────────────────────────────────────────────
colors = ["#27ae60" if v > THRESHOLD else "#e74c3c" for v in dmin_vals]

# ── Layout: 3 panel ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Distribusi D_min Pasangan M2-M3 — ToothFairy2 (Maxillary)", 
             fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs[0, :])   # top: scatter per pasangan
ax2 = fig.add_subplot(gs[1, 0])   # bottom-left: histogram frekuensi
ax3 = fig.add_subplot(gs[1, 1])   # bottom-right: pie chart

# ── Panel 1: D_min per pasangan gigi (scatter + hline) ───────────────────────
x = range(len(dmin_vals))
ax1.scatter(x, sorted(dmin_vals), c=sorted(colors, key=lambda c: dmin_vals[sorted(colors).index(c)] if c in sorted(colors) else 0), 
            s=80, zorder=3, edgecolors="white", linewidths=0.5)

# Rebuild properly
sorted_data = sorted(zip(dmin_vals, colors, files))
sv, sc, sf = zip(*sorted_data)

ax1.bar(range(len(sv)), sv, color=sc, width=0.8, alpha=0.85, zorder=2)
ax1.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, zorder=4,
            label=f"Threshold {THRESHOLD} mm")
ax1.set_xlabel("Pasangan Gigi (diurutkan dari D_min terkecil)", fontsize=10)
ax1.set_ylabel("D_min (mm)", fontsize=10)
ax1.set_title(f"D_min Per Pasangan — {len(dmin_vals)} pasangan valid dari {len(rows)} total", fontsize=11)
ax1.set_xticks(range(len(sv)))
ax1.set_xticklabels(sf, rotation=45, ha="right", fontsize=7)
ax1.set_ylim(0, max(sv) * 1.2)

# Annotasi zona
ax1.axhspan(0, THRESHOLD, alpha=0.06, color="#e74c3c", zorder=1)
ax1.axhspan(THRESHOLD, max(sv)*1.2, alpha=0.06, color="#27ae60", zorder=1)
ax1.text(len(sv)*0.02, THRESHOLD*0.5, "Zona Kontak/Resorpsi", 
         color="#c0392b", fontsize=8, va="center")
ax1.text(len(sv)*0.02, THRESHOLD*1.08, "Zona Independent", 
         color="#1e8449", fontsize=8, va="bottom")

patch_i = mpatches.Patch(color="#27ae60", label=f"INDEPENDENT (>{THRESHOLD} mm)  n={len(indep)}")
patch_k = mpatches.Patch(color="#e74c3c", label=f"CONTACT/RESORPTION (<={THRESHOLD} mm)  n={len(kontak)}")
ax1.legend(handles=[patch_i, patch_k,
                    plt.Line2D([0],[0], color="black", linestyle="--", label=f"Threshold {THRESHOLD} mm")],
           fontsize=8, loc="upper left")

# ── Panel 2: Histogram frekuensi ──────────────────────────────────────────────
bins = np.arange(0, max(dmin_vals) + 0.15, 0.05)
n_indep, _, _ = ax2.hist(indep,  bins=bins, color="#27ae60", alpha=0.8, label="INDEPENDENT", edgecolor="white")
n_kont,  _, _ = ax2.hist(kontak, bins=bins, color="#e74c3c", alpha=0.8, label="CONTACT/RESORPTION", edgecolor="white")
ax2.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5)
ax2.set_xlabel("D_min (mm)", fontsize=10)
ax2.set_ylabel("Jumlah Pasangan", fontsize=10)
ax2.set_title("Histogram Frekuensi D_min", fontsize=11)
ax2.legend(fontsize=8)

# Annotasi statistik
stats_txt = (f"Min : {min(dmin_vals):.3f} mm\n"
             f"Max : {max(dmin_vals):.3f} mm\n"
             f"Mean: {np.mean(dmin_vals):.3f} mm\n"
             f"Med : {np.median(dmin_vals):.3f} mm\n"
             f"Std : {np.std(dmin_vals):.3f} mm")
ax2.text(0.97, 0.97, stats_txt, transform=ax2.transAxes, fontsize=8,
         va="top", ha="right", family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ── Panel 3: Pie chart status ─────────────────────────────────────────────────
n_absent_m3   = sum(1 for r in absent if r["class"] == "ABSENT_M3")
n_absent_m2   = sum(1 for r in absent if r["class"] == "ABSENT_M2")
n_absent_both = sum(1 for r in absent if r["class"] == "ABSENT_BOTH")

pie_vals   = [len(indep), len(kontak), n_absent_m3, n_absent_m2, n_absent_both]
pie_labels = [
    f"INDEPENDENT\n({len(indep)} pasang)",
    f"KONTAK/RESORPSI\n({len(kontak)} pasang)",
    f"M3 Absen\n({n_absent_m3} pasang)",
    f"M2 Absen\n({n_absent_m2} pasang)",
    f"Keduanya Absen\n({n_absent_both} pasang)",
]
pie_colors = ["#27ae60", "#e74c3c", "#95a5a6", "#bdc3c7", "#ecf0f1"]
pie_exp    = [0.05, 0.05, 0, 0, 0]

wedges, texts, autotexts = ax3.pie(
    [v for v in pie_vals if v > 0],
    labels=[l for l,v in zip(pie_labels, pie_vals) if v > 0],
    colors=[c for c,v in zip(pie_colors, pie_vals) if v > 0],
    explode=[e for e,v in zip(pie_exp, pie_vals) if v > 0],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 8}
)
ax3.set_title(f"Komposisi Seluruh Pasangan\n(total {len(rows)} entri)", fontsize=11)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, tidak butuh display
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Visual disimpan ke: {OUT_PATH}")
