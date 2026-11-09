"""Plot Comprehensive Training Curves for 2-Class Vision Transformer Ensemble.

Generates:
1. Training Loss vs Validation Loss (Gradient Descent Curves) for all 3 models.
2. Validation Accuracy and Macro-F1 curves.
3. Overfitting Gap Analysis (Train Loss vs Val Loss divergence).
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIST_PATH = Path("reports/binary_ensemble_training_history.json")
OUT_IMG = Path("reports/binary_ensemble_curves.png")

with open(HIST_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

models = ["swin_s_mha", "swin_small", "vit_base_16"]
titles = ["Swin-Small + MHA(head)", "Swin-Small (Standard)", "ViT-Base/16 (Standard)"]
colors = ["#38bdf8", "#10b981", "#f59e0b"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0a0c16")

for col, (m_key, title, colr) in enumerate(zip(models, titles, colors)):
    hist = data[m_key]["history"]
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_loss = [h["val_loss"] for h in hist]
    val_acc = [h["val_acc"] * 100 for h in hist]
    val_f1 = [h["val_f1"] * 100 for h in hist]
    best_ep = data[m_key]["best_epoch"]

    # 1. Row 1: Loss Curve (Gradient Descent)
    ax_loss = axes[0, col]
    ax_loss.set_facecolor("#111528")
    ax_loss.plot(epochs, train_loss, "o-", color="#ef4444", lw=2, label="Train Loss (CrossEntropy)")
    ax_loss.plot(epochs, val_loss, "s--", color=colr, lw=2, label="Val Loss")
    ax_loss.axvline(best_ep, color="#facc15", linestyle=":", lw=1.8, label=f"Best Checkpoint (Ep {best_ep})")
    ax_loss.set_title(f"{title}\nLoss / Gradient Descent", color="white", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax_loss.set_ylabel("Loss", color="#94a3b8", fontsize=9)
    ax_loss.tick_params(colors="#cbd5e1")
    ax_loss.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
    ax_loss.legend(loc="upper right", facecolor="#1e2540", edgecolor="#334155", labelcolor="white", fontsize=8)

    # 2. Row 2: Accuracy & F1-Score
    ax_acc = axes[1, col]
    ax_acc.set_facecolor("#111528")
    ax_acc.plot(epochs, val_acc, "d-", color="#38bdf8", lw=2, label="Validation Accuracy (%)")
    ax_acc.plot(epochs, val_f1, "^--", color="#34d399", lw=2, label="Validation Macro-F1 (%)")
    ax_acc.axvline(best_ep, color="#facc15", linestyle=":", lw=1.8, label=f"Best Model (Ep {best_ep})")
    ax_acc.set_title(f"{title}\nAccuracy & Macro-F1", color="white", fontsize=11, fontweight="bold")
    ax_acc.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax_acc.set_ylabel("Percentage (%)", color="#94a3b8", fontsize=9)
    ax_acc.set_ylim(50, 95)
    ax_acc.tick_params(colors="#cbd5e1")
    ax_acc.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
    ax_acc.legend(loc="lower right", facecolor="#1e2540", edgecolor="#334155", labelcolor="white", fontsize=8)

plt.suptitle("Kurva Pembelajaran (Loss Gradient Descent, Akurasi, dan F1) Model Ensemble 2-Kelas",
             color="white", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_IMG, dpi=160, facecolor="#0a0c16")
print(f"Curves chart saved to: {OUT_IMG}")
