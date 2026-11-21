"""Plot convergence curves from results/history/<run>.json -> results/convergence.png

    python plot_history.py                       # seed 42, three main models
    python plot_history.py --seed 7
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
runs = {"Mel-CNN": f"mel_cnn_seed{a.seed}", "W2V linear": f"w2v_linear_L12_seed{a.seed}",
        "W2V attention": f"w2v_attn_L12_seed{a.seed}"}
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), dpi=300, sharex=False)
for label, run in runs.items():
    f = Path("results/history") / f"{run}.json"
    if not f.exists():
        print("missing", f); continue
    h = json.loads(f.read_text()); ep = [r["epoch"] for r in h]
    axes[0].plot(ep, [r["train_acc"] for r in h], "--", lw=0.9)
    axes[0].plot(ep, [r["val_acc"] for r in h], lw=1.2, label=label)
    axes[1].plot(ep, [r["val_loss"] for r in h], lw=1.2, label=label)
axes[0].set_title("Accuracy (dashed = train, solid = validation)", fontsize=7)
axes[1].set_title("Validation loss", fontsize=7)
for ax in axes:
    ax.set_xlabel("Epoch", fontsize=7); ax.tick_params(labelsize=6)
axes[0].legend(fontsize=6, frameon=False)
plt.tight_layout(); Path("results").mkdir(exist_ok=True); plt.savefig("results/convergence.png")
print("wrote results/convergence.png")
