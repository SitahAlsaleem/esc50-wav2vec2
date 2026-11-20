"""Collect results/exp/*.json into paper-ready tables (mean ± std over seeds).

    python aggregate_results.py            -> results/summary_main.csv, summary_ablation.csv,
                                              tuning_log.csv, tables.md
Std is the sample std (ddof=1). Tuning rows are ranked by VALIDATION accuracy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

rows = [json.loads(f.read_text()) for f in sorted(Path("results/exp").glob("*.json"))]
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results in results/exp yet")
df["name"] = df["run_name"]
tune = df[df.name.str.startswith("tune_")].copy()
main = df[~df.name.str.startswith("tune_")].copy()

def config(r):
    if r.model == "mel_cnn":
        return "Mel-CNN" + ("" if r.augment is not False else " (no augmentation)")
    base = {"w2v_linear": "W2V linear", "w2v_uniform": "W2V uniform-weight", "w2v_attn": "W2V attention"}[r.model]
    return f"{base} (layer {int(r.layer)})"

main["config"] = main.apply(config, axis=1)
cols = ["test_acc", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall",
        "best_val_acc", "train_acc_at_best", "best_epoch", "train_minutes"]
g = main.groupby("config")
summary = g[cols].agg(["mean", "std"])
summary["n_seeds"] = g.size()
summary["trainable_params"] = g["trainable_params"].first()
summary.columns = ["_".join(c).strip("_") for c in summary.columns]
summary.to_csv("results/summary_all.csv")

def fmt(m, s): return f"{100*m:.1f} ± {100*s:.1f}"
lines = ["| Configuration | Params | Seeds | Test acc (%) | Macro-F1 (%) | Val acc (%) | Train acc (%) |",
         "|---|---|---|---|---|---|---|"]
order = ["Mel-CNN", "W2V linear (layer 12)", "W2V uniform-weight (layer 12)", "W2V attention (layer 12)",
         "Mel-CNN (no augmentation)", "W2V linear (layer 8)", "W2V attention (layer 8)"]
for c in [o for o in order if o in summary.index] + [c for c in summary.index if c not in order]:
    r = summary.loc[c]
    lines.append(f"| {c} | {int(r.trainable_params):,} | {int(r.n_seeds)} | "
                 f"{fmt(r.test_acc_mean, r.test_acc_std)} | {fmt(r.test_macro_f1_mean, r.test_macro_f1_std)} | "
                 f"{fmt(r.best_val_acc_mean, r.best_val_acc_std)} | "
                 f"{fmt(r.train_acc_at_best_mean, r.train_acc_at_best_std)} |")
out = ["## Main + ablation results (mean ± std over seeds)", "", *lines, ""]

if not tune.empty:
    tune = tune.sort_values(["model", "best_val_acc"], ascending=[True, False])
    tune[["run_name", "model", "lr", "dropout", "best_val_acc", "best_epoch", "test_acc"]].to_csv(
        "results/tuning_log.csv", index=False)
    out += ["## Tuning log (seed 42; choose by validation accuracy)", "",
            "| Model | LR | Dropout | Val acc (%) | Best epoch | Test acc (%) |", "|---|---|---|---|---|---|"]
    for r in tune.itertuples():
        out.append(f"| {r.model} | {r.lr:g} | {r.dropout:g} | {100*r.best_val_acc:.1f} | {r.best_epoch} | {100*r.test_acc:.1f} |")
    out.append("")
Path("results/tables.md").write_text("\n".join(out))
print("\n".join(out))
