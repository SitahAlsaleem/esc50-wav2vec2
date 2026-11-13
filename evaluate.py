"""Evaluate a trained model on the held-out TEST fold (fold 5).

Usage:
    python evaluate.py --model mel_cnn   --root ESC-50
    python evaluate.py --model w2v_linear --layer 12
    python evaluate.py --model w2v_attn   --layer 12
Outputs:
    results/eval_<tag>.json   accuracy, macro/weighted F1, precision, recall, ROC-AUC
    results/cm_<tag>.png      50x50 confusion matrix
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from data.cached import CachedFeatures
from data.esc50 import TEST_FOLDS, ESC50Dataset
from train import build_model, is_w2v
from utils.seed import set_seed


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mel_cnn", "w2v_linear", "w2v_attn"])
    p.add_argument("--root", default="ESC-50")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", default="cache")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = (f"{args.model}_L{args.layer}_seed{args.seed}" if is_w2v(args.model)
           else f"{args.model}_seed{args.seed}")

    if is_w2v(args.model):
        ds = CachedFeatures(Path(args.cache_dir) / f"w2v_L{args.layer}_test.pt")
    else:
        ds = ESC50Dataset(args.root, TEST_FOLDS, feature="mel", augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(Path("checkpoints") / f"{tag}.pt", map_location=device))

    probs, labels = get_probs(model, loader, device)
    preds = probs.argmax(axis=1)

    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None

    meta = pd.read_csv(Path(args.root) / "meta" / "esc50.csv")
    class_names = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    cm = confusion_matrix(labels, preds, labels=list(range(50)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    results = {
        "model": args.model, "tag": tag, "n_test": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": auc,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results", f"eval_{tag}.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(50)); ax.set_yticks(range(50))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test fold 5): {tag}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(Path("results") / f"cm_{tag}.png", dpi=150)

    print(json.dumps(results, indent=2))
    order = np.argsort(per_class_acc)
    print("\nHardest classes:", ", ".join(f"{class_names[i]} ({per_class_acc[i]:.0%})" for i in order[:5]))
    print("Easiest classes:", ", ".join(f"{class_names[i]} ({per_class_acc[i]:.0%})" for i in order[::-1][:5]))


if __name__ == "__main__":
    main()
