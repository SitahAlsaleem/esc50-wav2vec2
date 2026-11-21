"""Layer-wise linear probe of the FROZEN Wav2Vec 2.0 backbone (extra analysis).

For each of the 13 hidden states (0 = transformer input, 1-12 = transformer layers):
mean+std pooling over time -> StandardScaler -> LogisticRegression.
Fit on train folds (1-3), choose layer (and C) on validation fold 4,
and report the test fold 5 score of the validation-selected setting only.

Usage: python layer_sweep.py --root ESC-50            (about 5-10 minutes on a T4)
       python layer_sweep.py --root ESC-50 --Cs 0.01 0.1 1.0   (also tunes C)
Outputs: cache/w2v_pooled_all_layers.npz, results/layer_sweep.csv,
         results/layer_sweep_best.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import Wav2Vec2Model

from data.esc50 import TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS, ESC50Dataset
from utils.seed import set_seed


@torch.no_grad()
def pooled_features(model, waves, device, batch_size=16):
    """(N, 80000) waveforms -> (N, 13, 1536): mean and std over time for every layer."""
    out = []
    for i in tqdm(range(0, len(waves), batch_size)):
        x = waves[i:i + batch_size].to(device)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True) + 1e-7)
        hs = torch.stack(model(x, output_hidden_states=True).hidden_states)  # (13,B,T,768)
        feats = torch.cat([hs.mean(2), hs.std(2)], dim=-1)                   # (13,B,1536)
        out.append(feats.permute(1, 0, 2).cpu())
    return torch.cat(out).numpy()


def load_or_extract(args, device):
    cache = Path(args.cache_dir) / "w2v_pooled_all_layers.npz"
    if cache.exists():
        return dict(np.load(cache))
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    for p in model.parameters():
        p.requires_grad = False                                   # frozen backbone
    data = {}
    for split, folds in (("train", TRAIN_FOLDS), ("val", VAL_FOLDS), ("test", TEST_FOLDS)):
        ds = ESC50Dataset(args.root, folds, feature="wave", augment=False)
        data[f"X_{split}"] = pooled_features(model, ds.waves, device)
        data[f"y_{split}"] = ds.labels.numpy()
    Path(args.cache_dir).mkdir(exist_ok=True)
    np.savez(cache, **data)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="ESC-50")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--Cs", type=float, nargs="+", default=[0.1])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_or_extract(args, device)
    y_tr, y_va, y_te = d["y_train"], d["y_val"], d["y_test"]

    rows = []
    for layer in range(d["X_train"].shape[1]):
        scaler = StandardScaler().fit(d["X_train"][:, layer])      # fit on train only
        Xtr = scaler.transform(d["X_train"][:, layer])
        Xva = scaler.transform(d["X_val"][:, layer])
        Xte = scaler.transform(d["X_test"][:, layer])
        for C in args.Cs:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=args.seed).fit(Xtr, y_tr)
            pred_te = clf.predict(Xte)
            row = dict(
                layer=layer, C=C,
                val_acc=accuracy_score(y_va, clf.predict(Xva)),
                test_acc=accuracy_score(y_te, pred_te),
                test_macro_f1=f1_score(y_te, pred_te, average="macro"),
                test_roc_auc=roc_auc_score(y_te, clf.predict_proba(Xte),
                                           multi_class="ovr", average="macro"),
            )
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/layer_sweep.csv", index=False)
    best = res.loc[res.val_acc.idxmax()]          # chosen on VALIDATION only
    summary = {k: float(v) for k, v in best.items()}
    Path("results/layer_sweep_best.json").write_text(json.dumps(summary, indent=2))
    print("\nBest (selected on validation):", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
