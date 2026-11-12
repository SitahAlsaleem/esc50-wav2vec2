"""Run the FROZEN Wav2Vec 2.0 backbone once and cache its frame features.

Usage: python cache_features.py --root ESC-50 --layer 12
Output: cache/w2v_L<layer>_{train,val,test}.pt  (features fp16: N x 249 x 768)

--layer 12 = last transformer layer. Earlier layers (e.g. 6-9) often transfer better
to non-speech audio, so a cheap extra experiment is to repeat with --layer 8.
"""
import argparse
from pathlib import Path

import torch
from transformers import Wav2Vec2Model

from data.esc50 import TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS, ESC50Dataset
from utils.seed import set_seed

SPLITS = {"train": TRAIN_FOLDS, "val": VAL_FOLDS, "test": TEST_FOLDS}


@torch.no_grad()
def extract(model, waves, layer, device, batch_size=16):
    chunks = []
    for i in range(0, len(waves), batch_size):
        x = waves[i:i + batch_size].to(device)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True) + 1e-7)
        hidden = model(x, output_hidden_states=True).hidden_states[layer]
        chunks.append(hidden.half().cpu())
    return torch.cat(chunks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="ESC-50")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--model_name", default="facebook/wav2vec2-base")
    args = p.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device).eval()
    for param in model.parameters():          # frozen backbone
        param.requires_grad = False

    Path(args.cache_dir).mkdir(exist_ok=True)
    for split, folds in SPLITS.items():
        ds = ESC50Dataset(args.root, folds, feature="wave", augment=False)
        feats = extract(model, ds.waves, args.layer, device)
        out = Path(args.cache_dir) / f"w2v_L{args.layer}_{split}.pt"
        torch.save({"feats": feats, "labels": ds.labels, "filenames": ds.filenames}, out)
        print(f"{split}: {tuple(feats.shape)} -> {out}")


if __name__ == "__main__":
    main()
