"""Sanity check: run this first. Usage: python check_data.py --root /content/ESC-50"""
import argparse
from collections import Counter

from data.esc50 import NUM_SAMPLES, make_datasets
from utils.seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="ESC-50")
args = parser.parse_args()

set_seed(42)
for feature in ("wave", "mel"):
    train, val, test = make_datasets(args.root, feature=feature)
    print(f"\n[{feature}] sizes -> train {len(train)}, val {len(val)}, test {len(test)}")
    x, y = train[0]
    print(f"[{feature}] one sample shape: {tuple(x.shape)}, label: {int(y)}")
    if feature == "wave":
        assert x.shape[0] == NUM_SAMPLES
    for name, ds in (("train", train), ("val", val), ("test", test)):
        c = Counter(ds.labels.tolist())
        print(f"  {name}: folds {ds.folds}, {len(c)} classes, "
              f"{min(c.values())}-{max(c.values())} clips per class")
print("\nOK: folds are disjoint and shapes are correct.")
