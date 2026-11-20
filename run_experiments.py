"""Resumable experiment runner for the paper (main results, ablation, tuning).

    python run_experiments.py --stages main ablation tune --mirror /content/drive/MyDrive/esc50_results

Every run is a call to train.py with --run_name; a run whose results/exp/<name>.json
already exists is skipped, so the script can be restarted after a Colab disconnect.
--mirror copies results/ (and best checkpoints) to a second folder (e.g. Drive) after each run.
Test-set numbers are recorded for every run, but tuning choices are made on VALIDATION accuracy only.
"""
import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 7, 123, 2024, 31]


def runs(stage, seeds, layers):
    r = []  # (run_name, extra args)
    if stage == "main":
        for sd in seeds:
            r.append((f"mel_cnn_seed{sd}", ["--model", "mel_cnn", "--seed", str(sd)]))
            for m in ("w2v_linear", "w2v_attn"):
                r.append((f"{m}_L12_seed{sd}", ["--model", m, "--layer", "12", "--seed", str(sd)]))
    elif stage == "ablation":
        for sd in seeds:
            r.append((f"w2v_uniform_L12_seed{sd}", ["--model", "w2v_uniform", "--layer", "12", "--seed", str(sd)]))
            r.append((f"mel_cnn_noaug_seed{sd}", ["--model", "mel_cnn", "--no_augment", "--seed", str(sd)]))
            for m in ("w2v_linear", "w2v_attn"):
                r.append((f"{m}_L8_seed{sd}", ["--model", m, "--layer", "8", "--seed", str(sd)]))
    elif stage == "tune":  # small grid, single seed 42, head models on layer 12
        for m, lr, dp in itertools.product(("w2v_linear", "w2v_attn"), (1e-3, 3e-4, 1e-4), (0.3, 0.5)):
            r.append((f"tune_{m}_lr{lr:g}_dp{dp:g}",
                      ["--model", m, "--layer", "12", "--seed", "42", "--lr", str(lr), "--dropout", str(dp)]))
    return r


def mirror(dst):
    dst = Path(dst)
    for sub in ("results", "checkpoints"):
        src = Path(sub)
        if src.exists():
            shutil.copytree(src, dst / sub, dirs_exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["main"], choices=["main", "ablation", "tune"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--root", default="ESC-50")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--mirror", default=None)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    todo = [x for st in a.stages for x in runs(st, a.seeds, None)]
    print(f"{len(todo)} runs planned")
    for i, (name, extra) in enumerate(todo, 1):
        if Path("results/exp", f"{name}.json").exists():
            print(f"[{i}/{len(todo)}] skip {name} (done)")
            continue
        cmd = [sys.executable, "train.py", "--run_name", name, "--root", a.root,
               "--cache_dir", a.cache_dir] + extra
        print(f"[{i}/{len(todo)}] {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        if subprocess.run(cmd).returncode != 0:
            print(f"!! {name} failed, continuing")
            continue
        if a.mirror:
            mirror(a.mirror)
    print("done")


if __name__ == "__main__":
    main()
