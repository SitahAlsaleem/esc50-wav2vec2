"""Train and evaluate a model on ESC-50 (official folds: train 1-3, val 4, test 5).

Usage (Colab):
    python train.py --model mel_cnn --root ESC-50
Outputs:
    runs/<model>_seed<seed>/          TensorBoard logs
    checkpoints/<model>_seed<seed>.pt best weights (chosen on the VALIDATION fold)
    results/<model>_seed<seed>.json   final test metrics
The test fold is used exactly once, at the end.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from data.cached import CachedFeatures
from data.esc50 import NUM_CLASSES, make_datasets, make_loaders
from models.mel_cnn import MelCNN
from models.wav2vec_heads import AttentionHead, LinearHead
from utils.seed import set_seed


def build_model(name: str) -> nn.Module:
    if name == "mel_cnn":
        return MelCNN(num_classes=NUM_CLASSES)
    if name == "w2v_linear":
        return LinearHead(num_classes=NUM_CLASSES)
    if name == "w2v_attn":
        return AttentionHead(num_classes=NUM_CLASSES)
    raise ValueError(f"Unknown model: {name}")


def is_w2v(name: str) -> bool:
    return name.startswith("w2v")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            n += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mel_cnn",
                   choices=["mel_cnn", "w2v_linear", "w2v_attn"])
    p.add_argument("--layer", type=int, default=12, help="cached Wav2Vec layer")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--root", default="ESC-50")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = (f"{args.model}_L{args.layer}_seed{args.seed}" if is_w2v(args.model)
           else f"{args.model}_seed{args.seed}")
    for d in ("runs", "checkpoints", "results"):
        Path(d).mkdir(exist_ok=True)

    if is_w2v(args.model):
        cache = Path(args.cache_dir)
        train_ds, val_ds, test_ds = (
            CachedFeatures(cache / f"w2v_L{args.layer}_{s}.pt")
            for s in ("train", "val", "test")
        )
        workers = 0
    else:
        train_ds, val_ds, test_ds = make_datasets(args.root, feature="mel")
        workers = 2
    train_dl, val_dl, test_dl = make_loaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, seed=args.seed,
        num_workers=workers,
    )

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | model: {args.model} | trainable params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(f"runs/{tag}")

    best_val_acc, best_epoch, bad_epochs = 0.0, 0, 0
    ckpt_path = Path("checkpoints") / f"{tag}.pt"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, val_dl, criterion, device)
        scheduler.step()

        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("accuracy", {"train": tr_acc, "val": va_acc}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    # Final evaluation on the held-out test fold, using the best-validation weights.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    te_loss, te_acc, preds, labels = run_epoch(model, test_dl, criterion, device)
    te_f1 = f1_score(labels, preds, average="macro")
    results = {
        "model": args.model, "layer": args.layer if is_w2v(args.model) else None,
        "seed": args.seed, "trainable_params": n_params,
        "best_epoch": best_epoch, "best_val_acc": best_val_acc,
        "test_acc": te_acc, "test_macro_f1": te_f1,
        "train_minutes": round((time.time() - start) / 60, 2),
    }
    Path("results", f"{tag}.json").write_text(json.dumps(results, indent=2))
    writer.add_hparams({"model": args.model, "lr": args.lr, "seed": args.seed},
                       {"hparam/test_acc": te_acc, "hparam/test_macro_f1": te_f1})
    writer.close()
    print("\nFINAL", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
