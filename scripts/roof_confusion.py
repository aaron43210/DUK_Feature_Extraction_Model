#!/usr/bin/env python3
"""
Roof-type confusion matrix and per-class precision / recall / F1.

Rebuilds the validation split used in training (random 20% tile-level split,
seed 42, as in run_ddp.sh) and runs the checkpoint over it. Pixels are counted
the same way as the training validation loop (all pixels, background included),
and a second matrix excludes NoData pixels via valid_mask.

Usage (on the machine that holds the imagery):
    python scripts/roof_confusion.py --train_dirs ../DATA/MAP1 ... ../DATA/MAP9 \
        --checkpoint checkpoint/best_inference.pt --out results/roof_confusion
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import create_dataloaders  # noqa: E402
from inference.predict import _extract_state_dict, _strip_prefixes  # noqa: E402
from models.model import EnsembleDUKModel  # noqa: E402

CLASSES = ["Background", "RCC", "Tiled", "Tin", "Other"]


def parse_args():
    p = argparse.ArgumentParser("Roof-type confusion matrix")
    p.add_argument("--train_dirs", nargs="+", required=True, help="All nine MAP directories")
    p.add_argument("--checkpoint", default="checkpoint/best_inference.pt")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--split_mode", default="tile", choices=["tile", "map"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="results/roof_confusion")
    return p.parse_args()


def scores(cm: np.ndarray):
    """Per-class precision, recall, F1 from a confusion matrix (rows = true, cols = predicted)."""
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1


def write_outputs(cm: np.ndarray, out: Path, tag: str):
    prec, rec, f1 = scores(cm)
    with open(out / f"confusion_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + CLASSES)
        for name, row in zip(CLASSES, cm):
            w.writerow([name] + [int(v) for v in row])
    with open(out / f"scores_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "pixels"])
        for i, name in enumerate(CLASSES):
            w.writerow([name, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(cm[i].sum())])
        roof = slice(1, None)
        w.writerow(["macro (roof classes)", f"{prec[roof].mean():.4f}",
                    f"{rec[roof].mean():.4f}", f"{f1[roof].mean():.4f}", int(cm[roof].sum())])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(norm * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(5), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j] * 100:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\n[{tag}] per-class scores (%)")
    for i, name in enumerate(CLASSES):
        print(f"  {name:10s}  P {prec[i] * 100:5.1f}  R {rec[i] * 100:5.1f}  F1 {f1[i] * 100:5.1f}")


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = create_dataloaders(
        train_dirs=[Path(d) for d in args.train_dirs],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    model = EnsembleDUKModel(pretrained=False, num_roof_classes=5)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(_strip_prefixes(_extract_state_dict(ckpt)), strict=True)
    model.float().to(device).eval()

    cm_all = torch.zeros(5, 5, dtype=torch.long, device=device)
    cm_valid = torch.zeros(5, 5, dtype=torch.long, device=device)
    for n, batch in enumerate(val_loader, 1):
        pred = model(batch["image"].to(device))["roof_type_mask"].argmax(dim=1)
        true = batch["roof_type_mask"].to(device).long().clamp(0, 4)
        if true.ndim == 4:
            true = true.squeeze(1)
        valid = batch["valid_mask"].to(device) > 0.5
        if valid.ndim == 4:
            valid = valid.squeeze(1)
        idx = true * 5 + pred
        cm_all += torch.bincount(idx.flatten(), minlength=25).view(5, 5)
        cm_valid += torch.bincount(idx[valid], minlength=25).view(5, 5)
        if n % 100 == 0:
            print(f"{n}/{len(val_loader)} batches")

    cm_all, cm_valid = cm_all.cpu().numpy(), cm_valid.cpu().numpy()
    assert cm_all.sum() >= cm_valid.sum() > 0, "no pixels were counted"
    write_outputs(cm_all, out, "all_pixels")
    write_outputs(cm_valid, out, "valid_pixels")
    # Same definition as the training validation loop, so this can be checked against the
    # per-class recall stored in the checkpoint (98.9 / 98.5 / 92.2 / 96.1 / 84.9 %).
    print("\nRecall check vs checkpoint metrics:",
          " ".join(f"{r * 100:.1f}" for r in scores(cm_all)[1]))


if __name__ == "__main__":
    main()
