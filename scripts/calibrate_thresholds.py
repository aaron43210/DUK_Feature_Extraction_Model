#!/usr/bin/env python3
"""
Calibrate per-task binary thresholds on validation tiles to maximize IoU.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import create_dataloaders
from models.model import EnsembleSvamitvaModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("calibrate_thresholds")

BINARY_TASKS = [
    "building",
    "road",
    "road_centerline",
    "waterbody",
    "waterbody_line",
    "waterbody_point",
    "utility_line",
    "utility_point",
    "bridge",
    "railway",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate per-task segmentation thresholds."
    )
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--train_dirs", nargs="+", default=["data/MAP1"])
    p.add_argument("--val_dir", default=None)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--split_mode", choices=["map", "tile"], default="map")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--tile_overlap", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_thr", type=float, default=0.2)
    p.add_argument("--max_thr", type=float, default=0.8)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--sam2_checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument(
        "--sam2_model_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    p.add_argument("--out", default="checkpoints/thresholds.json")
    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()


def _device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = EnsembleSvamitvaModel(
        pretrained=True,
        checkpoint_path=args.sam2_checkpoint,
        model_cfg=args.sam2_model_cfg,
    ).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _init_stats(thresholds: List[float]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for task in BINARY_TASKS:
        out[task] = {
            "tp": [0.0 for _ in thresholds],
            "fp": [0.0 for _ in thresholds],
            "fn": [0.0 for _ in thresholds],
        }
    return out


@torch.no_grad()
def evaluate_thresholds(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    thresholds: List[float],
    device: torch.device,
) -> Dict[str, Dict[str, List[float]]]:
    stats = _init_stats(thresholds)

    for batch in val_loader:
        image = batch["image"].to(device)
        valid_mask = batch.get("valid_mask")
        valid_mask_t = (
            (valid_mask.to(device) > 0.5)
            if isinstance(valid_mask, torch.Tensor)
            else torch.ones_like(image[:, :1], dtype=torch.bool).squeeze(1)
        )

        preds = model(image, task="all")
        for task in BINARY_TASKS:
            key = f"{task}_mask"
            if key not in preds or key not in batch:
                continue

            prob = torch.sigmoid(preds[key]).squeeze(1)
            tgt = batch[key].to(device).float()
            if tgt.ndim == 4:
                tgt = tgt.squeeze(1)
            tgt_bin = tgt > 0.5
            valid = valid_mask_t

            for i, thr in enumerate(thresholds):
                pred_bin = prob > thr
                tp = (pred_bin & tgt_bin & valid).sum().item()
                fp = (pred_bin & (~tgt_bin) & valid).sum().item()
                fn = ((~pred_bin) & tgt_bin & valid).sum().item()
                stats[task]["tp"][i] += float(tp)
                stats[task]["fp"][i] += float(fp)
                stats[task]["fn"][i] += float(fn)

    return stats


def summarize(
    thresholds: List[float], stats: Dict[str, Dict[str, List[float]]]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    best_thresholds: Dict[str, float] = {}
    details: Dict[str, Dict[str, float]] = {}

    for task in BINARY_TASKS:
        best_iou = -1.0
        best_thr = 0.5
        for i, thr in enumerate(thresholds):
            tp = stats[task]["tp"][i]
            fp = stats[task]["fp"][i]
            fn = stats[task]["fn"][i]
            iou = tp / (tp + fp + fn + 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_thr = thr

        best_thresholds[f"{task}_mask"] = round(float(best_thr), 4)
        details[f"{task}_mask"] = {
            "best_iou": float(best_iou),
            "threshold": float(best_thr),
        }

    return best_thresholds, details


def main() -> None:
    args = parse_args()
    thresholds: List[float] = []
    t = args.min_thr
    while t <= args.max_thr + 1e-9:
        thresholds.append(round(float(t), 4))
        t += args.step

    device = _device(args.force_cpu)
    logger.info("Using device: %s", device)

    _, val_loader = create_dataloaders(
        train_dirs=[Path(d) for d in args.train_dirs],
        val_dir=Path(args.val_dir) if args.val_dir else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        val_split=args.val_split,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    model = _load_model(args, device=device)
    stats = evaluate_thresholds(model, val_loader, thresholds, device=device)
    best_thresholds, details = summarize(thresholds, stats)

    out = {
        "thresholds": best_thresholds,
        "details": details,
        "search": {
            "min": args.min_thr,
            "max": args.max_thr,
            "step": args.step,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved calibrated thresholds to %s", out_path)


if __name__ == "__main__":
    main()
