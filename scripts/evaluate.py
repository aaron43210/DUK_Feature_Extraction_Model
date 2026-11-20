#!/usr/bin/env python3
"""
SVAMITVA Validation & Accuracy Report Generator
Computes IoU, Dice, and Pixel Accuracy on a trained model over a validation dataset.

Usage:
    python scripts/evaluate.py --val_dirs ../DATA/MAP1 --checkpoint checkpoint/best_inference.pt
"""

import argparse
import logging
import torch
from pathlib import Path

from train_engine.config import TrainingConfig
from data.augmentation import get_val_transforms
from data.dataset import SvamitvaDataset
from train_engine.trainer import Trainer
from models.model import EnsembleDUKModel
from inference.predict import _extract_state_dict, _strip_prefixes
from models.losses import MultiTaskLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

def parse_args():
    p = argparse.ArgumentParser("Generate Validation Report")
    p.add_argument("--val_dirs", nargs="+", required=True, help="Validation MAP directories")
    p.add_argument("--checkpoint", type=str, default="checkpoint/best_inference.pt", help="Path to checkpoint")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainingConfig()
    cfg.batch_size = args.batch_size

    # Every tile of the given maps is evaluated; no split is applied.
    logger.info(f"Loading datasets from {args.val_dirs}...")
    val_ds = SvamitvaDataset(
        [Path(d) for d in args.val_dirs],
        image_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        transform=get_val_transforms(cfg.tile_size),
        mode="val",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = EnsembleDUKModel(pretrained=False, num_roof_classes=cfg.num_roof_classes)
    logger.info(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(_strip_prefixes(_extract_state_dict(ckpt)), strict=True)
    model.float().eval()

    trainer = Trainer(model, val_loader, val_loader, MultiTaskLoss(), cfg)

    logger.info("Running evaluation...")
    val_loss, metrics = trainer._validate_epoch(epoch=0)

    logger.info("=" * 50)
    logger.info(f"Loss: {val_loss:.4f}")
    for k in sorted(metrics):
        logger.info(f"  {k.ljust(30)}: {metrics[k] * 100:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
