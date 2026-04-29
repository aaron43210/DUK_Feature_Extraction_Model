#!/usr/bin/env python3
"""
SVAMITVA Validation & Accuracy Report Generator
Computes IoU, Dice, and Pixel Accuracy on a trained model over a validation dataset.

Usage:
    python scripts/evaluate.py --val_dirs ../DATA/MAP1 --checkpoint check/segmentation_best.pt
"""

import argparse
import logging
import torch
import numpy as np
from pathlib import Path

from train_engine.config import TrainingConfig
from data.dataset import create_dataloaders
from train_engine.trainer import Trainer
from models.model import EnsembleSvamitvaModel
from models.losses import MultiTaskLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

def parse_args():
    p = argparse.ArgumentParser("Generate Validation Report")
    p.add_argument("--val_dirs", nargs="+", required=True, help="Validation MAP directories")
    p.add_argument("--checkpoint", type=str, default="check/segmentation_best.pt", help="Path to best.pt")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainingConfig()
    cfg.batch_size = args.batch_size
    
    logger.info(f"Loading datasets from {args.val_dirs}...")
    
    # We load only the validation dataloader by mapping train_dirs to a dummy 
    # and validating on val_dir, or just use val_dirs directly.
    # Dataset splits:
    _, val_loader = create_dataloaders(
        train_dirs=[Path(d) for d in args.val_dirs],
        val_dir=None,
        val_split=0.99, # Treat almost all as validation
        batch_size=args.batch_size,
        image_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        num_workers=cfg.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Init Model
    model = EnsembleSvamitvaModel(
        pretrained=False,
        num_roof_classes=cfg.num_roof_classes
    )
    
    logger.info(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    model.to(device)
    model.eval()
    
    loss_fn = MultiTaskLoss(device=device)
    
    # Dummy Trainer just for _validate_epoch usage
    trainer = Trainer(model, val_loader, val_loader, loss_fn, cfg)
    trainer.device = device
    
    logger.info("Running evaluation over validation set...")
    val_loss, val_iou, val_dice = trainer._validate_epoch(epoch=0)
    
    logger.info("\n" + "="*50)
    logger.info("FINAL VALIDATION & ACCURACY REPORT")
    logger.info("="*50)
    logger.info(f"Overall Validation Loss: {val_loss:.4f}")
    
    logger.info("\n--- Intersection over Union (IoU) ---")
    for k, v in val_iou.items():
        logger.info(f"  - {k.ljust(25)}: {v*100:.2f}%")
        
    logger.info("\n--- Dice Coefficient (F1 Score) ---")
    for k, v in val_dice.items():
        logger.info(f"  - {k.ljust(25)}: {v*100:.2f}%")
    logger.info("="*50)

if __name__ == "__main__":
    main()