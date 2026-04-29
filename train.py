#!/usr/bin/env python3
"""
SVAMITVA Feature Extraction — Segmentation Training Entry Point
Digital University Kerala (DUK)

Orchestrates SegFormer+UPerFPN training for buildings, roads,
waterbodies, and roof-type classification.

Usage:
    # Single-node, auto GPU selection
    python train.py --train_dirs data/MAP1 data/MAP2 --epochs 150

    # DGX / multi-GPU (recommended)
    bash run_ddp.sh data/MAP1 data/MAP2 ...

    # Resume from last checkpoint
    python train.py --resume --train_dirs data/MAP1
"""

import argparse
import logging
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="SVAMITVA Segmentation Training")

    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=[],
        help="Directories containing MAP*.tif + shapefiles",
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--resume", action="store_true", help="Resume from latest.pt")
    p.add_argument("--checkpoint_dir", default="check")
    p.add_argument("--name", default="segmentation_v1")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--quick_test",
        action="store_true",
        help="3-epoch smoke test with small batch",
    )
    return p.parse_args()


def _check_disk(project_root: Path, min_gb: int = 10):
    _, _, free = shutil.disk_usage(project_root)
    free_gb = free // (2 ** 30)
    if free_gb < min_gb:
        logger.error(
            "Low disk space: %d GB free, need at least %d GB. "
            "Free up space and retry.",
            free_gb,
            min_gb,
        )
        sys.exit(1)
    logger.info("Disk check passed: %d GB free", free_gb)


def _run(cmd: List[str], label: str):
    logger.info("Starting: %s", label)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Step '%s' failed: %s", label, e)
        sys.exit(1)
    logger.info("Done: %s (%.1f min)", label, (time.time() - t0) / 60)


def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    project_root = Path(__file__).resolve().parent

    logger.info("=" * 60)
    logger.info("SVAMITVA Segmentation Training")
    logger.info("  Train dirs : %s", args.train_dirs)
    logger.info("  Epochs     : %d", args.epochs)
    logger.info("  Batch size : %d", args.batch_size)
    logger.info("=" * 60)

    def _handle_signal(sig, _frame):
        logger.warning("Received signal %d, shutting down.", sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _check_disk(project_root)

    check_dir = project_root / args.checkpoint_dir
    check_dir.mkdir(parents=True, exist_ok=True)

    ddp_script = project_root / "run_ddp.sh"
    if ddp_script.exists():
        logger.info("Delegating to run_ddp.sh for multi-GPU launch")
        seg_cmd = ["bash", str(ddp_script)] + [str(d) for d in args.train_dirs]
    else:
        seg_cmd = [
            sys.executable,
            str(project_root / "train_engine" / "train_segmentation.py"),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--train_dirs",
        ] + args.train_dirs + [
            "--checkpoint_dir", args.checkpoint_dir,
            "--name", args.name,
            "--lr", str(args.lr),
            "--tile_size", str(args.tile_size),
            "--num_workers", str(args.num_workers),
        ]
        if args.resume:
            seg_cmd.append("--resume")
        if args.quick_test:
            seg_cmd.append("--quick_test")

    _run(seg_cmd, "Segmentation")

    # Consolidate best weights into a predictable location
    candidates = [
        check_dir / "best.pt",
        check_dir / args.name / "best.pt",
    ]
    for src in candidates:
        if src.exists():
            dest = check_dir / "segmentation_best.pt"
            shutil.copy2(src, dest)
            logger.info("Best weights saved to: %s", dest)
            break

    logger.info("=" * 60)
    logger.info("Training complete. Checkpoints in: %s", check_dir.resolve())
    logger.info("Launch the dashboard with: streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
