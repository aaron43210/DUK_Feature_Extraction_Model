#!/usr/bin/env python3
"""
Inference benchmark for the TerraX checkpoint.

Times one forward pass of a 512x512 tile in FP32 and FP16 and reports seconds
per tile, peak device memory, and seconds per km^2 of orthophoto for a given
GSD and tile overlap.

Usage:
    python scripts/benchmark.py --checkpoint checkpoint/best_inference.pt
"""

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.predict import _extract_state_dict, _strip_prefixes  # noqa: E402
from models.model import EnsembleDUKModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("TerraX inference benchmark")
    p.add_argument("--checkpoint", default="checkpoint/best_inference.pt")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--gsd_cm", type=float, nargs="+", default=[3.04, 3.50])
    p.add_argument("--overlap", type=int, default=192, help="Tile overlap in px")
    p.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"],
                   help="Run one dtype per process for a clean memory reading on MPS")
    return p.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        # MPS has no peak counter; driver-allocated memory after the run is the closest proxy.
        return torch.mps.driver_allocated_memory() / 2**20
    return float("nan")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"Apple GPU (MPS), {platform.machine()}"
    return platform.processor() or "CPU"


@torch.no_grad()
def time_model(model, device, dtype, batch, warmup, iters):
    x = torch.randn(batch, 3, 512, 512, device=device, dtype=dtype)
    for _ in range(warmup):
        model(x)
    sync(device)
    reset_peak(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        sync(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return median / batch, peak_mb(device)


def tiles_per_km2(gsd_cm: float, overlap: int) -> float:
    stride_m = (512 - overlap) * gsd_cm / 100.0
    return 1e6 / stride_m**2


def main():
    args = parse_args()
    device = pick_device()

    model = EnsembleDUKModel(pretrained=False, num_roof_classes=5)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(_strip_prefixes(_extract_state_dict(ckpt)), strict=True)
    model.eval()

    print(f"Device: {device_name(device)} | torch {torch.__version__} | batch {args.batch_size}")
    print(f"Tile 512x512, overlap {args.overlap} px (stride {512 - args.overlap} px)")
    rows = []
    chosen = {"fp32": (torch.float32, "FP32"), "fp16": (torch.float16, "FP16")}
    for dtype, label in (chosen[d] for d in args.dtypes):
        if device.type == "cpu" and dtype == torch.float16:
            continue  # FP16 convolutions are not a meaningful CPU benchmark
        model.to(device=device, dtype=dtype)
        sec, mem = time_model(model, device, dtype, args.batch_size, args.warmup, args.iters)
        rows.append((label, sec, mem))
        print(f"{label}: {sec * 1000:.1f} ms/tile, peak memory {mem:.0f} MB")
        for gsd in args.gsd_cm:
            n = tiles_per_km2(gsd, args.overlap)
            print(f"    GSD {gsd:.2f} cm: {n:,.0f} tiles/km^2 -> {sec * n / 60:.1f} min/km^2")

    assert all(math.isfinite(r[1]) and r[1] > 0 for r in rows), "benchmark produced invalid timings"


if __name__ == "__main__":
    main()
