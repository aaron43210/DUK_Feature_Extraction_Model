"""
Tiled inference engine for SVAMITVA segmentation outputs.

Runs SegFormer+UPerFPN in a sliding-window fashion over large
GeoTIFF orthophotos or standard images (JPG/PNG).

Outputs per-pixel probability maps for:
  building_mask, roof_type_mask, road_mask, road_centerline_mask,
  waterbody_mask, waterbody_line_mask, utility_line_mask
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Preprocessing helpers ──────────────────────────────────────────────────────


def _percentile_stretch(
    image: np.ndarray, limits: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    image = image.astype(np.float32)
    vmin, vmax = np.percentile(image, limits)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    return np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)


def _to_rgb(tile: np.ndarray) -> np.ndarray:
    if tile.ndim != 3:
        raise ValueError(f"Expected HxWxC tile, got {tile.shape}")
    c = tile.shape[2]
    if c == 1:
        return np.repeat(tile, 3, axis=2)
    if c == 2:
        return np.concatenate([tile, tile[:, :, :1]], axis=2)
    if c > 3:
        return tile[:, :, :3]
    return tile


def _gaussian_kernel_2d(size: int, sigma: float = 0.0) -> np.ndarray:
    if sigma <= 0:
        sigma = size / 4
    x        = np.arange(size) - size / 2 + 0.5
    k1d      = np.exp(-0.5 * (x / sigma) ** 2)
    k2d      = np.outer(k1d, k1d)
    return k2d / k2d.max()


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x  = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(np.clip(x, -50, 50))
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-8)


# ── Checkpoint loading helpers ─────────────────────────────────────────────────


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            v = ckpt.get(key)
            if isinstance(v, dict):
                return dict(v)
        return dict(ckpt)
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _strip_prefixes(state: Dict[str, Any]) -> Dict[str, Any]:
    for prefix in ("module.", "model."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


# ── Tiled predictor ────────────────────────────────────────────────────────────


class TiledPredictor:
    """
    Sliding-window inference over large rasters.

    Applies Gaussian-weighted blending at tile boundaries to eliminate
    visible seams in the probability maps.
    """

    BINARY_KEYS: List[str] = [
        "building_mask",
        "road_mask",
        "road_centerline_mask",
        "waterbody_mask",
        "waterbody_line_mask",
        "utility_line_mask",
    ]
    ROOF_KEY = "roof_type_mask"
    ALL_KEYS: List[str] = BINARY_KEYS + [ROOF_KEY]

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        tile_size: int = 512,
        overlap: int = 192,
        threshold: float = 0.5,
        use_tta: bool = False,
    ):
        self.model      = model.to(device).eval()
        self.device     = device
        self.tile_size  = tile_size
        self.overlap    = overlap
        self.threshold  = threshold
        self.use_tta    = use_tta

        self.blend_kernel = _gaussian_kernel_2d(tile_size).astype(np.float32)
        self._tta_dims: Sequence[Tuple[int, ...]] = (
            [(), (3,), (2,), (2, 3)] if use_tta else [()]
        )

    # ── Normalization ──────────────────────────────────────────────────────────

    def _normalize(self, tile: np.ndarray) -> torch.Tensor:
        rgb    = _to_rgb(tile)
        rgb    = _percentile_stretch(rgb)
        rgb    = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb    = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        return tensor.unsqueeze(0).to(self.device)

    # ── Model forward (with optional TTA) ─────────────────────────────────────

    def _forward(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        n = len(self._tta_dims)
        for dims in self._tta_dims:
            aug   = torch.flip(tensor, dims=list(dims)) if dims else tensor
            out   = self.model(aug, task="all")
            for k, v in out.items():
                if not isinstance(v, torch.Tensor):
                    continue
                restored = torch.flip(v, dims=list(dims)) if dims else v
                merged[k] = merged[k] + restored if k in merged else restored.clone()
        for k in list(merged):
            merged[k] = merged[k] / float(n)
        return merged

    def _predict_tile(self, tile_img: np.ndarray) -> Dict[str, np.ndarray]:
        tensor = self._normalize(tile_img)
        with torch.no_grad():
            outputs = self._forward(tensor)
        result: Dict[str, np.ndarray] = {}
        for k, v in outputs.items():
            arr        = v.detach().cpu().numpy()
            result[k]  = arr[0] if arr.ndim >= 3 else arr
        return result

    # ── Valid-data thumbnail for skipping empty tiles ──────────────────────────

    def _valid_mask(self, tif_path: Path) -> np.ndarray:
        try:
            with rasterio.open(str(tif_path)) as src:
                h, w  = src.height, src.width
                scale = min(1024.0 / max(h, w), 1.0)
                th, tw = max(1, int(h * scale)), max(1, int(w * scale))
                thumb  = src.read(
                    out_shape=(src.count, th, tw),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                return np.any(thumb > 0, axis=0)
        except Exception as exc:
            logger.warning("Thumbnail scan failed: %s", exc)
            return np.ones((1, 1), dtype=bool)

    # ── GeoTIFF inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_tif(
        self,
        tif_path: Path,
        selected_masks: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        selected = set(selected_masks or self.ALL_KEYS) & set(self.ALL_KEYS)
        if not selected:
            selected = set(self.ALL_KEYS)

        valid_thumb = self._valid_mask(tif_path)

        with rasterio.open(str(tif_path)) as src:
            h, w = src.height, src.width
            logger.info("Predicting %s (%dx%d)", tif_path.name, w, h)

            th_h, th_w = valid_thumb.shape
            scale_y, scale_x = th_h / h, th_w / w

            model_accum = {
                k: np.zeros((h, w), dtype=np.float32)
                for k in self.BINARY_KEYS if k in selected
            }
            roof_accum = (
                np.zeros((5, h, w), dtype=np.float32)
                if self.ROOF_KEY in selected else None
            )
            weight_map = np.zeros((h, w), dtype=np.float32)

            stride  = max(1, self.tile_size - self.overlap)
            windows = [
                (x, y, min(self.tile_size, w - x), min(self.tile_size, h - y))
                for y in range(0, h, stride)
                for x in range(0, w, stride)
                if valid_thumb[
                    min(int(y * scale_y), th_h - 1),
                    min(int(x * scale_x), th_w - 1),
                ]
            ]

            for idx, (x0, y0, tw_act, th_act) in enumerate(
                tqdm(windows, desc="Inference", leave=False)
            ):
                if progress_callback:
                    progress_callback(idx, len(windows))

                win      = Window(x0, y0, self.tile_size, self.tile_size)
                part     = src.read(window=win, boundless=True, fill_value=0)
                tile_img = np.transpose(part, (1, 2, 0))

                if float(np.any(tile_img[:th_act, :tw_act] > 0, axis=2).mean()) < 0.01:
                    continue

                blend = self.blend_kernel[:th_act, :tw_act]
                weight_map[y0:y0+th_act, x0:x0+tw_act] += blend

                tile_out = self._predict_tile(tile_img)

                for k in list(model_accum):
                    if k not in tile_out:
                        continue
                    logits = tile_out[k]
                    if logits.ndim == 3 and logits.shape[0] == 1:
                        logits = logits[0]
                    elif logits.ndim != 2:
                        continue
                    prob = _sigmoid_np(logits[:th_act, :tw_act])
                    model_accum[k][y0:y0+th_act, x0:x0+tw_act] += prob * blend

                if roof_accum is not None and self.ROOF_KEY in tile_out:
                    r = tile_out[self.ROOF_KEY]
                    if r.ndim == 3 and r.shape[0] >= 2:
                        rp = _softmax_np(r[:, :th_act, :tw_act], axis=0)
                        roof_accum[:, y0:y0+th_act, x0:x0+tw_act] += rp * blend[None]

        weight_map = np.maximum(weight_map, 1e-8)
        final: Dict[str, Any] = {k: v / weight_map for k, v in model_accum.items()}

        if roof_accum is not None:
            roof_probs = roof_accum / weight_map[None]
            roof_mask  = np.argmax(roof_probs, axis=0).astype(np.uint8)
            if "building_mask" in final:
                from inference.postprocess import refine_roof_types
                bld_bin   = (final["building_mask"] > self.threshold).astype(np.uint8)
                roof_mask = refine_roof_types(bld_bin, roof_mask)
                roof_mask[final["building_mask"] <= self.threshold] = 0
            final[self.ROOF_KEY] = roof_mask

        for k in selected:
            if k not in final:
                dtype     = np.uint8 if k == self.ROOF_KEY else np.float32
                final[k]  = np.zeros((h, w), dtype=dtype)

        return final

    # ── Standard image inference ───────────────────────────────────────────────

    @torch.no_grad()
    def predict_image(
        self,
        image_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ext = image_path.suffix.lower()
        if ext in {".tif", ".tiff"}:
            return self.predict_tif(image_path, selected_masks)

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required: pip install Pillow")

        selected = set(selected_masks or self.ALL_KEYS) & set(self.ALL_KEYS)
        if not selected:
            selected = set(self.ALL_KEYS)

        pil_img    = Image.open(image_path).convert("RGB")
        full_image = np.array(pil_img, dtype=np.float32) / 255.0
        h, w       = full_image.shape[:2]
        logger.info("Predicting %s (%dx%d)", image_path.name, w, h)

        model_accum = {
            k: np.zeros((h, w), dtype=np.float32)
            for k in self.BINARY_KEYS if k in selected
        }
        roof_accum = (
            np.zeros((5, h, w), dtype=np.float32) if self.ROOF_KEY in selected else None
        )
        weight_map = np.zeros((h, w), dtype=np.float32)

        stride  = max(1, self.tile_size - self.overlap)
        windows = [
            (x, y, min(self.tile_size, w - x), min(self.tile_size, h - y))
            for y in range(0, h, stride)
            for x in range(0, w, stride)
        ]

        for x0, y0, tw_act, th_act in tqdm(windows, desc="Inference", leave=False):
            crop = full_image[y0:y0+th_act, x0:x0+tw_act]
            if float(crop.mean()) < 0.01:
                continue

            tile_img = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.float32)
            tile_img[:th_act, :tw_act] = crop

            blend = self.blend_kernel[:th_act, :tw_act]
            weight_map[y0:y0+th_act, x0:x0+tw_act] += blend

            tile_out = self._predict_tile(tile_img)

            for k in list(model_accum):
                if k not in tile_out:
                    continue
                logits = tile_out[k]
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits[0]
                elif logits.ndim != 2:
                    continue
                prob = _sigmoid_np(logits[:th_act, :tw_act])
                model_accum[k][y0:y0+th_act, x0:x0+tw_act] += prob * blend

            if roof_accum is not None and self.ROOF_KEY in tile_out:
                r = tile_out[self.ROOF_KEY]
                if r.ndim == 3 and r.shape[0] >= 2:
                    for c in range(min(5, r.shape[0])):
                        roof_accum[c][y0:y0+th_act, x0:x0+tw_act] += r[c][:th_act, :tw_act] * blend

        safe_w = np.maximum(weight_map, 1e-6)
        final: Dict[str, Any] = {
            k: (v / safe_w >= self.threshold).astype(np.uint8)
            for k, v in model_accum.items()
        }

        if roof_accum is not None:
            for c in range(5):
                roof_accum[c] /= safe_w
            roof_mask = np.argmax(_softmax_np(roof_accum, axis=0), axis=0).astype(np.uint8)
            if "building_mask" in final:
                from inference.postprocess import refine_roof_types
                bld_bin   = (final["building_mask"] > 0.5).astype(np.uint8)
                roof_mask = refine_roof_types(bld_bin, roof_mask)
                roof_mask[final["building_mask"] <= 0.5] = 0
            final[self.ROOF_KEY] = roof_mask

        for k in selected:
            if k not in final:
                dtype    = np.uint8 if k == self.ROOF_KEY else np.float32
                final[k] = np.zeros((h, w), dtype=dtype)

        return final


# ── Pipeline loader ────────────────────────────────────────────────────────────


def _find_weights(weights_path: str) -> Optional[Path]:
    p = Path(weights_path)
    if p.exists():
        return p
    for cand in [Path("check/best.pt"), Path("checkpoints/best.pt")]:
        if cand.exists():
            logger.warning("Weights not found at %s; using fallback %s", p, cand)
            return cand
    return None


def load_segmentation_pipeline(
    weights_path: str,
    device: torch.device = torch.device("cuda"),
    use_tta: bool = False,
    tile_size: int = 512,
    overlap: int = 192,
) -> TiledPredictor:
    from models.model import EnsembleDUKModel

    model = EnsembleDUKModel(pretrained=False)

    resolved = _find_weights(weights_path)
    if resolved is None:
        raise FileNotFoundError(
            f"No segmentation checkpoint found at '{weights_path}'. "
            "Train the model first or point --weights to a valid .pt file."
        )

    ckpt       = torch.load(resolved, map_location="cpu", weights_only=False)
    state_dict = _strip_prefixes(_extract_state_dict(ckpt))
    info       = model.load_state_dict(state_dict, strict=False)

    total  = len(model.state_dict())
    loaded = total - len(info.missing_keys)
    ratio  = loaded / max(total, 1)

    if ratio < 0.80:
        raise RuntimeError(
            f"Checkpoint appears incompatible: loaded {loaded}/{total} keys "
            f"({ratio:.1%}). Check that the checkpoint was saved from this architecture."
        )

    if info.missing_keys or info.unexpected_keys:
        logger.warning(
            "Partial load: %d missing, %d unexpected keys (%.1f%% loaded)",
            len(info.missing_keys),
            len(info.unexpected_keys),
            ratio * 100,
        )

    logger.info("Loaded weights from %s (%.1f%% matched)", resolved, ratio * 100)

    return TiledPredictor(
        model=model,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        use_tta=use_tta,
    )


# Backwards-compat alias used by older scripts
load_ensemble_pipeline = load_segmentation_pipeline
