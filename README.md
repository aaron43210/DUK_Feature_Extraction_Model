# SVAMITVA Feature Extraction
**Digital University Kerala — Hackathon Submission**

Semantic segmentation pipeline for extracting geospatial features from SVAMITVA drone orthophotos. Targets buildings, roads, waterbodies, and roof-type classification at high precision using a SegFormer-B4 backbone with a UPerNet-style FPN decoder.

---

## Architecture

```
Input GeoTIFF / Image
        │
        ▼
  Tiled Inference (512×512, 192 px overlap, Gaussian blend)
        │
        ▼
  SegFormer-B4 Encoder  (nvidia/segformer-b4-finetuned-cityscapes-1024-1024)
  Multi-scale features: S/4 · S/8 · S/16 · S/32
        │
        ▼
  UPerFPN Decoder  (lateral convs → top-down fusion → CBAM → 256-ch feature map)
        │
   ┌────┴──────────────────────────┐
   │   Task Heads                  │
   │   BuildingHead  → building_mask + roof_type_mask  │
   │   BinaryHead    → road_mask, waterbody_mask        │
   │   LineHead      → road_centerline, waterbody_line  │
   │   LineHead      → utility_line_mask                │
   └───────────────────────────────┘
        │
        ▼
  Post-processing  (Lovász-IoU thresholds · morphological clean-up · FER)
        │
        ▼
  GeoPackage / Shapefile export (QGIS / ArcGIS ready)
```

### Encoder — SegFormer-B4

Mix-Transformer backbone pre-trained on ImageNet-1K and Cityscapes. Outputs four hierarchical feature maps without positional-encoding coupling, which makes it robust to the variable ground-sampling distances found in SVAMITVA orthophotos.

| Feature map | Stride | Resolution (512 input) | Channels |
|-------------|--------|------------------------|----------|
| feat_s1     | 4      | 128 × 128              | 64       |
| feat_s2     | 8      | 64 × 64                | 128      |
| feat_s3     | 16     | 32 × 32                | 320      |
| feat_s4     | 32     | 16 × 16                | 512      |

### Decoder — UPerFPN + CBAM

Lateral 1×1 convolutions collapse all four encoder channels to 256. A top-down pathway propagates global context to fine-resolution maps with bilinear interpolation. A CBAM block refines both channel and spatial attention before all levels are upsampled to H/4, concatenated, and compressed back to 256 channels.

### Task heads

| Head         | Type        | Output                          |
|--------------|-------------|---------------------------------|
| BuildingHead | Shared trunk + dual 1×1 | `(B,1,H,W)` binary + `(B,5,H,W)` roof class |
| BinaryHead   | Conv + residual skip    | `(B,1,H,W)` for roads and waterbodies |
| LineHead     | D-LinkNet (dilated 1,2,4,8) | `(B,1,H,W)` for centrelines and utility lines |

### Loss

- **Lovász-Hinge** — differentiable surrogate for Jaccard/IoU; directly optimises the competition metric.
- **Focal Loss** — down-weights easy background pixels so the model focuses on object boundaries.
- **Boundary Loss** — Sobel-gradient penalty on mask edges to counter the "blobby" tendency of ViT decoders.

---

## Output layers

| Key                    | Type        | Description                         |
|------------------------|-------------|-------------------------------------|
| `building_mask`        | float32 map | Building footprint probability      |
| `roof_type_mask`       | uint8 class | 0=bg 1=RCC 2=Tiled 3=Tin 4=Others  |
| `road_mask`            | float32 map | Road polygon probability            |
| `road_centerline_mask` | float32 map | Road centreline probability         |
| `waterbody_mask`       | float32 map | Waterbody polygon probability       |
| `waterbody_line_mask`  | float32 map | Waterbody shoreline probability     |
| `utility_line_mask`    | float32 map | Overhead utility line probability   |

---

## Setup

```bash
git clone <repo-url>
cd FEATURE
pip install -r requirements.txt
```

Python 3.10+ and a CUDA GPU are required.

---

## Training

### Single node (auto GPU)

```bash
python train.py \
    --train_dirs data/MAP1 data/MAP2 ... \
    --epochs 150 \
    --batch_size 8
```

### DGX / multi-GPU (recommended)

The `run_ddp.sh` script uses `torchrun` with elastic GPU discovery. It automatically detects free GPUs (>20 GB VRAM), sets DDP environment variables, and restarts on preemption.

```bash
bash run_ddp.sh data/MAP1 data/MAP2 ...
```

Key settings in `run_ddp.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EPOCHS` | 150 | Total training epochs |
| `PER_GPU_BATCH` | 12 | Batch size per GPU |
| `WORKERS` | 2 | DataLoader workers per GPU |

### Resume after crash

```bash
python train.py --resume --train_dirs data/MAP1 ...
```

The trainer saves `check/latest.pt` after every epoch and `check/best.pt` whenever validation IoU improves. The DDP loop in `run_ddp.sh` will automatically restart and resume from `latest.pt`.

### Quick smoke test

```bash
python train.py --quick_test --train_dirs data/MAP1
```

Runs 3 epochs at 256×256 with batch size 2 — useful for verifying the environment before a full run.

---

## Checkpoints

| File | Contents |
|------|----------|
| `check/best.pt` | Best validation IoU — use for inference |
| `check/latest.pt` | Last completed epoch — use for resume |
| `check/best_inference.pt` | FP16 inference-only copy (~120 MB) |

---

## Inference

### Streamlit dashboard

```bash
streamlit run app.py
```

Upload a GeoTIFF or JPG/PNG, select the feature layers to extract, and click **Run Extraction**. Results are shown as colour-coded overlays. GIS export produces a ZIP of `.gpkg` or `.shp` files ready for QGIS/ArcGIS.

### Programmatic

```python
import torch
from inference.predict import load_segmentation_pipeline

predictor = load_segmentation_pipeline(
    weights_path="check/best.pt",
    device=torch.device("cuda"),
    tile_size=512,
    overlap=192,
)
results = predictor.predict_tif("orthophoto.tif")
# results["building_mask"]  → np.float32 probability map
# results["roof_type_mask"] → np.uint8 class map
```

---

## Repository structure

```
.
├── app.py                          # Streamlit dashboard
├── train.py                        # Training entry point
├── run_ddp.sh                      # Multi-GPU DDP launcher (DGX)
├── models/
│   ├── model.py                    # EnsembleDUKModel (encoder + decoder + heads)
│   ├── segformer_encoder.py        # HuggingFace SegFormer wrapper
│   ├── decoder.py                  # UPerFPN + CBAM
│   ├── heads.py                    # Task-specific prediction heads
│   └── losses.py                   # Lovász + Focal + Boundary losses
├── train_engine/
│   ├── trainer.py                  # Training loop (AMP, DDP, early stopping)
│   ├── config.py                   # TrainingConfig dataclass
│   ├── metrics.py                  # IoU / Dice / Roof accuracy
│   └── train_segmentation.py       # CLI entry point for torchrun
├── inference/
│   ├── predict.py                  # TiledPredictor + load_segmentation_pipeline
│   ├── postprocess.py              # Mask refinement, FER orthogonalisation
│   ├── export.py                   # GeoPackage / Shapefile vectorisation
│   └── fer.py                      # Feature Edge Reconstruction
└── scripts/
    ├── evaluate.py                 # Standalone validation script
    ├── calibrate_thresholds.py     # Per-class threshold search
    └── class_balance_analysis.py   # Dataset statistics
```

---

## Post-processing

**Buildings** — Douglas-Peucker simplification, dominant-angle extraction, frame-field snapping to force 90° corners (Feature Edge Reconstruction). Produces clean rectangular and L-shaped footprints equivalent to PolyMapper output without requiring a separate network.

**Roads** — Morphological closing (7 px kernel) to bridge canopy gaps. Centreline extracted with `skan` skeleton pruning and Chaikin corner-cutting.

**Waterbodies** — Large morphological closing (9 px) for smooth shorelines; convex hull for small isolated ponds.

---

*Developed by students of Digital University Kerala for the SVAMITVA Feature Extraction Hackathon.*
