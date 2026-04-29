"""
SVAMITVA Feature Extraction Dashboard
Digital University Kerala (DUK)
"""

import atexit
import io
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
import torch
from inference.predict import load_segmentation_pipeline
from inference.postprocess import refine_mask, get_threshold

st.set_page_config(
    page_title="DUK — SVAMITVA Feature Extraction",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; color: #e0e0e6; }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Inter', sans-serif;
                  font-weight: 700; letter-spacing: -0.025em; }
    .stButton>button {
        background: linear-gradient(90deg, #1d6fa4 0%, #1a9ed4 100%);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 158, 212, 0.4);
    }
    [data-testid="stSidebar"] {
        background-color: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .stMetric {
        background: rgba(255,255,255,0.05); padding: 1rem;
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

FEATURES = {
    "building_mask":          ("Built-up Area",             (255, 100,  50)),
    "roof_type_mask":         ("Roof Classification",        (255,   0, 180)),
    "road_mask":              ("Road (Polygon)",             (255, 255, 100)),
    "road_centerline_mask":   ("Road Centre Line",          (255, 220,   0)),
    "waterbody_mask":         ("Water Body",                ( 50, 150, 255)),
    "waterbody_line_mask":    ("Waterbody Line",            (  0, 200, 255)),
    "utility_line_mask":      ("Utility Lines",             ( 50, 220, 100)),
}

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required — CPU execution is not supported.")
DEVICE = torch.device("cuda")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

_temp_files: list = []


def _cleanup_temp_files():
    for f in _temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass


atexit.register(_cleanup_temp_files)


def get_best_ckpt() -> str:
    candidates = [
        "check/best.pt",
        "check/segmentation_best.pt",
        "check/best_latest.pt",
        "checkpoints/best.pt",
        "best.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]


def main():
    st.title("SVAMITVA Feature Extraction — DUK")

    with st.sidebar:
        st.header("Model")
        ckpt_path = st.text_input("Segmentation Weights", get_best_ckpt())

        st.divider()
        st.subheader("Extraction Layers")
        selected_masks = []
        for key, (name, _) in FEATURES.items():
            if st.checkbox(name, value=True, key=f"feat_{key}"):
                selected_masks.append(key)

        st.divider()
        st.subheader("Legend")
        for key, (name, color) in FEATURES.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
                f'<div style="width:14px;height:14px;background:rgb{color};'
                f'border-radius:3px;margin-right:8px;border:1px solid rgba(255,255,255,0.2);"></div>'
                f'<span style="font-size:0.83rem;color:#b0b0b8;">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if key == "roof_type_mask":
                st.markdown(
                    '<div style="margin-left:22px;font-size:0.73rem;'
                    'border-left:1px solid rgba(255,255,255,0.1);padding-left:8px;">'
                    '• <span style="color:#ff7f0e">RCC</span> | '
                    '• <span style="color:#2ca02c">Tiled</span> | '
                    '• <span style="color:#d62728">Tin</span> | '
                    '• <span style="color:#8c8c8c">Others</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Inference")
        threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.50)
        use_tta   = st.checkbox("Enable TTA (higher quality, slower)", value=False)
        tile_size = st.select_slider(
            "Tile Size", options=[512, 768, 1024, 1280, 1536], value=512
        )
        overlap = st.slider("Tile Overlap (px)", 64, 384, 192, step=32)
        alpha   = st.slider("Overlay Opacity",  0.1, 0.9,  0.5)

        st.divider()
        st.subheader("Export")
        export_format = st.radio(
            "GIS Format",
            ["GeoPackage (.gpkg)", "ESRI Shapefile (.shp)"],
            index=0,
        )
        export_ext = "SHP" if "Shapefile" in export_format else "GPKG"

    st.markdown("### Upload Image")
    uploaded = st.file_uploader(
        "GeoTIFF, JPG, or PNG",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
    )

    if not uploaded:
        return

    if len(uploaded.getvalue()) > MAX_UPLOAD_BYTES:
        st.error("File exceeds 10 GB limit.")
        st.stop()

    ext = Path(uploaded.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        st.error(f"Unsupported file type: {ext}")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tif_path = Path(tmp.name)
        _temp_files.append(tmp.name)

    is_geospatial = ext in {".tif", ".tiff"}

    col_run, _ = st.columns([1, 2])
    with col_run:
        run_clicked = st.button("Run Extraction", type="primary", use_container_width=True)

    if run_clicked:
        if not selected_masks:
            st.warning("Select at least one feature layer.")
            st.stop()

        progress = st.progress(0)
        status   = st.empty()
        t0       = time.time()
        status.info("Loading model weights...")

        def _progress(current, total):
            pct = (current + 1) / total
            progress.progress(pct)
            elapsed = time.time() - t0
            if pct > 0:
                remaining = elapsed / pct - elapsed
                m, s = divmod(int(remaining), 60)
                status.markdown(f"**{pct:.1%}** complete — ~{m}m {s}s remaining")

        with st.spinner("Running tiled inference..."):
            predictor = load_segmentation_pipeline(
                weights_path=ckpt_path,
                device=DEVICE,
                use_tta=use_tta,
                tile_size=tile_size,
                overlap=overlap,
            )
            predictor.threshold = threshold

            if is_geospatial:
                st.session_state.results = predictor.predict_tif(
                    tif_path,
                    selected_masks=selected_masks,
                    progress_callback=_progress,
                )
            else:
                st.session_state.results = predictor.predict_image(
                    tif_path,
                    selected_masks=selected_masks,
                )

        progress.empty()
        status.empty()
        st.session_state.tif_path        = tif_path
        st.session_state.is_geospatial   = is_geospatial
        st.success("Extraction complete.")

    results = st.session_state.get("results")
    if not results:
        return

    # Build display thumbnail
    import cv2

    with rasterio.open(str(tif_path)) as src:
        H, W  = src.height, src.width
        scale = min(1200.0 / max(H, W), 1.0)
        th    = max(1, int(H * scale))
        tw    = max(1, int(W * scale))
        raw   = src.read(
            out_shape=(src.count, th, tw),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        thumb = np.transpose(raw, (1, 2, 0))
        if thumb.shape[2] > 3:
            thumb = thumb[:, :, :3]
        if thumb.shape[2] == 1:
            thumb = np.repeat(thumb, 3, axis=2)
        if thumb.dtype != np.uint8:
            t    = thumb.astype(np.float32)
            vmax = float(np.percentile(t, 99.0)) or 1.0
            thumb = np.clip(t / vmax, 0.0, 1.0) * 255.0
        thumb = thumb.astype(np.uint8)

    tab_overview, tab_layer, tab_compare = st.tabs(
        ["Overview", "Layer Inspect", "Comparison"]
    )

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab_overview:
        overlay = thumb.copy()
        for key, (_, color) in FEATURES.items():
            if key not in results:
                continue
            interp = cv2.INTER_NEAREST if key == "roof_type_mask" else cv2.INTER_LINEAR
            m = cv2.resize(
                results[key], (thumb.shape[1], thumb.shape[0]), interpolation=interp
            )
            binary = m > 0 if key == "roof_type_mask" else m > threshold
            for c in range(3):
                overlay[binary, c] = overlay[binary, c] * (1 - alpha) + color[c] * alpha
        st.image(overlay.astype(np.uint8), use_container_width=True)

    # ── Layer Inspect ─────────────────────────────────────────────────────────
    with tab_layer:
        available = [k for k in FEATURES if k in results]
        if not available:
            st.info("Run extraction to inspect individual layers.")
        else:
            sel = st.selectbox(
                "Layer", available, format_func=lambda x: FEATURES[x][0]
            )
            f_name, f_color = FEATURES[sel]
            m_raw = results[sel].copy()

            if sel != "roof_type_mask":
                binary  = (m_raw > threshold).astype(np.uint8)
                refined = refine_mask(binary, sel)
                m_raw   = np.where(refined > 0, m_raw, 0)

            interp = cv2.INTER_NEAREST if sel == "roof_type_mask" else cv2.INTER_LINEAR
            m_disp = cv2.resize(
                m_raw, (thumb.shape[1], thumb.shape[0]), interpolation=interp
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(thumb.astype(np.uint8), caption="Original")
            with col_b:
                if sel == "roof_type_mask":
                    import matplotlib.pyplot as plt
                    cmap  = plt.get_cmap("tab10")
                    c_map = (cmap(m_disp)[:, :, :3] * 255).astype(np.uint8)
                    c_map[m_disp == 0] = 0
                    st.image(c_map, caption=f"{f_name} Map")
                else:
                    c_mask = np.zeros_like(thumb)
                    binary = m_disp > threshold
                    for i in range(3):
                        c_mask[binary, i] = f_color[i]
                    st.image(c_mask.astype(np.uint8), caption=f"{f_name} Mask")

            # Fused overlay
            f_ovl    = thumb.copy()
            is_roof  = sel == "roof_type_mask"
            bin_mask = m_disp > 0 if is_roof else m_disp > threshold
            if is_roof:
                import matplotlib.pyplot as plt
                cmap   = plt.get_cmap("tab10")
                c_roof = (cmap(m_disp)[:, :, :3] * 255).astype(np.uint8)
                f_ovl[bin_mask] = (
                    f_ovl[bin_mask] * (1 - alpha) + c_roof[bin_mask] * alpha
                )
            else:
                for i in range(3):
                    f_ovl[bin_mask, i] = (
                        f_ovl[bin_mask, i] * (1 - alpha) + f_color[i] * alpha
                    )
            st.image(f_ovl.astype(np.uint8), use_container_width=True)

    # ── Comparison ────────────────────────────────────────────────────────────
    with tab_compare:
        available = [k for k in FEATURES if k in results]
        if not available:
            st.info("Run extraction to compare layers.")
        else:
            target = st.selectbox(
                "Layer to Verify",
                available,
                format_func=lambda x: FEATURES[x][0],
                key="cmp_layer",
            )
            f_name, f_color = FEATURES[target]
            m_raw  = results[target]
            interp = cv2.INTER_NEAREST if target == "roof_type_mask" else cv2.INTER_LINEAR
            m_disp = cv2.resize(
                m_raw, (thumb.shape[1], thumb.shape[0]), interpolation=interp
            )

            mode = st.radio(
                "Mode", ["Side-by-Side", "Opacity Blend"], horizontal=True
            )

            if mode == "Side-by-Side":
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original")
                    st.image(thumb.astype(np.uint8), use_container_width=True)
                with c2:
                    st.caption(f"AI Extraction: {f_name}")
                    mask_view = np.zeros_like(thumb)
                    binary    = m_disp > threshold
                    for i in range(3):
                        mask_view[binary, i] = f_color[i]
                    st.image(mask_view, use_container_width=True)
            else:
                blend_a = st.slider("Opacity", 0.0, 1.0, 0.5)
                ovl     = thumb.copy().astype(np.float32)
                is_roof = target == "roof_type_mask"
                bm      = m_disp > 0 if is_roof else m_disp > threshold
                if is_roof:
                    import matplotlib.pyplot as plt
                    cmap   = plt.get_cmap("tab10")
                    c_roof = (cmap(m_disp)[:, :, :3] * 255).astype(np.uint8)
                    ovl[bm] = ovl[bm] * (1 - blend_a) + c_roof[bm] * blend_a
                else:
                    for i in range(3):
                        ovl[bm, i] = ovl[bm, i] * (1 - blend_a) + f_color[i] * blend_a
                st.image(
                    ovl.astype(np.uint8),
                    use_container_width=True,
                    caption=f"{f_name} at {blend_a*100:.0f}% opacity",
                )

    # ── GIS Export ────────────────────────────────────────────────────────────
    if is_geospatial:
        st.divider()
        if st.button("Generate GIS Layers"):
            from inference.export import export_predictions

            with tempfile.TemporaryDirectory() as out_dir:
                export_predictions(
                    results,
                    tif_path,
                    Path(out_dir),
                    threshold=threshold,
                    roof_type_mask=results.get("roof_type_mask"),
                    export_format=export_ext,
                )
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for p in Path(out_dir).rglob("*.*"):
                        if p.suffix.lower() in {".gpkg", ".shp", ".dbf", ".shx", ".prj", ".cpg"}:
                            zf.write(p, p.name)
                st.download_button(
                    f"Download {export_ext} ZIP",
                    zip_buf.getvalue(),
                    f"svamitva_export_{export_ext.lower()}.zip",
                )
    else:
        st.info("GIS export is only available for GeoTIFF inputs.")


if __name__ == "__main__":
    main()
