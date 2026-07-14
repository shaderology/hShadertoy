"""
Image metrics for the render-compare pipeline.

The user-facing verdict is PERCEPTUAL, not pixel-exact: the two renderers
legitimately differ in texture filtering, mip levels, half-pixel fragCoord
alignment and color pipelines, so raw per-pixel error is expected to be
non-zero even for a semantically perfect transpile. We therefore gate on:

  * SSIM  - structural similarity (skimage, gaussian-weighted) on grayscale.
  * dMAE  - mean absolute error AFTER box-downsampling both images ~10x
            (800x450 -> 80x45). Downsampling integrates away sub-pixel
            shifts, filtering differences and high-frequency noise while
            keeping genuine color/shape divergence. Reported in 0..255 units.

Plain MAE and PSNR are also computed and stored for forensics, but do not
drive the verdict. All metrics are computed on RGB only - the Shadertoy site
forces alpha=1 on the image pass while hShadertoy preserves user alpha, so
alpha is a known, uninteresting divergence.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DOWNSAMPLE = (45, 80)  # (rows, cols) ~ /10

# Verdict gates - initial values, tuned on the first corpus batch.
GATE_PASS_SSIM = 0.85
GATE_PASS_DMAE = 4.0
GATE_WARN_SSIM = 0.60
GATE_WARN_DMAE = 12.0


def load_rgb(path) -> np.ndarray:
    """PNG -> float32 HxWx3 in [0,1] (whatever encoding the file carries)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def box_downsample(img: np.ndarray, shape=DOWNSAMPLE) -> np.ndarray:
    """Area-average downsample via PIL (exact box filter, any channel count)."""
    rows, cols = shape
    arr = np.clip(img, 0.0, 1.0)
    pil = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    small = pil.resize((cols, rows), Image.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def compute_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    """ref/test: float HxWx3 in [0,1], same shape. Returns metric dict."""
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {test.shape}")

    diff = np.abs(ref - test)
    mae = float(diff.mean() * 255.0)
    mse = float(np.mean((ref - test) ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))

    from skimage.metrics import structural_similarity
    ssim = float(structural_similarity(
        to_gray(ref), to_gray(test), data_range=1.0, gaussian_weights=True))

    dmae = float(np.abs(
        box_downsample(ref) - box_downsample(test)).mean() * 255.0)

    return {"mae": mae, "psnr": psnr, "ssim": ssim, "dmae": dmae}


def verdict(metrics: dict) -> str:
    """PASS / WARN / FAIL from the perceptual gates."""
    s, d = metrics["ssim"], metrics["dmae"]
    if s >= GATE_PASS_SSIM and d <= GATE_PASS_DMAE:
        return "PASS"
    if s >= GATE_WARN_SSIM and d <= GATE_WARN_DMAE:
        return "WARN"
    return "FAIL"


def save_diff_heatmap(ref: np.ndarray, test: np.ndarray, path) -> None:
    """Amplified |diff| visualisation: black=equal, yellow->red=diverging."""
    err = np.abs(ref - test).max(axis=2)  # worst channel per pixel
    amp = np.clip(err * 4.0, 0.0, 1.0)    # 0.25 error saturates
    heat = np.zeros(err.shape + (3,), dtype=np.float32)
    heat[..., 0] = amp                     # red ramps with error
    heat[..., 1] = np.clip(amp * 2.0, 0.0, 1.0) * (1.0 - amp)  # yellow mids
    Image.fromarray((heat * 255.0 + 0.5).astype(np.uint8)).save(path)


def compare_files(ref_path, test_path, diff_path=None) -> dict:
    """Load two PNGs, compute metrics + verdict, optionally write heatmap."""
    ref = load_rgb(ref_path)
    test = load_rgb(test_path)
    if ref.shape != test.shape:
        return {"verdict": "ERROR",
                "error": f"shape mismatch: {ref.shape} vs {test.shape}"}
    m = compute_metrics(ref, test)
    m["verdict"] = verdict(m)
    if diff_path is not None:
        save_diff_heatmap(ref, test, diff_path)
    return m
