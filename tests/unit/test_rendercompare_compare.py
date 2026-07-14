"""Unit tests for the render-compare metrics (tests/rendercompare/compare.py).

Pure numpy/PIL/skimage - no wgpu, no Houdini.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rendercompare"))
import compare  # noqa: E402
import common  # noqa: E402


@pytest.fixture
def gradient():
    """800x450 uv-gradient image, float [0,1] RGB."""
    h, w = 450, 800
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = np.linspace(0, 1, w)[None, :]
    img[..., 1] = np.linspace(0, 1, h)[:, None]
    return img


def test_identical_images_are_perfect(gradient):
    m = compare.compute_metrics(gradient, gradient.copy())
    assert m["mae"] == 0.0
    assert m["dmae"] == 0.0
    assert m["psnr"] == float("inf")
    assert m["ssim"] == pytest.approx(1.0)
    assert compare.verdict(m) == "PASS"


def test_known_mae_offset(gradient):
    shifted = np.clip(gradient + 10.0 / 255.0, 0.0, 1.0)
    m = compare.compute_metrics(gradient, shifted)
    # most pixels move exactly 10/255; clipping at the bright edge lowers it a touch
    assert 8.0 < m["mae"] <= 10.01


def test_ssim_is_real_not_placeholder(gradient):
    """Guard against the fake 1/(1+mse) SSIM in tests/helpers/image_comparison.py."""
    noisy = np.clip(
        gradient + np.random.default_rng(0).normal(0, 0.25, gradient.shape)
        .astype(np.float32), 0, 1)
    m = compare.compute_metrics(gradient, noisy)
    fake_ssim = 1.0 / (1.0 + np.mean((gradient - noisy) ** 2))
    assert abs(m["ssim"] - fake_ssim) > 0.05
    assert m["ssim"] < 0.6  # heavy noise destroys structure


def test_yflip_is_detected(gradient):
    """A vertically flipped image must not pass the perceptual gates."""
    m = compare.compute_metrics(gradient, gradient[::-1].copy())
    assert compare.verdict(m) != "PASS"


def test_subpixel_style_shift_passes(gradient):
    """A 1-pixel translation (filtering/half-pixel divergence) must PASS."""
    shifted = np.roll(gradient, 1, axis=1)
    shifted[:, 0] = gradient[:, 0]
    m = compare.compute_metrics(gradient, shifted)
    assert compare.verdict(m) == "PASS"


def test_uniform_color_change_fails(gradient):
    """Grossly wrong colors must FAIL even though structure matches."""
    wrong = gradient.copy()
    wrong[..., 2] = 0.8  # blue channel blown out everywhere
    m = compare.compute_metrics(gradient, wrong)
    assert m["dmae"] > compare.GATE_WARN_DMAE
    assert compare.verdict(m) == "FAIL"


def test_shape_mismatch_raises(gradient):
    with pytest.raises(ValueError):
        compare.compute_metrics(gradient, gradient[:-1])


def test_compare_files_roundtrip(tmp_path, gradient):
    from PIL import Image
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    d = tmp_path / "d.png"
    Image.fromarray((gradient * 255).astype(np.uint8)).save(a)
    Image.fromarray((gradient * 255).astype(np.uint8)).save(b)
    m = compare.compare_files(a, b, diff_path=d)
    assert m["verdict"] == "PASS"
    assert d.exists()
    # heatmap of identical images is black
    heat = compare.load_rgb(d)
    assert heat.max() == 0.0


def test_shader_flags_detection():
    shader = {"Shader": {"info": {}, "renderpass": [
        {"type": "image", "name": "Image", "inputs": [
            {"ctype": "webcam", "channel": 0}],
         "code": "void mainImage(){ float t = iTime + iDate.w; }"}]}}
    flags = common.shader_flags(shader)
    assert "uses_iDate" in flags
    assert "input_webcam" in flags
    assert "multipass" not in flags


def test_itime_contract():
    assert common.itime_for_frame(601) == pytest.approx(10.0)
    assert common.itime_for_frame(1) == 0.0
