"""The builder must map the API sampler block (filter/wrap/vflip) onto the
HDA's per-channel sampler parms - found by the render-compare pipeline:
without this every texture keeps the HDA defaults (vflip=true, wrap=repeat,
filter=mipmap) and vflip=false shaders render upside-down vs the site.

builder.py imports hou at module scope; stub it so the pure param-building
logic is testable under plain pytest.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "houdini" / "scripts" / "python"))

if "hou" not in sys.modules:
    _hou = types.ModuleType("hou")
    _hou.__getattr__ = lambda name: type(name, (), {})  # Node, Error, ...
    sys.modules["hou"] = _hou

from hshadertoy.builder.builder import _build_renderpass_params  # noqa: E402


ASSETS_MAP = {
    5: {"hda": {"folder": {"token": 1}, "asset": {"token": 10}}},
}


def _renderpass(sampler):
    return {
        "name": "Image", "type": "image",
        "code": "void mainImage(out vec4 c, in vec2 f) { c = vec4(1.0); }",
        "inputs": [{"id": 5, "channel": 0, "ctype": "texture",
                    "sampler": sampler}],
    }


def _params(sampler):
    return _build_renderpass_params(
        _renderpass(sampler), 0, ASSETS_MAP,
        transpiler_func=lambda code, mode=None: code,
        transpiler_mode="Transpile")


def test_vflip_false_is_forwarded():
    params = _params({"filter": "mipmap", "wrap": "repeat", "vflip": "false"})
    assert params["vflip_rp0_ch0"] is False


def test_vflip_true_is_forwarded():
    params = _params({"vflip": "true"})
    assert params["vflip_rp0_ch0"] is True


def test_wrap_mapping():
    # HDA wrap menu tokens: Clamp='1', Wrap='3' (probed from the HDA)
    assert _params({"wrap": "clamp"})["wrap_rp0_ch0"] == "1"
    assert _params({"wrap": "repeat"})["wrap_rp0_ch0"] == "3"


def test_filter_mapping():
    # HDA filter menu: 0=nearest 1=linear 2=mipmap
    assert _params({"filter": "nearest"})["filter_rp0_ch0"] == 0
    assert _params({"filter": "linear"})["filter_rp0_ch0"] == 1
    assert _params({"filter": "mipmap"})["filter_rp0_ch0"] == 2


def test_shadertoy_defaults_when_sampler_missing():
    """No sampler block -> the site defaults (mipmap/repeat/vflip=true)."""
    params = _params({})
    assert params["vflip_rp0_ch0"] is True
    assert params["wrap_rp0_ch0"] == "3"
    assert params["filter_rp0_ch0"] == 2


def test_common_prefix_naming():
    rp = _renderpass({"vflip": "false"})
    params = _build_renderpass_params(
        rp, "common", ASSETS_MAP,
        transpiler_func=lambda code, mode=None: code,
        transpiler_mode="Transpile")
    assert params["vflip_common_ch0"] is False
