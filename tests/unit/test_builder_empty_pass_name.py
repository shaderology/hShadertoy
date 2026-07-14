"""Old (pre-~2016) Shadertoy API shaders return renderpass name="" (79 image
passes + 2 sound passes in the campaign corpus, e.g. 4l2XWw "phyllotaxis 2D").
The builder used to skip any pass with a falsy name -> it silently built a
DEFAULT-state HDA ("Configured 0 parameters") while showing Build Complete.

The fix derives the canonical name from the pass `type` when `name` is empty.
Buffers stay skipped (cannot disambiguate A-D from type alone; never seen
nameless in the corpus) - but that skip must warn, not silently continue.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "houdini" / "scripts" / "python"))

if "hou" not in sys.modules:
    _hou = types.ModuleType("hou")
    _hou.__getattr__ = lambda name: type(name, (), {})
    sys.modules["hou"] = _hou

from hshadertoy.builder.builder import (  # noqa: E402
    _get_renderpass_token,
    _resolve_renderpass_name,
)


def test_named_pass_keeps_its_name():
    assert _resolve_renderpass_name(
        {"name": "Buffer A", "type": "buffer"}) == "Buffer A"


@pytest.mark.parametrize("rp_type,expected", [
    ("image", "Image"),
    ("common", "Common"),
    ("cubemap", "Cube A"),
    ("sound", "Sound"),
])
def test_empty_name_derived_from_type(rp_type, expected):
    assert _resolve_renderpass_name({"name": "", "type": rp_type}) == expected
    assert _resolve_renderpass_name({"type": rp_type}) == expected  # missing key


def test_derived_names_are_valid_tokens():
    for rp_type in ("image", "common", "cubemap", "sound"):
        name = _resolve_renderpass_name({"name": "", "type": rp_type})
        _get_renderpass_token(name)  # must not raise


def test_nameless_buffer_stays_unresolved():
    # A-D cannot be told apart from type alone - caller warns and skips.
    assert _resolve_renderpass_name({"name": "", "type": "buffer"}) == ""


def test_unknown_type_stays_unresolved():
    assert _resolve_renderpass_name({"name": "", "type": "mystery"}) == ""
