"""
Unit tests for category Z — sampler params in user functions.

GLSL lets a user function take a channel as a `sampler2D`/`samplerCube`/
`sampler3D` parameter (the classic triplanar `tex3D(sampler2D t, vec3 p, vec3 n)`
idiom). The Houdini/Copernicus runtime has no sampler types: channels are
`const IMX_Layer*` (see houdini/ocl/include/textureHelpers.h — every texture
builtin takes `const IMX_Layer* layer`), and `iChannel0..3` are already declared
as `static const IMX_Layer*` in the runtime header.

Fix (Session 9): map the GLSL sampler types to `const IMX_Layer*` in the
transformer's type_map. The parameter is NOT marked is_pointer — that flag is
reserved for out/inout params and would drag sampler params into the
auto-dereference machinery (`texture(*s, ...)` would be wrong: the runtime
builtins take the pointer itself). Call sites need no change: `iChannel0` is
already the right pointer type and flows through as a plain argument.
"""

import sys
from pathlib import Path

import pytest

# tests/transpile.py is the campaign transpiler (header/kernel split path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def tp(glsl):
    """Transpile and return (header, kernel_body)."""
    result = transpile(glsl)
    return result.get_header(), result.get_kernel()


MAIN_CALLS_TEX3D = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
    "    vec2 uv = fragCoord / iResolution.xy;\n"
    "    fragColor = vec4(tex3D(iChannel0, vec3(uv, 0.0), vec3(0.0, 0.0, 1.0)), 1.0);\n"
    "}\n"
)


# ============================================================================
# 1. Sampler param types map to the runtime layer pointer type
# ============================================================================

def test_sampler2d_param_maps_to_layer_ptr():
    """The classic triplanar idiom: tex3D(sampler2D t, vec3 p, vec3 n)."""
    glsl = (
        "vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){\n"
        "    return (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y"
        " + texture(tex, p.xy)*n.z).xyz;\n"
        "}\n" + MAIN_CALLS_TEX3D
    )
    header, kernel = tp(glsl)
    assert "float3 tex3D(const IMX_Layer* tex, float3 p, float3 n)" in header
    assert "sampler2D" not in header
    assert "sampler2D" not in kernel


def test_samplercube_param_maps_to_layer_ptr():
    """samplerCube params use the same layer pointer (cube is a packed 2D layer)."""
    glsl = (
        "vec4 getSpec(samplerCube samp, vec3 n){\n"
        "    return texture(samp, n);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = getSpec(iChannel1, vec3(0.0, 1.0, 0.0));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float4 getSpec(const IMX_Layer* samp, float3 n)" in header
    assert "samplerCube" not in header


def test_sampler3d_param_maps_to_layer_ptr():
    """sampler3D params map the same way (runtime packs 3D into a 2D layer)."""
    glsl = (
        "vec4 s3d(sampler3D T, vec3 U){\n"
        "    return texture(T, U);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = s3d(iChannel0, vec3(0.5));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float4 s3d(const IMX_Layer* T, float3 U)" in header
    assert "sampler3D" not in header


def test_in_qualifier_sampler_param():
    """`in sampler2D chan` (4ljyRc pattern): `in` drops, type still maps."""
    glsl = (
        "vec3 graffiti(in vec2 uv, in sampler2D chan){\n"
        "    return texture(chan, uv).rgb;\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(graffiti(fragCoord, iChannel0), 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float3 graffiti(float2 uv, const IMX_Layer* chan)" in header


def test_const_sampler_param_no_duplicate_const():
    """`const sampler2D` must not emit `const const IMX_Layer*`."""
    glsl = (
        "vec4 look(const sampler2D s, vec2 uv){\n"
        "    return texture(s, uv);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = look(iChannel0, fragCoord);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "const IMX_Layer* s" in header
    assert "const const" not in header


# ============================================================================
# 2. No pointer machinery: body reads and call sites stay plain
# ============================================================================

def test_body_texture_call_not_dereferenced():
    """The sampler param must NOT join pointer_params: no `texture(*tex, ...)`."""
    glsl = (
        "vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){\n"
        "    return texture(tex, p.xy).xyz;\n"
        "}\n" + MAIN_CALLS_TEX3D
    )
    header, kernel = tp(glsl)
    assert "texture(tex," in header.replace(" ", "").replace("texture(tex,", "texture(tex,") or \
        "texture(tex" in header
    assert "*tex" not in header


def test_call_site_passes_channel_unchanged():
    """mainImage passes iChannel0 as-is — no & and no * at the call site."""
    glsl = (
        "vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){\n"
        "    return texture(tex, p.xy).xyz;\n"
        "}\n" + MAIN_CALLS_TEX3D
    )
    header, kernel = tp(glsl)
    assert "tex3D(iChannel0," in kernel.replace(" ", "").replace("tex3D(iChannel0,", "tex3D(iChannel0,") or \
        "tex3D(iChannel0" in kernel
    assert "&iChannel0" not in kernel
    assert "*iChannel0" not in kernel


def test_helper_to_helper_sampler_pass():
    """A sampler param forwarded to another helper stays a plain argument."""
    glsl = (
        "vec4 fetch(sampler2D s, vec2 uv){\n"
        "    return texture(s, uv);\n"
        "}\n"
        "vec4 blur(sampler2D chan, vec2 uv){\n"
        "    return fetch(chan, uv) + fetch(chan, uv + 0.01);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = blur(iChannel0, fragCoord);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float4 fetch(const IMX_Layer* s, float2 uv)" in header
    assert "float4 blur(const IMX_Layer* chan, float2 uv)" in header
    assert "fetch(chan," in header.replace("fetch( chan", "fetch(chan") or "fetch(chan" in header
    assert "&chan" not in header
    assert "*chan" not in header


def test_out_param_alongside_sampler_still_pointer():
    """A real out-param next to a sampler param keeps its pointer treatment."""
    glsl = (
        "void sample2(sampler2D s, vec2 uv, out vec4 col){\n"
        "    col = texture(s, uv);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec4 c;\n"
        "    sample2(iChannel0, fragCoord, c);\n"
        "    fragColor = c;\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "const IMX_Layer* s" in header
    assert "float4* col" in header
    # out-param write inside the body is dereferenced; sampler read is not.
    assert "*col" in header
    assert "*s" not in header
