"""
Characterization tests for the transpile() entry-point contract.

Written BEFORE the single-translation-unit restructure of tests/transpile.py
(entry-point redesign, see docs/handover/ENTRYPOINT_REDESIGN.md). These pin
the externally observable header/kernel contract that the campaign and the
Houdini kernel scaffold rely on, so the pipeline internals (text split,
synthetic re-wrap, re-parse, substring surgery) can be replaced without
changing behavior:

- header = every top-level declaration except the mainImage definition
  (post-mainImage code included — category S; alternate entry definitions
  after mainImage excluded, before it kept),
- kernel = mainImage body statements only (no signature, no braces), with
  fragColor/fragCoord referenced as plain kernel locals (never pointerized),
- custom entry param names (out vec4 O, vec2 U) bridged via alias
  declarations + final fragColor write,
- category-A hoisted global initializers prepended to the kernel body.
"""

import sys
from pathlib import Path

import pytest

# tests/transpile.py is the campaign transpiler (header/kernel split path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile, TranspileError  # noqa: E402


def tp(glsl, common=""):
    """Transpile and return (header, kernel_body)."""
    result = transpile(glsl, common=common)
    return result.get_header(), result.get_kernel()


MAIN = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
    "    fragColor = vec4(0.0);\n"
    "}\n"
)


# ============================================================================
# 1. Basic split contract
# ============================================================================

def test_helper_goes_to_header_body_to_kernel():
    glsl = (
        "float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = fragCoord / iResolution.xy;\n"
        "    fragColor = vec4(lum(vec3(uv, 0.0)));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float lum(float3 c)" in header
    assert "lum" in kernel  # call survives
    # kernel is body-only: no signature, no mainImage definition
    # (a "// Shadertoy void mainImage(...)" marker comment is allowed)
    assert "void mainImage(out" not in kernel
    assert "void mainImage(__private" not in kernel
    assert "uv = fragCoord / iResolution.xy" in kernel.replace("  ", " ")


def test_kernel_has_shadertoy_markers():
    header, kernel = tp(MAIN)
    assert "// ---- SHADERTOY CODE BEGIN ----" in kernel
    assert "// ---- SHADERTOY CODE END ----" in kernel


def test_entry_params_never_pointerized_in_kernel():
    """fragColor/fragCoord are @KERNEL locals: no derefs may survive."""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(fragCoord, 0.0, 1.0);\n"
        "    fragColor.x += 0.5;\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    assert "*fragColor" not in kernel
    assert "*fragCoord" not in kernel
    assert "&fragColor" not in kernel
    assert "&fragCoord" not in kernel


def test_missing_mainimage_raises():
    with pytest.raises(TranspileError):
        transpile("float f(float x) { return x; }\n")


# ============================================================================
# 2. Helper out-params keep pointer semantics (category B) in both sections
# ============================================================================

def test_helper_out_param_pointerized_but_entry_not():
    glsl = (
        "void tint(out vec4 col) { col = vec4(1.0); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    tint(fragColor);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    # helper keeps pointer out-param in header
    assert "float4* col" in header or "float4 * col" in header
    # call site takes the address of the kernel-local fragColor
    assert "tint(&fragColor)" in kernel


# ============================================================================
# 3. Custom entry parameter names (golf shaders)
# ============================================================================

def test_custom_param_names_aliased():
    glsl = (
        "void mainImage(out vec4 O, vec2 U) {\n"
        "    O = vec4(U, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    # alias declarations bridge the names
    assert "float4 O" in kernel
    assert "float2 U = fragCoord" in kernel
    # final write back to the kernel-local fragColor
    assert "fragColor = O;" in kernel
    # body identifiers kept verbatim
    assert "O = (float4)(U, 0.0f, 1.0f);" in kernel


def test_standard_names_get_no_aliases():
    _, kernel = tp(MAIN)
    assert "fragCoord = fragCoord" not in kernel
    assert "fragColor = fragColor" not in kernel


def test_custom_param_types_drive_inference_conversion():
    """Entry params must be registered under GLSL type names so that
    inference-driven transforms (category N conversions) fire inside the
    entry body — the old pipeline got this via GLSL alias declarations."""
    glsl = (
        "void mainImage(out vec4 O, vec2 u) {\n"
        "    ivec2 U = ivec2(30. * u / iResolution.xy);\n"
        "    O = vec4(vec2(U), 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    # float2 -> int2 must go through convert_int2, not a C-style vector cast
    assert "convert_int2(30.f * u / iResolution.xy)" in kernel
    assert "(int2)(30.f" not in kernel


def test_custom_param_types_drive_matrix_lowering():
    """mat2 * customParam-derived vec2 must lower to GLSL_mul_mat2_vec2."""
    glsl = (
        "void mainImage(out vec4 o, vec2 u) {\n"
        "    vec2 v = mat2(1.0, 0.0, 0.0, 1.0) * (u / iResolution.y - 0.5);\n"
        "    o = vec4(v, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    assert "GLSL_mul_mat2_vec2" in kernel


# ============================================================================
# 4. Code after mainImage (category S) and alternate entry points
# ============================================================================

def test_post_mainimage_definition_kept_in_header():
    glsl = (
        "float Fn(float x);\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(Fn(1.0));\n"
        "}\n"
        "float Fn(float x) { return x * 2.0; }\n"
    )
    header, kernel = tp(glsl)
    assert "float Fn(float x);" in header          # prototype
    assert "return x * 2.0f;" in header             # definition body
    assert "Fn(1.0f)" in kernel


def test_pre_main_code_precedes_post_main_code_in_header():
    glsl = (
        "float A(float x) { return x; }\n"
        "float Fn(float x);\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(A(Fn(1.0)));\n"
        "}\n"
        "float Fn(float x) { return x * 2.0; }\n"
    )
    header, _ = tp(glsl)
    assert header.index("float A(") < header.index("return x * 2.0f;")


def test_alternate_entry_after_mainimage_excluded():
    glsl = (
        MAIN +
        "void mainVR(out vec4 fragColor, in vec2 fragCoord,"
        " in vec3 ro, in vec3 rd) {\n"
        "    fragColor = vec4(rd, 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "mainVR" not in header
    assert "mainVR" not in kernel


def test_alternate_entry_before_mainimage_kept():
    """Some shaders define mainVR first and CALL it from mainImage."""
    glsl = (
        "void mainVR(out vec4 fragColor, in vec2 fragCoord,"
        " in vec3 ro, in vec3 rd) {\n"
        "    fragColor = vec4(rd, 1.0);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    mainVR(fragColor, fragCoord, vec3(0.0), vec3(0.0, 0.0, 1.0));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "void mainVR" in header
    assert "mainVR(" in kernel


# ============================================================================
# 5. Category-A hoisting still lands at the top of the kernel body
# ============================================================================

def test_hoisted_global_init_prepended_to_kernel():
    glsl = (
        "vec3 sunDir = normalize(vec3(0.2, 0.1, 0.4));\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    assert "float3 sunDir;" in header               # bare declaration
    assert "hoisted global initializers" in kernel  # marker comment
    # hoist must precede the body statements
    assert kernel.index("sunDir =") < kernel.index("fragColor =")


# ============================================================================
# 6. Common tab merge
# ============================================================================

def test_common_helpers_available_to_pass():
    common = "float saturate1(float x) { return clamp(x, 0.0, 1.0); }\n"
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(saturate1(2.0));\n"
        "}\n"
    )
    header, kernel = tp(glsl, common=common)
    assert "float saturate1(float x)" in header
    assert "saturate1(2.0f)" in kernel


# ============================================================================
# 7. Top-level comments (license/attribution headers) are preserved
# ============================================================================

def test_top_level_comments_preserved_in_header():
    """Shadertoy code is CC-licensed: attribution/license comment blocks at
    file scope must survive into the emitted header (the pre-single-TU
    pipeline kept them as raw text)."""
    glsl = (
        "// SPDX-License-Identifier: MIT\n"
        "// Copyright (c) 2026 @somebody\n"
        "float lum(vec3 c) { return c.x; }\n"
        "/* block comment survives too */\n"
        + MAIN
    )
    header, _ = tp(glsl)
    assert "// SPDX-License-Identifier: MIT" in header
    assert "// Copyright (c) 2026 @somebody" in header
    assert "/* block comment survives too */" in header


# ============================================================================
# 8. Structs at top level keep their trailing semicolon
# ============================================================================

def test_struct_definition_in_header():
    glsl = (
        "struct Ray { vec3 ro; vec3 rd; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    Ray r = Ray(vec3(0.0), vec3(0.0, 0.0, 1.0));\n"
        "    fragColor = vec4(r.rd, 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "struct" in header and "Ray" in header
    assert "};" in header or "} Ray;" in header


# ============================================================================
# 9. gl_FragCoord builtin in the entry body (category Q)
# ============================================================================
# gl_FragCoord is a fragment-shader builtin: gl_FragCoord.xy == fragCoord, the
# pixel-center coordinate the entry receives. Shaders that reference it in the
# entry body (rather than the mainImage `fragCoord` param) hit "undeclared
# identifier 'gl_FragCoord'". Fix: inject a body-local derived from fragCoord.

def _norm(s):
    return " ".join(s.split())


def test_gl_fragcoord_in_entry_body_gets_local():
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = gl_FragCoord.xy / iResolution.xy;\n"
        "    fragColor = vec4(uv, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    # A body-local vec4 gl_FragCoord built from fragCoord (exact Shadertoy value)
    assert "float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f)" in _norm(kernel)
    # It must be declared BEFORE the first use.
    k = _norm(kernel)
    assert k.index("float4 gl_FragCoord =") < k.index("gl_FragCoord.xy / iResolution")


def test_gl_fragcoord_local_via_alias_macro_use():
    # Custom entry param name: fragCoord is still provided by the host, so the
    # injected local references `fragCoord` regardless of the user's param name.
    glsl = (
        "void mainImage(out vec4 O, in vec2 U) {\n"
        "    vec2 uv = gl_FragCoord.xy / iResolution.xy;\n"
        "    O = vec4(uv, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    assert "float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f)" in _norm(kernel)


def test_no_gl_fragcoord_no_injection():
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(fragCoord / iResolution.xy, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    assert "gl_FragCoord" not in kernel


def test_gl_fragcoord_user_define_not_reinjected():
    # `#define gl_FragCoord fragCoord` already compiles on the site; injecting a
    # `float4 gl_FragCoord` would expand to `float4 fragCoord` -> redefinition.
    glsl = (
        "#define gl_FragCoord fragCoord\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = gl_FragCoord.xy / iResolution.xy;\n"
        "    fragColor = vec4(uv, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    assert "float4 gl_FragCoord =" not in kernel


def test_gl_fragcoord_user_declared_not_reinjected():
    # A shader that declares its own gl_FragCoord must not be double-declared.
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec4 gl_FragCoord = vec4(fragCoord, 0.0, 1.0);\n"
        "    fragColor = vec4(gl_FragCoord.xy / iResolution.xy, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    # exactly one declaration of gl_FragCoord (the user's), not two
    assert _norm(kernel).count("gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f)") == 1
