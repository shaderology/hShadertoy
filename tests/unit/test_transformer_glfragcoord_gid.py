"""
Category Q (Design B) — gid-derived gl_FragCoord accessor for HELPER functions.

The entry-body case is handled by an injected `float4 gl_FragCoord =
(float4)(fragCoord, 0.0f, 1.0f);` local (see test_transpile_entrypoint.py §9).
Helper functions cannot see that kernel local, so a helper referencing
`gl_FragCoord` fails to compile with "undeclared identifier 'gl_FragCoord'".

Design B threads NO extra parameters. Instead:
  * a runtime header (houdini/ocl/include/glslHelpers.h) provides a
    program-scope `static int2 GLSL_glFragCoord_off;` and an accessor
    `static float4 GLSL_glFragCoord(void)` that reconstructs the pixel
    coordinate from `get_global_id()` + the offset;
  * the offset is seeded by the HDA setter `shadertoy_bind_inputs()` at the top
    of EVERY kernel body (host header, `SHADERTOY_INPUTS` macro) — so the
    transpiler emits NO offset seed of its own (retired Session 58; the setter
    write and the retired entry-body write produced identical values);
  * each HELPER that references gl_FragCoord (directly or via a
    `#define F gl_FragCoord` object-macro alias) gets a first-statement local
    `float4 gl_FragCoord = GLSL_glFragCoord();`.

These tests pin the transpiler output. The header helper is exercised by the
corpus compile (it is invisible to string assertions here).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def tp(glsl, common=""):
    result = transpile(glsl, common=common)
    return result.get_header(), result.get_kernel()


def _norm(s):
    return " ".join(s.split())


OFFSET = ("GLSL_glFragCoord_off = (int2)(AT_ix - (int)get_global_id(0), "
          "AT_iy - (int)get_global_id(1))")
HELPER_LOCAL = "float4 gl_FragCoord = GLSL_glFragCoord();"
ENTRY_LOCAL = "float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f)"


# ---------------------------------------------------------------------------
# Direct helper use
# ---------------------------------------------------------------------------

def test_helper_using_gl_fragcoord_gets_accessor_local():
    glsl = (
        "float rnd() { return fract(sin(gl_FragCoord.x) * 43758.5453); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(rnd());\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    # helper (goes to header) gets the accessor-derived local
    assert HELPER_LOCAL in _norm(header)
    # the offset is seeded by the HDA setter, NOT by the transpiler entry body
    assert OFFSET not in _norm(kernel)


def test_two_helpers_each_get_the_local():
    glsl = (
        "float a() { return gl_FragCoord.x; }\n"
        "float b() { return gl_FragCoord.y; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(a() + b());\n"
        "}\n"
    )
    header, _ = tp(glsl)
    assert _norm(header).count(HELPER_LOCAL) == 2


def test_helper_not_using_gl_fragcoord_gets_no_local():
    glsl = (
        "float nouse() { return 1.0; }\n"
        "float rnd() { return gl_FragCoord.x; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(nouse() + rnd());\n"
        "}\n"
    )
    header, _ = tp(glsl)
    # only the one helper that references gl_FragCoord is instrumented
    assert _norm(header).count(HELPER_LOCAL) == 1


# ---------------------------------------------------------------------------
# Offset gating: entry-only use must NOT seed the offset static
# ---------------------------------------------------------------------------

def test_entry_only_use_keeps_fragcoord_injection_and_no_offset():
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = gl_FragCoord.xy / iResolution.xy;\n"
        "    fragColor = vec4(uv, 0.0, 1.0);\n"
        "}\n"
    )
    _, kernel = tp(glsl)
    # entry keeps the proven fragCoord-based local
    assert ENTRY_LOCAL in _norm(kernel)
    # no helper needs the accessor -> no offset static, no accessor call
    assert "GLSL_glFragCoord_off" not in kernel
    assert "GLSL_glFragCoord()" not in kernel


def test_no_gl_fragcoord_no_injection_at_all():
    glsl = (
        "float lum(vec3 c) { return dot(c, vec3(0.3)); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(lum(vec3(fragCoord / iResolution.xy, 0.0)));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "gl_FragCoord" not in header
    assert "gl_FragCoord" not in kernel


# ---------------------------------------------------------------------------
# #define alias (object-macro): #define F gl_FragCoord
# ---------------------------------------------------------------------------

def test_alias_macro_in_helper_gets_local_no_transpiler_offset():
    glsl = (
        "#define F gl_FragCoord\n"
        "float sharpen() { vec2 uv = F.xy; return uv.x; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 u = F.xy;\n"
        "    fragColor = vec4(sharpen() + u.x);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    # helper reached via alias gets the accessor local (CPP later maps F->gl_FragCoord)
    assert HELPER_LOCAL in _norm(header)
    # entry also uses F -> keeps the fragCoord-based local
    assert ENTRY_LOCAL in _norm(kernel)
    # the offset is seeded by the HDA setter, NOT by the transpiler entry body
    assert OFFSET not in _norm(kernel)


def test_alias_macro_in_common_pass_reaches_helper():
    # 3dK3zR shape: the `#define F gl_FragCoord` lives in the Common tab, which
    # is string-merged into every pass before parsing.
    common = "#define F gl_FragCoord\n"
    glsl = (
        "float ssao() { vec2 uv = F.xy / iResolution.xy; return uv.x; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(ssao());\n"
        "}\n"
    )
    header, kernel = tp(glsl, common=common)
    assert HELPER_LOCAL in _norm(header)
    # the offset is seeded by the HDA setter, NOT by the transpiler entry body
    assert OFFSET not in _norm(kernel)


# ---------------------------------------------------------------------------
# Skip-gate: shader supplies its own gl_FragCoord
# ---------------------------------------------------------------------------

def test_user_define_gl_fragcoord_disables_helper_injection():
    glsl = (
        "#define gl_FragCoord fragCoord\n"
        "float rnd() { return fract(sin(gl_FragCoord.x) * 43758.5); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(rnd());\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "GLSL_glFragCoord()" not in header
    assert "GLSL_glFragCoord_off" not in kernel


def test_user_declared_gl_fragcoord_disables_helper_injection():
    glsl = (
        "float rnd() { vec4 gl_FragCoord = vec4(1.0); return gl_FragCoord.x; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(rnd());\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "GLSL_glFragCoord()" not in header
    assert "GLSL_glFragCoord_off" not in kernel
