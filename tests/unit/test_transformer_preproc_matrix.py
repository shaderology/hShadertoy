"""
Unit tests for category E residual (Session 53) — matrix multiplication in
preprocessor territory.

The S19 AST matrix path (`v*M`/`M*v`/`M*M`/`v*=M` -> GLSL_mul) is correct for
plain code, but two textual regions never reached it:

Part 1 — statement-level `#if`/`#ifdef` blocks in FUNCTION BODIES: tree-sitter
parses their contents as structured statement children, but the transformer
flattened the whole node to raw text (`_transform_preprocessor`), so nothing
inside was ever typed or lowered. Now the children are routed through the
normal AST pipeline and re-wrapped in the original directives
(IR.PreprocessorBlock); any parse error / unknown child falls back to the old
raw-text behavior, so the worst case per block is the status quo. Program-scope
(header-level) blocks keep the raw-text path.

Part 2 — `#define` macro bodies stay textual by nature; the macro-body pass
now wraps a `*`/`*=` whose operand is a literal `GLSL_matN(...)` constructor
into GLSL_mul (positive matrix evidence only — GLSL_mul has no vec·vec or
scalar·scalar overload, so speculative wrapping is forbidden). The same pass
runs on code lines inside `#if` blocks, which composes with (and pre-empts)
Part 1 for ctor-literal lines.
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


def wrap_main(body):
    return (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        + body
        + "\n    fragColor = vec4(0.0);\n}\n"
    )


# ============================================================================
# Part 1 — #if/#ifdef blocks in function bodies are AST-routed
# ============================================================================

ROT2_FN = (
    "mat2 rot2(float a) {\n"
    "    float c = cos(a), s = sin(a);\n"
    "    return mat2(c, -s, s, c);\n"
    "}\n"
)


def test_ifdef_user_fn_matrix_mul_routed():
    """3sd3Rj shape: `vec2 q = rot2(a)*p;` inside #ifndef — the user function's
    mat2 return type must drive GLSL_mul once the block is AST-routed."""
    header, kernel = tp(ROT2_FN + wrap_main(
        "    vec2 p = fragCoord;\n"
        "    float a = 1.0;\n"
        "#ifndef STANDARD\n"
        "    vec2 q = rot2(a)*p;\n"
        "    p = q;\n"
        "#endif\n"
    ))
    assert "#ifndef STANDARD" in kernel
    assert "#endif" in kernel
    assert "GLSL_mul" in kernel
    assert "rot2(a)*p" not in kernel.replace(" ", "")


def test_ifdef_global_matrix_var_mul_routed():
    """MdycRK shape: a matrix read whose var is AST-declared OUTSIDE the block,
    multiplied INSIDE the block."""
    header, kernel = tp(wrap_main(
        "    mat3 vuMat = mat3(1.0);\n"
        "    vec3 rd = vec3(0.0);\n"
        "#if 1\n"
        "    rd = vuMat * normalize(vec3(fragCoord, 1.0));\n"
        "#endif\n"
    ))
    assert "#if 1" in kernel
    assert "GLSL_mul" in kernel


def test_ifdef_decl_inside_use_outside():
    """wtKXWV/wll3z2 shape: matrix DECLARED inside #if (both branches), used
    outside — the routed declarations must register the type so the outside
    use lowers to GLSL_mul."""
    header, kernel = tp(wrap_main(
        "    vec2 ro = fragCoord;\n"
        "#ifdef HQ\n"
        "    mat2 m = mat2(1.0, 0.0, 0.0, 1.0);\n"
        "#else\n"
        "    mat2 m = mat2(2.0, 0.0, 0.0, 2.0);\n"
        "#endif\n"
        "    ro.xy *= m;\n"
    ))
    assert "#ifdef HQ" in kernel
    assert "#else" in kernel
    assert "GLSL_mul" in kernel


def test_ifdef_else_branch_statements_transformed():
    """Both branches must be transformed (ctor lowered), directives kept."""
    header, kernel = tp(wrap_main(
        "    vec2 p = fragCoord;\n"
        "#ifdef FLIP\n"
        "    p = vec2(1.0, 2.0);\n"
        "#else\n"
        "    p = vec2(3.0, 4.0);\n"
        "#endif\n"
    ))
    assert "#ifdef FLIP" in kernel
    assert "#else" in kernel
    assert "(float2)(1.0f, 2.0f)" in kernel
    assert "(float2)(3.0f, 4.0f)" in kernel


def test_ifdef_matrix_macro_call_routed():
    """XsBcDc shape: `d.xy *= r2d(...)` inside #if where r2d is a
    matrix-returning #define — matrix_macros seeding must fire for the
    routed statement."""
    header, kernel = tp(
        "#define r2d(r) mat2(cos(r), -sin(r), sin(r), cos(r))\n"
        + wrap_main(
            "    vec2 d = fragCoord;\n"
            "#if 1\n"
            "    d.xy *= r2d(1.0);\n"
            "#endif\n"
        ))
    assert "GLSL_mul" in kernel


def test_ifdef_unrouted_child_falls_back_raw():
    """A block containing a statement type outside the routing allowlist
    (switch — not in the transformer's dispatch map) must keep today's
    raw-text passthrough: directives AND content preserved, nothing dropped.
    (Outright-broken content is a different path: the whole-file parse fails
    and the S38 conditional stripper handles it — G-family, unchanged.)"""
    header, kernel = tp(wrap_main(
        "    float x = 1.0;\n"
        "    int i = 0;\n"
        "#ifdef WEIRD\n"
        "    switch (i) { default: x = 2.0; }\n"
        "#endif\n"
    ))
    assert "#ifdef WEIRD" in kernel
    # The raw content is preserved (not silently dropped).
    assert "switch (i)" in kernel


def test_header_level_ifdef_keeps_raw_path():
    """Program-scope #ifdef blocks (wrapping declarations/functions) keep the
    raw-text passthrough this session."""
    header, kernel = tp(
        "#ifdef USE_HELPER\n"
        "float helper(float x) { return x * 2.0; }\n"
        "#endif\n"
        + wrap_main("    float y = 1.0;\n"))
    assert "#ifdef USE_HELPER" in header
    assert "helper" in header


def test_ifdef_stage0_cast_survives_routing():
    """Stage-0 pre-rewrites `vec2(...)` -> `(float2)(...)` on #if-block lines
    BEFORE parsing; the routed AST must emit the cast intact (the transformer
    used to have no cast handler and silently dropped the node)."""
    header, kernel = tp(wrap_main(
        "    vec2 p = fragCoord;\n"
        "#ifdef X\n"
        "    p = vec2(1.0, 2.0) * 3.0;\n"
        "#endif\n"
    ))
    assert "(float2)(1.0f, 2.0f)" in kernel
    assert "3.0f" in kernel


# ============================================================================
# Part 2 — #define macro bodies: GLSL_mul wrap on literal matrix-ctor operands
# ============================================================================

def test_define_compound_assign_matrix_ctor_wrapped():
    """4tSSzt/XlKSWG shape: `#define r(v,t) v *= mat2(...)`."""
    header, kernel = tp(
        "#define r(v,t) v *= mat2(cos(t), sin(t), -sin(t), cos(t))\n"
        + wrap_main(
            "    vec2 p = fragCoord;\n"
            "    r(p.xy, .13);\n"
        ))
    combined = header + kernel
    assert "v = GLSL_mul(v, GLSL_mat2(" in combined
    assert "v *= GLSL_mat2(" not in combined


def test_define_mul_matrix_ctor_rhs_wrapped():
    """wstXz8/wdSXW1/4dVXWz shape:
    `#define shash(p) fract(sin((p)*mat2(...))*43758.5453)`."""
    header, kernel = tp(
        "#define shash(p) fract(sin((p)*mat2(127.1, 311.7, 269.5, 183.3))*43758.5453)\n"
        + wrap_main(
            "    vec2 K = vec2(shash(fragCoord.xy), 0.0);\n"
        ))
    combined = header + kernel
    assert "GLSL_mul((p), GLSL_mat2(127.1f, 311.7f, 269.5f, 183.3f))" in combined


def test_define_mul_matrix_ctor_lhs_wrapped():
    """Ctor on the LEFT of `*`: `mat2(...)*v`."""
    header, kernel = tp(
        "#define RV(t) (mat2(1.0, 0.0, 0.0, 1.0)*vec2(t, t))\n"
        + wrap_main(
            "    vec2 p = RV(0.5);\n"
        ))
    combined = header + kernel
    assert "GLSL_mul(GLSL_mat2(1.0f, 0.0f, 0.0f, 1.0f), (float2)(t, t))" in combined


def test_define_plain_scalar_mul_untouched():
    """No matrix evidence -> never wrap (GLSL_mul has no scalar/vector-only
    overloads)."""
    header, kernel = tp(
        "#define DBL(x) ((x)*2.0)\n"
        + wrap_main(
            "    float y = DBL(3.0);\n"
        ))
    combined = header + kernel
    assert "GLSL_mul" not in combined
    assert "#define DBL(x) ((x)*2.0f)" in combined


def test_define_matrix_returning_macro_body_unchanged():
    """A bare matrix-ctor body (`#define rot(a) mat2(...)`) must stay a plain
    ctor (it feeds matrix_macros seeding); only a mul AROUND a ctor wraps."""
    header, kernel = tp(
        "#define rot(a) mat2(cos(a), -sin(a), sin(a), cos(a))\n"
        + wrap_main(
            "    vec2 p = fragCoord;\n"
            "    p *= rot(1.0);\n"
        ))
    combined = header + kernel
    assert "#define rot(a) GLSL_mat2(" in combined
    # The AST use-site still lowers via matrix_macros seeding.
    assert "GLSL_mul" in combined


# ============================================================================
# Part 3 — PROGRAM-SCOPE #if/#elif blocks that wrap whole FUNCTION DEFINITIONS
# (Session 63) — the case S53 deferred ("program-scope blocks keep the raw
# path"). A `#if DFx / #elif DFy` chain selecting one of several definitions of
# the same helper is a common Shadertoy idiom (e.g. mslfR2 "cubes"). Its bodies
# must be AST-routed just like statement-level blocks so out/inout params,
# matrix `*=`, and vec ctors lower correctly. Entry-point definitions inside a
# conditional stay on the raw path (S59 owns that flow).
# ============================================================================

_CUBES_HELPERS = (
    "mat3 g_rot = mat3(1.0);\n"
    "float torus(vec3 p, vec2 t){ return length(p)-t.x; }\n"
)


def test_program_scope_ifelif_function_out_param_routed():
    """mslfR2 shape: an out-param helper defined inside a program-scope
    #if/#elif chain must strip out -> pointer (not leak `out` into OpenCL)."""
    header, kernel = tp(
        "#define DF2\n"
        + _CUBES_HELPERS
        + "#if defined(DF1)\n"
        "float dfeffect(vec3 p, out float ogd) { ogd = 1.0; return p.x; }\n"
        "#elif defined(DF2)\n"
        "float dfeffect(vec3 p, out float ogd) {\n"
        "  vec3 p1 = p;\n"
        "  p1 *= g_rot;\n"
        "  float d1 = torus(p1, 10.0*vec2(1.0, 0.0125));\n"
        "  ogd = d1;\n"
        "  return d1;\n"
        "}\n"
        "#endif\n"
        + wrap_main(
            "  float g;\n"
            "  float d = dfeffect(vec3(fragCoord, 0.0), g);\n"
        )
    )
    combined = header + kernel
    # Directives preserved so the OpenCL preprocessor still picks one branch.
    assert "#if defined(DF1)" in combined
    assert "#elif defined(DF2)" in combined
    # out-param lowered to a pointer in BOTH the definition and the call site.
    assert "out float ogd" not in combined
    assert "float* ogd" in combined or "float *ogd" in combined
    assert "*ogd = d1" in combined
    # matrix compound-assign lowered.
    assert "p1 *= g_rot" not in combined
    assert "GLSL_mul" in combined
    # vec2 ctor lowered (no bare GLSL `vec2(` survives).
    assert "vec2(1.0" not in combined


def test_program_scope_ifelif_call_site_gets_address_of():
    """The top-level caller of a conditionally-defined out-param helper must
    pass the argument by address."""
    header, kernel = tp(
        "#define DF2\n"
        + _CUBES_HELPERS
        + "#if defined(DF1)\n"
        "float dfeffect(vec3 p, out float ogd) { ogd = 1.0; return p.x; }\n"
        "#elif defined(DF2)\n"
        "float dfeffect(vec3 p, out float ogd) { ogd = p.y; return p.x; }\n"
        "#endif\n"
        "float df(vec3 p){ float gd; float d = dfeffect(p, gd); return d + gd; }\n"
        + wrap_main("  float d = df(vec3(fragCoord, 0.0));\n")
    )
    combined = header + kernel
    assert "dfeffect(p, &gd)" in combined


def test_program_scope_ifdef_entry_point_stays_raw():
    """Regression guard for S59: an entry point trapped in a program-scope
    conditional must NOT be routed into a PreprocessorBlock (that flow is owned
    by transpile.py `_entry_trapped_in_conditional`). It must still transpile."""
    header, kernel = tp(
        "#define SIMPLE\n"
        "#ifdef SIMPLE\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "  fragColor = vec4(fragCoord, 0.0, 1.0);\n"
        "}\n"
        "#else\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "  fragColor = vec4(0.0);\n"
        "}\n"
        "#endif\n"
    )
    combined = header + kernel
    # Entry point resolved (S59 path) — SOME mainImage/kernel body emitted.
    assert "fragColor" in combined
