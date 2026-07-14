"""
Unit tests for GLSL integer-vector `abs()` / `clamp()` overloads (UNKNOWN
sub-cluster, Session 31).

GLSL `abs()` and `clamp()` accept `genIType` (int, ivec2..4) as well as float
vectors. The runtime header `houdini/ocl/include/glslHelpers.h` previously
defined `GLSL_abs` / `GLSL_clamp` for `float`/`floatN` only, so an integer-
vector call had NO viable overload:

    error: no matching function for call to 'GLSL_abs'
        int2 q = GLSL_abs((rd + 1) / 2 * 7 - ro);   // arg is int2

    error: no matching function for call to 'GLSL_clamp'
        ... GLSL_clamp(uv, (int2)(0), convert_int2(iResolution.xy)) ...

Corpus casualties (sole-blockers): 4lBcRd (abs int3), lstXzs (clamp int2).

The transpiler already ROUTES these to `GLSL_abs`/`GLSL_clamp` (verified by
these tests); the fix is purely additive integer-vector overloads in the
runtime header (proven by the campaign, mirroring the category-M texture-bias
header fix). We deliberately add only the *vector* integer overloads: a
currently-passing shader cannot contain an int-vector arg to abs/clamp (it
would have had no viable overload = compile error), so no passing shader's
overload resolution can change — zero regression risk. Scalar-int overloads are
intentionally omitted (they would be the only resolution-shifting case).

These tests lock in the transpiler-side contract the header depends on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def test_int_vector_abs_routes_to_glsl_abs():
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        ivec3 b = ivec3(1, 2, 3);
        ivec3 s = abs(b);
        O = vec4(float(s.x));
    }
    """
    kernel = _kernel(src)
    assert "GLSL_abs(b)" in kernel


def test_int_vector_clamp_routes_to_glsl_clamp():
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        ivec2 uv = ivec2(U);
        ivec2 c = clamp(uv, ivec2(0), ivec2(100));
        O = vec4(float(c.x));
    }
    """
    kernel = _kernel(src)
    assert "GLSL_clamp(uv, (int2)(0), (int2)(100))" in kernel


def test_float_vector_abs_unchanged():
    # The float path must be untouched by the integer overloads.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        vec3 v = vec3(-1.0, 2.0, -3.0);
        O = vec4(abs(v), 1.0);
    }
    """
    kernel = _kernel(src)
    assert "GLSL_abs(v)" in kernel
