"""
Unit tests for the UNKNOWN "struct out-param `&`" sub-cluster (Session 28).

A user struct passed to an `out`/`inout` parameter must have its address taken
at the call site (the callee param is pointerized to `Struct *`). The out-arg
`&`-insertion in `_transform_call_expression` handled plain identifiers and
array elements but excluded ALL member-access args, on the (correct-for-vectors)
grounds that `&v.xy` is not a valid swizzle address. That over-excluded struct
field access — `&cam.ray` IS valid — so a struct field passed to an out/inout
param compiled to:

    error: passing 'Ray' to parameter of incompatible type 'Ray *';
           take the address with &

Corpus casualties (all sole-blockers): MtdXRS, llcXRS, MldSW8 — each calls
`marchRay(cam.ray, col)` where `marchRay(inout Ray ray, inout vec4 colour)`.

Fix: distinguish a struct-field member access (base type is a user struct →
addressable) from a vector swizzle (base is a vector → not addressable), and
take the address of the former.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def test_struct_field_outarg_gets_address():
    # The MtdXRS/llcXRS/MldSW8 shape: struct field passed to an inout param.
    src = """
    struct Ray { vec3 o; vec3 d; };
    struct Cam { Ray ray; };
    void marchRay(inout Ray ray, inout vec4 colour) { colour.x += ray.o.x; }
    void mainImage(out vec4 O, in vec2 U) {
        Cam cam;
        vec4 col = vec4(0.0);
        marchRay(cam.ray, col);
        O = col;
    }
    """
    kernel = _kernel(src)
    assert "marchRay(&cam.ray, &col)" in kernel


def test_vector_swizzle_outarg_not_addressed():
    # A vector swizzle passed to an out param must NOT gain '&' — &v.xy is
    # invalid. (Such shaders don't compile on Shadertoy either, but the point
    # is the fix must not start emitting an illegal swizzle address.)
    src = """
    void bump(inout vec2 p) { p += 1.0; }
    void mainImage(out vec4 O, in vec2 U) {
        vec4 v = vec4(0.0);
        bump(v.xy);
        O = v;
    }
    """
    kernel = _kernel(src)
    assert "&v.xy" not in kernel


def test_plain_identifier_outarg_still_addressed():
    # Regression guard: the existing identifier out-arg path is untouched.
    src = """
    void bump(inout float x) { x += 1.0; }
    void mainImage(out vec4 O, in vec2 U) {
        float a = 0.0;
        bump(a);
        O = vec4(a);
    }
    """
    kernel = _kernel(src)
    assert "bump(&a)" in kernel
