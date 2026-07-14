"""
Unit tests for category B residual: a vector SWIZZLE passed to an out/inout
parameter (Session 36).

The hg_sdf domain-operator idiom is ubiquitous in raymarching shaders:

    void pR(inout vec2 p, float a) { ... }   // rotate
    pR(p.xz, iTime);                          // called on a swizzle

A vector swizzle (`p.xz`, even single-component `p.z`) is NOT an addressable
lvalue in OpenCL — `&p.xz` is illegal ("address of vector element"). So the
out-arg `&`-insertion in `_transform_call_expression` skipped swizzles, emitting
`pR(p.xz, ...)` → `error: passing 'float2' to parameter of incompatible type
'float2 *'`.

GLSL defines a swizzle out-arg by copy-in/copy-out. The fix lowers the call
statement to a block:

    { float2 _cico0 = p.xz; pR(&_cico0, ...); p.xz = _cico0; }

Corpus casualties (sole-blockers): MstBR4, ldKBRt, lsGyDt, ll3SDN, lttGzs,
XtGBDh, lljBzz, MscGzs, MltcDS, WdB3Dw, ... — all the `pR`/`pMod1`/`pMirror`
rotate/mirror helpers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def test_multi_component_swizzle_outarg_copy_in_out():
    src = """
    void pR(inout vec2 p, float a) { p = p + a; }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 p = vec3(U, 0.0);
        pR(p.zy, 1.0);
        O = vec4(p, 1.0);
    }
    """
    kernel = _kernel(src)
    # copy-in temp, call takes its address, copy-out writeback
    assert "= p.zy;" in kernel          # copy-in
    assert "pR(&" in kernel             # address of the temp
    assert "p.zy =" in kernel           # copy-out writeback
    # and the illegal address-of-swizzle must NOT appear
    assert "&p.zy" not in kernel
    assert "pR(p.zy" not in kernel


def test_single_component_swizzle_outarg_copy_in_out():
    # Single-component vector swizzle `p.x` is also non-addressable.
    src = """
    void pMod1(inout float p, float s) { p = p + s; }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 p = vec3(U, 0.0);
        pMod1(p.x, 2.0);
        O = vec4(p, 1.0);
    }
    """
    kernel = _kernel(src)
    assert "= p.x;" in kernel
    assert "pMod1(&" in kernel
    assert "p.x =" in kernel
    assert "&p.x" not in kernel


def test_swizzle_outarg_in_decl_init_copy_in_out():
    # Session 39: the same hg_sdf idiom in a DECLARATION initializer
    # (`float c = pMod1(p.z, 7.5);`) fell through the S36 gate, which only
    # drained at bare expression statements. The declaration cannot be
    # block-wrapped (the binding must stay in scope), so the prelude/writeback
    # are spliced as sibling statements around it.
    # Corpus: XtGBDh ldKBRt lljBzz ltcBzN MdVfWw 4d3BDM.
    src = """
    float pMod1(inout float p, float size) { p = p + size; return p; }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 p = vec3(U, 0.0);
        float c = pMod1(p.z, 7.5);
        O = vec4(p, c);
    }
    """
    kernel = _kernel(src)
    assert "= p.z;" in kernel            # copy-in temp before the declaration
    assert "pMod1(&" in kernel           # call takes the temp's address
    assert "p.z =" in kernel             # writeback after the declaration
    assert "pMod1(p.z" not in kernel
    assert "&p.z" not in kernel
    # ordering: copy-in < declaration < writeback, all in the same scope
    ci = kernel.index("= p.z;")
    decl = kernel.index("float c = pMod1")
    wb = kernel.index("p.z =")
    assert ci < decl < wb


def test_multi_component_swizzle_outarg_in_decl_init():
    # 2-component variant (`pModPolar(p.xz, ...)` — MdVfWw).
    src = """
    float pModPolar(inout vec2 p, float n) { p = p * n; return n; }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 p = vec3(U, 0.0);
        float ofs = pModPolar(p.xz, 5.0);
        O = vec4(p, ofs);
    }
    """
    kernel = _kernel(src)
    assert "= p.xz;" in kernel
    assert "pModPolar(&" in kernel
    assert "p.xz =" in kernel
    assert "pModPolar(p.xz" not in kernel


def test_plain_decl_init_call_unchanged():
    # Regression guard: a declaration whose initializer passes a plain
    # identifier out-arg keeps the direct `&a` path - no temps, no splicing.
    src = """
    float bump(inout float x) { x += 1.0; return x; }
    void mainImage(out vec4 O, in vec2 U) {
        float a = 0.0;
        float b = bump(a);
        O = vec4(a, b, 0.0, 1.0);
    }
    """
    kernel = _kernel(src)
    assert "bump(&a)" in kernel
    assert "_cico" not in kernel


def test_plain_identifier_outarg_unchanged():
    # Regression guard: a plain identifier out-arg is still `&a`, no copy-out.
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
    assert "= a;" not in kernel.replace("float a = 0.0f;", "")  # no spurious copy-in
