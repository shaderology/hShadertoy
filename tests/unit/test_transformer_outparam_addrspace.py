"""
Unit tests for category B residual: address-space mismatch on out/inout pointer
params (Session 37).

`_transform_parameter` emitted out/inout params with an explicit `__private`
address-space qualifier: `void f(__private float4* p)`. A program-scope global
(category A leaves compile-time-constant-init globals at file scope, where
OpenCL places them in the `__global` address space) passed by address then
fails:

    float4 gState = (float4)(0.0f);      // program scope -> __global
    void save(__private float4* c){...}
    save(&gState);   // &gState is '__global float4*'
    -> error: passing '__global float4 *' to parameter of type '... *'
       changes address space of pointer

A probe on the campaign build target (no -cl-std) established: a BARE pointer
param (`float4* p`) accepts BOTH a `__global` arg (the global) AND a `__private`
arg (a local), while `__private float4* p` rejects the global. So the fix is to
drop the explicit `__private` qualifier and emit a bare pointer param.

Corpus casualties (sole-blockers): 4lSyRm, 4tGGzd, MlVSz1, MlyXzD, XltGDr,
XlycWh — multipass shaders that pass a program-scope global (or fragColor-like
buffer) to an inout helper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _full(src: str) -> str:
    r = transpile(src)
    return r.get_header() + "\n" + r.get_kernel()


def test_outparam_pointer_has_no_private_qualifier():
    src = """
    void f(out vec2 r) { r = vec2(1.0); }
    void mainImage(out vec4 O, in vec2 U) {
        vec2 a;
        f(a);
        O = vec4(a, 0.0, 1.0);
    }
    """
    out = _full(src)
    assert "float2* r" in out or "float2 *r" in out
    assert "__private float2* r" not in out
    assert "__private" not in out


def test_inout_pointer_has_no_private_qualifier():
    src = """
    void g(inout vec3 v) { v += vec3(1.0); }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 p = vec3(0.0);
        g(p);
        O = vec4(p, 1.0);
    }
    """
    out = _full(src)
    assert "float3* v" in out or "float3 *v" in out
    assert "__private" not in out


def test_global_passed_by_address_still_takes_address():
    # The MlVSz1 shape: a program-scope global passed to an inout helper. The
    # call site still takes the address; the param is now a bare pointer so the
    # __global address of the global is accepted.
    src = """
    vec4 gState = vec4(0.0);
    void save(inout vec4 c) { c = gState; }
    void mainImage(out vec4 O, in vec2 U) {
        save(gState);
        O = gState;
    }
    """
    out = _full(src)
    assert "void save(float4* c)" in out or "void save(float4 *c)" in out
    assert "save(&gState)" in out
    assert "__private" not in out
