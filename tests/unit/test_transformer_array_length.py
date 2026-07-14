"""
Unit tests for the GLSL array `.length()` method (UNKNOWN sub-cluster,
Session 26).

GLSL arrays expose a compile-time `.length()` method returning the element
count. OpenCL C has no such method — `arr.length()` was previously emitted as
`arr.GLSL_length()` (the post-process builtin-prefix regex in
tests/transpile.py matched `.length(`), which the OpenCL compiler rejected:

    error: member reference base type '__global float2 [N]' is not a
    structure or union

Corpus casualties (all sole-blockers): 4tVcDK, MsVBzW, MstBR7, tdlGW8.

Fix (Session 26): the transformer rewrites `<array>.length()` to the standard
C element-count idiom `(sizeof(<array>) / sizeof(<array>[0]))`, which is a
compile-time constant and needs no size tracking. Guarded on a zero-argument
call whose callee is a field access named `length` — the only shape GLSL's
array `.length()` takes (the free function `length(v)` has an identifier
callee, so it is untouched).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def test_local_array_length_uses_sizeof_idiom():
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        float arr[4];
        int n = arr.length();
        O = vec4(float(n));
    }
    """
    kernel = _kernel(src)
    assert "sizeof(arr) / sizeof(arr[0])" in kernel
    # The broken member-call form must be gone.
    assert "arr.GLSL_length()" not in kernel
    assert ".length()" not in kernel


def test_array_length_as_loop_bound():
    # The MstBR7 shape: `for (i < poly.length() - 1; ...)`.
    src = """
    vec2 poly[3];
    void mainImage(out vec4 O, in vec2 U) {
        float d = 0.0;
        for (int i = 0; i < poly.length() - 1; ++i) {
            d += float(i);
        }
        O = vec4(d);
    }
    """
    kernel = transpile(src).full
    assert "sizeof(poly) / sizeof(poly[0])" in kernel
    assert ".GLSL_length()" not in kernel


def test_free_length_function_untouched():
    # The geometric builtin length(v) must still become GLSL_length(v).
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        float d = length(U);
        O = vec4(d);
    }
    """
    kernel = _kernel(src)
    assert "GLSL_length(U)" in kernel
    assert "sizeof" not in kernel
