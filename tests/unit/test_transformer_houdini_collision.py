"""
Unit tests for the D2 sub-cluster — user function name collides with a Houdini
builtin (Session 29).

The mass-test/fix campaign originally filed this under "D2 — overloadable
forward declarations", but the real root cause is different: a shader defines a
function whose name matches a function in one of the Houdini OpenCL headers that
`main_header.cl` `#include`s (e.g. `rotate2D` in <matrix.h>, `lerp` in
<interpolate.h>, `fit` in <interpolate.h>). Session 1 marks every user function
definition `__attribute__((overloadable))`; the Houdini builtin of the same name
is UNMARKED, so clang rejects the pair:

    error: redeclaration of 'rotate2D' must not have the 'overloadable' attribute
    error: redefinition of 'rotate2D'

(Forward declarations were already handled separately in category S via
`_transform_function_prototype`, so they are NOT the cause here.)

Corpus casualties (sole-blockers): 4l2XWw (rotate2D), Mssfz4 (rotate2D),
4tsXWn (lerp).

Fix: rename user functions whose name collides with a Houdini reserved builtin
to a `sh_`-prefixed name — at the definition, the forward-declaration prototype,
and every call site. A shader that defines such a name currently ALWAYS fails to
compile, so this rename can only fix, never regress.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _full(src: str) -> str:
    return transpile(src).full


def test_rotate2d_definition_and_call_renamed():
    # The 4l2XWw/Mssfz4 shape: a user rotate2D collides with matrix.h::rotate2D.
    src = """
    float2 rotate2D(float2 p, float t) {
        float c = cos(t), s = sin(t);
        return (float2)(c * p.x - s * p.y, s * p.x + c * p.y);
    }
    void mainImage(out vec4 O, in vec2 U) {
        float2 uv = rotate2D(U, 1.0);
        O = vec4(uv, 0.0, 1.0);
    }
    """
    full = _full(src)
    # Definition renamed
    assert "sh_rotate2D(float2 p, float t)" in full
    # Call site renamed
    assert "sh_rotate2D(U" in full
    # The colliding name must not survive as a bare definition/call
    assert "float2 rotate2D(" not in full
    assert "= rotate2D(" not in full


def test_lerp_renamed():
    # The 4tsXWn shape: a user lerp collides with interpolate.h::lerp.
    src = """
    float lerp(float a, float b, float s) { return a + (b - a) * s; }
    void mainImage(out vec4 O, in vec2 U) {
        float v = lerp(0.2, 0.8, U.x);
        O = vec4(v);
    }
    """
    full = _full(src)
    assert "sh_lerp(float a, float b, float s)" in full
    assert "sh_lerp(0.2f" in full
    assert "float lerp(" not in full


def test_recursive_self_call_renamed():
    # A colliding function that calls itself must rename the recursive call too.
    src = """
    float fit(float x, int n) {
        if (n <= 0) return x;
        return fit(x * 0.5, n - 1);
    }
    void mainImage(out vec4 O, in vec2 U) {
        O = vec4(fit(U.x, 3));
    }
    """
    full = _full(src)
    assert "sh_fit(float x, int n)" in full
    # both the outer call and the recursive self-call renamed
    assert full.count("sh_fit(") >= 3


def test_forward_declared_collision_renamed_consistently():
    # Prototype + later definition of a colliding name: both renamed, and the
    # call site between them too.
    src = """
    float2 rotate2D(float2 p, float t);
    void mainImage(out vec4 O, in vec2 U) {
        O = vec4(rotate2D(U, 1.0), 0.0, 1.0);
    }
    float2 rotate2D(float2 p, float t) {
        return p * t;
    }
    """
    full = _full(src)
    # prototype
    assert "sh_rotate2D(float2 p, float t);" in full
    # definition
    assert "sh_rotate2D(float2 p, float t) {" in full
    # call
    assert "sh_rotate2D(U" in full
    assert "float2 rotate2D(" not in full


def test_non_colliding_user_function_unchanged():
    # A user function whose name is NOT a Houdini builtin keeps its name.
    src = """
    float myHelper(float x) { return x * 2.0; }
    void mainImage(out vec4 O, in vec2 U) {
        O = vec4(myHelper(U.x));
    }
    """
    full = _full(src)
    assert "myHelper(float x)" in full
    assert "myHelper(U.x)" in full
    assert "sh_myHelper" not in full
