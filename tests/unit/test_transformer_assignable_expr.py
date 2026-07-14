"""
Unit tests for the UNKNOWN "expression is not assignable" sub-cluster
(Session 27). Two independent emitter operator-emission bugs, both surfacing
as OpenCL's `error: expression is not assignable`.

1. Ternary with assignment branches — `cond ? a = b : c = d`.
   GLSL's grammar makes the third `?:` operand an `assignment_expression`, so
   `cond ? a=b : c=d` parses as `cond ? (a=b) : (c=d)`. C/OpenCL's third
   operand is only a `conditional-expression`, so the same text parses as
   `(cond ? (a=b) : c) = d` — the ternary result is not an lvalue:
       error: expression is not assignable
   Corpus casualties (sole-blockers): XtB3Dm, XsVyDh.
   Fix: parenthesize a ternary branch that is itself an assignment.

2. Adjacent unary operators — `- -1` / `+ +1`.
   A unary minus over a `-1` operand was emitted with no separator as `--1`,
   which C lexes as the pre-decrement operator applied to a literal:
       error: expression is not assignable
   Corpus casualties (sole-blockers): 4tffD8, 4dsfRn (both via constant-folded
   macro expansions like `vec2(u-1, v-1)`).
   Fix: emit a space between a `+`/`-`/`++`/`--` operator and an operand whose
   emitted form starts with `+`/`-`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


# ---------------------------------------------------------------------------
# 1. Ternary with assignment branches
# ---------------------------------------------------------------------------

def test_ternary_assignment_branches_parenthesized():
    # The XtB3Dm / XsVyDh shape: `cond ? a = b : c = d;`
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        vec3 m = vec3(0.0);
        float d = U.x, sgn = 1.0;
        d * sgn < 0. ? m.z = m.y : m.x = m.y;
        O = vec4(m, 1.0);
    }
    """
    kernel = _kernel(src)
    # Both assignment branches must be parenthesized so C keeps GLSL semantics.
    assert "(m.z = m.y)" in kernel
    assert "(m.x = m.y)" in kernel
    # The broken unparenthesized form must be gone.
    assert "? m.z = m.y : m.x = m.y" not in kernel


def test_ternary_nonassignment_branches_unchanged():
    # A plain value-returning ternary must NOT gain parentheses.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        float x = U.x > 0.5 ? 1.0 : 2.0;
        O = vec4(x);
    }
    """
    kernel = _kernel(src)
    assert "U.x > 0.5f ? 1.0f : 2.0f" in kernel


# ---------------------------------------------------------------------------
# 2. Adjacent unary operators
# ---------------------------------------------------------------------------

def test_adjacent_unary_minus_gets_space():
    # `- -1` must not collapse into `--1` (C pre-decrement).
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        vec2 v = vec2(-1, - -1);
        O = vec4(v, 0.0, 1.0);
    }
    """
    kernel = _kernel(src)
    assert "--1" not in kernel
    assert "- -1" in kernel


def test_single_unary_minus_unchanged():
    # A lone unary minus must stay tight: `-1`, not `- 1`.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        float a = -1.0;
        O = vec4(a);
    }
    """
    kernel = _kernel(src)
    assert "-1.0f" in kernel
    assert "- 1.0f" not in kernel
