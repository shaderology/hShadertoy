"""
Unit tests for the GLSL comma (sequence) operator and comments inside a
parenthesized expression (category AD, Session 45).

Both bugs manifested identically: a parenthesized sub-expression collapsed to
empty `()`, so clang reported `error: expected expression`.

1. Comma operator — `(A, B, C)` is a valid GLSL/C sequence expression that
   evaluates each operand and yields the last. tree-sitter parses it as a
   `comma_expression` node, but the transformer had no handler for that node
   type, so `_transform_node` returned None and the parens emitted empty.
   Corpus casualties (sole-blockers): 4lyyW1 (`vec3((1.25, 1., 1.2) - tint)`),
   ldtBW2 (`float b = (min(...), min(...));`).

2. Comment inside the parens — `if ( //note\n x >= 0 )`. tree-sitter keeps the
   `comment` as the FIRST named child of the `parenthesized_expression`, and the
   transformer emitted `named_children[0]` (the comment → nothing) instead of the
   real condition. Corpus casualty (sole-blocker): MsVBzW.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def test_comma_expression_as_initializer():
    # ldtBW2 shape: initializer is a parenthesized comma expression.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        vec2 xy = U;
        float b = (
            min(abs(xy.x - 0.5), abs(xy.x + 0.5)),
            min(abs(xy.y - 0.5), abs(xy.y + 0.5))
        );
        O = vec4(b);
    }
    """
    kernel = _kernel(src)
    assert "= ()" not in kernel
    # The last operand must survive; the sequence must be emitted verbatim.
    assert "GLSL_min(GLSL_abs(xy.y - 0.5f), GLSL_abs(xy.y + 0.5f))" in kernel
    # Both operands separated by a comma inside preserved parentheses.
    assert "float b = (GLSL_min(GLSL_abs(xy.x - 0.5f), GLSL_abs(xy.x + 0.5f)), " in kernel


def test_comma_expression_in_constructor_argument():
    # 4lyyW1 shape: (1.25, 1., 1.2) - tint  inside a vec3 constructor.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        vec3 tint = vec3(0.1);
        O = vec4(vec3((1.25, 1., 1.2) - tint), 1.0);
    }
    """
    kernel = _kernel(src)
    assert "() - tint" not in kernel
    assert "(1.25f, 1.f, 1.2f) - tint" in kernel


def test_comment_inside_if_condition_parens():
    # MsVBzW shape: a leading // comment inside the if-condition parentheses.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        ivec2 triidx = ivec2(0);
        for (int i = 0; i < 4; i++) {
            if ( //early out
                triidx.x >= 0 ) break;
        }
        O = vec4(0.0);
    }
    """
    kernel = _kernel(src)
    assert "if ()" not in kernel
    assert "if (triidx.x >= 0)" in kernel


def test_plain_parenthesized_expression_unchanged():
    # Guard: a normal single-expression paren must still round-trip.
    src = """
    void mainImage(out vec4 O, in vec2 U) {
        float x = (1.0 + 2.0) * 3.0;
        O = vec4(x);
    }
    """
    kernel = _kernel(src)
    assert "(1.0f + 2.0f) * 3.0f" in kernel
