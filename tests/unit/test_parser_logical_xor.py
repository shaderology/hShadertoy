"""
Unit tests for category P, cluster 3 — logical XOR `^^` (Session 23).

tree-sitter-glsl does not recognise GLSL's logical-XOR operator `^^`, so any
shader using it fails to parse. `a ^^ b` on bools is exactly `a != b`, but a
naive token swap is unsafe: `^^` binds looser than almost everything (only
`||` is looser), whereas `!=` binds at equality level. So the parser wraps each
operand in parentheses to preserve grouping:

    A ^^ B   ->   (A) != (B)

This is correct regardless of operand precedence, including the mixed case
`b ^^ f(...) == 1` (GLSL: `b ^^ (f(...) == 1)`), which a bare `b != f(...) == 1`
would mis-group as `(b != f(...)) == 1`.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.parser.glsl_parser import _normalize_array_syntax


@pytest.fixture
def parser():
    return GLSLParser()


def _wrap(body):
    return "void mainImage(out vec4 o, in vec2 g){ " + body + " o = vec4(0.0); }"


# ---------------------------------------------------------------------------
# The bug: these must now parse instead of raising ParseError
# ---------------------------------------------------------------------------

def test_xor_relational_parses(parser):
    parser.parse(_wrap("bool x = (g.x > 0.5) ^^ (g.y > 0.5);"))


def test_xor_nested_paren_operands_parses(parser):
    parser.parse(_wrap("bool x = (mod(g.x, 2.0) < 0.5) ^^ (mod(g.y, 2.0) > 0.5);"))


def test_xor_no_spaces_parses(parser):
    parser.parse(_wrap("bool x; if (g.x>0.5^^g.y>0.5) x = true;"))


def test_xor_mixed_equality_chain_parses(parser):
    parser.parse(_wrap("bool b = true; b = b ^^ int(g.x) == 1;"))


# ---------------------------------------------------------------------------
# The rewrite itself — wrapping preserves grouping
# ---------------------------------------------------------------------------

def test_normalize_wraps_simple_operands():
    assert _normalize_array_syntax("x = a ^^ b;") == "x = (a) != (b);"


def test_normalize_wraps_relational_operands():
    assert _normalize_array_syntax("if (p>0.5^^q>0.5)") == "if ((p>0.5) != (q>0.5))"


def test_normalize_mixed_equality_keeps_rhs_grouped():
    # RHS `f(x) == 1` must stay entirely on the right of !=
    out = _normalize_array_syntax("b = b ^^ f(x) == 1;")
    assert out == "b = (b) != (f(x) == 1);"


def test_normalize_nested_paren_operands():
    out = _normalize_array_syntax("(mod(a,2.0)<0.5)^^(mod(b,2.0)>0.5)")
    assert out == "((mod(a,2.0)<0.5)) != ((mod(b,2.0)>0.5))"


# ---------------------------------------------------------------------------
# Guards: `^^` inside comments must not be rewritten
# ---------------------------------------------------------------------------

def test_normalize_ignores_line_comment():
    src = "float x = 1.0; // arrow ^^^ up here\n"
    assert _normalize_array_syntax(src) == src


def test_normalize_ignores_block_comment():
    src = "/* see ^^ note */ float x = 1.0;"
    assert _normalize_array_syntax(src) == src
