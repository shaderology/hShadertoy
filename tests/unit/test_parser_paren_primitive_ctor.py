"""
Unit tests for category P, cluster 1 — a parenthesized primitive-type
constructor mis-parsed as a C-style cast (Session 22).

tree-sitter-glsl reads a *grouping* paren immediately followed by a scalar
primitive constructor — `(float( ... )` — as a C-style cast `(float)( ... )`
and raises a ParseError, even though GLSL has no C casts and this is a perfectly
legal constructor call inside a parenthesised expression. The idiom is extremely
common in loops: `(float(i) / float(N))`, `(float(j) * 1.79)`.

The parser normalizes it (like the existing array/qualifier rewrites) by
inserting a parse-neutral, semantically inert unary `+` after the opening paren:

    (float( ... )   ->   (+float( ... )

Unary `+` is identity on every numeric type and emits fine as OpenCL. `bool` is
excluded — unary `+` is illegal on bool.
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

def test_paren_float_ctor_parses(parser):
    parser.parse(_wrap("float k = (float(3) * 1.79);"))


def test_paren_float_ratio_parses(parser):
    # the canonical loop idiom
    parser.parse(_wrap("for (int i = 0; i < 4; i++) { float t = (float(i) / float(4)); }"))


def test_paren_int_ctor_parses(parser):
    parser.parse(_wrap("int n = (int(g.x) + 1);"))


def test_paren_uint_ctor_parses(parser):
    parser.parse(_wrap("uint u = (uint(g.x) << 2u);"))


def test_paren_double_ctor_parses(parser):
    parser.parse(_wrap("float d = float((double(g.x) + 1.0));"))


def test_nested_and_double_paren_parses(parser):
    parser.parse(_wrap("float k = ((float(2)) + float(3));"))


# ---------------------------------------------------------------------------
# The rewrite itself
# ---------------------------------------------------------------------------

def test_normalize_inserts_unary_plus():
    assert _normalize_array_syntax("x = (float(3));") == "x = (+float(3));"


def test_paren_bool_ctor_parses(parser):
    # Category P 4sjcz1: `(bool(x) ? a : b)` — a grouping paren before a bool
    # constructor is mis-read as the C-cast `(bool)(x)`. Unary `+` is illegal
    # on bool, so this is disambiguated with the identity `!!` instead.
    parser.parse(_wrap("float k = (bool(g.x) ? 1.0 : 0.0);"))


def test_normalize_inserts_double_not_on_bool_ctor():
    # `!!` is identity on bool and forces expression context (cannot be a cast)
    assert _normalize_array_syntax("x = (bool(a));") == "x = (!!bool(a));"


def test_normalize_leaves_call_arg_semantically_noop():
    # `vec4(float(...))` already parsed fine; over-matching only adds a
    # harmless no-op unary + inside the call arg — still valid.
    out = _normalize_array_syntax("c = vec4(float(a));")
    assert out == "c = vec4(+float(a));"


# ---------------------------------------------------------------------------
# Guard: a construct that already parsed must keep working (no regression)
# ---------------------------------------------------------------------------

def test_plain_ctor_without_wrapping_paren_still_parses(parser):
    parser.parse(_wrap("float k = float(3) * 1.79;"))


def test_vec_ctor_after_paren_untouched(parser):
    # `(vec2(...))` was never mis-parsed (not a scalar primitive) and must
    # remain unchanged.
    src = "x = (vec2(a) * 2.0);"
    assert _normalize_array_syntax(src) == src
    parser.parse(_wrap(src))
