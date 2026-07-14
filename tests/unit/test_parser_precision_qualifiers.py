"""
Unit tests for category P, cluster 2 — GLSL ES precision qualifiers (Session 23).

tree-sitter-glsl rejects the GLSL ES precision qualifiers `highp`/`mediump`/
`lowp` wherever they prefix a type (return types, parameters, const/local
declarations) AND the default-precision statement `precision highp float;`.
Both are meaningless in OpenCL, so the parser strips them before parsing:

    highp float hash(vec2 co)   ->   float hash(vec2 co)
    lowp vec2 A                 ->   vec2 A
    precision highp int;        ->   (removed)

Only horizontal whitespace is consumed, so no source line is ever merged and
ParseError line numbers stay accurate.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.parser.glsl_parser import _normalize_array_syntax


@pytest.fixture
def parser():
    return GLSLParser()


# ---------------------------------------------------------------------------
# The bug: these must now parse instead of raising ParseError
# ---------------------------------------------------------------------------

def test_precision_return_type_parses(parser):
    parser.parse("highp float hash(vec2 co) { return 0.0; }")


def test_precision_param_parses(parser):
    parser.parse("bool Is(vec2 uv, lowp vec2 A, lowp vec2 B) { return true; }")


def test_precision_local_decl_parses(parser):
    parser.parse("void f() { lowp vec3 AB, BC, AC; }")


def test_precision_const_decl_parses(parser):
    parser.parse("void f() { highp float a = 12.9898; }")


def test_precision_statement_parses(parser):
    parser.parse("precision highp float;\nconst float pi = 3.0;\nvoid f() {}")


def test_precision_bare_decl_parses(parser):
    # 4llcWn's `highp float;` -> `float;` (a valid empty declaration)
    parser.parse("highp float;\nconst float pi = 3.0;\nvoid f() {}")


# ---------------------------------------------------------------------------
# The rewrite itself
# ---------------------------------------------------------------------------

def test_normalize_strips_inline_qualifier():
    assert _normalize_array_syntax("highp float hash(") == "float hash("


def test_normalize_strips_lowp_param():
    assert _normalize_array_syntax("f(lowp vec2 A)") == "f(vec2 A)"


def test_normalize_removes_precision_statement():
    assert _normalize_array_syntax("precision highp int;") == ""


def test_normalize_preserves_line_numbers():
    src = "precision highp float;\nlowp vec3 x;\nfoo"
    out = _normalize_array_syntax(src)
    assert out.count("\n") == src.count("\n")
    assert out.splitlines()[1] == "vec3 x;"


def test_normalize_leaves_identifier_suffix_untouched():
    # `mediump`/`highp` embedded in a longer identifier must not be stripped
    src = "float highpValue = mediumpX;"
    assert _normalize_array_syntax(src) == src
