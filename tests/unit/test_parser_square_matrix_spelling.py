"""
Unit tests for category X — square full-name matrix type spellings (Session 54).

GLSL allows the redundant `matNxN` spelling as an exact synonym of `matN`
(`mat3x3` == `mat3`, `mat2x2` == `mat2`, `mat4x4` == `mat4`). tree-sitter-glsl
and the OpenCL compiler know only `matN`, so `mat3x3` leaked through verbatim
and clang errored `unknown type name 'mat3x3'`. The parser now normalizes the
square spellings before parsing, in both type-name and constructor position:

    mat3x3 m = mat3x3(a, b, c);   ->   mat3 m = mat3(a, b, c);

Non-square spellings (`mat2x4`, `mat3x2`, `mat4x2`) are genuinely distinct
types and are deliberately left untouched (deferred — need real struct types).
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.parser.glsl_parser import _normalize_array_syntax


@pytest.fixture
def parser():
    return GLSLParser()


# ---------------------------------------------------------------------------
# The string-level normalization
# ---------------------------------------------------------------------------

def test_mat3x3_type_and_ctor_normalized():
    out = _normalize_array_syntax("mat3x3 m = mat3x3(a, b, c);")
    assert "mat3x3" not in out
    assert "mat3 m = mat3(a, b, c);" in out


def test_mat2x2_normalized():
    assert _normalize_array_syntax("mat2x2 m;") == "mat2 m;"


def test_mat4x4_normalized():
    assert _normalize_array_syntax("mat4x4 m = mat4x4(1.0);") == "mat4 m = mat4(1.0);"


def test_nonsquare_matrix_untouched():
    # mat2x4 / mat3x2 / mat4x2 are distinct types — must NOT be rewritten.
    src = "mat2x4 a; mat3x2 b; mat4x2 c;"
    assert _normalize_array_syntax(src) == src


def test_word_boundary_not_over_matching():
    # A longer identifier containing the pattern must be left alone.
    src = "float mat3x3_scale = 1.0;"
    assert "mat3x3_scale" in _normalize_array_syntax(src)


# ---------------------------------------------------------------------------
# End-to-end: the bug used to make these unparseable / mistranspiled
# ---------------------------------------------------------------------------

def test_mat3x3_declaration_parses(parser):
    parser.parse("void f() { mat3x3 cam_mat; }")


def test_mat3x3_ctor_parses(parser):
    parser.parse("void f() { mat3x3 m = mat3x3(vec3(1.0), vec3(2.0), vec3(3.0)); }")
