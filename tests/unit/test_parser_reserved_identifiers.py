"""
Unit tests for category U — user identifier collides with an OpenCL/C reserved
word (Session 41).

A GLSL shader may legally name a variable, parameter, or function `char`,
`kernel`, `local`, ... — none of those are GLSL keywords. But every one of them
IS a keyword in OpenCL C, so the emitted kernel either fails to compile
(`uint char = ...;` -> "cannot combine with previous declaration specifier") or,
in parameter/expression position, tree-sitter-glsl itself rejects the token and
the shader never parses (`float printChar(vec2 uv, uint char)`).

The fix is a whole-source pre-parse rename in `_normalize_array_syntax`: a
reserved word used as an identifier is suffixed with `_` (`char` -> `char_`)
everywhere it appears (declaration AND uses stay consistent). Because these
words are never valid GLSL type/qualifier keywords, any occurrence in GLSL source
is necessarily a user identifier, and any shader that uses one is already failing
downstream — so the rename can only fix, never regress.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.parser.glsl_parser import _normalize_array_syntax


@pytest.fixture
def parser():
    return GLSLParser()


# ---------------------------------------------------------------------------
# Parse-side: tree-sitter-glsl rejects the reserved word as an identifier
# (lscBW4: `uint char` parameter used in an expression).
# ---------------------------------------------------------------------------

def test_reserved_param_parses(parser):
    # A bare-paren use `(char >> 4)` is read by tree-sitter as a C-cast `(char)`
    # and rejected; the rename to `char_` restores it to an expression.
    parser.parse(
        "float printChar(vec2 uv, uint char) {\n"
        "    return textureLod(iChannel1, 0xFU - (char >> 4), 0.).a;\n"
        "}"
    )


def test_reserved_local_in_expression_parses(parser):
    parser.parse("void f() { uint char = 3u; float x = 0xFU - (char >> 4); }")


# ---------------------------------------------------------------------------
# Compile-side: parses today, but emits a keyword where an identifier belongs.
# ---------------------------------------------------------------------------

def test_reserved_local_decl_parses(parser):
    # Md2fzV / MtsXRX: `uint char = ...;` / `int char = ...;`
    parser.parse("void f() { uint char = 7u; }")


def test_reserved_kernel_local_parses(parser):
    # XtBXRm: `float kernel = sphere + 2.2;`
    parser.parse("void f() { float kernel = 2.2; }")


def test_reserved_function_name_parses(parser):
    # XtsGRl: `float char(float ch, vec2 uv) { ... }`
    parser.parse("float char(float ch, vec2 uv) { return ch; }")


# ---------------------------------------------------------------------------
# The rewrite itself.
# ---------------------------------------------------------------------------

def test_normalize_renames_reserved_local():
    assert _normalize_array_syntax("uint char = 7u;") == "uint char_ = 7u;"


def test_normalize_renames_kernel():
    assert _normalize_array_syntax("float kernel = 2.2;") == "float kernel_ = 2.2;"


def test_normalize_renames_decl_and_uses_consistently():
    src = "float char(float ch) { return ch; }\nfloat y = char(1.0);"
    out = _normalize_array_syntax(src)
    assert "float char_(float ch)" in out
    assert "char_(1.0)" in out


def test_normalize_leaves_identifier_substring_untouched():
    # `char` inside a longer identifier must not be renamed.
    src = "float printChar = charge + character;"
    assert _normalize_array_syntax(src) == src


def test_normalize_ignores_reserved_word_in_comment():
    src = "float x = 1.0; // char is a keyword\nfloat y = 2.0;"
    assert _normalize_array_syntax(src) == src


def test_normalize_preserves_line_numbers():
    src = "void f() {\n  uint char = 1u;\n  float k = float(char);\n}"
    out = _normalize_array_syntax(src)
    assert out.count("\n") == src.count("\n")
