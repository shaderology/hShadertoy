"""
Unit tests for comment interference in expressions.

Bug Context:
- tree-sitter emits comments as *named* children. Several transformer/parser
  handlers picked operands by positional `named_children` index, so a comment
  sitting between two operands shifted the indices and the real operand was
  silently dropped (or the structure was rejected).
- Example: `a  // note\n * b` emitted `a * ;` (right operand of `*` lost).
- This pattern is common in hand-formatted / "golf" Shadertoy shaders, so it
  silently corrupted otherwise-valid shaders.
- Fix: select operands via tree-sitter field names (which skip comments) with a
  comment-filtered positional fallback. Covers binary, assignment, member,
  subscript, and ternary expressions.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


@pytest.fixture
def parser():
    return GLSLParser()


@pytest.fixture
def transformer():
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker)


@pytest.fixture
def emitter():
    return OpenCLEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    return emitter.emit(transformed)


# ============================================================================
# Binary expressions
# ============================================================================

def test_binary_comment_between_operands_multiline(parser, transformer, emitter):
    """`a  // c\\n * b` must keep both operands of the multiplication."""
    glsl = "void f() {\n  float a, b;\n  float r = a // comment\n    * b;\n}"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "a * b" in out
    # The right operand must not be dropped (regression: produced `a * ;`).
    assert "* ;" not in out
    assert ";" not in out.split("a * b")[0][-3:]  # no stray empty operand


def test_binary_comment_does_not_drop_function_operand(parser, transformer, emitter):
    """Right operand that is a builtin call survives a preceding comment."""
    glsl = (
        "void f() {\n  float x;\n"
        "  float r = cos(x) // angular\n    * cos(x);\n}"
    )
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "GLSL_cos(x) * GLSL_cos(x)" in out


def test_binary_no_comment_still_works(parser, transformer, emitter):
    """Control: no comment, multiline binary still emits both operands."""
    glsl = "void f() {\n  float a, b;\n  float r = a\n    * b;\n}"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "a * b" in out


# ============================================================================
# Assignment expressions
# ============================================================================

def test_assignment_comment_before_value(parser, transformer, emitter):
    """`z = // c\\n 5.0;` must keep the assigned value."""
    glsl = "void f() {\n  float z;\n  z = // comment\n    5.0;\n}"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "z = 5.0f" in out
    assert "z = ;" not in out


# ============================================================================
# Member access (swizzle / field)
# ============================================================================

def test_member_access_comment_before_field(parser, transformer, emitter):
    """`v // c\\n .x` must keep the `.x` member access."""
    glsl = "void f() {\n  float4 v;\n  float r = v // comment\n    .x;\n}"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "v.x" in out
    assert "v." not in out.replace("v.x", "")  # no dangling `v.`


# ============================================================================
# Subscript
# ============================================================================

def test_subscript_comment_before_index(parser, transformer, emitter):
    """`a // c\\n [1]` must keep the index."""
    glsl = "void f() {\n  float a[2];\n  float r = a // comment\n    [1];\n}"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "a[1]" in out
    assert "a[]" not in out


# ============================================================================
# Ternary
# ============================================================================

def test_ternary_comment_before_branch(parser, transformer, emitter):
    """`c ? // note\\n 1.0 : 2.0` must transform without dropping branches."""
    glsl = "void f() {\n  bool c;\n  float r = c ? // note\n    1.0 : 2.0;\n}"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert "1.0f" in out and "2.0f" in out
    assert "?" in out and ":" in out
