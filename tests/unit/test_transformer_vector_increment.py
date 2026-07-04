"""
Unit tests for two Wave-1 transformer fixes:

- Category AA: OpenCL forbids ++/-- on vector types ("cannot increment value of
  type 'float4'"). A vector v++/v--/++v/--v must become v += 1 / v -= 1; a
  scalar increment must be left unchanged.

- Category L: a hex integer literal like 0x9e3853U contains the hex digit 'e',
  which the number-literal transform misread as a float exponent and appended
  'f' (-> invalid 'Uf' suffix, or a silently wrong value like 0xE9 -> 0xE9f).
  Hex literals must pass through untouched.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
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


def t(glsl_code, parser, transformer, emitter):
    ast = parser.parse(glsl_code)
    return emitter.emit(transformer.transform(ast))


# ---- AA: vector ++/-- ------------------------------------------------------

def test_vector_postinc_to_compound(parser, transformer, emitter):
    out = t("void f() { float4 v = (float4)(0.0); v++; }", parser, transformer, emitter)
    assert 'v += 1' in out
    assert '++' not in out


def test_vector_postdec_to_compound(parser, transformer, emitter):
    out = t("void f() { float4 v = (float4)(0.0); v--; }", parser, transformer, emitter)
    assert 'v -= 1' in out
    assert '--' not in out


def test_vector_preinc_to_compound(parser, transformer, emitter):
    out = t("void f() { float3 v = (float3)(0.0); ++v; }", parser, transformer, emitter)
    assert 'v += 1' in out


def test_vector_swizzle_decrement(parser, transformer, emitter):
    """A multi-component swizzle is a vector lvalue too -> compound assignment."""
    out = t("void f() { float4 O = (float4)(0.0); O.gb--; }", parser, transformer, emitter)
    assert 'O.gb -= 1' in out


def test_scalar_increment_unchanged(parser, transformer, emitter):
    out = t("void f() { int i = 0; i++; float s = 0.0; s++; }", parser, transformer, emitter)
    assert 'i++' in out or '++i' in out
    assert 's++' in out or '++s' in out
    assert '+= 1' not in out


def test_scalar_swizzle_component_unchanged(parser, transformer, emitter):
    """A single-component swizzle (O.a) is a scalar -> keep ++ (not required to
    change; correctness is preserved either way)."""
    out = t("void f() { float4 O = (float4)(0.0); O.a++; }", parser, transformer, emitter)
    # Either form is acceptable for a scalar; just ensure it still transpiles and
    # references the component.
    assert 'O.a' in out


# ---- L: hex literals not corrupted -----------------------------------------

def test_hex_with_e_digit_not_floatified(parser, transformer, emitter):
    out = t("void f() { uint h = 0x9e3853U; }", parser, transformer, emitter)
    assert '0x9e3853U' in out
    assert '0x9e3853Uf' not in out


def test_hex_uppercase_e_not_floatified(parser, transformer, emitter):
    out = t("void f() { int h = 0xE9; }", parser, transformer, emitter)
    assert '0xE9' in out
    assert '0xE9f' not in out


def test_hex_constant_mask_intact(parser, transformer, emitter):
    out = t("void f() { uint m = 0x7fffffffU; uint n = 0xffffffU; }", parser, transformer, emitter)
    assert '0x7fffffffU' in out and '0xffffffU' in out


def test_real_float_still_gets_suffix(parser, transformer, emitter):
    """Guard: genuine floats (decimal / exponent) still get the 'f' suffix."""
    out = t("void f() { float a = 2.0; float b = 1e5; }", parser, transformer, emitter)
    assert '2.0f' in out
    assert '1e5f' in out
