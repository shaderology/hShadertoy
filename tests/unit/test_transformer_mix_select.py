"""
Unit tests for mix(a, b, bool-vector) -> select (Bug-fix campaign follow-up).

GLSL mix(a, b, m) where m is a bool vector is component-wise SELECT
(m[i] ? b[i] : a[i]), NOT linear interpolation. We emit GLSL_mix(a, b, m) for
every mix, so a bvec/comparison mask hits "no matching function for call to
'GLSL_mix'" (first seen MdVBDV, surfaced once comparisons became int-vectors).

Fix: when mix's 3rd arg is a bool/int-vector mask (a relational expression or a
bvec-typed value), emit OpenCL select(a, b, mask). OpenCL `select` picks `b`
where the mask's sign bit is set — matching our -1-for-true relational results
and the GLSL mix(a,b,bvec) "pick b where true" semantics.

The float-`t` interpolation path (the overwhelmingly common case) MUST stay
GLSL_mix.
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


HEAD = ("void f() {\n"
        "  vec3 a = vec3(0.0); vec3 b = vec3(1.0);\n"
        "  vec3 x = vec3(0.1); vec3 y = vec3(0.2);\n")
TAIL = "\n}\n"


def test_mix_with_inline_comparison_to_select(parser, transformer, emitter):
    out = t(HEAD + "  vec3 r = mix(a, b, lessThan(x, y));" + TAIL,
            parser, transformer, emitter)
    assert 'select(a, b, (x < y))' in out
    assert 'GLSL_mix' not in out


def test_mix_with_bvec_variable_to_select(parser, transformer, emitter):
    out = t(HEAD + "  bvec3 m = lessThan(x, y);\n  vec3 r = mix(a, b, m);" + TAIL,
            parser, transformer, emitter)
    assert 'select(a, b, m)' in out
    assert 'GLSL_mix(a, b, m)' not in out


def test_mix_with_float_scalar_factor_unchanged(parser, transformer, emitter):
    """The interpolation path is sacred: float t must stay GLSL_mix."""
    out = t(HEAD + "  float w = 0.5;\n  vec3 r = mix(a, b, w);" + TAIL,
            parser, transformer, emitter)
    assert 'GLSL_mix(a, b, w)' in out
    assert 'select(' not in out


def test_mix_with_vector_interp_factor_unchanged(parser, transformer, emitter):
    """A float-vector interpolation factor is NOT a mask -> stays GLSL_mix."""
    out = t(HEAD + "  vec3 r = mix(a, b, vec3(0.5));" + TAIL,
            parser, transformer, emitter)
    assert 'GLSL_mix(a, b,' in out
    assert 'select(' not in out


def test_nested_mix_with_bvec_masks(parser, transformer, emitter):
    """MdVBDV pattern: mix(a, mix(a, b, m1), m2) with bvec masks -> nested select."""
    out = t(HEAD + "  bvec3 m1 = lessThan(x, y); bvec3 m2 = greaterThan(x, y);\n"
                   "  vec3 r = mix(a, mix(a, b, m1), m2);" + TAIL,
            parser, transformer, emitter)
    assert out.count('select(') == 2
    assert 'GLSL_mix' not in out
