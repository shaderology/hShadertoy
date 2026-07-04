"""
Unit tests for missing-GLSL-builtin mappings (Bug-fix campaign category X).

Category X = a GLSL builtin we don't provide is emitted verbatim, so the OpenCL
compiler reports "implicit declaration of function 'X'" / ptxas "unresolved
extern function". Two transformer-only sub-fixes, both mapping to native OpenCL
builtins/operators (no GLSL_* runtime helper, so no Houdini round-trip):

1. Bit-cast reinterprets, size-suffixed by argument width:
     uintBitsToFloat / intBitsToFloat -> as_float / as_float2/3/4
     floatBitsToUint                  -> as_uint  / as_uint2/3/4
     floatBitsToInt                   -> as_int   / as_int2/3/4

2. Component-wise comparison functions -> OpenCL relational operators
   (which return int-vectors per component):
     lessThan -> <   lessThanEqual -> <=   greaterThan -> >
     greaterThanEqual -> >=   equal -> ==   notEqual -> !=
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


def _fn(body):
    return "float f(uint n, float x, uvec3 u3, vec4 v4) {\n" + body + "\n}\n"


# ---- bit-casts -------------------------------------------------------------

def test_uintBitsToFloat_scalar(parser, transformer, emitter):
    out = t(_fn("float r = uintBitsToFloat(n); return r;"), parser, transformer, emitter)
    assert 'as_float(n)' in out
    assert 'uintBitsToFloat' not in out


def test_intBitsToFloat_scalar(parser, transformer, emitter):
    out = t(_fn("int i = 3; float r = intBitsToFloat(i); return r;"), parser, transformer, emitter)
    assert 'as_float(i)' in out
    assert 'intBitsToFloat' not in out


def test_floatBitsToUint_scalar(parser, transformer, emitter):
    out = t(_fn("uint r = floatBitsToUint(x); return float(r);"), parser, transformer, emitter)
    assert 'as_uint(x)' in out
    assert 'floatBitsToUint' not in out


def test_floatBitsToInt_scalar(parser, transformer, emitter):
    out = t(_fn("int r = floatBitsToInt(x); return float(r);"), parser, transformer, emitter)
    assert 'as_int(x)' in out
    assert 'floatBitsToInt' not in out


def test_bitcast_vec3_width_suffix(parser, transformer, emitter):
    out = t(_fn("vec3 r = uintBitsToFloat(u3); return r.x;"), parser, transformer, emitter)
    assert 'as_float3(u3)' in out


def test_bitcast_vec4_width_suffix(parser, transformer, emitter):
    out = t(_fn("uvec4 r = floatBitsToUint(v4); return float(r.x);"), parser, transformer, emitter)
    assert 'as_uint4(v4)' in out


def test_bitcast_vec2_constructor_arg(parser, transformer, emitter):
    out = t(_fn("vec2 r = uintBitsToFloat(uvec2(1u, 2u)); return r.x;"),
            parser, transformer, emitter)
    assert 'as_float2(' in out


# ---- comparison functions --------------------------------------------------

def test_lessThan_to_operator(parser, transformer, emitter):
    out = t(_fn("bvec2 b = lessThan(vec2(x), vec2(1.0)); return b.x ? 1.0 : 0.0;"),
            parser, transformer, emitter)
    assert 'lessThan' not in out
    assert '<' in out


def test_greaterThanEqual_to_operator(parser, transformer, emitter):
    out = t(_fn("bvec2 b = greaterThanEqual(vec2(x), vec2(1.0)); return b.x ? 1.0 : 0.0;"),
            parser, transformer, emitter)
    assert 'greaterThanEqual' not in out
    assert '>=' in out


def test_equal_notEqual_to_operator(parser, transformer, emitter):
    out = t(_fn("bvec3 a = equal(u3, u3); bvec3 c = notEqual(u3, u3); return a.x||c.x ? 1.0:0.0;"),
            parser, transformer, emitter)
    assert 'equal(' not in out and 'notEqual(' not in out
    assert '==' in out and '!=' in out


def test_all_lessThan_compiles(parser, transformer, emitter):
    """The all(lessThan(..)) pattern: all((a < b)) — relational result feeds all()."""
    out = t(_fn("bool g = all(lessThan(vec2(x), vec2(1.0))); return g ? 1.0 : 0.0;"),
            parser, transformer, emitter)
    assert 'lessThan' not in out
    assert 'all(' in out
    assert '<' in out


def test_user_function_named_equal_not_hijacked(parser, transformer, emitter):
    """A user-defined function shadowing a comparison name must NOT be rewritten."""
    glsl = """
    bool equal(float a, float b) { return a == b; }
    float g(float x) { return equal(x, 1.0) ? 1.0 : 0.0; }
    """
    out = t(glsl, parser, transformer, emitter)
    # the call site must still call the user function, not become `(x == 1.0)`
    assert 'equal(x, 1.0f)' in out
