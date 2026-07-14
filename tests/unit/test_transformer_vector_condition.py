"""
Unit tests for GLSL aggregate vector comparison operators (Bug-fix campaign
category O).

In GLSL, `==` / `!=` applied to whole vectors yield a SCALAR bool:
`v1 == v2` is "all components equal", `v1 != v2` is "any component differs".
The transpiler lowered them to the OpenCL component-wise relational operators,
which yield an int-vector mask — invalid wherever a scalar is required:

    if (tex.xy == vec2(0.))         "statement requires expression of scalar
                                     type ('int ext_vector_type(2)')"
    I == vec2(1) ? a : b            "vector condition type ... and result type
                                     ... do not have the same number of elements"
    return a == b;   (bool fn)      "returning int2 from a function with
                                     incompatible result type 'bool'"

Fix at the PRODUCER in `_transform_binary_expression`: a vector `==` becomes
`all(l == r)` and a vector `!=` becomes `any(l != r)` (scalar int 0/1 — the
OpenCL relational's -1-for-true sets the MSB that all()/any() test), so every
consumption site (if / ternary / && / return / bool assignment) is scalar.

The lessThan/equal/... builtin lowering constructs its mask BinaryOp directly
(not via _transform_binary_expression) and MUST keep yielding a raw mask —
those are GLSL bvec producers consumed by any()/all()/vec ctors explicitly.
Scalar comparisons are untouched.
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
    return ("vec4 f(vec2 uv, vec3 p3, vec4 p4, ivec2 ip2, uvec2 up2) {\n"
            + body + "\n}\n")


# ---- if (vec == vec) / if (vec != vec) --------------------------------------

def test_if_vec2_equality(parser, transformer, emitter):
    """The dominant O shape: if (tex.xy == vec2(0.))."""
    out = t(_fn("if (uv == vec2(0.0)) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(uv ==' in out


def test_if_vec3_nan_check(parser, transformer, emitter):
    """4lS3zw / ldG3Wh shape: if (col != col) — NaN check."""
    out = t(_fn("if (p3 != p3) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'any(p3 != p3)' in out


def test_if_swizzle_equality(parser, transformer, emitter):
    """Type inferred through a swizzle: if (p4.xy == vec2(0.))."""
    out = t(_fn("if (p4.xy == vec2(0.0)) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(p4.xy ==' in out


def test_if_builtin_call_equality(parser, transformer, emitter):
    """MsBczy shape: if (floor(u) == floor(m))."""
    out = t(_fn("if (floor(uv) == floor(uv)) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(GLSL_floor(uv) == GLSL_floor(uv))' in out


def test_if_uvec2_equality(parser, transformer, emitter):
    """XsKBWV shape: if (xy == uvec2(0u, 0u)) — uint vectors."""
    out = t(_fn("if (up2 == uvec2(0u, 0u)) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(up2 ==' in out


def test_if_ivec2_equality(parser, transformer, emitter):
    """tsfGW4 shape: int vectors."""
    out = t(_fn("if (ip2 == ivec2(1)) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(ip2 ==' in out


# ---- logical chains ----------------------------------------------------------

def test_and_or_of_vector_comparisons(parser, transformer, emitter):
    """MlByW3 shape: if (uv == vec3(0) && init != vec3(0) || flag)."""
    out = t(_fn("bool flag = false;\n"
                "if (p3 == vec3(0.0) && p3 != vec3(1.0) || flag)"
                " return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(p3 ==' in out
    assert 'any(p3 !=' in out


# ---- ternary condition -------------------------------------------------------

def test_ternary_vector_condition(parser, transformer, emitter):
    """Ml2fWG / Xty3Wz shape: I == vec2(1) ? a : b."""
    out = t(_fn("float k = uv == vec2(1.0) ? 1.0 : 0.0; return vec4(k);"),
            parser, transformer, emitter)
    assert 'all(uv ==' in out


def test_ternary_builtin_equality_condition(parser, transformer, emitter):
    """lstXzs shape: clamp(uv,0.,1.) == uv ? texture(...) : vec4(0.)."""
    out = t(_fn("vec4 c = clamp(uv, 0.0, 1.0) == uv ? vec4(1.0) : vec4(0.0);"
                " return c;"),
            parser, transformer, emitter)
    assert 'all(GLSL_clamp(uv, 0.0f, 1.0f) == uv)' in out


# ---- return from a bool function ---------------------------------------------

def test_return_bool_from_vector_equality(parser, transformer, emitter):
    """lstXzs shape: bool inside(ivec2 uv) { return a == b; }."""
    out = t("bool eq(vec2 a, vec2 b) { return a == b; }\n",
            parser, transformer, emitter)
    assert 'return all(a == b);' in out


def test_bool_declaration_from_vector_equality(parser, transformer, emitter):
    out = t(_fn("bool e = uv == vec2(0.0); return vec4(e ? 1.0 : 0.0);"),
            parser, transformer, emitter)
    assert 'all(uv ==' in out


# ---- must NOT wrap -----------------------------------------------------------

def test_scalar_comparison_untouched(parser, transformer, emitter):
    out = t(_fn("if (uv.x == 0.0) return vec4(1.0); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'all(' not in out
    assert 'uv.x == 0.0f' in out


def test_lessThan_family_mask_untouched(parser, transformer, emitter):
    """lessThan/equal builtins are bvec producers — masks must stay raw."""
    out = t(_fn("if (any(equal(uv, vec2(0.0)))) return vec4(1.0);"
                " return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'any((uv ==' in out
    assert 'any(all' not in out


def test_vector_ctor_from_relational_mask_untouched(parser, transformer, emitter):
    """Category-N mask normalization (vec4(a < b) -> convert &1) unaffected."""
    out = t(_fn("vec4 s = vec4(p4 < vec4(0.0)); return s;"),
            parser, transformer, emitter)
    assert 'convert_float4(' in out
    assert '& 1' in out
    assert 'all(' not in out
