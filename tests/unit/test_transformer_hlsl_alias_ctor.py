"""
Unit tests for HLSL-style vector-type aliases used as constructors
(bug-fix campaign category J, second sub-cluster).

Shadertoy authors porting from HLSL commonly write::

    #define float2 vec2
    #define float3 vec3
    #define float4 vec4

then call the constructor by its OpenCL/HLSL spelling: ``float2(x, y)``.
tree-sitter parses ``float2`` as an ordinary identifier (it is not a GLSL
type), so the AST transformer saw an unknown call and passed it through
unchanged. In OpenCL ``float2`` IS a type, so ``float2(x, y)`` compiles to
"unexpected type name 'float2': expected expression".

Fix: `_transform_call_expression` normalizes an OpenCL vector-type callee
(float2/3/4, int2/3/4, uint2/3/4) to its GLSL name via OPENCL_TO_GLSL_NAME,
so the existing constructor logic (incl. category-N single-arg conversions)
emits the correct ``(float2)(...)`` cast / conversion.
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
    return ("vec4 f(vec2 uv, vec3 p3, ivec3 ip3) {\n" + body + "\n}\n")


def test_float2_alias_ctor_two_args(parser, transformer, emitter):
    """float2(a, b) -> (float2)(a, b) — the lv() shape."""
    out = t(_fn("float2 dpp = float2(uv.y, -uv.x); return float4(0.0);"),
            parser, transformer, emitter)
    assert "(float2)(uv.y, -uv.x)" in out
    # No bare constructor-call spelling survives (that is the compile error).
    assert "= float2(" not in out


def test_float3_alias_ctor_three_args(parser, transformer, emitter):
    """float3(x, y, z) -> (float3)(x, y, z)."""
    out = t(_fn("float3 c = float3(1.0, 2.0, 3.0); return float4(c, 1.0);"),
            parser, transformer, emitter)
    assert "(float3)(1.0f, 2.0f, 3.0f)" in out


def test_float4_alias_ctor_four_args(parser, transformer, emitter):
    """float4(...) -> (float4)(...)."""
    out = t(_fn("return float4(0.1, 0.2, 0.7, 1.0);"),
            parser, transformer, emitter)
    assert "(float4)(0.1f, 0.2f, 0.7f, 1.0f)" in out


def test_int2_alias_single_vector_arg_converts(parser, transformer, emitter):
    """int2(vec2) still routes through the category-N conversion path."""
    out = t(_fn("int2 q = int2(uv); return float4(0.0);"),
            parser, transformer, emitter)
    assert "convert_int2(uv)" in out


def test_float2_alias_scalar_broadcast(parser, transformer, emitter):
    """float2(scalar) broadcast -> (float2)(scalar)."""
    out = t(_fn("float2 v = float2(0.5); return float4(0.0);"),
            parser, transformer, emitter)
    assert "(float2)(0.5f)" in out


# ---------------------------------------------------------------------------
# Category J: `p *= rot(...)` where rot is a matrix-returning #define.
# The preprocessor records rot as matrix-returning and transpile() seeds it
# into the AST's user_function_return_types, so the matmul dispatcher fires.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def test_vec_compound_mul_by_matrix_macro():
    """p *= rot(a) with rot a #define'd mat2 -> GLSL_mul_vec2_mat2 (was the
    unmasked-E blocker of the mat2-rotation-macro shaders).

    This shader parses cleanly, so the Session-24 function-like macro expander is
    gated OFF for it — the matrix-macro-return-type tracking path (which records
    rot as matrix-returning and seeds user_function_return_types) still owns it."""
    glsl = (
        "#define rot(a) mat2(cos(a),-sin(a),sin(a),cos(a))\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 p = fragCoord;\n"
        "    p *= rot(0.2);\n"
        "    fragColor = vec4(p, 0.0, 1.0);\n"
        "}\n"
    )
    result = transpile(glsl)
    body = result.get_header() + result.get_kernel()
    assert "GLSL_mul_vec2_mat2(p, rot(0.2f))" in body
    assert "p *= rot(" not in body
