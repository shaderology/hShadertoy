"""
Category W — scalar-argument geometric builtins.

GLSL permits dot()/normalize()/length()/distance() on *scalars*
(dot(a,b)==a*b, normalize(x)==sign(x), length(x)==|x|, distance(a,b)==|a-b|).
Real corpus shaders call e.g. `dot(length(uv), 4.35602)` — both args float.

The transpiler already emits the GLSL_-prefixed wrapper regardless of arg type;
these tests pin that transpile-side contract. The *actual* W bug (clang picked
"ambiguous" among the three vector overloads because glslHelpers.h had no scalar
overload) is only visible at OpenCL compile time and is proven by the corpus
re-test — a unit test cannot exercise the header. See fixcampaign PROGRESS.md.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


def transform_and_emit(glsl_code: str) -> str:
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    emitter = CodeEmitter()
    ast = parser.parse(glsl_code)
    return emitter.emit(transformer.transform(ast))


def test_scalar_dot_emits_glsl_dot():
    """dot() on two scalar floats still routes through GLSL_dot (scalar overload)."""
    glsl = "void test() { float a = 1.0; float b = 2.0; float x = dot(a, b); }"
    opencl = transform_and_emit(glsl)
    assert 'GLSL_dot(a, b)' in opencl


def test_scalar_normalize_emits_glsl_normalize():
    """normalize() on a scalar float still routes through GLSL_normalize."""
    glsl = "void test() { float a = 3.0; float x = normalize(a); }"
    opencl = transform_and_emit(glsl)
    assert 'GLSL_normalize(a)' in opencl


def test_nested_scalar_dot():
    """Nested scalar dot (Xld3Ws idiom) — inner and outer both scalar."""
    glsl = "void test() { vec4 c = vec4(1.0); float x = dot(dot(c.x, c.y), dot(c.z, c.w)); }"
    opencl = transform_and_emit(glsl)
    assert opencl.count('GLSL_dot') == 3
