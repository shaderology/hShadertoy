"""
Unit tests for user-function overloading (Bug-fix campaign category D).

OpenCL C has no function overloading by default. A GLSL shader that defines
two same-named functions with different signatures (e.g. hash(vec2) and
hash(vec3)), or redefines a name, compiles in GLSL but emits
"conflicting types for X" / "redefinition of X" in OpenCL.

Fix: mark every user function definition with __attribute__((overloadable))
so the OpenCL compiler resolves them as overloads (the GLSL_* runtime helpers
already use this attribute).

mainImage is the kernel entry point (its body is extracted by transpile.py and
re-wrapped in a fixed signature) and must NOT carry the attribute.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter

ATTR = "__attribute__((overloadable))"


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
    """Helper: parse, transform, and emit code."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    return emitter.emit(transformed)


def test_overloaded_functions_marked_overloadable(parser, transformer, emitter):
    """Two same-named user functions of different signatures both get the attribute."""
    glsl = """
    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(1.0, 2.0))) * 1e4);
    }
    float hash(vec3 p) {
        return fract(sin(dot(p, vec3(1.0, 2.0, 3.0))) * 1e4);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Both definitions must carry the overloadable attribute.
    assert opencl.count(ATTR) == 2
    assert f"{ATTR}\nfloat hash(float2 p)" in opencl
    assert f"{ATTR}\nfloat hash(float3 p)" in opencl


def test_single_user_function_marked_overloadable(parser, transformer, emitter):
    """A plain (non-overloaded) user function is still marked (harmless, uniform)."""
    glsl = """
    float sdf(float x) {
        return x * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert f"{ATTR}\nfloat sdf(float x)" in opencl


def test_mainimage_not_marked_overloadable(parser, transformer, emitter):
    """mainImage is the entry point and must NOT get the attribute."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        fragColor = vec4(fragCoord, 0.0, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert ATTR not in opencl


def test_overloadable_via_code_emitter(parser, transformer):
    """The transformer.code_emitter path emits the attribute too (used by other tests)."""
    glsl = """
    float saw(float x) { return x; }
    float saw(vec2 x) { return x.x; }
    """
    ast = parser.parse(glsl)
    transformed = transformer.transform(ast)
    opencl = CodeEmitter().emit(transformed)
    assert opencl.count(ATTR) == 2


def test_maincubemap_not_marked_overloadable(parser, transformer, emitter):
    """mainCubemap is a renderpass entry point — its signature is replaced by the
    Houdini @KERNEL wrapper, so marking it would leave a dangling attribute
    before the kernel (regression seen with shader wfffRN's cubemap pass)."""
    glsl = """
    float max3(vec3 rd) { return max(max(rd.x, rd.y), rd.z); }
    void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) {
        fragColor = vec4(rayDir, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # max3 (a real user function) IS marked; mainCubemap is NOT.
    assert opencl.count(ATTR) == 1
    assert f"{ATTR}\nfloat max3" in opencl
    assert f"{ATTR}\nvoid mainCubemap" not in opencl


def test_mainsound_not_marked_overloadable(parser, transformer, emitter):
    """mainSound is also a renderpass entry point and must stay unmarked."""
    glsl = """
    vec2 mainSound(int samp, float time) {
        return vec2(sin(time), cos(time));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert ATTR not in opencl
