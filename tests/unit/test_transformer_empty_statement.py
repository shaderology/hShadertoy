"""
Unit tests for empty-statement tolerance (Bug-fix campaign category R).

A stray empty statement — most commonly a trailing double semicolon
`uv = fract(uv);;` — parses as an `expression_statement` with no expression.
The transformer used to raise TransformationError("Empty expression statement"),
which aborted the ENTIRE header/kernel transform and failed the whole shader.

Fix: skip an empty expression statement (return None). The compound-statement
and top-level declaration loops already filter None, so the empty statement is
simply dropped — matching GLSL/C semantics (a bare `;` is a no-op).
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


def transform_and_emit(glsl_code, parser, transformer, emitter):
    """Helper: parse, transform, and emit code (raises on transform failure)."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    return emitter.emit(transformed)


def test_trailing_double_semicolon_in_function(parser, transformer, emitter):
    """A trailing `;;` must not abort the transform; the real statement survives."""
    glsl = """
    float f(float x) {
        float y = x * 2.0;;
        return y;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float y = x * 2.0f;' in opencl
    assert 'return y;' in opencl


def test_double_semicolon_realistic_kernel(parser, transformer, emitter):
    """Realistic mainImage body with a `;;` (the corpus pattern)."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 uv = fragCoord / iResolution.xy;
        uv = 2.0 * fract(uv) - 1.0;;
        fragColor = vec4(uv, 0.0, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'fragColor' in opencl
    assert 'GLSL_fract' in opencl


def test_empty_statement_between_functions(parser, transformer, emitter):
    """A bare `;` at top level (between functions) must be skipped, not crash."""
    glsl = """
    float a(float x) { return x; }
    ;
    float b(float x) { return x + 1.0; }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float a(float x)' in opencl
    assert 'float b(float x)' in opencl


def test_infinite_for_loop_still_works(parser, transformer, emitter):
    """Regression guard: empty for-clauses (`for(;;)`) must still transform."""
    glsl = """
    float f() {
        float s = 0.0;
        for (;;) {
            s += 1.0;
            if (s > 5.0) break;
        }
        return s;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'for (' in opencl
    assert 'break;' in opencl
