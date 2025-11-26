"""
Unit tests for do-while loops and jump statements (break, continue, discard).

Tests:
- Do-while loops
- Break statements
- Continue statements
- Discard statements (GLSL fragment shader jump -> return)
- Nested loops with jumps
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
    """Create GLSL parser."""
    return GLSLParser()


@pytest.fixture
def transformer():
    """Create transformer with type checker."""
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker)


@pytest.fixture
def emitter():
    """Create OpenCL code emitter."""
    return OpenCLEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    """Helper: parse, transform, and emit code."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)
    return opencl


# ============================================================================
# 1. Do-While Loops
# ============================================================================

def test_do_while_simple(parser, transformer, emitter):
    """Test simple do-while loop."""
    glsl = """
    void test() {
        int i = 0;
        do {
            i++;
        } while (i < 10);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'do {' in opencl
    assert 'i = 0' in opencl
    assert '++i' in opencl or 'i++' in opencl
    assert 'while (i < 10)' in opencl


def test_do_while_with_condition(parser, transformer, emitter):
    """Test do-while with complex condition."""
    glsl = """
    void test() {
        int j = 0;
        float a = 0.0;
        do {
            a += 0.1;
            j++;
        } while (j < 10);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'do {' in opencl
    assert 'a += 0.1f' in opencl
    assert 'while (j < 10)' in opencl


def test_do_while_with_break(parser, transformer, emitter):
    """Test do-while with break statement."""
    glsl = """
    void test() {
        int j = 0;
        float a = 0.0;
        do {
            a += 0.1;
            if (j > 8) break;
            j++;
        } while (j < 10);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'do {' in opencl
    assert 'if (j > 8)' in opencl
    assert 'break;' in opencl
    assert 'while (j < 10)' in opencl


def test_do_while_with_continue(parser, transformer, emitter):
    """Test do-while with continue statement."""
    glsl = """
    void test() {
        int i = 0;
        do {
            if (i < 5) continue;
            i++;
        } while (i < 10);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'do {' in opencl
    assert 'if (i < 5)' in opencl
    assert 'continue;' in opencl
    assert 'while (i < 10)' in opencl


# ============================================================================
# 2. Break Statements
# ============================================================================

def test_break_in_for_loop(parser, transformer, emitter):
    """Test break statement in for loop."""
    glsl = """
    void test() {
        for (int i = 0; i < 8; ++i) {
            if (i < 8) break;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'for (int i = 0; i < 8; ++i)' in opencl
    assert 'if (i < 8)' in opencl
    assert 'break;' in opencl


def test_break_in_while_loop(parser, transformer, emitter):
    """Test break statement in while loop."""
    glsl = """
    void test() {
        int i = 0;
        while (i < 10) {
            if (i > 5) break;
            i++;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'while (i < 10)' in opencl
    assert 'if (i > 5)' in opencl
    assert 'break;' in opencl


def test_break_with_complex_condition(parser, transformer, emitter):
    """Test break with complex condition."""
    glsl = """
    void test(vec2 uv) {
        for (int i = 0; i < 10; i++) {
            if (uv.x > 0.9 && uv.y < 0.1) break;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (uv.x > 0.9f && uv.y < 0.1f)' in opencl
    assert 'break;' in opencl


# ============================================================================
# 3. Continue Statements
# ============================================================================

def test_continue_in_for_loop(parser, transformer, emitter):
    """Test continue statement in for loop."""
    glsl = """
    void test() {
        for (int i = 0; i < 8; ++i) {
            if (i < 4) {
                continue;
            }
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'for (int i = 0; i < 8; ++i)' in opencl
    assert 'if (i < 4)' in opencl
    assert 'continue;' in opencl


def test_continue_in_while_loop(parser, transformer, emitter):
    """Test continue statement in while loop."""
    glsl = """
    void test() {
        int i = 0;
        while (i < 10) {
            i++;
            if (i < 5) continue;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'while (i < 10)' in opencl
    assert 'if (i < 5)' in opencl
    assert 'continue;' in opencl


def test_continue_with_vector_condition(parser, transformer, emitter):
    """Test continue with vector comparison."""
    glsl = """
    void test(vec2 uv) {
        for (int i = 0; i < 8; ++i) {
            if (uv.x < uv.y) {
                continue;
            }
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (uv.x < uv.y)' in opencl
    assert 'continue;' in opencl


# ============================================================================
# 4. Discard Statements
# ============================================================================

def test_discard_statement(parser, transformer, emitter):
    """Test discard statement transforms to return."""
    glsl = """
    void test() {
        float x = 1.0;
        if (x > 0.9) discard;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (x > 0.9f)' in opencl
    assert 'return;' in opencl
    # Make sure it's NOT "discard;"
    assert 'discard;' not in opencl


def test_discard_in_block(parser, transformer, emitter):
    """Test discard inside a block."""
    glsl = """
    void test(vec2 uv) {
        vec3 col = vec3(0.0);
        if (uv.y > 0.9) {
            col = vec3(0.0);
            discard;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (uv.y > 0.9f)' in opencl
    assert 'col = (float3)(0.0f)' in opencl
    assert 'return;' in opencl
    assert 'discard;' not in opencl


def test_discard_with_return(parser, transformer, emitter):
    """Test both discard and return in same function."""
    glsl = """
    void test(vec2 uv) {
        if (uv.y > 0.9) {
            return;
        }
        if (uv.x > 0.9) discard;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Both should become return;
    assert opencl.count('return;') == 2
    assert 'discard;' not in opencl


# ============================================================================
# 5. Nested and Combined Cases
# ============================================================================

def test_nested_loops_with_break_continue(parser, transformer, emitter):
    """Test nested loops with both break and continue."""
    glsl = """
    void test() {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                if (j < 5) continue;
                if (j > 8) break;
            }
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (j < 5)' in opencl
    assert 'continue;' in opencl
    assert 'if (j > 8)' in opencl
    assert 'break;' in opencl


def test_all_jumps_in_function(parser, transformer, emitter):
    """Test function with all jump types."""
    glsl = """
    float somefunc(float x) {
        for (int i = 0; i < 8; ++i) {
            if (i < 4) {
                continue;
            } else if (i == 5) {
                return x;
            }
        }
        return x;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'if (i < 4)' in opencl
    assert 'continue;' in opencl
    assert 'else if (i == 5)' in opencl
    assert 'return x;' in opencl


def test_do_while_nested_in_for(parser, transformer, emitter):
    """Test do-while nested inside for loop."""
    glsl = """
    void test() {
        for (int i = 0; i < 5; i++) {
            int j = 0;
            do {
                j++;
                if (j > 3) break;
            } while (j < 10);
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'for (int i = 0; i < 5' in opencl
    assert 'do {' in opencl
    assert 'while (j < 10)' in opencl
    assert 'break;' in opencl
