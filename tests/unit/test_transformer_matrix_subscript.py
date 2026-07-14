"""
Unit tests for category F: matrix column subscript M[i] -> M.cols[i] (Session 17).

GLSL indexes a matrix column with M[i], which returns the i-th column vector
(M[i][j] is the element at column i, row j). The OpenCL matrix types
(matrix2x2 / matrix3x3 / matrix4x4 from matrix_types.h) are STRUCTS whose
columns live in a float2/3/4 array named `cols`, so a bare M[i] is rejected by
clang ("subscripted value is not an array, pointer, or vector"). The fix maps a
subscript whose base is proven matrix-typed to M.cols[i]; a genuine array or
vector subscript is left untouched.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


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
    return CodeEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    return emitter.emit(transformed)


# ============================================================================
# Matrix column subscript -> .cols[i]
# ============================================================================

def test_mat2_local_column_subscript(parser, transformer, emitter):
    """mat2 M; M[0] -> M.cols[0]."""
    glsl = """
    vec2 f() {
        mat2 M = mat2(1.0);
        return M[0];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M.cols[0]' in opencl


def test_mat3_variable_index_column(parser, transformer, emitter):
    """mat3 M; M[i] -> M.cols[i] (runtime index)."""
    glsl = """
    vec3 f(int i) {
        mat3 M = mat3(1.0);
        return M[i];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M.cols[i]' in opencl


def test_matrix_element_access_double_subscript(parser, transformer, emitter):
    """mat4 M; M[i][j] -> M.cols[i][j] (column then vector component)."""
    glsl = """
    float f(int i, int j) {
        mat4 M = mat4(1.0);
        return M[i][j];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M.cols[i][j]' in opencl


def test_matrix_parameter_column_subscript(parser, transformer, emitter):
    """A mat2 parameter also gets .cols[] on subscript."""
    glsl = """
    vec2 col(mat2 m) {
        return m[1];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'm.cols[1]' in opencl


# ============================================================================
# Guards: genuine array / vector subscripts must be UNCHANGED
# ============================================================================

def test_float_array_subscript_unchanged(parser, transformer, emitter):
    """float a[4]; a[0] stays a[0] — never .cols."""
    glsl = """
    float f() {
        float a[4];
        return a[0];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'a[0]' in opencl
    assert '.cols' not in opencl


def test_matrix_array_element_unchanged(parser, transformer, emitter):
    """mat3 arr[2]; arr[0] indexes the ARRAY (element is a matrix), not a column."""
    glsl = """
    void f() {
        mat3 arr[2];
        mat3 first = arr[0];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'arr[0]' in opencl
    assert '.cols' not in opencl


def test_vector_component_subscript_unchanged(parser, transformer, emitter):
    """A vector subscript v[0] is a valid OpenCL component access — unchanged."""
    glsl = """
    float f(vec3 v) {
        return v[0];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'v[0]' in opencl
    assert '.cols' not in opencl
