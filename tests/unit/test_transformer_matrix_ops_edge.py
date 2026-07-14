"""
Unit tests for matrix-op edge cases found by the Session 19 pipeline review.

OpenCL matrix types (matrix2x2/3x3/4x4) are STRUCTS, so every GLSL operator
that silently works on a real matrix type needs an explicit lowering. The
review probed the whole matrix pipeline and found these unhandled shapes:

  - unary minus            -M            (raw struct negation -> clang error)
  - increment/decrement    M++ / --M     (raw struct ++ -> clang error)
  - aggregate equality     A == B        (raw struct == -> clang error)
  - outerProduct(a, b)     unmapped GLSL builtin (implicit declaration)
  - matrixCompMult(A, B)   missing from glsl_builtins -> emitted raw
  - inverse(M) / transpose(M) on an OpenCL-named param ('matrix3x3'):
    return-type inference did TYPE_NAME_MAP.get without normalization ->
    downstream detection (matmul, M[i] -> .cols[i]) silently missed
  - transpose(x) on an UNTYPEABLE arg emitted the bare GLSL_transpose, which
    only existed for matrix2x2 -> overloadable bare-name wrappers added for
    all sizes (matrix_ops.h), same pattern as the GLSL_mul dispatcher
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
# Unary minus on a matrix
# ============================================================================

def test_unary_neg_matrix(parser, transformer, emitter):
    """-M componentwise-negates: GLSL_mat3_muls(M, -1.0f)."""
    glsl = """
    mat3 f(mat3 M) {
        return -M;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_muls(M, -1.0f)' in opencl


def test_neg_matrix_times_vec(parser, transformer, emitter):
    """-M * v: negation lowered AND matmul still detected."""
    glsl = """
    vec3 f(mat3 M, vec3 v) {
        return -M * v;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat3_vec3(GLSL_mat3_muls(M, -1.0f), v)' in opencl


def test_neg_parenthesized_matrix(parser, transformer, emitter):
    """-(A) also lowers (paren must not defeat detection)."""
    glsl = """
    mat2 f(mat2 A) {
        return -(A);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_muls' in opencl


def test_neg_vector_stays_native(parser, transformer, emitter):
    """-v on a vector is valid OpenCL and stays native."""
    glsl = """
    vec3 f(vec3 v) {
        return -v;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'return -v;' in opencl
    assert 'muls' not in opencl


# ============================================================================
# Increment / decrement on a matrix
# ============================================================================

def test_matrix_increment(parser, transformer, emitter):
    """M++ adds 1 to every element: M = GLSL_mat2_adds(M, 1)."""
    glsl = """
    void f() {
        mat2 M = mat2(1.0);
        M++;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M = GLSL_mat2_adds(M, 1)' in opencl


def test_matrix_decrement_prefix(parser, transformer, emitter):
    """--M: M = GLSL_mat3_subs(M, 1)."""
    glsl = """
    void f(mat3 M) {
        --M;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M = GLSL_mat3_subs(M, 1)' in opencl


def test_vector_increment_unchanged(parser, transformer, emitter):
    """v++ keeps the existing vector rewrite (v += 1)."""
    glsl = """
    void f(vec3 v) {
        v++;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'v += 1' in opencl


# ============================================================================
# Aggregate matrix equality
# ============================================================================

def test_matrix_equality(parser, transformer, emitter):
    """A == B on matrices is aggregate equality: GLSL_mat_eq(A, B)."""
    glsl = """
    bool f(mat2 A, mat2 B) {
        return A == B;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat_eq(A, B)' in opencl


def test_matrix_inequality(parser, transformer, emitter):
    """A != B: !GLSL_mat_eq(A, B)."""
    glsl = """
    bool f(mat3 A, mat3 B) {
        return A != B;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '!GLSL_mat_eq(A, B)' in opencl


def test_vector_equality_unchanged(parser, transformer, emitter):
    """Vector == keeps the category-O all() lowering, not mat_eq."""
    glsl = """
    bool f(vec3 a, vec3 b) {
        return a == b;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'all(a == b)' in opencl
    assert 'GLSL_mat_eq' not in opencl


# ============================================================================
# outerProduct / matrixCompMult mapping + dispatch
# ============================================================================

def test_outer_product_mapped(parser, transformer, emitter):
    """outerProduct(a, b) -> GLSL_outerProduct(a, b) (overloadable helper)."""
    glsl = """
    mat3 f(vec3 a, vec3 b) {
        return outerProduct(a, b);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_outerProduct(a, b)' in opencl


def test_outer_product_result_typed(parser, transformer, emitter):
    """outerProduct result is a matrix: outerProduct(a,b) * v detects matmul."""
    glsl = """
    vec3 f(vec3 a, vec3 b, vec3 v) {
        return outerProduct(a, b) * v;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat3_vec3(GLSL_outerProduct(a, b), v)' in opencl


def test_compmult_param_dispatch(parser, transformer, emitter):
    """matrixCompMult on mat3 PARAMS ('matrix3x3' names) dispatches _mat3."""
    glsl = """
    mat3 f(mat3 A, mat3 B) {
        return matrixCompMult(A, B);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrixCompMult_mat3(A, B)' in opencl


# ============================================================================
# Builtin matrix return-type inference (OpenCL-named args)
# ============================================================================

def test_inverse_param_result_typed(parser, transformer, emitter):
    """inverse(M) on a param is typed mat3 -> inverse(M) * v detects matmul."""
    glsl = """
    vec3 f(mat3 M, vec3 v) {
        return inverse(M) * v;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat3_vec3(GLSL_inverse_mat3(M), v)' in opencl


def test_transpose_param_subscript(parser, transformer, emitter):
    """transpose(M)[0] on a param resolves the column: .cols[0]."""
    glsl = """
    vec3 f(mat3 M) {
        return transpose(M)[0];
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_transpose_mat3(M).cols[0]' in opencl


# ============================================================================
# Deref'd out-param member precedence (emitter mirror of opencl_emitter)
# ============================================================================

def test_out_param_matrix_subscript_parenthesized(parser, transformer, emitter):
    """(out mat3 M): M[0] = ... must emit (*M).cols[0], never *M.cols[0]."""
    glsl = """
    void f(out mat3 M) {
        M[0] = vec3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '(*M).cols[0]' in opencl
