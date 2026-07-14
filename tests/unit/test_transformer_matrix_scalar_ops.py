"""
Unit tests for category H: componentwise matrix arithmetic (Session 18).

GLSL allows scalar-broadcast and elementwise arithmetic on matrices:
`M * s`, `s * M`, `M / s`, `M + s`, `s - M`, `M1 + M2`, `M1 - M2`, all
componentwise (GLSL `M + s` adds `s` to EVERY element, not just the diagonal).
The OpenCL matrix types (matrix2x2 / matrix3x3 / matrix4x4 from matrix_types.h)
are STRUCTS, so a native `M * s` / `A + B` is rejected by clang ("invalid
operands to binary expression ('float' and 'matrix3x3')" / "('matrix2x2' and
'matrix2x2')"). The fix rewrites these shapes to GLSL_matN_* helper calls;
matrix*matrix `*` stays matrix multiplication and vector/scalar ops are
untouched.
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
# Matrix * scalar  /  scalar * matrix
# ============================================================================

def test_mat2_times_scalar(parser, transformer, emitter):
    """mat2 M; M * 2.0 -> GLSL_mat2_muls(M, 2.0f)."""
    glsl = "mat2 f() { mat2 M = mat2(1.0); return M * 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_muls(M, 2.0f)' in out
    assert 'M * 2.0f' not in out


def test_scalar_times_mat2_puts_matrix_first(parser, transformer, emitter):
    """2.0 * M -> GLSL_mat2_muls(M, 2.0f) (matrix arg first, commutative)."""
    glsl = "mat2 f() { mat2 M = mat2(1.0); return 2.0 * M; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_muls(M, 2.0f)' in out


def test_mat3_times_scalar(parser, transformer, emitter):
    """mat3 M; M * 2.0 -> GLSL_mat3_muls."""
    glsl = "mat3 f() { mat3 M = mat3(1.0); return M * 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_muls(M, 2.0f)' in out


def test_mat4_times_scalar(parser, transformer, emitter):
    """mat4 M; M * 2.0 -> GLSL_mat4_muls."""
    glsl = "mat4 f() { mat4 M = mat4(1.0); return M * 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat4_muls(M, 2.0f)' in out


# ============================================================================
# Matrix / scalar
# ============================================================================

def test_mat2_div_scalar(parser, transformer, emitter):
    """M / s -> GLSL_mat2_divs(M, s)."""
    glsl = "mat2 f() { mat2 M = mat2(1.0); return M / 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_divs(M, 2.0f)' in out


# ============================================================================
# Matrix +/- scalar  (broadcast to every element)
# ============================================================================

def test_mat3_plus_scalar(parser, transformer, emitter):
    """M + s -> GLSL_mat3_adds(M, s)."""
    glsl = "mat3 f() { mat3 M = mat3(1.0); return M + 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_adds(M, 2.0f)' in out


def test_scalar_plus_mat3(parser, transformer, emitter):
    """s + M -> GLSL_mat3_adds(M, s) (commutative, matrix first)."""
    glsl = "mat3 f() { mat3 M = mat3(1.0); return 2.0 + M; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_adds(M, 2.0f)' in out


def test_mat3_minus_scalar(parser, transformer, emitter):
    """M - s -> GLSL_mat3_subs(M, s)."""
    glsl = "mat3 f() { mat3 M = mat3(1.0); return M - 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_subs(M, 2.0f)' in out


def test_scalar_minus_mat3_is_rsub(parser, transformer, emitter):
    """s - M -> GLSL_mat3_rsub(s, M) (order matters, not commutative)."""
    glsl = "mat3 f() { mat3 M = mat3(1.0); return 2.0 - M; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_rsub(2.0f, M)' in out


# ============================================================================
# Matrix +/- matrix  (elementwise)
# ============================================================================

def test_mat2_plus_mat2(parser, transformer, emitter):
    """A + B -> GLSL_mat2_add(A, B)."""
    glsl = "mat2 f() { mat2 A = mat2(1.0); mat2 B = mat2(2.0); return A + B; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_add(A, B)' in out


def test_mat2_minus_mat2(parser, transformer, emitter):
    """A - B -> GLSL_mat2_sub(A, B)."""
    glsl = "mat2 f() { mat2 A = mat2(1.0); mat2 B = mat2(2.0); return A - B; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_sub(A, B)' in out


# ============================================================================
# Compound assignment (A op= B)
# ============================================================================

def test_mat2_plus_equals_matrix(parser, transformer, emitter):
    """AtA += mat2(...) -> AtA = GLSL_mat2_add(AtA, GLSL_mat2(...))."""
    glsl = """
    mat2 f() {
        mat2 AtA = mat2(0.0);
        AtA += mat2(1.0, 2.0, 3.0, 4.0);
        return AtA;
    }
    """
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'AtA = GLSL_mat2_add(AtA, GLSL_mat2(' in out


def test_mat3_div_equals_scalar(parser, transformer, emitter):
    """M /= s -> M = GLSL_mat3_divs(M, s)."""
    glsl = """
    mat3 f() {
        mat3 M = mat3(1.0);
        M /= 2.0;
        return M;
    }
    """
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M = GLSL_mat3_divs(M, 2.0f)' in out


def test_mat2_times_equals_matrix_stays_multiplication(parser, transformer, emitter):
    """A *= B stays matrix multiplication (linear algebra), not componentwise."""
    glsl = """
    mat2 f() {
        mat2 A = mat2(1.0);
        mat2 B = mat2(2.0);
        A *= B;
        return A;
    }
    """
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'A = GLSL_mul_mat2_mat2(A, B)' in out


# ============================================================================
# Constructor-result / untyped scalar divisor (M / s where s is untyped)
# ============================================================================

def test_matrix_ctor_divided_by_untyped_scalar(parser, transformer, emitter):
    """mat2(...) / (M[0][0]) — the divisor's type does not infer, but a
    matrix under `/` can only be scaled by a scalar, so it must be a divs."""
    glsl = """
    mat2 f() {
        mat2 AtA = mat2(1.0, 2.0, 3.0, 4.0);
        mat2 inv = mat2(AtA[1][1], -AtA[0][1], -AtA[1][0], AtA[0][0])
                   / (AtA[0][0] * AtA[1][1] - AtA[1][0] * AtA[0][1]);
        return inv;
    }
    """
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_divs(GLSL_mat2(' in out


# ============================================================================
# Guards: these shapes MUST be left untouched
# ============================================================================

def test_matrix_times_matrix_stays_multiplication(parser, transformer, emitter):
    """A * B is matrix MULTIPLICATION, not componentwise."""
    glsl = "mat2 f() { mat2 A = mat2(1.0); mat2 B = mat2(2.0); return A * B; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat2_mat2(A, B)' in out
    assert '_muls' not in out


def test_matrix_times_vector_stays_matvec(parser, transformer, emitter):
    """M * v stays matrix-vector multiplication."""
    glsl = "vec2 f() { mat2 M = mat2(1.0); vec2 v = vec2(1.0); return M * v; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat2_vec2(M, v)' in out


def test_vector_times_scalar_untouched(parser, transformer, emitter):
    """vec2 * scalar is native OpenCL — no GLSL_mat helper."""
    glsl = "vec2 f() { vec2 v = vec2(1.0); return v * 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'v * 2.0f' in out
    assert 'GLSL_mat' not in out


def test_scalar_ops_untouched(parser, transformer, emitter):
    """float * float is native — no matrix helper."""
    glsl = "float f() { float a = 1.0; return a * 2.0; }"
    out = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'a * 2.0f' in out
    assert 'GLSL_mat' not in out


# ============================================================================
# Qualifier-tolerant matrix detection (const matrix3x3 / __global matrix4x4)
# ============================================================================

def test_is_matrix_type_sees_through_qualifiers(transformer):
    """Qualified OpenCL matrix type names (const / __global) still detect."""
    assert transformer._is_matrix_type('const matrix3x3')
    assert transformer._is_matrix_type('__global matrix4x4')
    assert transformer._is_matrix_type('matrix2x2')
    assert not transformer._is_matrix_type('const float3')
