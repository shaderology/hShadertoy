"""
Unit tests for category C: matrix constructors resolved by TOTAL COMPONENT
count, not argument count (Session 15).

GLSL matrix constructors consume components column-major from a flat list of
scalar/vector arguments; the argument shapes are unconstrained as long as the
total component count matches. The old dispatch only accepted 1 arg
(diagonal/cast), N same-width column vectors, or N*N scalars, and RAISED on
everything else, killing whole shaders at transpile time. Corpus shapes fixed
here:

  mat2(a, -a.y, a.x)          vec2 + 2 scalars (complex-number idiom)
  mat2(sin(t + vec4(...)))    single vec4 (rotation trick) — was emitted as
                              GLSL_matrix2x2_diagonal(float4) -> compile error
  mat3(r, u, -f)              unary minus made a column untypeable -> raise
  mat2(r3[0].st, r3[1].st)    stpq swizzles: untypeable + invalid in OpenCL
  mat3(f(a), g(b), h(c))      untypeable args but arg count == column count
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
# Mixed scalar/vector runs -> flatten to GLSL_matN component list
# ============================================================================

def test_mat2_vec2_plus_two_scalars(parser, transformer, emitter):
    """mat2(a, -a.y, a.x) — the complex-multiply idiom (4dG3zd, wslSRr...)."""
    glsl = """
    mat2 cm(vec2 a) {
        return mat2(a, -a.y, a.x);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2(a.x, a.y, -a.y, a.x)' in opencl


def test_mat2_scalars_then_vec2(parser, transformer, emitter):
    """mat2(s, t, v) — vector run at the tail."""
    glsl = """
    mat2 f(vec2 v, float s, float t) {
        return mat2(s, t, v);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2(s, t, v.x, v.y)' in opencl


def test_mat3_mixed_vec2_scalar_runs(parser, transformer, emitter):
    """mat3(a, b.yz, c, d.xy, e) — mixed runs summing to 9."""
    glsl = """
    mat3 f(float a, vec3 b, float c, vec4 d, vec3 e) {
        return mat3(a, b.yz, c, d.xy, e);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert ('GLSL_mat3(a, b.yz.x, b.yz.y, c, d.xy.x, d.xy.y, e.x, e.y, e.z)'
            in opencl)


def test_mat2_flatten_wraps_complex_args(parser, transformer, emitter):
    """A non-postfix argument (binary op) gets parenthesized before .x."""
    glsl = """
    mat2 f(vec2 a, vec2 b) {
        return mat2(a + b, 0.0, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2((a + b).x, (a + b).y, 0.0f, 1.0f)' in opencl


# ============================================================================
# mat2 from a single vec4
# ============================================================================

def test_mat2_from_vec4_variable(parser, transformer, emitter):
    glsl = """
    mat2 f(vec4 v) {
        return mat2(v);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_from_vec4(v)' in opencl


def test_mat2_from_vec4_expression(parser, transformer, emitter):
    """mat2(sin(t + vec4(...))) — the animated-rotation trick (3sB3WG...)."""
    glsl = """
    mat2 rot(float t) {
        return mat2(sin(t + vec4(0.0, 33.0, 11.0, 0.0)));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_from_vec4(GLSL_sin(' in opencl
    assert 'diagonal' not in opencl


def test_mat2_from_scaled_vec4(parser, transformer, emitter):
    """mat2(sqrt(2.)*vec4(-1,1,1,1)) (XsBBD3)."""
    glsl = """
    mat2 f() {
        return mat2(sqrt(2.0) * vec4(-1.0, 1.0, 1.0, 1.0));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_from_vec4(' in opencl


# ============================================================================
# Column constructors whose args were previously untypeable
# ============================================================================

def test_mat3_columns_with_unary_minus(parser, transformer, emitter):
    """mat3(r, u, -f) (lsffzS) — unary minus must not defeat column typing."""
    glsl = """
    mat3 cam(vec3 r, vec3 u, vec3 f) {
        return mat3(r, u, -f);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_cols(r, u, -f)' in opencl


def test_mat3_columns_untyped_call_args(parser, transformer, emitter):
    """3 untypeable args for mat3 -> assume columns (valid-GLSL fallback)."""
    glsl = """
    mat3 cam(vec3 prp, vec3 vrp) {
        return mat3(normalize(prp), vec3(0.0, 1.0, 0.0), normalize(vrp - prp));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_cols(' in opencl


def test_mat2_columns_from_binary_ops(parser, transformer, emitter):
    """mat2(A/M, a/m) (ltXfRr) — 2 args, must resolve to columns."""
    glsl = """
    mat2 f(vec2 A, vec2 a, float M, float m) {
        return mat2(A / M, a / m);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_cols(' in opencl


# ============================================================================
# stpq swizzles (4sdXRl): typed AND remapped to xyzw for OpenCL
# ============================================================================

def test_mat2_columns_from_st_swizzles(parser, transformer, emitter):
    glsl = """
    mat2 f(vec3 p, vec3 q) {
        return mat2(p.st, q.st);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_cols(p.xy, q.xy)' in opencl


def test_stpq_swizzle_remapped_outside_ctor(parser, transformer, emitter):
    glsl = """
    vec2 f(vec4 v) {
        return v.pq;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'v.zw' in opencl


def test_single_t_field_on_struct_not_remapped(parser, transformer, emitter):
    """.t on a non-vector (struct) must stay a struct field access."""
    glsl = """
    struct Hit { float t; };
    float f(Hit h) {
        return h.t;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'h.t' in opencl
    assert 'h.y' not in opencl


# ============================================================================
# Matrix-from-matrix: identity passthrough + normalized cast names
# ============================================================================

def test_mat3_identity_cast_passthrough(parser, transformer, emitter):
    """mat3(m) where m is mat3 is the identity — no bogus _from_ call."""
    glsl = """
    mat3 f(mat3 m) {
        return mat3(m);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '_from_' not in opencl
    assert 'diagonal' not in opencl


def test_mat3_from_mat4_param_opencl_type_name(parser, transformer, emitter):
    """Parameter types register OpenCL names (matrix4x4); the cast helper
    name must still be the GLSL-name form GLSL_mat3_from_mat4."""
    glsl = """
    mat3 f(mat4 m) {
        return mat3(m);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_from_mat4(m)' in opencl
    assert 'matrix4x4' not in opencl.split('GLSL_mat3_from_')[1][:12]


# ============================================================================
# Guards: existing shapes unchanged
# ============================================================================

def test_guard_mat2_diagonal(parser, transformer, emitter):
    glsl = """
    void test() {
        mat2 M = mat2(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrix2x2_diagonal(1.0f)' in opencl


def test_guard_mat2_full_scalars(parser, transformer, emitter):
    glsl = """
    void test() {
        mat2 M = mat2(1.0, 2.0, 3.0, 4.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2(1.0f, 2.0f, 3.0f, 4.0f)' in opencl


def test_guard_mat3_full_scalars_with_ints(parser, transformer, emitter):
    """mat3(1,0,0, 0,c,s, 0,-s,c) — int literals in scalar list (XsXfz2)."""
    glsl = """
    mat3 rx(float c, float s) {
        return mat3(1, 0, 0, 0, c, s, 0, -s, c);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3(1, 0, 0, 0, c, s, 0, -s, c)' in opencl


def test_guard_mat2_column_vectors(parser, transformer, emitter):
    glsl = """
    void test() {
        vec2 a = vec2(1.0, 0.0);
        vec2 b = vec2(0.0, 1.0);
        mat2 M = mat2(a, b);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat2_cols(a, b)' in opencl


def test_guard_mat4_from_mat3_cast(parser, transformer, emitter):
    glsl = """
    void test() {
        mat3 M3 = mat3(1.0);
        mat4 M4 = mat4(M3);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat4_from_mat3(M3)' in opencl


def test_guard_mat3_diagonal_variable(parser, transformer, emitter):
    glsl = """
    void test() {
        float s = 2.0;
        mat3 M = mat3(s);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_matrix3x3_diagonal(s)' in opencl
