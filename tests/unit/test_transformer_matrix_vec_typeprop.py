"""
Unit tests for category E: v*M / M*v mis-detected because an operand's type
was not propagated (Session 19).

The matmul branch in _transform_binary_expression only fires when BOTH operand
types resolve. Corpus shapes where one side stayed untyped and the raw OpenCL
`M * v` leaked through (clang: "cannot convert between vector and non-scalar
values"):

  - struct field of matrix/vector type   cylinder.r * v      (4ltcRn)
  - compound assign on a struct field    h.p *= rotY(-T)     (4tdSW8)
  - swizzle on a deref'd inout param     (*ro).yz = m2 * (*ro).yz  (4lBcRd)
  - subscript of an array parameter      Minv * (points[0]-P)      (4tBBDK)
  - parenthesized ternary                (k ? A : B) * v
  - macro/undeclared identifier partner  rotation * v1       (MsGBDD; v1 is a
    #define — statically untypeable, dispatched via overloadable GLSL_mul)
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
# Struct fields (struct_types registry must feed member-access typing)
# ============================================================================

def test_struct_field_matrix_times_vec_expr(parser, transformer, emitter):
    """cylinder.r (mat3 field) * (vec3 expr) -> GLSL_mul_mat3_vec3."""
    glsl = """
    struct Cylinder { vec3 p; mat3 r; };
    vec3 f(Cylinder cylinder, vec3 orig) {
        return cylinder.r * (orig - cylinder.p);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat3_vec3' in opencl


def test_vec_times_struct_field_matrix(parser, transformer, emitter):
    """local_normal * cylinder.r -> GLSL_mul_vec3_mat3."""
    glsl = """
    struct Cylinder { vec3 p; mat3 r; };
    vec3 f(Cylinder cylinder, vec3 n) {
        return n * cylinder.r;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_vec3_mat3' in opencl


def test_struct_field_vec_compound_assign_matrix(parser, transformer, emitter):
    """h.p (vec3 field) *= rotY(-T) (mat3 call) -> h.p = GLSL_mul_vec3_mat3(...)."""
    glsl = """
    struct Hit { vec3 p; };
    mat3 rotY(float a) { return mat3(1.0); }
    void f(float T) {
        Hit h;
        h.p *= rotY(-T);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'h.p = GLSL_mul_vec3_mat3(h.p, rotY(-T))' in opencl


def test_nested_struct_field_matrix(parser, transformer, emitter):
    """b.a.m (mat2 field through a nested struct) * v -> GLSL_mul_mat2_vec2."""
    glsl = """
    struct A { mat2 m; };
    struct B { A a; };
    vec2 f(B b, vec2 v) {
        return b.a.m * v;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat2_vec2' in opencl


# ============================================================================
# Deref'd pointer params and array-param subscripts
# ============================================================================

def test_swizzle_on_inout_deref(parser, transformer, emitter):
    """(*ro).yz (inout vec3 param) = mat2 ctor * (*ro).yz -> GLSL_mul_mat2_vec2."""
    glsl = """
    void rot(inout vec3 ro, float c, float s) {
        ro.yz = mat2(c, -s, s, c) * ro.yz;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat2_vec2' in opencl


def test_array_param_subscript_operand(parser, transformer, emitter):
    """Minv * (points[0] - P) where points is a vec3[4] param -> GLSL_mul_mat3_vec3."""
    glsl = """
    vec3 f(mat3 Minv, vec3 points[4], vec3 P) {
        return Minv * (points[0] - P);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat3_vec3' in opencl


def test_struct_array_param_field(parser, transformer, emitter):
    """hits[0].p (vec3 field of a struct-array element) * M -> GLSL_mul_vec3_mat3."""
    glsl = """
    struct Hit { vec3 p; };
    vec3 f(Hit hits[2], mat3 M) {
        return hits[0].p * M;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_vec3_mat3' in opencl


# ============================================================================
# Ternary operand
# ============================================================================

def test_parenthesized_ternary_matrix_operand(parser, transformer, emitter):
    """(k ? A : B) * v with mat2 branches -> GLSL_mul_mat2_vec2."""
    glsl = """
    vec2 f(mat2 A, mat2 B, vec2 v, bool k) {
        return (k ? A : B) * v;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul_mat2_vec2' in opencl


# ============================================================================
# Statically untypeable partner -> overloadable GLSL_mul dispatcher
# ============================================================================

def test_unknown_partner_right_generic_mul(parser, transformer, emitter):
    """mat4 M * v1 (undeclared, e.g. a #define) -> GLSL_mul(M, v1)."""
    glsl = """
    vec4 f(mat4 M) {
        return M * v1;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul(M, v1)' in opencl


def test_unknown_partner_left_generic_mul(parser, transformer, emitter):
    """v1 (undeclared) * mat4 M -> GLSL_mul(v1, M)."""
    glsl = """
    vec4 f(mat4 M) {
        return v1 * M;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mul(v1, M)' in opencl


def test_unknown_value_compound_assign_matrix_target(parser, transformer, emitter):
    """mat3 M *= unknown -> M = GLSL_mul(M, unknown) (scalar or matrix at runtime)."""
    glsl = """
    void f() {
        mat3 M = mat3(1.0);
        M *= scale_factor;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'M = GLSL_mul(M, scale_factor)' in opencl


# ============================================================================
# Guards: shapes that must stay native / keep their existing lowering
# ============================================================================

def test_vec_times_vec_stays_native(parser, transformer, emitter):
    """A genuine vector*vector stays a native componentwise multiply."""
    glsl = """
    vec3 f(vec3 a, vec3 b) {
        return a * b;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'a * b' in opencl
    assert 'GLSL_mul' not in opencl


def test_unknown_times_unknown_stays_native(parser, transformer, emitter):
    """Two untypeable operands must not be dispatched to GLSL_mul."""
    glsl = """
    float f() {
        return foo * bar;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'foo * bar' in opencl
    assert 'GLSL_mul' not in opencl


def test_vec_times_unknown_stays_native(parser, transformer, emitter):
    """vec * unknown stays native (a vector's partner may legally be scalar)."""
    glsl = """
    vec3 f(vec3 v) {
        return v * k;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'v * k' in opencl
    assert 'GLSL_mul' not in opencl


def test_matrix_times_scalar_still_componentwise(parser, transformer, emitter):
    """mat * typed scalar keeps the direct H helper, not the generic dispatcher."""
    glsl = """
    mat3 f(mat3 M, float s) {
        return M * s;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_muls(M, s)' in opencl
    assert 'GLSL_mul(' not in opencl


def test_vector_component_subscript_is_scalar(parser, transformer, emitter):
    """v[0] is a scalar component: v[0] * M must be componentwise, not matmul."""
    glsl = """
    mat3 f(vec3 v, mat3 M) {
        return v[0] * M;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'GLSL_mat3_muls(M, v[0])' in opencl
    assert 'GLSL_mul_vec3_mat3' not in opencl


def test_struct_field_named_s_survives(parser, transformer, emitter):
    """A struct field named 's' (stpq letter) is not remapped or mistyped."""
    glsl = """
    struct P { float s; };
    float f(P p) {
        return p.s * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'p.s * 2.0f' in opencl
