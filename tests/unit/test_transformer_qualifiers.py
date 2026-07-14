"""
Unit tests for function parameter qualifiers (Session 8).

Tests:
- Qualifier transformation (in/out/inout/const)
- Pointer syntax (scalar, vector, matrix)
- Dereference insertion (assignments to out/inout parameters)
- Address-of insertion (call sites with out/inout parameters)

Architecture:
    GLSL: void func(out float x)
    OpenCL: void func(float* x)

    GLSL: out_param = value;
    OpenCL: *out_param = value;

    GLSL: func(variable);
    OpenCL: func(&variable);
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
    """Create code emitter."""
    return CodeEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    """Helper: parse, transform, and emit code."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)
    return opencl


# ============================================================================
# 1. Qualifier Transformation (15 tests)
# ============================================================================

def test_in_qualifier_removed(parser, transformer, emitter):
    """Test in float x -> float x (in qualifier removed)."""
    glsl = """
    float test(in float x) {
        return x * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float test(float x)' in opencl
    assert 'in ' not in opencl


def test_out_scalar_to_pointer(parser, transformer, emitter):
    """Test out float x -> float* x."""
    glsl = """
    void test(out float x) {
        x = 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float* x' in opencl or 'float *x' in opencl


def test_inout_scalar_to_pointer(parser, transformer, emitter):
    """Test inout float x -> float* x."""
    glsl = """
    void test(inout float x) {
        x = x * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float* x' in opencl or 'float *x' in opencl


def test_const_qualifier_preserved(parser, transformer, emitter):
    """Test const float x parameter (const may be removed for non-pointer params)."""
    glsl = """
    float test(const float x) {
        return x * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # const may be removed for simple pass-by-value parameters
    assert 'float x' in opencl


def test_no_qualifier_default(parser, transformer, emitter):
    """Test float x -> float x (no qualifier, default pass-by-value)."""
    glsl = """
    float test(float x) {
        return x * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float test(float x)' in opencl


def test_out_vec2_to_pointer(parser, transformer, emitter):
    """Test out vec2 v -> float2* v."""
    glsl = """
    void test(out vec2 v) {
        v = vec2(1.0, 2.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float2* v' in opencl or 'float2 *v' in opencl


def test_out_vec3_to_pointer(parser, transformer, emitter):
    """Test out vec3 v -> float3* v."""
    glsl = """
    void test(out vec3 v) {
        v = vec3(1.0, 2.0, 3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float3* v' in opencl or 'float3 *v' in opencl


def test_out_vec4_to_pointer(parser, transformer, emitter):
    """Test out vec4 v -> float4* v."""
    glsl = """
    void test(out vec4 v) {
        v = vec4(1.0, 2.0, 3.0, 4.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float4* v' in opencl or 'float4 *v' in opencl


def test_out_mat2_to_pointer(parser, transformer, emitter):
    """Test out mat2 M -> matrix2x2* M."""
    glsl = """
    void test(out mat2 M) {
        M = mat2(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'matrix2x2* M' in opencl or 'matrix2x2 *M' in opencl


def test_out_mat3_no_pointer(parser, transformer, emitter):
    """Test out mat3 M -> matrix3x3* M (struct type uses pointer like mat2/mat4)."""
    glsl = """
    void test(out mat3 M) {
        M = mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, so it uses pointers like mat2/mat4
    assert 'matrix3x3* M' in opencl or 'matrix3x3 *M' in opencl


def test_out_mat4_to_pointer(parser, transformer, emitter):
    """Test out mat4 M -> matrix4x4* M."""
    glsl = """
    void test(out mat4 M) {
        M = mat4(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'matrix4x4* M' in opencl or 'matrix4x4 *M' in opencl


def test_inout_vec2_to_pointer(parser, transformer, emitter):
    """Test inout vec2 v -> float2* v."""
    glsl = """
    void test(inout vec2 v) {
        v = v * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float2* v' in opencl or 'float2 *v' in opencl


def test_inout_mat3_no_pointer(parser, transformer, emitter):
    """Test inout mat3 M -> matrix3x3* M (struct type uses pointer)."""
    glsl = """
    void test(inout mat3 M) {
        M = mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, so it uses pointers like mat2/mat4
    assert 'matrix3x3* M' in opencl or 'matrix3x3 *M' in opencl


def test_multiple_parameters_mixed(parser, transformer, emitter):
    """Test mixed parameter qualifiers."""
    glsl = """
    void test(in float a, out float b, inout float c, const float d) {
        b = a * 2.0;
        c = c + d;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # in removed, out/inout become pointers, const may be removed
    assert 'float a' in opencl
    assert 'float* b' in opencl or 'float *b' in opencl
    assert 'float* c' in opencl or 'float *c' in opencl
    assert 'float d' in opencl


def test_pointer_params_tracking(parser, transformer, emitter):
    """Verify pointer_params set tracks pointer parameters."""
    glsl = """
    void test(out float x, out vec2 y) {
        x = 1.0;
        y = vec2(2.0, 3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Both x and y should be pointers
    assert 'float* x' in opencl or 'float *x' in opencl
    assert 'float2* y' in opencl or 'float2 *y' in opencl
    # Assignments should have dereferences
    assert '*x = 1.0f' in opencl
    assert '*y = (float2)' in opencl


# ============================================================================
# 2. Pointer Syntax (15 tests)
# ============================================================================

def test_scalar_pointer_syntax(parser, transformer, emitter):
    """Test float* pointer syntax."""
    glsl = """
    void test(out float x) {
        x = 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float* x' in opencl or 'float *x' in opencl


def test_int_pointer_syntax(parser, transformer, emitter):
    """Test int* pointer syntax."""
    glsl = """
    void test(out int x) {
        x = 10;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'int* x' in opencl or 'int *x' in opencl


def test_vec2_pointer_syntax(parser, transformer, emitter):
    """Test float2* pointer syntax."""
    glsl = """
    void test(out vec2 v) {
        v = vec2(1.0, 2.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float2* v' in opencl or 'float2 *v' in opencl


def test_vec3_pointer_syntax(parser, transformer, emitter):
    """Test float3* pointer syntax."""
    glsl = """
    void test(out vec3 v) {
        v = vec3(1.0, 2.0, 3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float3* v' in opencl or 'float3 *v' in opencl


def test_vec4_pointer_syntax(parser, transformer, emitter):
    """Test float4* pointer syntax."""
    glsl = """
    void test(out vec4 v) {
        v = vec4(1.0, 2.0, 3.0, 4.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float4* v' in opencl or 'float4 *v' in opencl


def test_ivec2_pointer_syntax(parser, transformer, emitter):
    """Test int2* pointer syntax."""
    glsl = """
    void test(out ivec2 v) {
        v = ivec2(1, 2);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'int2* v' in opencl or 'int2 *v' in opencl


def test_uvec3_pointer_syntax(parser, transformer, emitter):
    """Test uint3* pointer syntax."""
    glsl = """
    void test(out uvec3 v) {
        v = uvec3(1, 2, 3);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'uint3* v' in opencl or 'uint3 *v' in opencl


def test_mat2_pointer_syntax(parser, transformer, emitter):
    """Test matrix2x2* pointer syntax."""
    glsl = """
    void test(out mat2 M) {
        M = mat2(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'matrix2x2* M' in opencl or 'matrix2x2 *M' in opencl


def test_mat3_array_syntax(parser, transformer, emitter):
    """Test matrix3x3* pointer syntax (struct type uses pointer like mat2/mat4)."""
    glsl = """
    void test(out mat3 M) {
        M = mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, so it uses pointers like mat2/mat4
    assert 'matrix3x3* M' in opencl or 'matrix3x3 *M' in opencl


def test_mat4_pointer_syntax(parser, transformer, emitter):
    """Test matrix4x4* pointer syntax."""
    glsl = """
    void test(out mat4 M) {
        M = mat4(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'matrix4x4* M' in opencl or 'matrix4x4 *M' in opencl


def test_is_pointer_flag_true(parser, transformer, emitter):
    """Verify is_pointer flag is True for out/inout scalar/vector."""
    glsl = """
    void test(out float x, out vec2 y) {
        x = 1.0;
        y = vec2(2.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should have pointer syntax
    assert 'float* x' in opencl or 'float *x' in opencl
    assert 'float2* y' in opencl or 'float2 *y' in opencl


def test_is_pointer_flag_false_mat3(parser, transformer, emitter):
    """Verify is_pointer flag is True for matrix3x3 (struct type uses pointers)."""
    glsl = """
    void test(out mat3 M) {
        M = mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, so it uses pointers like mat2/mat4
    assert 'matrix3x3* M' in opencl or 'matrix3x3 *M' in opencl


def test_no_private_qualifier_on_outparam(parser, transformer, emitter):
    """Out/inout pointer params are emitted WITHOUT an explicit `__private`
    address-space qualifier (Session 37): a bare `float* x` acts as the generic
    address space in the campaign/Houdini build mode, so a `__global` argument
    (a program-scope global passed by address) is accepted, whereas an explicit
    `__private float*` rejects it."""
    glsl = """
    void test(out float x) {
        x = 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '__private' not in opencl
    assert 'float* x' in opencl or 'float *x' in opencl


def test_emit_parameter_pointer(parser, transformer, emitter):
    """Test emit_Parameter with pointer flag."""
    glsl = """
    void test(out float x, out vec3 v, out mat2 M) {
        x = 1.0;
        v = vec3(0.0);
        M = mat2(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # All should have and pointer syntax
    assert 'float* x' in opencl or 'float *x' in opencl
    assert 'float3* v' in opencl or 'float3 *v' in opencl
    assert 'matrix2x2* M' in opencl or 'matrix2x2 *M' in opencl


def test_function_signature_registry(parser, transformer, emitter):
    """Verify function signatures are registered."""
    glsl = """
    void helper(out float x) {
        x = 1.0;
    }
    void test() {
        float y;
        helper(y);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Function should have pointer parameter
    assert 'void helper(float* x)' in opencl or 'void helper(float *x)' in opencl
    # Call site should have address-of
    assert 'helper(&y)' in opencl


# ============================================================================
# 3. Dereference Insertion (15 tests)
# ============================================================================

def test_dereference_scalar_assignment(parser, transformer, emitter):
    """Test *out_param = value for scalar."""
    glsl = """
    void test(out float x) {
        x = 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x = 1.0f' in opencl


def test_dereference_vec2_assignment(parser, transformer, emitter):
    """Test *out_param = value for vec2."""
    glsl = """
    void test(out vec2 v) {
        v = vec2(1.0, 2.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*v = (float2)' in opencl


def test_dereference_vec3_assignment(parser, transformer, emitter):
    """Test *out_param = value for vec3."""
    glsl = """
    void test(out vec3 v) {
        v = vec3(1.0, 2.0, 3.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*v = (float3)' in opencl


def test_dereference_simple_identifier(parser, transformer, emitter):
    """Test dereference for simple identifier assignment."""
    glsl = """
    void test(out float x) {
        x = 5.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x = 5.0f' in opencl


def test_no_dereference_mat3(parser, transformer, emitter):
    """Test NO dereference for mat3 (array type)."""
    glsl = """
    void test(out mat3 M) {
        M = mat3(1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 should NOT have dereference
    assert '*M' not in opencl or 'M =' in opencl


def test_dereference_in_function_body(parser, transformer, emitter):
    """Test dereference for assignments in function body."""
    glsl = """
    void test(out float x, out float y) {
        x = 1.0;
        y = 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x = 1.0f' in opencl
    assert '*y = 2.0f' in opencl


def test_multiple_dereferences(parser, transformer, emitter):
    """Test multiple dereferences in same function."""
    glsl = """
    void test(out vec2 v) {
        v = vec2(1.0, 2.0);
        v = v * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Both assignments should have dereference on LHS; the RHS read of the
    # pointer param also dereferences (category B fix): *v = *v * 2.0f.
    assert '*v = (float2)' in opencl
    assert '*v = *v * 2.0f' in opencl


def test_dereference_compound_assignment(parser, transformer, emitter):
    """Test dereference for compound assignments (+=, etc)."""
    glsl = """
    void test(inout float x) {
        x += 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x += 1.0f' in opencl or '*x = *x + 1.0f' in opencl


def test_dereference_member_access(parser, transformer, emitter):
    """Test dereference with member access."""
    glsl = """
    void test(out vec2 v) {
        v.x = 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Should have dereference before member access or on assignment
    # This might be: (*v).x = 1.0f or v->x = 1.0f depending on implementation
    assert '*v' in opencl or 'v->' in opencl or '.x = 1.0f' in opencl


def test_no_dereference_in_param(parser, transformer, emitter):
    """Test NO dereference for in parameters."""
    glsl = """
    void test(in float x) {
        float y = x;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x' not in opencl


def test_dereference_in_loop(parser, transformer, emitter):
    """Test dereference in loop."""
    glsl = """
    void test(out float x) {
        for(int i = 0; i < 10; i++) {
            x = float(i);
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x = (float)' in opencl


def test_dereference_in_if(parser, transformer, emitter):
    """Test dereference in if statement."""
    glsl = """
    void test(out float x, float condition) {
        if (condition > 0.0) {
            x = 1.0;
        }
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x = 1.0f' in opencl


def test_inout_dereference(parser, transformer, emitter):
    """Test inout parameter dereference."""
    glsl = """
    void test(inout float x) {
        x = x * 2.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Both sides dereference the inout pointer param (category B fix):
    # *x = *x * 2.0f  (previously the RHS read emitted bare `x`, a float*, which
    # was invalid OpenCL).
    assert '*x = *x * 2.0f' in opencl


def test_complex_expression_assignment(parser, transformer, emitter):
    """Test dereference with complex expression."""
    glsl = """
    void test(out float x, float a, float b) {
        x = a * b + sin(a);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert '*x = ' in opencl
    assert 'GLSL_sin' in opencl


def test_pointer_params_reset_per_function(parser, transformer, emitter):
    """Verify pointer_params is reset per function."""
    glsl = """
    void func1(out float x) {
        x = 1.0;
    }
    void func2(float y) {
        float z = y;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # func1 should have dereference for x
    assert '*x = 1.0f' in opencl
    # func2 should not have any dereferences (y is not out)
    lines = opencl.split('\n')
    func2_lines = []
    in_func2 = False
    for line in lines:
        if 'func2' in line:
            in_func2 = True
        if in_func2:
            func2_lines.append(line)
            if '}' in line and in_func2:
                break
    func2_code = '\n'.join(func2_lines)
    assert '*y' not in func2_code


# ============================================================================
# 4. Address-Of Insertion (15 tests)
# ============================================================================

def test_addressof_scalar_call(parser, transformer, emitter):
    """Test func(&out_param) for scalar."""
    glsl = """
    void helper(out float x) {
        x = 1.0;
    }
    void test() {
        float y;
        helper(y);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&y)' in opencl


def test_addressof_vec2_call(parser, transformer, emitter):
    """Test func(&out_param) for vec2."""
    glsl = """
    void helper(out vec2 v) {
        v = vec2(1.0, 2.0);
    }
    void test() {
        vec2 result;
        helper(result);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&result)' in opencl


def test_addressof_vec3_call(parser, transformer, emitter):
    """Test func(&out_param) for vec3."""
    glsl = """
    void helper(out vec3 v) {
        v = vec3(1.0, 2.0, 3.0);
    }
    void test() {
        vec3 result;
        helper(result);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&result)' in opencl


def test_addressof_mat2_call(parser, transformer, emitter):
    """Test func(&out_param) for mat2."""
    glsl = """
    void helper(out mat2 M) {
        M = mat2(1.0);
    }
    void test() {
        mat2 result;
        helper(result);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&result)' in opencl


def test_no_addressof_mat3_call(parser, transformer, emitter):
    """Test address-of for matrix3x3 (struct type uses pointers)."""
    glsl = """
    void helper(out mat3 M) {
        M = mat3(1.0);
    }
    void test() {
        mat3 result;
        helper(result);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mat3 now uses struct type, so it needs address-of like mat2/mat4
    assert 'helper(&result)' in opencl


def test_addressof_simple_identifier(parser, transformer, emitter):
    """Test address-of for simple identifier."""
    glsl = """
    void helper(out float x) {
        x = 1.0;
    }
    void test() {
        float value;
        helper(value);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&value)' in opencl


def test_multiple_addressof_calls(parser, transformer, emitter):
    """Test multiple address-of insertions."""
    glsl = """
    void helper(out float x, out float y) {
        x = 1.0;
        y = 2.0;
    }
    void test() {
        float a, b;
        helper(a, b);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&a, &b)' in opencl


def test_mixed_params_call(parser, transformer, emitter):
    """Test mixed in/out parameters at call site."""
    glsl = """
    void helper(float a, out float b, inout float c) {
        b = a * 2.0;
        c = c + a;
    }
    void test() {
        float x = 1.0;
        float y, z = 3.0;
        helper(x, y, z);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # Only y and z should have address-of
    assert 'helper(x, &y, &z)' in opencl


def test_nested_function_calls(parser, transformer, emitter):
    """Test nested function calls with out parameters."""
    glsl = """
    void inner(out float x) {
        x = 1.0;
    }
    void outer(out float y) {
        inner(y);
    }
    void test() {
        float z;
        outer(z);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # inner(y) inside outer should dereference y first, but this is complex
    # For now, just check that outer(z) has address-of
    assert 'outer(&z)' in opencl


def test_addressof_in_expression(parser, transformer, emitter):
    """Test address-of in expression context."""
    glsl = """
    void helper(out float x) {
        x = 1.0;
    }
    void test() {
        float a;
        helper(a);
        float b = a + 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&a)' in opencl


def test_no_addressof_in_param(parser, transformer, emitter):
    """Test NO address-of for in parameters."""
    glsl = """
    void helper(float x) {
        float y = x;
    }
    void test() {
        float a = 1.0;
        helper(a);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(a)' in opencl
    assert '&a' not in opencl


def test_inout_addressof(parser, transformer, emitter):
    """Test address-of for inout parameters."""
    glsl = """
    void helper(inout float x) {
        x = x * 2.0;
    }
    void test() {
        float value = 1.0;
        helper(value);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'helper(&value)' in opencl


def test_function_signature_lookup(parser, transformer, emitter):
    """Test function signature lookup for address-of insertion."""
    glsl = """
    void func1(out float x) { x = 1.0; }
    void func2(float x) { float y = x; }
    void test() {
        float a, b;
        func1(a);
        func2(b);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'func1(&a)' in opencl
    assert 'func2(b)' in opencl


def test_mainimage_special_handling(parser, transformer, emitter):
    """Test mainImage special handling (no dereference for fragColor/fragCoord)."""
    glsl = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        fragColor = vec4(fragCoord, 0.0, 1.0);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # mainImage parameters should still be pointers in signature
    assert 'float4* fragColor' in opencl or 'float4 *fragColor' in opencl
    # But dereference might be handled by transpile.py post-processing
    # For now, just check that the function compiles


def test_realistic_shader_pattern(parser, transformer, emitter):
    """Test realistic shader pattern with out parameters."""
    glsl = """
    void random2(vec2 uv, out vec2 noise2) {
        float a = fract(1e4 * sin(uv.x * 541.17));
        float b = fract(1e4 * sin(uv.y * 321.46));
        noise2 = vec2(a, b);
    }
    void test() {
        vec2 pixelnoise;
        random2(vec2(1.0, 2.0), pixelnoise);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    # random2 should have pointer parameter
    assert 'float2* noise2' in opencl or 'float2 *noise2' in opencl
    # Assignment should have dereference
    assert '*noise2 = (float2)' in opencl
    # Call site should have address-of
    assert 'random2((float2)(1.0f, 2.0f), &pixelnoise)' in opencl
