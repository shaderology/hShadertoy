"""
Unit tests for basic AST transformations.

Tests transformation of:
- Float literal suffixes (1.0 -> 1.0f)
- Type names (vec2 -> float2)
- Simple expressions
- Simple statements
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer import transformed_ast as IR
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def parser():
    """Create GLSL parser."""
    return GLSLParser()


@pytest.fixture
def transformer():
    """Create AST transformer with type checker."""
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker)


@pytest.fixture
def emitter():
    """Create code emitter."""
    return CodeEmitter()


def transform_and_emit(glsl_code: str) -> str:
    """Helper: Parse, transform, and emit GLSL code."""
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    emitter = CodeEmitter()

    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    opencl = emitter.emit(transformed)

    return opencl


# ============================================================================
# Float Literal Transformation Tests (10 tests)
# ============================================================================

def test_float_literal_decimal_point():
    """Test float literal with decimal point gets 'f' suffix."""
    glsl = "void test() { float x = 1.0; }"
    opencl = transform_and_emit(glsl)

    assert '1.0f' in opencl
    assert '1.0;' not in opencl  # No un-suffixed float


def test_float_literal_trailing_decimal():
    """Test float literal with trailing decimal (8.) gets 'f' suffix."""
    glsl = "void test() { float x = 8.; }"
    opencl = transform_and_emit(glsl)

    assert '8.f' in opencl


def test_float_literal_leading_decimal():
    """Test float literal with leading decimal (.5) gets 'f' suffix."""
    glsl = "void test() { float x = .5; }"
    opencl = transform_and_emit(glsl)

    assert '.5f' in opencl


def test_float_literal_scientific_notation():
    """Test float literal in scientific notation gets 'f' suffix."""
    glsl = "void test() { float x = 1.5e-3; }"
    opencl = transform_and_emit(glsl)

    assert '1.5e-3f' in opencl


def test_float_literal_negative():
    """Test negative float literal gets 'f' suffix."""
    glsl = "void test() { float x = -1.0; }"
    opencl = transform_and_emit(glsl)

    assert '1.0f' in opencl  # Negative is unary operator


def test_float_literal_already_suffixed():
    """Test float literal that already has 'f' suffix is unchanged."""
    glsl = "void test() { float x = 1.0f; }"
    opencl = transform_and_emit(glsl)

    assert '1.0f' in opencl
    assert opencl.count('1.0f') == 1  # No double-suffix


def test_float_literal_uppercase_f_suffix():
    """Test float literal with uppercase 'F' suffix is normalized to 'f'.

    GLSL accepts both 0.951F and 0.951f; OpenCL / our FloatLiteral require
    lowercase 'f'. Regression: shader 3lX3Rr used `0.95100F` (category P).
    """
    glsl = "void test() { float x = 0.95100F; }"
    opencl = transform_and_emit(glsl)

    assert '0.95100f' in opencl
    assert '0.95100F' not in opencl  # uppercase suffix normalized away


def test_integer_literal_unchanged():
    """Test integer literals don't get 'f' suffix."""
    glsl = "void test() { int x = 42; }"
    opencl = transform_and_emit(glsl)

    assert '42' in opencl
    assert '42f' not in opencl


def test_multiple_float_literals():
    """Test multiple float literals all get 'f' suffix."""
    glsl = "void test() { float a = 1.0; float b = 2.5; float c = 3.14; }"
    opencl = transform_and_emit(glsl)

    assert '1.0f' in opencl
    assert '2.5f' in opencl
    assert '3.14f' in opencl


def test_float_literals_in_expression():
    """Test float literals in expressions get 'f' suffix."""
    glsl = "void test() { float x = 1.0 + 2.0 * 3.0; }"
    opencl = transform_and_emit(glsl)

    assert '1.0f' in opencl
    assert '2.0f' in opencl
    assert '3.0f' in opencl


def test_float_literals_in_function_call():
    """Test float literals in function calls get 'f' suffix."""
    glsl = "void test() { float x = sin(1.0); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_sin(1.0f)' in opencl


# ============================================================================
# Type Name Transformation Tests (10 tests)
# ============================================================================

def test_vec2_to_float2():
    """Test vec2 type name transformed to float2."""
    glsl = "void test() { vec2 v = vec2(1.0, 2.0); }"
    opencl = transform_and_emit(glsl)

    assert 'float2 v' in opencl
    assert '(float2)' in opencl  # Constructor
    assert 'vec2' not in opencl


def test_vec3_to_float3():
    """Test vec3 type name transformed to float3."""
    glsl = "void test() { vec3 v = vec3(1.0, 2.0, 3.0); }"
    opencl = transform_and_emit(glsl)

    assert 'float3 v' in opencl
    assert '(float3)' in opencl
    assert 'vec3' not in opencl


def test_vec4_to_float4():
    """Test vec4 type name transformed to float4."""
    glsl = "void test() { vec4 v = vec4(1.0, 2.0, 3.0, 4.0); }"
    opencl = transform_and_emit(glsl)

    assert 'float4 v' in opencl
    assert '(float4)' in opencl
    assert 'vec4' not in opencl


def test_ivec2_to_int2():
    """Test ivec2 type name transformed to int2."""
    glsl = "void test() { ivec2 v = ivec2(1, 2); }"
    opencl = transform_and_emit(glsl)

    assert 'int2 v' in opencl
    assert '(int2)' in opencl
    assert 'ivec2' not in opencl


def test_mat2_unchanged():
    """Test mat2 transforms to matrix2x2."""
    glsl = "void test() { mat2 m = mat2(1.0); }"
    opencl = transform_and_emit(glsl)

    assert 'matrix2x2 m' in opencl


def test_mat3_unchanged():
    """Test mat3 transforms to matrix3x3."""
    glsl = "void test() { mat3 m = mat3(1.0); }"
    opencl = transform_and_emit(glsl)

    assert 'matrix3x3 m' in opencl


def test_float_type_unchanged():
    """Test scalar float type unchanged."""
    glsl = "void test() { float x = 1.0; }"
    opencl = transform_and_emit(glsl)

    assert 'float x' in opencl


def test_int_type_unchanged():
    """Test scalar int type unchanged."""
    glsl = "void test() { int x = 42; }"
    opencl = transform_and_emit(glsl)

    assert 'int x' in opencl


def test_function_return_type_vec3():
    """Test function return type vec3 transformed to float3."""
    glsl = "vec3 compute() { return vec3(0.0); }"
    opencl = transform_and_emit(glsl)

    assert 'float3 compute()' in opencl
    assert 'vec3' not in opencl


def test_function_parameter_type_vec2():
    """Test function parameter type vec2 transformed to float2."""
    glsl = "void compute(vec2 pos) { }"
    opencl = transform_and_emit(glsl)

    assert 'float2 pos' in opencl
    assert 'vec2' not in opencl


# ============================================================================
# Type Constructor Transformation Tests (10 tests)
# ============================================================================

def test_vec2_constructor_cast_syntax():
    """Test vec2(...) becomes (float2)(...)."""
    glsl = "void test() { vec2 v = vec2(1.0, 2.0); }"
    opencl = transform_and_emit(glsl)

    assert '(float2)(1.0f, 2.0f)' in opencl


def test_vec3_constructor_cast_syntax():
    """Test vec3(...) becomes (float3)(...)."""
    glsl = "void test() { vec3 v = vec3(1.0, 2.0, 3.0); }"
    opencl = transform_and_emit(glsl)

    assert '(float3)(1.0f, 2.0f, 3.0f)' in opencl


def test_vec4_constructor_cast_syntax():
    """Test vec4(...) becomes (float4)(...)."""
    glsl = "void test() { vec4 v = vec4(1.0, 2.0, 3.0, 4.0); }"
    opencl = transform_and_emit(glsl)

    assert '(float4)(1.0f, 2.0f, 3.0f, 4.0f)' in opencl


def test_vec2_constructor_single_arg():
    """Test vec2(x) with single argument (splat)."""
    glsl = "void test() { vec2 v = vec2(1.0); }"
    opencl = transform_and_emit(glsl)

    assert '(float2)(1.0f)' in opencl


def test_ivec2_constructor():
    """Test ivec2(...) becomes (int2)(...)."""
    glsl = "void test() { ivec2 v = ivec2(1, 2); }"
    opencl = transform_and_emit(glsl)

    assert '(int2)(1, 2)' in opencl


def test_nested_vector_constructor():
    """Test nested vector constructors."""
    glsl = "void test() { vec3 v = vec3(vec2(1.0, 2.0), 3.0); }"
    opencl = transform_and_emit(glsl)

    assert '(float3)((float2)(1.0f, 2.0f), 3.0f)' in opencl


def test_vector_constructor_in_expression():
    """Test vector constructor in expression."""
    glsl = "void test() { vec2 v = vec2(1.0, 2.0) + vec2(3.0, 4.0); }"
    opencl = transform_and_emit(glsl)

    assert '(float2)(1.0f, 2.0f) + (float2)(3.0f, 4.0f)' in opencl


def test_vector_constructor_in_return():
    """Test vector constructor in return statement."""
    glsl = "vec2 test() { return vec2(1.0, 2.0); }"
    opencl = transform_and_emit(glsl)

    assert 'return (float2)(1.0f, 2.0f);' in opencl


def test_matrix_constructor_single_arg():
    """Test mat2(1.0) constructor (diagonal matrix)."""
    glsl = "void test() { mat2 m = mat2(1.0); }"
    opencl = transform_and_emit(glsl)

    # Transformed to matrix2x2
    assert 'matrix2x2' in opencl


def test_vector_constructor_as_function_arg():
    """Test vector constructor as function argument."""
    glsl = "void test() { float x = length(vec3(1.0, 2.0, 3.0)); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_length((float3)(1.0f, 2.0f, 3.0f))' in opencl


# ============================================================================
# Built-in Function Transformation Tests (10 tests)
# ============================================================================

def test_sin_function_prefix():
    """Test sin() becomes GLSL_sin()."""
    glsl = "void test() { float x = sin(1.0); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_sin(1.0f)' in opencl
    # Check that sin appears with GLSL_ prefix, not unprefixed
    assert opencl.count('sin(') == 1  # Only one occurrence
    assert opencl.count('GLSL_sin(') == 1  # And it has the prefix


def test_cos_function_prefix():
    """Test cos() becomes GLSL_cos()."""
    glsl = "void test() { float x = cos(1.0); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_cos(1.0f)' in opencl


def test_normalize_function_prefix():
    """Test normalize() becomes GLSL_normalize()."""
    glsl = "void test() { vec3 v = normalize(vec3(1.0, 2.0, 3.0)); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_normalize' in opencl


def test_dot_function_prefix():
    """Test dot() becomes GLSL_dot()."""
    glsl = "void test() { float x = dot(vec3(1.0), vec3(2.0)); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_dot' in opencl


def test_mod_function_prefix():
    """Test mod() becomes GLSL_mod() (different semantics from OpenCL)."""
    glsl = "void test() { float x = mod(5.0, 3.0); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_mod(5.0f, 3.0f)' in opencl


def test_mix_function_prefix():
    """Test mix() becomes GLSL_mix()."""
    glsl = "void test() { float x = mix(1.0, 2.0, 0.5); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_mix(1.0f, 2.0f, 0.5f)' in opencl


def test_clamp_function_prefix():
    """Test clamp() becomes GLSL_clamp()."""
    glsl = "void test() { float x = clamp(5.0, 0.0, 1.0); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_clamp(5.0f, 0.0f, 1.0f)' in opencl


def test_nested_function_calls():
    """Test nested function calls all get GLSL_ prefix."""
    glsl = "void test() { float x = sin(cos(1.0)); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_sin(GLSL_cos(1.0f))' in opencl


def test_function_call_with_vector():
    """Test function call with vector argument."""
    glsl = "void test() { float x = length(vec3(1.0, 2.0, 3.0)); }"
    opencl = transform_and_emit(glsl)

    assert 'GLSL_length((float3)(1.0f, 2.0f, 3.0f))' in opencl


def test_user_function_no_prefix():
    """Test user-defined functions don't get GLSL_ prefix."""
    glsl = """
    float myFunc(float x) { return x; }
    void test() { float y = myFunc(1.0); }
    """
    opencl = transform_and_emit(glsl)

    # User function should NOT have GLSL_ prefix
    assert 'myFunc(1.0f)' in opencl
    assert 'GLSL_myFunc' not in opencl
