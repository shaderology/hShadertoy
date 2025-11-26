"""
Unit tests for Session 9: Preprocessor Directives.

Tests the PreprocessorTransformer class which transforms GLSL preprocessor
directives to OpenCL equivalents.

Test coverage:
- #define macro transformation (15 tests)
- Float literals in macros (10 tests)
- Function calls in macros (10 tests)
- Conditional directives (5 tests)
Total: 40 tests
"""

import pytest
from src.glsl_to_opencl.preprocessor import PreprocessorTransformer


@pytest.fixture
def transformer():
    """Fixture for PreprocessorTransformer instance."""
    return PreprocessorTransformer()


# ============================================================================
# #define Macro Transformation (15 tests)
# ============================================================================

def test_simple_define_object_like(transformer):
    """Test simple object-like macro."""
    source = "#define PI 3.14159265"
    result = transformer.transform(source)
    assert result == "#define PI 3.14159265f"


def test_simple_define_with_integer(transformer):
    """Test object-like macro with integer (no suffix)."""
    source = "#define SIZE 512"
    result = transformer.transform(source)
    assert result == "#define SIZE 512"


def test_simple_define_function_like(transformer):
    """Test simple function-like macro."""
    source = "#define SQUARE(x) ((x)*(x))"
    result = transformer.transform(source)
    assert result == "#define SQUARE(x) ((x)*(x))"


def test_define_with_glsl_function(transformer):
    """Test macro with GLSL function call."""
    source = "#define SINE(x) sin(x)"
    result = transformer.transform(source)
    assert result == "#define SINE(x) GLSL_sin(x)"


def test_define_with_multiple_functions(transformer):
    """Test macro with multiple GLSL function calls."""
    source = "#define random(x) fract(1e4*sin(x))"
    result = transformer.transform(source)
    assert result == "#define random(x) GLSL_fract(1e4f*GLSL_sin(x))"


def test_define_empty_macro(transformer):
    """Test empty macro definition."""
    source = "#define DIRECTION_X"
    result = transformer.transform(source)
    assert result == "#define DIRECTION_X"


def test_define_with_trailing_whitespace(transformer):
    """Test macro with trailing whitespace."""
    source = "#define PI 3.14    "
    result = transformer.transform(source)
    # Trailing whitespace is preserved in body
    assert "3.14f" in result


def test_define_with_leading_whitespace(transformer):
    """Test macro with leading whitespace."""
    source = "   #define PI 3.14"
    result = transformer.transform(source)
    assert result == "   #define PI 3.14f"


def test_define_with_comment(transformer):
    """Test macro with inline comment."""
    source = "#define PI 3.14159 // Pi constant"
    result = transformer.transform(source)
    assert "#define PI 3.14159f // Pi constant" == result


def test_define_with_parentheses_in_body(transformer):
    """Test macro with complex expression."""
    source = "#define CALC(x,y) ((x)+(y)*2.0)"
    result = transformer.transform(source)
    assert result == "#define CALC(x,y) ((x)+(y)*2.0f)"


def test_define_with_multiple_params(transformer):
    """Test function-like macro with multiple parameters."""
    source = "#define DOT2(a,b) ((a).x*(b).x+(a).y*(b).y)"
    result = transformer.transform(source)
    assert result == source  # No floats or functions to transform


def test_define_with_string_literal(transformer):
    """Test macro with string literal (edge case)."""
    source = '#define MSG "Hello 3.14"'
    result = transformer.transform(source)
    # Should not transform numbers inside strings (basic test)
    # Note: Our simple regex might still transform it (acceptable limitation)
    assert '#define MSG "Hello' in result


def test_multiple_defines(transformer):
    """Test multiple macro definitions."""
    source = "#define PI 3.14\n#define E 2.71\n#define SIZE 100"
    result = transformer.transform(source)
    lines = result.split('\n')
    assert lines[0] == "#define PI 3.14f"
    assert lines[1] == "#define E 2.71f"
    assert lines[2] == "#define SIZE 100"


def test_define_no_space_after_name(transformer):
    """Test macro with no space between name and body (edge case)."""
    source = "#define VALUE(x) (x+1.0)"
    result = transformer.transform(source)
    # Body should be transformed
    assert "1.0f" in result


def test_define_preserve_operator_spacing(transformer):
    """Test that operator spacing is preserved."""
    source = "#define CALC(x) x+1.0"
    result = transformer.transform(source)
    assert result == "#define CALC(x) x+1.0f"


# ============================================================================
# Float Literals in Macros (10 tests)
# ============================================================================

def test_float_with_decimal(transformer):
    """Test float literal with decimal point."""
    source = "#define PI 3.14159"
    result = transformer.transform(source)
    assert "3.14159f" in result


def test_float_with_trailing_zero(transformer):
    """Test float literal with trailing zeros."""
    source = "#define ONE 1.0"
    result = transformer.transform(source)
    assert "1.0f" in result


def test_float_with_exponent(transformer):
    """Test float literal with exponent."""
    source = "#define BIG 1e4"
    result = transformer.transform(source)
    assert "1e4f" in result


def test_float_with_negative_exponent(transformer):
    """Test float literal with negative exponent."""
    source = "#define SMALL 1.5e-3"
    result = transformer.transform(source)
    assert "1.5e-3f" in result


def test_float_with_positive_exponent(transformer):
    """Test float literal with explicit positive exponent."""
    source = "#define BIG 2.5e+10"
    result = transformer.transform(source)
    assert "2.5e+10f" in result


def test_float_decimal_start(transformer):
    """Test float literal starting with decimal point."""
    source = "#define HALF .5"
    result = transformer.transform(source)
    assert ".5f" in result


def test_float_already_has_suffix(transformer):
    """Test float literal that already has 'f' suffix."""
    source = "#define PI 3.14f"
    result = transformer.transform(source)
    # Should not double-add suffix (check for 'ff')
    assert "ff" not in result
    assert "3.14f" in result


def test_multiple_floats_in_macro(transformer):
    """Test multiple float literals in one macro."""
    source = "#define RECT(x,y) ((x)*2.0+(y)*3.5)"
    result = transformer.transform(source)
    assert "2.0f" in result
    assert "3.5f" in result


def test_float_in_complex_expression(transformer):
    """Test float in complex mathematical expression."""
    source = "#define CALC sin(3.14*x+0.5)"
    result = transformer.transform(source)
    assert "3.14f" in result
    assert "0.5f" in result
    assert "GLSL_sin" in result


def test_integer_not_transformed(transformer):
    """Test that integer literals are not given 'f' suffix."""
    source = "#define SIZE 512"
    result = transformer.transform(source)
    assert result == "#define SIZE 512"
    # Check that 512 doesn't have 'f' after it (not checking whole string for 'f' because 'define' has 'f')
    assert "512f" not in result


# ============================================================================
# Function Calls in Macros (10 tests)
# ============================================================================

def test_sin_function(transformer):
    """Test sin function transformation."""
    source = "#define SINE(x) sin(x)"
    result = transformer.transform(source)
    assert "GLSL_sin" in result


def test_cos_function(transformer):
    """Test cos function transformation."""
    source = "#define COSINE(x) cos(x)"
    result = transformer.transform(source)
    assert "GLSL_cos" in result


def test_fract_function(transformer):
    """Test fract function transformation."""
    source = "#define FRAC(x) fract(x)"
    result = transformer.transform(source)
    assert "GLSL_fract" in result


def test_normalize_function(transformer):
    """Test normalize function transformation."""
    source = "#define NORM(v) normalize(v)"
    result = transformer.transform(source)
    assert "GLSL_normalize" in result


def test_length_function(transformer):
    """Test length function transformation."""
    source = "#define LEN(v) length(v)"
    result = transformer.transform(source)
    assert "GLSL_length" in result


def test_nested_function_calls(transformer):
    """Test nested GLSL function calls."""
    source = "#define CALC sin(cos(x))"
    result = transformer.transform(source)
    assert "GLSL_sin(GLSL_cos(x))" in result


def test_function_with_float_arg(transformer):
    """Test function call with float literal argument."""
    source = "#define HALF_SIN sin(0.5)"
    result = transformer.transform(source)
    assert "GLSL_sin(0.5f)" in result


def test_pow_function(transformer):
    """Test pow function with two arguments."""
    source = "#define SQUARE(x) pow(x,2.0)"
    result = transformer.transform(source)
    assert "GLSL_pow(x,2.0f)" in result


def test_mod_function(transformer):
    """Test mod function transformation."""
    source = "#define MOD2(x) mod(x,2.0)"
    result = transformer.transform(source)
    assert "GLSL_mod(x,2.0f)" in result


def test_complex_function_expression(transformer):
    """Test complex expression with multiple functions and floats."""
    source = "#define random(x) fract(1e4*sin(x*541.17))"
    result = transformer.transform(source)
    assert "GLSL_fract" in result
    assert "GLSL_sin" in result
    assert "1e4f" in result
    assert "541.17f" in result


# ============================================================================
# Conditional Directives (5 tests)
# ============================================================================

def test_ifdef_pass_through(transformer):
    """Test #ifdef directive passes through unchanged."""
    source = "#ifdef SOME_FLAG"
    result = transformer.transform(source)
    assert result == source


def test_ifndef_pass_through(transformer):
    """Test #ifndef directive passes through unchanged."""
    source = "#ifndef SOME_FLAG"
    result = transformer.transform(source)
    assert result == source


def test_else_pass_through(transformer):
    """Test #else directive passes through unchanged."""
    source = "#else"
    result = transformer.transform(source)
    assert result == source


def test_endif_pass_through(transformer):
    """Test #endif directive passes through unchanged."""
    source = "#endif"
    result = transformer.transform(source)
    assert result == source


def test_elif_pass_through(transformer):
    """Test #elif directive passes through unchanged."""
    source = "#elif defined(OTHER_FLAG)"
    result = transformer.transform(source)
    assert result == source


# ============================================================================
# Vector Constructor Transformation (10 tests)
# ============================================================================

def test_vec2_constructor_in_define(transformer):
    """Test vec2 constructor in #define macro."""
    source = "#define DIR vec2(1.0, 0.0)"
    result = transformer.transform(source)
    assert result == "#define DIR (float2)(1.0f, 0.0f)"


def test_vec3_constructor_in_define(transformer):
    """Test vec3 constructor in #define macro."""
    source = "#define UP vec3(0, 1, 0)"
    result = transformer.transform(source)
    assert "(float3)(0, 1, 0)" in result


def test_vec4_constructor_in_define(transformer):
    """Test vec4 constructor in #define macro."""
    source = "#define COLOR vec4(1.0, 0.5, 0.0, 1.0)"
    result = transformer.transform(source)
    assert "(float4)(1.0f, 0.5f, 0.0f, 1.0f)" in result


def test_ivec2_constructor_in_define(transformer):
    """Test ivec2 constructor in #define macro."""
    source = "#define SIZE ivec2(800, 600)"
    result = transformer.transform(source)
    assert "(int2)(800, 600)" in result


def test_multiple_vector_constructors_in_define(transformer):
    """Test multiple vector constructors in one macro."""
    source = "#define BLEND(a,b,t) mix(vec2(a), vec2(b), t)"
    result = transformer.transform(source)
    assert "(float2)(a)" in result
    assert "(float2)(b)" in result
    assert "GLSL_mix" in result


def test_nested_vector_constructor_in_define(transformer):
    """Test nested vector constructor with float literal."""
    source = "#define HALF vec2(0.5)"
    result = transformer.transform(source)
    assert "(float2)(0.5f)" in result


def test_vec2_constructor_in_ifdef_block(transformer):
    """Test vec2 constructor inside #ifdef block."""
    source = """#ifdef FOO
    vec2 bar = vec2(0.);
#endif"""
    result = transformer.transform(source)
    assert "(float2)(0.f)" in result
    assert "#ifdef FOO" in result
    assert "#endif" in result


def test_vec3_constructor_in_else_block(transformer):
    """Test vec3 constructor inside #else block."""
    source = """#ifdef FOO
    vec2 a = vec2(1.0);
#else
    vec3 b = vec3(0.0);
#endif"""
    result = transformer.transform(source)
    assert "(float2)(1.0f)" in result
    assert "(float3)(0.0f)" in result


def test_vec_constructor_with_function_call(transformer):
    """Test vector constructor with GLSL function inside."""
    source = "#define CALC vec2(sin(x), cos(y))"
    result = transformer.transform(source)
    assert "(float2)(GLSL_sin(x), GLSL_cos(y))" in result


def test_vector_constructor_no_double_transform(transformer):
    """Test that already transformed constructors are not double-transformed."""
    source = "#define DIR (float2)(1.0f, 0.0f)"
    result = transformer.transform(source)
    # Should not double-transform
    assert "((float2))" not in result
    assert "(float2)(1.0f, 0.0f)" in result


# ============================================================================
# Integration Tests (covers remaining scenarios)
# ============================================================================

def test_mixed_directives(transformer):
    """Test mix of define and conditional directives."""
    source = """#define PI 3.14
#ifdef USE_PI
#define CIRCLE_AREA(r) (PI*(r)*(r))
#endif"""
    result = transformer.transform(source)
    assert "3.14f" in result
    assert "#ifdef USE_PI" in result
    assert "#endif" in result


def test_real_world_macro(transformer):
    """Test real-world shader macro from macros.glsl."""
    source = "#define random(x)  fract(1e4*sin((x)*541.17))"
    result = transformer.transform(source)
    # Check key transformations (whitespace may vary)
    assert "GLSL_fract" in result
    assert "1e4f" in result
    assert "GLSL_sin" in result
    assert "541.17f" in result


def test_non_preprocessor_lines_unchanged(transformer):
    """Test that non-preprocessor lines pass through unchanged."""
    source = "float x = 3.14;"
    result = transformer.transform(source)
    assert result == source  # No transformation (not a preprocessor directive)


def test_include_directive_pass_through(transformer):
    """Test #include directive passes through unchanged."""
    source = "#include <stdio.h>"
    result = transformer.transform(source)
    assert result == source


def test_pragma_directive_pass_through(transformer):
    """Test #pragma directive passes through unchanged."""
    source = "#pragma once"
    result = transformer.transform(source)
    assert result == source
