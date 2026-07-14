"""
Unit tests for scalar type constructors (Bug-fix campaign category V).

Category V = GLSL scalar type constructors `float(x)`, `int(x)`, `uint(x)`,
`bool(x)` left as function-call syntax in TEXTUALLY-transformed regions —
`#define` macro bodies and code lines inside `#if`/`#ifdef` conditional
blocks, both handled by `PreprocessorTransformer._transform_macro_body`
(string/regex pass, not the AST). OpenCL has no `float(x)` function, so the
kernel fails with "expected expression". The AST path already converts
scalar ctors to C casts (`(float)(x)`) — guard tests below lock that in.

Real corpus shapes reproduced here:
  4dVXzR:  #define time ((saw(float(__LINE__))+.5)*(iTime/PI+12345.12345))
  ldX3R2:  #define id(i,j,k) (float(128+i)+256.*float(128+j)+65536.*float(k))
  ldy3D1:  #define CEIL(x) (float (int ((x) + 0.9999)))
  MlK3zt:  color = float(BooleanFunction_L(percent));   // inside #if block
  ldSGRW:  ltime = time*3.0 + float(i)*20.134;          // inside #if block
"""

import pytest
from src.glsl_to_opencl.preprocessor import PreprocessorTransformer
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


@pytest.fixture
def pp():
    """Fixture for PreprocessorTransformer instance."""
    return PreprocessorTransformer()


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


def t(glsl_code, parser, transformer, emitter):
    ast = parser.parse(glsl_code)
    return emitter.emit(transformer.transform(ast))


# ============================================================================
# Scalar ctors in #define bodies (the V core)
# ============================================================================

def test_float_ctor_in_define(pp):
    """float(x) in an object-like macro body -> (float)(x)."""
    source = "#define F float(x)"
    result = pp.transform(source)
    assert "(float)(x)" in result
    assert "float(x)" not in result.replace("(float)(x)", "")


def test_int_ctor_in_define(pp):
    """int(x) in a macro body -> (int)(x)."""
    source = "#define I(x) int(x)"
    result = pp.transform(source)
    assert "(int)(x)" in result


def test_uint_ctor_in_define(pp):
    """uint(x) in a macro body -> (uint)(x)."""
    source = "#define U(x) uint(x)"
    result = pp.transform(source)
    assert "(uint)(x)" in result


def test_bool_ctor_in_define(pp):
    """bool(x) in a macro body -> (bool)(x)."""
    source = "#define B(x) bool(x)"
    result = pp.transform(source)
    assert "(bool)(x)" in result


def test_float_ctor_of_line_macro(pp):
    """4dVXzR shape: float(__LINE__) inside a time macro."""
    source = "#define time ((saw(float(__LINE__))+.5)*(iTime/PI+12345.12345))"
    result = pp.transform(source)
    assert "saw((float)(__LINE__))" in result


def test_multiple_float_ctors_in_define(pp):
    """ldX3R2 shape: several float(expr) ctors in one function-like macro."""
    source = "#define id(i,j,k) (float(128+i)+256.*float(128+j)+65536.*float(k))"
    result = pp.transform(source)
    assert "(float)(128+i)" in result
    assert "(float)(128+j)" in result
    assert "(float)(k)" in result


def test_nested_scalar_ctors_with_space(pp):
    """ldy3D1 shape: float (int (...)) with whitespace before the paren."""
    source = "#define CEIL(x) (float (int ((x) + 0.9999)))"
    result = pp.transform(source)
    assert "(float)((int)((x) + 0.9999f))" in result


def test_float_ctor_of_expression(pp):
    """Ctor of a compound expression."""
    source = "#define G(a,b) float(a+b)"
    result = pp.transform(source)
    assert "(float)(a+b)" in result


# ============================================================================
# Scalar ctors in code lines inside #if / #ifdef blocks
# ============================================================================

def test_float_ctor_of_call_in_if_block(pp):
    """MlK3zt shape: float(userFn(arg)) on a code line inside #if."""
    source = """#if SHOW_2D_SHAPE
    color = float(BooleanFunction_L(percent));
#endif"""
    result = pp.transform(source)
    assert "(float)(BooleanFunction_L(percent))" in result


def test_float_ctor_of_loop_var_in_if_block(pp):
    """ldSGRW shape: float(i) arithmetic on a code line inside #if."""
    source = """#if 1
    ltime = time*3.0 + float(i)*20.134;
#endif"""
    result = pp.transform(source)
    assert "(float)(i)" in result
    assert "3.0f" in result


def test_int_ctor_in_ifdef_block(pp):
    source = """#ifdef FOO
    int n = int(f * 4.0);
#endif"""
    result = pp.transform(source)
    assert "(int)(f * 4.0f)" in result


# ============================================================================
# Guards: things that must NOT be touched
# ============================================================================

def test_existing_cast_not_double_transformed(pp):
    """(float)(x) already cast-style must stay unchanged."""
    source = "#define F(x) (float)(x)"
    result = pp.transform(source)
    assert "((float))" not in result
    assert "(float)(x)" in result


def test_vector_ctor_unchanged_by_scalar_rule(pp):
    """float2(...)/vec2(...) belong to the vector rule, not the scalar one."""
    source = "#define D vec2(1.0, 0.0)"
    result = pp.transform(source)
    assert "(float2)(1.0f, 0.0f)" in result
    # no stray (float)( injected
    assert "(float)(" not in result


def test_identifier_containing_type_name_untouched(pp):
    """User function names embedding a type name must not match."""
    source = "#define X intersect(ro, rd) + myfloat(a) + convert_float(a)"
    result = pp.transform(source)
    assert "intersect(ro, rd)" in result
    assert "myfloat(a)" in result
    assert "convert_float(a)" in result


def test_declaration_in_if_block_untouched(pp):
    """A declaration `float x = ...` has no paren after the type."""
    source = """#if 1
    float x = fract(y);
#endif"""
    result = pp.transform(source)
    assert "float x = GLSL_fract(y);" in result


def test_function_definition_in_if_block_untouched(pp):
    """`float rand(vec2 co)` — return type followed by a name, not a paren."""
    source = """#ifdef HQ
float rand(vec2 co) { return fract(co.x); }
#endif"""
    result = pp.transform(source)
    assert "float rand((float2)" not in result
    assert "float rand(" in result


# ============================================================================
# AST-path guards (scalar ctors in normal parsed code already work — lock in)
# ============================================================================

def _fn(body):
    return "vec4 f(float x, int i, uint u, vec2 uv) {\n" + body + "\n}\n"


def test_ast_float_ctor_of_int(parser, transformer, emitter):
    out = t(_fn("float a = float(i); return vec4(a);"),
            parser, transformer, emitter)
    assert "(float)(i)" in out or "convert_float(i)" in out
    assert "float(i)" not in out.replace("(float)(i)", "").replace("convert_float(i)", "")


def test_ast_int_ctor_of_float(parser, transformer, emitter):
    out = t(_fn("int b = int(x); return vec4(0.0);"),
            parser, transformer, emitter)
    assert "(int)(x)" in out or "convert_int(x)" in out


def test_ast_float_ctor_of_expression(parser, transformer, emitter):
    out = t(_fn("float c = float(i + 1); return vec4(c);"),
            parser, transformer, emitter)
    assert "float(i + 1)" not in out


def test_ast_nested_scalar_ctors(parser, transformer, emitter):
    out = t(_fn("float d = float(int(x)); return vec4(d);"),
            parser, transformer, emitter)
    assert "int(x)" not in out.replace("(int)(x)", "").replace("convert_int(x)", "")
