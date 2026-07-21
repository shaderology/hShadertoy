"""
Category N (Session 54), textual path — single-argument vector constructors
inside #define bodies / #if-region code lines route to the overloadable
GLSL_<type> dispatcher instead of the invalid C cast.

The AST never sees macro bodies, so `#define T(U) texelFetch(ch, ivec2(U), 0)`
was rewritten `ivec2(U)` -> `(int2)(U)`; when U expands to a float2 at the use
site that is the classic category-N `(int2)(float2)` compile error.

Scope (documented in PreprocessorTransformer._transform_macro_body):
  * Only SINGLE top-level argument constructors route (a 2+ arg component list
    is a legal OpenCL vector literal — kept as a cast).
  * Only when the argument text contains an identifier (a letter/underscore):
    a pure numeric-literal argument is provably a scalar broadcast, so the
    cast is already correct and is left untouched (keeps the blast radius to
    the constructors that can actually be vectors).
  * An unscannable (unbalanced-paren) argument list keeps the cast.
  * Scalar constructors (float/int/...) are NOT routed here — see
    test_transformer_scalar_ctor.py which locks the (float)(x) cast form.
"""

import pytest
from src.glsl_to_opencl.preprocessor import PreprocessorTransformer


@pytest.fixture
def pp():
    return PreprocessorTransformer()


def test_single_ident_ivec2_routes(pp):
    """The canonical texelFetch(ch, ivec2(U), 0) macro-body shape."""
    result = pp.transform("#define T(U) texelFetch(iChannel0, ivec2(U), 0)")
    assert "GLSL_ivec2(U)" in result
    assert "(int2)(U)" not in result


def test_single_ident_vec2_routes(pp):
    result = pp.transform("#define P(a) vec2(a)")
    assert "GLSL_vec2(a)" in result
    assert "(float2)(a)" not in result


def test_two_arg_ivec2_stays_cast(pp):
    """A component list is a legal OpenCL vector literal — keep the cast."""
    result = pp.transform("#define S ivec2(800, 600)")
    assert "(int2)(800, 600)" in result
    assert "GLSL_ivec2" not in result


def test_two_arg_vec2_with_ident_stays_cast(pp):
    result = pp.transform("#define D(a,b) vec2(a, b)")
    assert "(float2)(a, b)" in result
    assert "GLSL_vec2" not in result


def test_scalar_literal_single_arg_stays_cast(pp):
    """vec2(0.5): provably scalar broadcast — cast is correct, don't route."""
    result = pp.transform("#define HALF vec2(0.5)")
    assert "(float2)(0.5f)" in result
    assert "GLSL_vec2" not in result


def test_scalar_literal_in_ifdef_stays_cast(pp):
    result = pp.transform("#ifdef FOO\n    vec2 z = vec2(0.);\n#endif")
    assert "(float2)(0.f)" in result
    assert "GLSL_vec2" not in result


def test_ident_ctor_in_ifdef_routes(pp):
    result = pp.transform("#ifdef FOO\n    ivec2 q = ivec2(uv);\n#endif")
    assert "GLSL_ivec2(uv)" in result
    assert "(int2)(uv)" not in result


def test_hlsl_alias_single_ident_routes(pp):
    """HLSL-alias float2(k) single-arg -> GLSL_vec2 dispatcher (GLSL name)."""
    result = pp.transform("#define K(k) float2(k)")
    assert "GLSL_vec2(k)" in result
    assert "(float2)(k)" not in result


def test_hlsl_alias_two_arg_stays_cast(pp):
    """float2(k, 0.25) two-arg keyPressed idiom stays a cast."""
    result = pp.transform("#define keyPressed(k) (texture(ch,float2(k,0.25)).x>0.0)")
    assert "(float2)(k,0.25f)" in result
    assert "GLSL_vec2" not in result


def test_nested_ctor_arg_routes_and_inner_cast(pp):
    """vec3(int3(1,2,3)): outer single-arg (has letters) routes; inner
    component list stays a cast -> GLSL_vec3((int3)(1,2,3))."""
    result = pp.transform("#ifdef LIGHT\n    float3 v = float3(int3(1,2,3));\n#endif")
    assert "GLSL_vec3((int3)(1,2,3))" in result


def test_unbalanced_arg_keeps_cast(pp):
    """An unscannable (unbalanced) ctor arg list keeps today's cast."""
    result = pp.transform("#define OPEN vec2(a")
    assert "(float2)(a" in result
    assert "GLSL_vec2" not in result


def test_comma_list_object_macro_arg_keeps_cast(pp):
    """A known comma-list object macro as the single arg keeps the cast:
    vec3(COLOR_1) textually scans as one arg but EXPANDS to three — the
    dispatcher would become a 3-arg call with no overload, while the cast
    expands into a legal component list (the Xt23z3 shape)."""
    src = ("#define COLOR_1 0.50, 0.90, 0.95\n"
           "#define TINT(u) mix(vec3(COLOR_1), vec3(u), 0.5)")
    result = pp.transform(src)
    assert "(float3)(COLOR_1)" in result
    assert "GLSL_vec3(COLOR_1)" not in result
    # the parameter arg is still routed (params cannot be comma lists)
    assert "GLSL_vec3(u)" in result


def test_comma_body_with_parens_not_flagged(pp):
    """Commas inside parens don't make a macro a comma list — routing of its
    name stays enabled."""
    src = ("#define V vec2(1.0, 2.0)\n"
           "#define W(x) vec2(V)")
    result = pp.transform(src)
    # V's body is a ctor call (commas nested) -> vec2(V) single-arg routes
    assert "GLSL_vec2(V)" in result
