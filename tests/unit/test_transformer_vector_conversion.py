"""
Unit tests for vector size/type conversion constructors (Bug-fix campaign
category N).

Category N = a GLSL vector constructor with a SINGLE vector-valued argument
is emitted as a C-style vector cast, which OpenCL rejects. OpenCL's
`(T)(...)` vector-literal syntax only broadcasts scalars or assembles
component lists — it cannot change a whole vector's element type
("invalid conversion between ext-vector type 'int2' and 'float2'") nor
truncate a wider vector. Three transformer-only rewrites at the constructor
site (`_transform_call_expression`), driven by `_get_type_name` inference:

1. Element-type conversion (same width):
     ivec2(vec2_expr)  -> convert_int2(vec2_expr)
     vec3(ivec3_expr)  -> convert_float3(ivec3_expr)
2. Truncation (wider single vector arg):
     vec3(vec4_expr)   -> vec4_expr.xyz     (plus convert_* if the element
     vec2(vec3_expr)   -> vec3_expr.xy       base also differs)
3. Bool masks: OpenCL vector comparisons yield -1 for true; GLSL
   vec4(bvec4) must yield 1.0/0.0, so the mask is normalized with `& 1`
   before converting: vec4(a < b) -> convert_float4((a < b) & 1).

Scalar broadcasts vec3(1.0), component lists vec2(x, y), identity casts
vec2(vec2_expr) and unknown argument types keep the existing cast emission.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


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


def _fn(body):
    return ("vec4 f(vec2 uv, vec3 p3, vec4 p4, ivec3 ip3, uvec2 up2) {\n"
            + body + "\n}\n")


# ---- element-type conversion (cluster A) -----------------------------------

def test_ivec2_from_vec2_param(parser, transformer, emitter):
    """The texelFetch(ch, ivec2(fragCoord), 0) shape — the most common N."""
    out = t(_fn("ivec2 q = ivec2(uv); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(uv)' in out
    assert '(int2)(uv)' not in out


def test_ivec3_from_builtin_call(parser, transformer, emitter):
    """ivec3(floor(v)) — argument type inferred through a GLSL_ builtin."""
    out = t(_fn("ivec3 m = ivec3(floor(p3)); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int3(GLSL_floor(p3))' in out
    assert '(int3)(GLSL_floor' not in out


def test_vec3_from_ivec3_param(parser, transformer, emitter):
    out = t(_fn("vec3 v = vec3(ip3); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_float3(ip3)' in out
    assert '(float3)(ip3)' not in out


def test_vec2_from_uvec2_param(parser, transformer, emitter):
    out = t(_fn("vec2 v = vec2(up2); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_float2(up2)' in out


def test_ivec2_from_local_vec2_expr(parser, transformer, emitter):
    """Type inferred from a locally declared variable, not a parameter."""
    out = t(_fn("vec2 c = uv * 2.0; ivec2 q = ivec2(c); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(c)' in out


# ---- bool-mask conversion (cluster A, bvec sub-case) -----------------------

def test_vec4_from_vector_comparison(parser, transformer, emitter):
    """vec4(p < q): OpenCL comparison gives -1 for true, GLSL wants 1.0 —
    the mask must be &1-normalized, then converted."""
    out = t(_fn("vec4 s = vec4(p4 < vec4(0.0)); return s;"),
            parser, transformer, emitter)
    assert 'convert_float4(' in out
    assert '& 1' in out
    assert '(float4)(p4 <' not in out


def test_vec3_from_lessThanEqual(parser, transformer, emitter):
    """vec3(lessThanEqual(a, b)) — mask produced by comparison lowering."""
    out = t(_fn("vec3 s = vec3(lessThanEqual(p3, vec3(0.5))); return vec4(s, 1.0);"),
            parser, transformer, emitter)
    assert 'convert_float3(' in out
    assert '& 1' in out


def test_vec4_from_bvec_variable(parser, transformer, emitter):
    out = t(_fn("bvec4 b = lessThan(p4, vec4(0.0)); vec4 s = vec4(b); return s;"),
            parser, transformer, emitter)
    assert 'convert_float4((b & 1))' in out


# ---- truncation (cluster B) ------------------------------------------------

def test_vec3_from_vec4_param(parser, transformer, emitter):
    out = t(_fn("vec3 v = vec3(p4); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    assert 'p4.xyz' in out
    assert '(float3)(p4)' not in out


def test_vec2_from_vec3_param(parser, transformer, emitter):
    """The length(vec2(lightDir) - p) shape."""
    out = t(_fn("vec2 v = vec2(p3); return vec4(v, 0.0, 1.0);"),
            parser, transformer, emitter)
    assert 'p3.xy' in out
    assert '(float2)(p3)' not in out


def test_vec3_from_texture_call(parser, transformer, emitter):
    """vec3(texture(ch, uv)) — texture() returns vec4; needs .xyz."""
    glsl = ("vec4 f(sampler2D ch, vec2 uv) {\n"
            "  vec3 c = vec3(texture(ch, uv));\n"
            "  return vec4(c, 1.0);\n}\n")
    out = t(glsl, parser, transformer, emitter)
    assert 'texture(ch, uv).xyz' in out
    assert '(float3)(texture' not in out


def test_ivec2_from_vec3_truncate_and_convert(parser, transformer, emitter):
    """Size AND element base differ: swizzle first, then convert."""
    out = t(_fn("ivec2 q = ivec2(p3); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(p3.xy)' in out


# ---- shapes that must KEEP the existing cast emission ----------------------

def test_scalar_broadcast_unchanged(parser, transformer, emitter):
    out = t(_fn("vec3 v = vec3(1.0); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    assert '(float3)(1.0f)' in out
    assert 'convert_float3' not in out


def test_component_list_unchanged(parser, transformer, emitter):
    out = t(_fn("vec2 v = vec2(1.0, 2.0); return vec4(v, 0.0, 1.0);"),
            parser, transformer, emitter)
    assert '(float2)(1.0f, 2.0f)' in out


def test_vec4_from_vec3_and_scalar_unchanged(parser, transformer, emitter):
    """Multi-arg component list with a sub-vector is legal OpenCL."""
    out = t(_fn("vec4 v = vec4(p3, 1.0); return v;"),
            parser, transformer, emitter)
    assert '(float4)(p3, 1.0f)' in out


def test_identity_ctor_unchanged(parser, transformer, emitter):
    """vec2(vec2) is an identity — the cast is legal; don't touch it."""
    out = t(_fn("vec2 v = vec2(uv); return vec4(v, 0.0, 1.0);"),
            parser, transformer, emitter)
    assert '(float2)(uv)' in out
    assert 'convert_float2' not in out


def test_unknown_arg_type_unchanged(parser, transformer, emitter):
    """An identifier with un-inferrable type (e.g. a macro-defined value)
    must keep the cast (no guess)."""
    out = t(_fn("vec2 v = vec2(SOME_MACRO_VALUE); return vec4(v, 0.0, 1.0);"),
            parser, transformer, emitter)
    assert 'convert_float2' not in out
    assert '(float2)(SOME_MACRO_VALUE)' in out


def test_widening_left_alone(parser, transformer, emitter):
    """vec4(vec2) is invalid GLSL — never emitted by real shaders; do not
    rewrite it into something silently different."""
    out = t(_fn("vec4 v = vec4(uv); return v;"),
            parser, transformer, emitter)
    assert 'convert_float4' not in out


# ---- vector-vector conversion through a bitwise op (Session 21) -------------
# The arg's type must be inferred through &/|/^/<<>> for the ctor to convert.
# `iuv & 7` (int-vector mask / quantize idiom) was un-typed, so vec2(iuv & 7)
# fell back to the invalid (float2)(int2_expr) cast.

def test_vec_from_bitwise_and_vector(parser, transformer, emitter):
    """vec3(ip3 & 7): & result infers ivec3, so the ctor converts not casts."""
    out = t(_fn("vec3 v = vec3(ip3 & 7); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    assert 'convert_float3(ip3 & 7)' in out
    assert '(float3)(ip3 &' not in out


def test_vec_from_shift_vector(parser, transformer, emitter):
    """vec2(up2 >> 2u): shift result takes the left (uvec2) type."""
    out = t(_fn("vec2 v = vec2(up2 >> 2u); return vec4(v, 0.0, 1.0);"),
            parser, transformer, emitter)
    assert 'convert_float2(up2 >> 2u)' in out
    assert '(float2)(up2 >>' not in out


# ---- scalar-from-vector ctor: GLSL float(vecN) takes .x (Session 21) --------
# Same ctor site, scalar target. OpenCL rejects (float)(float3_expr); GLSL
# float(v)==v.x, int(v)==int(v.x). ~6 corpus shaders (hash-to-float idioms).

def test_float_from_vec3(parser, transformer, emitter):
    """float(vec3) -> vec3.x (base matches: no cast needed)."""
    out = t(_fn("float x = float(p3); return vec4(x);"),
            parser, transformer, emitter)
    assert 'p3.x' in out
    assert '(float)(p3)' not in out


def test_int_from_vec3(parser, transformer, emitter):
    """int(vec3): take .x then a scalar cast (element base differs)."""
    out = t(_fn("int i = int(p3); return vec4(0.0);"),
            parser, transformer, emitter)
    assert '(int)(p3.x)' in out
    assert '(int)(p3)' not in out


def test_float_from_uvec_expr(parser, transformer, emitter):
    """float(uvec2 & mask): .x of a parenthesized expr, then scalar cast."""
    out = t(_fn("float x = float(up2 & 3u); return vec4(x);"),
            parser, transformer, emitter)
    assert '(float)((up2 & 3u).x)' in out


def test_float_from_scalar_unchanged(parser, transformer, emitter):
    """float(intVar) is a normal scalar cast — must NOT grow a .x."""
    out = t(_fn("int n = 3; float x = float(n); return vec4(x);"),
            parser, transformer, emitter)
    assert '(float)(n)' in out
    assert 'n.x' not in out


# ---- Session 30: type-inference gaps that fell back to the invalid C cast ---

def test_ivec2_from_round_call(parser, transformer, emitter):
    """ivec2(round(uv)): `round` maps to the native OpenCL `round` (no GLSL_
    prefix), so its vector return type must still be inferred for the
    conversion to fire. Xs3fRB."""
    out = t(_fn("ivec2 g = ivec2(round(uv)); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(round(uv))' in out
    assert '(int2)(round' not in out


def test_ivec2_from_assignment_expr(parser, transformer, emitter):
    """ivec2(o /= .7): an assignment expression carries its target's (vector)
    type. 4dSfWD."""
    out = t(_fn("vec2 o = uv; ivec2 g = ivec2(o /= 0.7); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(' in out
    assert '(int2)(o' not in out


def test_ivec3_from_step_scalar_vector(parser, transformer, emitter):
    """ivec3(step(0.25, p3)): step()'s result type is the WIDEST arg (the
    vector), not arg[0] (a scalar). 4ljyRc."""
    out = t(_fn("ivec3 m = ivec3(step(0.25, p3)); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int3(GLSL_step(0.25f, p3))' in out
    assert '(int3)(GLSL_step' not in out


# ---- Session 50: binary op with a statically untypeable operand -------------
# An object-like macro (`#define scale 20.`) survives as a bare identifier
# (object macros are not inlined), so `vec / scale` had ONE typed operand and
# _infer_binary_op_type bailed -> the ctor arg lost its type -> invalid
# `(int2)(float2)` C cast. Vector-op-unknown now infers the vector type.

def test_ivec2_from_vec_div_macro_constant(parser, transformer, emitter):
    """ivec2(uv / scale) with `scale` an untypeable object-macro identifier.
    ts3GzX, ltScRm."""
    out = t(_fn("ivec2 pos = ivec2(uv / scale); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(uv / scale)' in out
    assert '(int2)(uv' not in out


def test_ivec2_from_macro_mul_vec(parser, transformer, emitter):
    """Untypeable operand on the LEFT: ivec2(scale * uv)."""
    out = t(_fn("ivec2 pos = ivec2(scale * uv); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_int2(scale * uv)' in out
    assert '(int2)(scale' not in out


def test_vec2_from_ivec_minus_macro(parser, transformer, emitter):
    """Base change the other way: vec2(ip3.xy - K) with K untypeable.
    The inner vec2 arg is an int vector minus an unknown -> stays int2 -> the
    float2 ctor must convert."""
    out = t(_fn("ivec2 iv = ivec2(0); vec2 p = vec2(iv - K); return vec4(0.0);"),
            parser, transformer, emitter)
    assert 'convert_float2(iv - K)' in out
    assert '(float2)(iv' not in out


def test_matrix_times_unknown_stays_uninferred(parser, transformer, emitter):
    """Guard: a MATRIX operand with an unknown partner must NOT be inferred as
    a vector — `mat * unknown` is ambiguous (matrix or vector). vec2(m2 * K)
    keeps the plain cast rather than wrongly emitting convert_int2."""
    out = t("vec4 f(mat2 m2) { ivec2 q = ivec2(m2 * K); return vec4(0.0); }",
            parser, transformer, emitter)
    assert 'convert_int2' not in out
