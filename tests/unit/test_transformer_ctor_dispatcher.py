"""
Category N (Session 54) — overloadable constructor dispatcher for
single-argument constructors whose argument type the transpiler cannot
statically determine.

The category-N lowering `_transform_vector_conversion_ctor` handles the cases
where the argument's type IS known (element conversion, truncation, bool
masks). When the argument type is *unknown* it used to fall through to the
plain C cast `(int2)(expr)` — which OpenCL rejects if `expr` turns out to be a
vector (`vec2(textureSize(ch,0))`, `ivec2(iChannelResolution[i].xy)`,
`int(READ(...))` where READ is a texture-returning macro).

Fix: emit a call to the overloadable `GLSL_<glslType>` dispatcher (defined in
glslHelpers.h) instead and let the OpenCL compiler's overload resolution
supply the type — the same precedent as GLSL_mul / GLSL_matN. A KNOWN scalar
argument still broadcasts correctly through the plain cast, so only genuinely
untypeable arguments are routed.
"""

from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


def _emit(glsl_code: str) -> str:
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    emitter = OpenCLEmitter()
    ast = parser.parse(glsl_code)
    return emitter.emit(transformer.transform(ast))


def _fn(body):
    return ("vec4 f(vec2 uv, vec3 p3, vec4 p4, ivec3 ip3, uvec2 up2) {\n"
            + body + "\n}\n")


# ---- routing: untypeable single-arg VECTOR ctors --------------------------

def test_vec2_of_unknown_call_routes_to_dispatcher():
    """vec2(f(...)) with f un-inferrable -> GLSL_vec2 dispatcher."""
    out = _emit(_fn("vec2 v = vec2(SOME_MACRO_FN(uv)); return vec4(v, 0.0, 1.0);"))
    assert 'GLSL_vec2(SOME_MACRO_FN(uv))' in out
    assert '(float2)(SOME_MACRO_FN(uv))' not in out


def test_ivec2_of_unknown_swizzle_routes():
    """ivec2(iChannelResolution[i].xy): swizzle of an untyped array element."""
    out = _emit(_fn("ivec2 q = ivec2(iChannelResolution[0].xy); return vec4(0.0);"))
    assert 'GLSL_ivec2(' in out
    assert '(int2)(' not in out


def test_vec2_of_texturesize_routes():
    """vec2(textureSize(ch, 0)): textureSize's ivec2 return isn't inferred."""
    glsl = ("vec4 f(sampler2D SRC) {\n"
            "  vec2 S = vec2(textureSize(SRC, 0));\n"
            "  return vec4(S, 0.0, 1.0);\n}\n")
    out = _emit(glsl)
    assert 'GLSL_vec2(textureSize(SRC, 0))' in out
    assert '(float2)(textureSize' not in out


# ---- routing: untypeable single-arg SCALAR ctors (XdVSRc) -----------------

def test_int_of_unknown_call_routes_to_dispatcher():
    """int(READ(...)) where READ is a texture-returning macro -> GLSL_int."""
    out = _emit(_fn("int n = int(READ(0)); return vec4(0.0);"))
    assert 'GLSL_int(READ(0))' in out
    assert '(int)(READ(0))' not in out


def test_float_of_unknown_call_routes_to_dispatcher():
    out = _emit(_fn("float x = float(UNKNOWN_FN(uv)); return vec4(x);"))
    assert 'GLSL_float(UNKNOWN_FN(uv))' in out
    assert '(float)(UNKNOWN_FN(uv))' not in out


# ---- guard: BARE IDENTIFIER args are NOT routed (Xt23z3 regression) --------
# An untypeable bare identifier is almost always an object-like macro, and a
# comma-list macro (`#define COLOR_1 .5, .9, .95`) expands GLSL_vec3(COLOR_1)
# into a 3-arg call with no overload — while (float3)(COLOR_1) expands into a
# legal component-list literal. Only call/member/subscript args (which cannot
# swallow a comma list) are routed.

def test_bare_identifier_vector_arg_not_routed():
    out = _emit(_fn("vec3 c = vec3(COLOR_1); return vec4(c, 1.0);"))
    assert 'GLSL_vec3' not in out
    assert '(float3)(COLOR_1)' in out


def test_bare_identifier_scalar_arg_not_routed():
    out = _emit(_fn("float x = float(COUNT); return vec4(x);"))
    assert 'GLSL_float' not in out
    assert '(float)(COUNT)' in out


# ---- guards: KNOWN types keep their existing lowering ---------------------

def test_known_vector_still_converts_not_routed():
    """ivec2(uv) with uv:vec2 stays convert_int2 (typed path wins)."""
    out = _emit(_fn("ivec2 q = ivec2(uv); return vec4(0.0);"))
    assert 'convert_int2(uv)' in out
    assert 'GLSL_ivec2' not in out


def test_known_scalar_broadcast_not_routed():
    """vec3(1.0) is an obvious scalar broadcast — keep the plain cast."""
    out = _emit(_fn("vec3 v = vec3(1.0); return vec4(v, 1.0);"))
    assert '(float3)(1.0f)' in out
    assert 'GLSL_vec3' not in out


def test_known_scalar_ctor_not_routed():
    """float(n) with n:int is a normal scalar cast — not routed."""
    out = _emit(_fn("int n = 3; float x = float(n); return vec4(x);"))
    assert 'GLSL_float' not in out
    assert '(float)(n)' in out


def test_widening_not_routed():
    """vec4(uv) is invalid GLSL widening — leave as-is (arg type is known)."""
    out = _emit(_fn("vec4 v = vec4(uv); return v;"))
    assert 'GLSL_vec4' not in out


def test_identity_ctor_not_routed():
    """vec2(uv) with uv:vec2 identity — plain cast, not the dispatcher."""
    out = _emit(_fn("vec2 v = vec2(uv); return vec4(v, 0.0, 1.0);"))
    assert '(float2)(uv)' in out
    assert 'GLSL_vec2' not in out


def test_component_list_not_routed():
    """Multi-arg ctors are legal OpenCL literals — never routed."""
    out = _emit(_fn("vec2 v = vec2(1.0, 2.0); return vec4(v, 0.0, 1.0);"))
    assert '(float2)(1.0f, 2.0f)' in out
    assert 'GLSL_vec2' not in out


def test_overloaded_fn_arg_not_routed():
    """AI guard: a type-overloaded user-fn arg keeps the broadcast cast."""
    glsl = (
        "float g(float x){ return x; }\n"
        "vec2  g(vec2  x){ return x; }\n"
        "vec4  g(vec4  x){ return x; }\n"
        "void mainImage(out vec4 O, in vec2 U){\n"
        "  vec3 c = vec3(g(U.x));\n"
        "  O = vec4(c, 1.0);\n"
        "}\n"
    )
    out = _emit(glsl)
    assert 'GLSL_vec3' not in out
    assert '(float3)(g(' in out
