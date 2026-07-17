"""
Category AI — vecN(overloadedUserFn(...)) must broadcast, not truncate.

A user function with type-overloads (`float f(float)` + `vec2 f(vec2)` +
`vec4 f(vec4)`) collapses to ONE entry in `user_function_return_types` (the
last definition wins, here vec4). So `vec3(f(scalar))` mis-infers the arg as
vec4 and the category-N constructor lowering truncates it to `f(...).xyz` —
but at runtime `f(scalar)` returns a *scalar*, and OpenCL rejects `.xyz` on a
scalar ("member reference base type 'float' is not a structure or union" —
shader MlySRh).

Fix: a width that traces to a *type-overloaded* user function is untrustworthy
(same precedent as `_truncate_overflow_ctor_args`), so the ctor lowering must
NOT truncate — it falls through to the plain cast `(float3)(...)`, which
broadcasts a scalar correctly and is an identity for a same-type vector.
"""

from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


def transform_and_emit(glsl_code: str) -> str:
    parser = GLSLParser()
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    transformer = ASTTransformer(type_checker)
    emitter = OpenCLEmitter()
    ast = parser.parse(glsl_code)
    return emitter.emit(transformer.transform(ast))


OVERLOADED = (
    "float f(float e0, float e1, float x) { return (x-e0)/(e1-e0); }\n"
    "vec2  f(vec2  e0, vec2  e1, vec2  x) { return (x-e0)/(e1-e0); }\n"
    "vec4  f(vec4  e0, vec4  e1, vec4  x) { return (x-e0)/(e1-e0); }\n"
)


def test_vec3_of_overloaded_scalar_call_broadcasts():
    """vec3(f(.,.,scalar)) with a type-overloaded f must NOT emit `.xyz`."""
    glsl = OVERLOADED + (
        "void mainImage(out vec4 O, in vec2 U){\n"
        "  vec4 grids = vec4(1.0);\n"
        "  vec3 c = vec3(f(0., 2., grids.y));\n"
        "  O = vec4(c, 1.0);\n"
        "}\n"
    )
    out = transform_and_emit(glsl)
    line = [l for l in out.splitlines() if 'vec3 c' in l or 'float3 c' in l][0]
    assert '.xyz' not in line, f"scalar swizzled: {line!r}"
    assert '(float3)(f(' in line, f"not a broadcast cast: {line!r}"


def test_non_overloaded_vec4_fn_still_truncates():
    """A NON-overloaded user fn returning vec4 keeps the vec3(v4)->v4.xyz
    truncation — the guard must be narrow (only true type-overloads)."""
    glsl = (
        "vec4 getColor(vec2 p) { return vec4(p, 0.0, 1.0); }\n"
        "void mainImage(out vec4 O, in vec2 U){\n"
        "  vec3 c = vec3(getColor(U));\n"
        "  O = vec4(c, 1.0);\n"
        "}\n"
    )
    out = transform_and_emit(glsl)
    line = [l for l in out.splitlines() if ' c =' in l and 'getColor' in l][0]
    assert 'getColor(U).xyz' in line, f"truncation lost: {line!r}"


def test_plain_vec3_of_vec4_local_still_truncates():
    """No user fn involved: vec3(v4) still lowers to v4.xyz (unchanged)."""
    glsl = (
        "void mainImage(out vec4 O, in vec2 U){\n"
        "  vec4 v = vec4(1.0);\n"
        "  vec3 c = vec3(v);\n"
        "  O = vec4(c, 1.0);\n"
        "}\n"
    )
    out = transform_and_emit(glsl)
    line = [l for l in out.splitlines() if ' c =' in l][0]
    assert 'v.xyz' in line, f"truncation lost: {line!r}"
