"""
Unit tests for category K — GLSL array constructors and struct-constructor
rvalues.

GLSL rvalue constructors have no valid direct OpenCL C spelling under the old
emitter:

  * struct ctor `S(a, b)` was emitted as a bare brace list `{a, b}`, which is
    only legal as a declaration initializer — `return {..};` / `s = {..};`
    are syntax errors ("expected expression").
  * array ctor `float[3](a, b, c)` fell through untransformed and was emitted
    verbatim.
  * type-first declarations `float[4] name = ...` and unsized ctors
    `int[](...)` don't parse at all (tree-sitter-glsl rejects them).
  * array-typed function parameters `vec3 pts[4]` lost their name
    ("parameter name omitted").
  * struct fields declared as arrays (`TextPage pages[18];`) raised
    "Field declaration ... missing name(s)".

Fix (Session 11): array ctors become IR.ArrayConstructor; in declaration-
initializer position both struct and array ctors emit brace lists `{...}`
(nested aggregates recurse), while in expression position they emit C99
compound literals `((S){...})` / `((float[]){...})` — verified accepted by
the campaign OpenCL compiler, including indexed unsized array literals.
Type-first declarations are normalized by a pre-parse rewrite in
GLSLParser.parse.
"""

import sys
from pathlib import Path

import pytest

# tests/transpile.py is the campaign transpiler (header/kernel split path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def tp(glsl):
    """Transpile and return (header, kernel_body)."""
    result = transpile(glsl)
    return result.get_header(), result.get_kernel()


MAIN_NOOP = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
    "    fragColor = vec4(0.0);\n"
    "}\n"
)


# ============================================================================
# 1. Array constructors (K2) — initializer position -> brace list
# ============================================================================

def test_local_array_ctor_sized_initializer():
    """float temp[3] = float[3](...) -> float temp[3] = {...}; (MtcBzs)"""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    float temp[3] = float[3](0.1, 0.2, 0.3);\n"
        "    fragColor = vec4(temp[0]);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float temp[3] = {0.1f, 0.2f, 0.3f};" in kernel
    assert "float[3](" not in kernel


def test_local_vec_array_ctor_initializer():
    """Vector-element array ctor keeps (float2)(...) literals inside braces."""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 lut[2] = vec2[2](vec2(1.0, 2.0), vec2(3.0, 4.0));\n"
        "    fragColor = vec4(lut[1], lut[0]);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float2 lut[2] = {(float2)(1.0f, 2.0f), (float2)(3.0f, 4.0f)};" in kernel


def test_global_const_array_ctor_stays_file_scope():
    """const vec2 LUT[2] = vec2[2](...) at file scope -> brace init (4sByDR)."""
    glsl = (
        "const vec2 LUT[2] = vec2[2](vec2(1.0, 2.0), vec2(3.0, 4.0));\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(LUT[0], LUT[1]);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert ("const float2 LUT[2] = {(float2)(1.0f, 2.0f), (float2)(3.0f, 4.0f)};"
            in header)
    assert "vec2[2](" not in header and "float2[2](" not in header


def test_array_ctor_rvalue_indexed():
    """Immediately-indexed ctor -> indexed compound literal ((float[]){...})[i]."""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    int i = int(fragCoord.x);\n"
        "    float d = float[4](0.1, 0.2, 0.3, 0.4)[i];\n"
        "    fragColor = vec4(d);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "((float[]){0.1f, 0.2f, 0.3f, 0.4f})[i]" in kernel


# ============================================================================
# 2. Parse-stage forms (K3) — pre-parse rewrite makes them parseable
# ============================================================================

def test_type_first_array_declaration():
    """const float[4] p = float[4](...) parses and emits p[4] (4dfBWM)."""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    const float[4] pattern = float[4](0.2, 0.6, 0.8, 0.4);\n"
        "    fragColor = vec4(pattern[0]);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "const float pattern[4] = {0.2f, 0.6f, 0.8f, 0.4f};" in kernel


def test_unsized_array_ctor_initializer():
    """int[512] p = int[](...) — unsized ctor side, sized decl side (4tKcDD)."""
    glsl = (
        "int[4] p = int[](1, 2, 3, 4);\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(float(p[0]));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "int p[4] = {1, 2, 3, 4};" in header


# ============================================================================
# 3. Struct constructor rvalues (K1) -> compound literals
# ============================================================================

STRUCT_RAY = (
    "struct RayData { float dist; float angle; };\n"
)


def test_struct_ctor_return_compound_literal():
    """return RayData(a, b); -> return ((RayData){a, b}); (3dlGDX)"""
    glsl = (
        STRUCT_RAY +
        "RayData castRay(float d) {\n"
        "    return RayData(d, 0.5);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    RayData r = castRay(fragCoord.x);\n"
        "    fragColor = vec4(r.dist);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "return ((RayData){d, 0.5f});" in header


def test_struct_ctor_assignment_compound_literal():
    """spheres[0] = Sphere(...); -> compound literal rvalue (WdXXz2/4dtGWB)."""
    glsl = (
        "struct Sphere { float radius; vec3 center; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    Sphere spheres[2];\n"
        "    spheres[0] = Sphere(1.0, vec3(0.0, -1.0, 3.0));\n"
        "    spheres[1] = Sphere(2.0, vec3(2.0, 0.0, 4.0));\n"
        "    fragColor = vec4(spheres[0].radius);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "spheres[0] = ((Sphere){1.0f, (float3)(0.0f, -1.0f, 3.0f)});" in kernel


def test_struct_ctor_declaration_keeps_braces():
    """Declaration initializer keeps the plain brace list (no churn)."""
    glsl = (
        STRUCT_RAY +
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    RayData r = RayData(1.0, 2.0);\n"
        "    fragColor = vec4(r.dist);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "RayData r = {1.0f, 2.0f};" in kernel


def test_struct_ctor_as_function_argument():
    """f(S(...)) -> f(((S){...})) — ctor as call argument."""
    glsl = (
        STRUCT_RAY +
        "float score(RayData r) { return r.dist + r.angle; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(score(RayData(1.0, 2.0)));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "score(((RayData){1.0f, 2.0f}))" in kernel


def test_struct_array_ctor_nested_braces():
    """Sphere[2](Sphere(..), Sphere(..)) initializer -> nested brace lists."""
    glsl = (
        "struct Sphere { float radius; vec3 center; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    Sphere spheres[2] = Sphere[2](Sphere(1.0, vec3(0.0)),"
        " Sphere(2.0, vec3(1.0)));\n"
        "    fragColor = vec4(spheres[1].radius);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert ("Sphere spheres[2] = {{1.0f, (float3)(0.0f)},"
            " {2.0f, (float3)(1.0f)}};" in kernel)


# ============================================================================
# 4. Array-typed function parameters (K4) — name must survive
# ============================================================================

def test_out_array_parameter_keeps_name():
    """void f(out vec3 pts[4]) -> float3 pts[4]; no pointer, no & at call (4tBBDK)."""
    glsl = (
        "void init_pts(out vec3 pts[4]) {\n"
        "    pts[0] = vec3(1.0);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec3 pts[4];\n"
        "    init_pts(pts);\n"
        "    fragColor = vec4(pts[0], 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "void init_pts(float3 pts[4])" in header
    assert "pts[0] = (float3)(1.0f);" in header  # body indexes directly, no deref
    assert "init_pts(pts);" in kernel            # call site passes array, no &


def test_in_array_parameter_keeps_name():
    """vec3 f(vec3 pts[4], int i) keeps the name and the array suffix."""
    glsl = (
        "vec3 pick(vec3 pts[4], int i) {\n"
        "    return pts[i];\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec3 pts[4];\n"
        "    fragColor = vec4(pick(pts, 0), 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float3 pick(float3 pts[4], int i)" in header
    assert "return pts[i];" in header


# ============================================================================
# 5. Struct fields declared as arrays (K5) — must not raise
# ============================================================================

def test_struct_with_array_fields():
    """struct T { vec4 data[4]; int n; }; transforms without error (Md2fzV)."""
    glsl = (
        "struct T { vec4 data[4]; int n; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    T t;\n"
        "    t.n = 2;\n"
        "    t.data[0] = vec4(1.0);\n"
        "    fragColor = t.data[0];\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float4 data[4];" in header
    assert "int n;" in header


def test_struct_array_field_of_struct_type():
    """Array fields of user-struct element type keep the struct type name."""
    glsl = (
        "struct Page { int start; int len; };\n"
        "struct Book { Page pages[3]; int count; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    Book b;\n"
        "    b.count = 1;\n"
        "    b.pages[0] = Page(0, 4);\n"
        "    fragColor = vec4(float(b.pages[0].len));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "Page pages[3];" in header
    assert "b.pages[0] = ((Page){0, 4});" in kernel


# ============================================================================
# 6. Out-param call sites with array-element arguments (K-adjacent)
# ============================================================================

def test_array_element_passed_to_out_param_gets_address():
    """foo(arr[0]) where the param is out -> foo(&arr[0]) (MdVfWw/ldKBRt)."""
    glsl = (
        "void bump(inout vec2 v) { v.x += 1.0; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 arr[2];\n"
        "    arr[0] = vec2(1.0, 2.0);\n"
        "    bump(arr[0]);\n"
        "    fragColor = vec4(arr[0], 0.0, 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "bump(&arr[0]);" in kernel
