"""
Unit tests for category S — function prototypes (forward declarations).

GLSL permits C-style function prototypes before mainImage:

    float PrSphDf (vec3 p, float r);
    ...
    float PrSphDf (vec3 p, float r) { return length (p) - r; }

tree-sitter parses the prototype as a `declaration` whose only named child
(besides the type) is a `function_declarator` — not an identifier /
init_declarator / array_declarator — so `_transform_declaration` raised
"Invalid declaration structure: no declarators found" (every category-S
failure in the corpus is this one shape; probed 2026-07-04, 258/258 hits).

Fix: route `declaration` nodes carrying a `function_declarator` to a prototype
transform that mirrors `_transform_function_definition` (same parameter
transformation, same `__attribute__((overloadable))` marking — the attribute
must agree between prototype and definition or OpenCL rejects the pair) and
emit `signature;` with no body. Prototypes also pre-register the function's
return type and out-param signature so calls that appear before the definition
(the whole point of a prototype) get correct type inference and `&` insertion.
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


MAIN_STUB = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
    "    fragColor = vec4(Fn(fragCoord.x), 0.0, 0.0, 1.0);\n"
    "}\n"
)


# ============================================================================
# 1. Basic prototype forms transpile instead of crashing
# ============================================================================

def test_scalar_prototype():
    """float Fn (float x); — the corpus shape (note space before parens)."""
    glsl = (
        "float Fn (float x);\n"
        "float Fn (float x) { return x + 1.0; }\n" + MAIN_STUB
    )
    header, kernel = tp(glsl)
    assert "float Fn(float x);" in header
    assert "float Fn(float x) {" in header


def test_vector_return_prototype():
    """vec2 Rot2D (vec2 q, float a); — type_identifier return type."""
    glsl = (
        "vec2 Rot2D (vec2 q, float a);\n"
        "vec2 Rot2D (vec2 q, float a) { return q * cos(a); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(Rot2D(fragCoord, 1.0), 0.0, 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float2 Rot2D(float2 q, float a);" in header


def test_void_empty_params_prototype():
    """void HexVorInit (); — empty parameter list."""
    glsl = (
        "float gv;\n"
        "void HexVorInit ();\n"
        "void HexVorInit () { gv = 1.0; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    HexVorInit();\n"
        "    fragColor = vec4(gv);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "void HexVorInit();" in header


def test_prototype_is_overloadable():
    """Prototype must carry the same overloadable attribute as the definition
    (OpenCL rejects an overloadable definition whose prior declaration is not)."""
    glsl = (
        "float Fn (float x);\n"
        "float Fn (float x) { return x + 1.0; }\n" + MAIN_STUB
    )
    header, kernel = tp(glsl)
    proto_idx = header.index("float Fn(float x);")
    attr_idx = header.rindex("__attribute__((overloadable))", 0, proto_idx)
    # the attribute directly precedes the prototype (same line-block)
    assert header[attr_idx:proto_idx].strip() == "__attribute__((overloadable))"


def test_overloaded_prototypes_coexist():
    """Same name, different signatures — the 4lSyRm 'cairo' library style."""
    glsl = (
        "void circle(vec2 p, float r);\n"
        "void circle(float x, float y, float r);\n"
        "void circle(vec2 p, float r) { }\n"
        "void circle(float x, float y, float r) { circle(vec2(x, y), r); }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    circle(fragCoord, 1.0);\n"
        "    fragColor = vec4(1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "void circle(float2 p, float r);" in header
    assert "void circle(float x, float y, float r);" in header


# ============================================================================
# 2. Qualifiers: in/out/inout/const transform exactly like definitions
# ============================================================================

def test_prototype_out_param_becomes_pointer():
    glsl = (
        "void Split (vec2 p, out float a, out float b);\n"
        "void Split (vec2 p, out float a, out float b) { a = p.x; b = p.y; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    float a, b;\n"
        "    Split(fragCoord, a, b);\n"
        "    fragColor = vec4(a, b, 0.0, 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "void Split(float2 p, float* a, float* b);" in header


def test_prototype_const_in_params():
    """const in float n — the 4dVGzw calcSoftshadow shape."""
    glsl = (
        "float calcSoftshadow( in vec3 ro, in vec3 rd );\n"
        "float calcSoftshadow( in vec3 ro, in vec3 rd ) { return ro.x + rd.x; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(calcSoftshadow(vec3(fragCoord, 0.0), vec3(1.0)));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float calcSoftshadow(float3 ro, float3 rd);" in header


def test_prototype_unnamed_params_keep_arity():
    """float Fn(vec3); — legal GLSL, the param list must not silently shrink."""
    glsl = (
        "float Fn(vec3);\n"
        "float Fn(vec3 p) { return p.x; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(Fn(vec3(fragCoord, 0.0)));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float Fn(float3);" in header


# ============================================================================
# 3. Prototype pre-registers signature: calls BEFORE the definition work
# ============================================================================

def test_call_before_definition_gets_address_of():
    """A function defined AFTER its caller (the reason prototypes exist):
    the caller's call site must still pass &x for the out param."""
    glsl = (
        "void GetVal (float x, out float v);\n"
        "float Wrap (float x) {\n"
        "    float v;\n"
        "    GetVal (x, v);\n"
        "    return v;\n"
        "}\n"
        "void GetVal (float x, out float v) { v = x * 2.0; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(Wrap(fragCoord.x));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "GetVal(x, &v);" in header


def test_call_before_definition_return_type_inferred():
    """vec2-returning forward-declared fn: caller swizzles the result, which
    needs the return type registered at prototype time."""
    glsl = (
        "vec2 HexToPix (vec2 h);\n"
        "float UseIt (vec2 h) { return HexToPix (h).x; }\n"
        "vec2 HexToPix (vec2 h) { return h * 2.0; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(UseIt(fragCoord));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float2 HexToPix(float2 h);" in header


# ============================================================================
# 4. Code AFTER mainImage — the reason the prototypes exist
# ============================================================================
# The dr2-style corpus shaders declare prototypes at the top, call them from
# mainImage, and put the DEFINITIONS after mainImage at the bottom of the file.
# extract_main_image_sections used to discard everything after mainImage, so
# the prototypes resolved to nothing (ptxas: unresolved extern function).

def test_helper_defined_after_mainimage_is_kept():
    glsl = (
        "float Fn (float x);\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(Fn(fragCoord.x), 0.0, 0.0, 1.0);\n"
        "}\n"
        "float Fn (float x)\n"
        "{\n"
        "    return x + 1.0;\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "float Fn(float x);" in header       # prototype
    assert "float Fn(float x) {" in header      # definition, no longer dropped


def test_global_after_mainimage_is_kept():
    glsl = (
        "float Fn (float x);\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(Fn(fragCoord.x), 0.0, 0.0, 1.0);\n"
        "}\n"
        "const float gK = 3.0;\n"
        "float Fn (float x) { return x + gK; }\n"
    )
    header, kernel = tp(glsl)
    assert "gK = 3.0f" in header
    assert "float Fn(float x) {" in header


def test_mainvr_before_mainimage_is_kept():
    """mainVR defined BEFORE mainImage was always included in the header, and
    some shaders (XlBGzm, lsVBDh, XscXzn) CALL mainVR from mainImage — it must
    stay included or they hit `ptxas: Unresolved extern function 'mainVR'`."""
    glsl = (
        "void mainVR(out vec4 fragColor, in vec2 fragCoord,"
        " in vec3 fragRayOri, in vec3 fragRayDir)\n"
        "{\n"
        "    fragColor = vec4(fragRayOri + fragRayDir, 1.0);\n"
        "}\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    mainVR(fragColor, fragCoord, vec3(0.0), vec3(0.0, 0.0, -1.0));\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "void mainVR(" in header


def test_mainvr_after_mainimage_stays_dropped():
    """Alternate entry points (mainVR/mainSound/mainCubemap) after mainImage
    were always dropped from the campaign build; keep that behavior so
    previously-passing VR shaders don't regress."""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(1.0);\n"
        "}\n"
        "void mainVR(out vec4 fragColor, in vec2 fragCoord,"
        " in vec3 fragRayOri, in vec3 fragRayDir)\n"
        "{\n"
        "    fragColor = vec4(fragRayOri + fragRayDir, 1.0);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    assert "mainVR" not in header


# ============================================================================
# 5. Prototypes must not disturb surrounding declarations
# ============================================================================

def test_prototype_between_globals():
    glsl = (
        "float gA = 1.0;\n"
        "float Fn (float x);\n"
        "float gB = 2.0;\n"
        "float Fn (float x) { return x + gA + gB; }\n" + MAIN_STUB
    )
    header, kernel = tp(glsl)
    assert "float Fn(float x);" in header
    assert "gA = 1.0f" in header
    assert "gB = 2.0f" in header
