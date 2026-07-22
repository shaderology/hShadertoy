"""
Unit tests for Houdini transpiler module
Tests the transpile_glsl.py module functionality
"""
import sys
sys.path.insert(0, 'C:/dev/hShadertoy/houdini/scripts/python')
sys.path.insert(0, 'C:/dev/hShadertoy')

import pytest
from hshadertoy.transpiler.transpile_glsl import (
    transpile,
    TranspilationError,
    _detect_renderpass_type,
    _format_for_houdini
)


class TestRenderpassDetection:
    """Test renderpass type detection"""

    def test_detect_mainImage(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { }"
        assert _detect_renderpass_type(glsl) == "mainImage"

    def test_detect_mainCubemap(self):
        glsl = "void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) { }"
        assert _detect_renderpass_type(glsl) == "mainCubemap"

    def test_detect_mainSound(self):
        glsl = "vec2 mainSound(in int sampleRate, in float time) { return vec2(0.0); }"
        assert _detect_renderpass_type(glsl) == "mainSound"

    def test_detect_common(self):
        glsl = "#define PI 3.14\nfloat helper() { return 1.0; }"
        assert _detect_renderpass_type(glsl) == "Common"

    def test_detect_common_empty(self):
        glsl = ""
        assert _detect_renderpass_type(glsl) == "Common"


class TestHeaderBodySplit:
    """Test the IR-level header/body split (single-TU entry-point model:
    no brace-counting over emitted text, no '*fragColor' surgery — the
    transformer never pointerizes the entry's params)."""

    def test_split_simple_shader(self):
        glsl = (
            "#define PI 3.14\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl)
        header, kernel = result.split("@KERNEL", 1)
        assert "#define PI 3.14f" in header
        assert "void mainImage" not in header
        assert "fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);" in kernel
        assert "*fragColor" not in kernel

    def test_split_with_functions(self):
        glsl = (
            "vec4 helper(float x) { return vec4(x, x, 0.0, 1.0); }\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = helper(0.5);\n"
            "}\n"
        )
        result = transpile(glsl)
        header, kernel = result.split("@KERNEL", 1)
        assert "float4 helper" in header
        assert "fragColor = helper(0.5f);" in kernel

    def test_split_keeps_code_after_mainimage(self):
        """Category S: post-mainImage definitions stay in the header."""
        glsl = (
            "float Fn(float x);\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(Fn(1.0));\n"
            "}\n"
            "float Fn(float x) { return x * 2.0; }\n"
        )
        result = transpile(glsl)
        header, kernel = result.split("@KERNEL", 1)
        assert "float Fn(float x);" in header
        assert "return x * 2.0f;" in header
        assert "Fn(1.0f)" in kernel

    def test_split_common_renderpass(self):
        glsl = "#define PI 3.14\nfloat helper() { return 1.0; }\n"
        result = transpile(glsl, mode="Common")
        assert "@KERNEL" not in result
        assert "#define PI 3.14f" in result
        assert "float helper()" in result


class TestHoistInjection:
    """Category A globals must be assigned at the top of the @KERNEL body
    (previously LOST in this host — TRANSPILER_REVIEW §0.2)."""

    def test_nonconst_global_hoisted_into_kernel(self):
        glsl = (
            "vec3 L = normalize(vec3(1.0, 0.9, 0.3));\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(L, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl)
        header, kernel = result.split("@KERNEL", 1)
        assert "float3 L;" in header
        assert "L = GLSL_normalize(" in kernel
        # hoist precedes the body statements
        assert kernel.index("L = GLSL_normalize(") < kernel.index("fragColor =")


class TestCustomParamNames:
    """Golf-style `out vec4 O, vec2 U` signatures are bridged via aliases."""

    def test_custom_names_aliased(self):
        glsl = (
            "void mainImage(out vec4 O, vec2 U) {\n"
            "    O = vec4(U, 0.0, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl)
        _, kernel = result.split("@KERNEL", 1)
        assert "float2 U = fragCoord;" in kernel
        assert "fragColor = O;" in kernel
        assert "O = (float4)(U, 0.0f, 1.0f);" in kernel


class TestEntryNormalization:
    """Unconventional entries are normalized before the pipeline."""

    def test_bare_main_gl_fragcolor(self):
        glsl = (
            "void main(void) {\n"
            "    gl_FragColor = vec4(gl_FragCoord.xy, 0.0, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl)
        assert "@KERNEL" in result
        assert "@fragColor.set(fragColor);" in result


class TestHoudiniFormatting:
    """Test Houdini output formatting"""

    def test_format_simple_shader(self):
        # The body arrives deref-free: the transformer never pointerizes the
        # entry's params (no '*fragColor' surgery exists anymore).
        header = "#define PI 3.14f"
        body = "fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);"
        result = _format_for_houdini(header, body, "mainImage")

        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result
        assert "@fragColor.set(fragColor);" in result
        assert "#define PI 3.14f" in result
        assert "fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);" in result

    def test_format_common_renderpass(self):
        header = "#define PI 3.14f\nfloat helper() { return 1.0f; }"
        body = ""
        result = _format_for_houdini(header, body, "Common")

        assert "@KERNEL" not in result
        assert result == header

    def test_format_with_empty_header(self):
        header = ""
        body = "fragColor = (float4)(1.0f, 0.0f, 0.0f, 1.0f);"
        result = _format_for_houdini(header, body, "mainImage")

        assert "@KERNEL" in result
        assert "// ---- HEADER:" not in result  # No header comment if empty


class TestTranspileFunction:
    """Test main transpile function"""

    def test_transpile_simple_mainImage(self):
        glsl = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv, 0.5, 1.0);
}
"""
        result = transpile(glsl)

        assert "@KERNEL" in result
        assert "SHADERTOY_INPUTS" in result
        assert "@fragColor.set(fragColor);" in result
        assert "float2" in result
        assert "float4" in result

    def test_transpile_with_helper_function(self):
        glsl = """
vec4 helper(float x) {
    return vec4(x, x, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = helper(0.5);
}
"""
        result = transpile(glsl)

        assert "float4 helper" in result
        assert "@KERNEL" in result

    def test_transpile_with_define(self):
        glsl = """
#define PI 3.14

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(PI, 0.0, 0.0, 1.0);
}
"""
        result = transpile(glsl)

        assert "#define PI 3.14f" in result
        assert "@KERNEL" in result

    def test_transpile_common_renderpass(self):
        glsl = """
#define PI 3.14
float helper() { return 1.0; }
"""
        result = transpile(glsl)

        assert "@KERNEL" not in result
        assert "#define PI 3.14f" in result

    def test_transpile_auto_detect_mode(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }"
        result = transpile(glsl, mode=None)  # Auto-detect

        assert "@KERNEL" in result

    def test_transpile_explicit_mode(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }"
        result = transpile(glsl, mode="mainImage")  # Explicit

        assert "@KERNEL" in result

    def test_transpile_invalid_syntax_raises_error(self):
        glsl = "this is not valid GLSL code @#$%"
        with pytest.raises(TranspilationError):
            transpile(glsl)


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_empty_body(self):
        glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { }"
        result = transpile(glsl)

        assert "@KERNEL" in result
        assert "@fragColor.set(fragColor);" in result

    def test_multiline_function_signature(self):
        glsl = """
void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord
) {
    fragColor = vec4(1.0);
}
"""
        result = transpile(glsl)

        assert "@KERNEL" in result

    def test_nested_braces(self):
        glsl = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    if (fragCoord.x > 0.5) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        fragColor = vec4(0.0, 1.0, 0.0, 1.0);
    }
}
"""
        result = transpile(glsl)

        assert "@KERNEL" in result
        assert "if" in result


class TestMatrixMacroParity:
    """The Houdini host must mirror the campaign host (tests/transpile.py) in
    seeding matrix-returning #define macros (category J) as user-function
    return types, so `v *= ROT(...)` — where ROT is `#define ROT(a) mat2(...)`
    — lowers to the GLSL_mul matmul helper instead of emitting a raw
    `float2 *= matrix2x2` (which OpenCL rejects). Regression: shadertoy
    mslfR2 "cubes" (tests/shaders/complex/cubes.glsl)."""

    def test_vec_compound_assign_matrix_macro_lowers(self):
        glsl = (
            "#define ROT(a) mat2(cos(a), sin(a), -sin(a), cos(a))\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    vec2 mp = fragCoord;\n"
            "    mp *= ROT(0.5);\n"
            "    fragColor = vec4(mp, 0.0, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl)
        assert "GLSL_mul" in result
        # The raw compound-assign against a matrix must NOT survive.
        assert "mp *= ROT" not in result

    def test_swizzle_compound_assign_matrix_macro_lowers(self):
        glsl = (
            "#define ROT(a) mat2(cos(a), sin(a), -sin(a), cos(a))\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    vec3 grd = vec3(fragCoord, 1.0);\n"
            "    grd.xy *= ROT(0.025);\n"
            "    fragColor = vec4(grd, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl)
        assert "GLSL_mul" in result
        assert "grd.xy *= ROT" not in result


class TestEntryTrappedInConditional:
    """Host A/B parity (S59): when the ONLY mainImage definition is trapped in
    a program-scope #ifdef/#ifndef (tree-sitter keeps it as one opaque raw
    blob so partition finds no top-level entry), the host must evaluate the
    constant conditional and rebuild the IR — otherwise the Houdini @KERNEL
    body comes out EMPTY and the shader renders nothing. Host A
    (tests/transpile.py) already did this; Host B must mirror it."""

    def test_mainimage_in_ifdef_produces_kernel_body(self):
        glsl = (
            "#define SIMPLE_VERSION\n"
            "#ifdef SIMPLE_VERSION\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n"
            "#else\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(0.0);\n"
            "}\n"
            "#endif\n"
        )
        result = transpile(glsl, mode="mainImage")
        # The @KERNEL body must contain the selected branch's assignment, not
        # be an empty shell.
        kernel_region = result[result.find("@KERNEL"):]
        assert "1.0f, 0.0f, 0.0f, 1.0f" in kernel_region

    def test_mainimage_in_ifndef_produces_kernel_body(self):
        glsl = (
            "#ifndef CFG_OFF\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(0.5, 0.25, 0.125, 1.0);\n"
            "}\n"
            "#endif\n"
        )
        result = transpile(glsl, mode="mainImage")
        kernel_region = result[result.find("@KERNEL"):]
        assert "0.5f, 0.25f, 0.125f, 1.0f" in kernel_region


class TestCommonCrossTabSignatures:
    """Houdini transpiles the Common tab as a SEPARATE node (code_common)
    injected before @KERNEL, so a renderpass that CALLS a Common-defined helper
    is transpiled without seeing that helper's definition. When the helper has
    an out/inout parameter, its call site must still take the argument's address
    (`&x`) or the emitted `float3* p` parameter is passed a bare `float3` and
    clang rejects the kernel. Host A (tests/transpile.py --common) merges Common
    into the pass TU and gets this for free; Host B restores parity by seeding
    the pass transformer with Common's harvested signatures via `common=`.

    Regression: shadertoy tsKXR3 "Multiscale MIP Fluid" — the Image pass calls
    `light(uv, BUMP, 0.5, dxy, iTime, avd)` where `light`'s 6th param is
    `inout vec3 avd`, defined in the Common tab.
    """

    def test_inout_arg_to_common_helper_gets_address_of(self):
        common = (
            "vec3 light(vec2 uv, float b, inout vec3 avd) {\n"
            "    avd = vec3(b);\n"
            "    return uv.xyx;\n"
            "}\n"
        )
        image = (
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    vec2 uv = fragCoord / iResolution.xy;\n"
            "    vec3 avd;\n"
            "    vec3 ld = light(uv, 3.0, avd);\n"
            "    fragColor = vec4(ld, 1.0);\n"
            "}\n"
        )
        result = transpile(image, mode="mainImage", common=common)
        kernel_region = result[result.find("@KERNEL"):]
        assert "light(uv, 3.0f, &avd)" in kernel_region, (
            "inout arg to a Common-defined helper must take its address; "
            f"got: {kernel_region!r}"
        )

    def test_by_value_arg_to_common_helper_unchanged(self):
        """A Common helper with no out/inout params must NOT get spurious &."""
        common = (
            "float scale(float x, float k) { return x * k; }\n"
        )
        image = (
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    float v = 2.0;\n"
            "    float r = scale(v, 3.0);\n"
            "    fragColor = vec4(r);\n"
            "}\n"
        )
        result = transpile(image, mode="mainImage", common=common)
        kernel_region = result[result.find("@KERNEL"):]
        assert "scale(v, 3.0f)" in kernel_region
        assert "&v" not in kernel_region

    def test_no_common_is_backward_compatible(self):
        """Omitting `common` keeps the previous single-arg behavior."""
        glsl = (
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        result = transpile(glsl, mode="mainImage")
        assert "@KERNEL" in result

    def test_common_signature_harvest_helper(self):
        """The harvest helper reports out/inout param info + return types."""
        from hshadertoy.transpiler.transpile_glsl import (
            _harvest_common_signatures,
        )
        common = (
            "vec3 light(vec2 uv, float b, inout vec3 avd) {\n"
            "    avd = vec3(b); return uv.xyx;\n"
            "}\n"
        )
        sigs, rets = _harvest_common_signatures(common)
        assert sigs["light"][3] == [
            ("uv", False), ("b", False), ("avd", True)
        ]
        assert rets["light"] == "vec3"

    def test_harvest_on_empty_common(self):
        from hshadertoy.transpiler.transpile_glsl import (
            _harvest_common_signatures,
        )
        assert _harvest_common_signatures("") == ({}, {})
        assert _harvest_common_signatures("   ") == ({}, {})


if __name__ == "__main__":
    # Allow running tests directly with: hython tests/unit/test_houdini_transpiler.py
    pytest.main([__file__, "-v"])
