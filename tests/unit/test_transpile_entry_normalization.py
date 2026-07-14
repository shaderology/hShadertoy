"""
Unit tests for entry-point normalization (entry-point redesign, S2).

Shadertoy itself never scans user code for mainImage: its GL header
forward-declares `void mainImage(out vec4 c, in vec2 f);`, `main()` calls
it, and the REAL GL preprocessor expands user macros. Two idioms in the
wild therefore bypass a literal `void mainImage(...)` definition:

(a) macro-entry (resources/examples/ProceduralNoiseCollection):

        #define gl_FragCoord fragCoord
        #define main() mainImage(out vec4 fragColor, vec2 fragCoord)
        void main() { ... fragColor = vec4(c, 1); }

(b) GLSL-Sandbox-style ports:

        void main(void) { gl_FragColor = vec4(uv, 0.0, 1.0); }

normalize_entry_point() rewrites both into conventional mainImage shaders
BEFORE the preprocessor/parse stages, so the rest of the pipeline never
needs to know. Shaders that already define mainImage are returned unchanged.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile, normalize_entry_point, TranspileError  # noqa: E402


# ============================================================================
# 1. Conventional shaders pass through untouched
# ============================================================================

def test_conventional_shader_unchanged():
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(0.0);\n"
        "}\n"
    )
    assert normalize_entry_point(glsl) == glsl


def test_no_entry_at_all_unchanged_and_raises():
    glsl = "float f(float x) { return x; }\n"
    assert normalize_entry_point(glsl) == glsl
    with pytest.raises(TranspileError):
        transpile(glsl)


# ============================================================================
# 2. Macro-entry idiom
# ============================================================================

MACRO_ENTRY = (
    "#define gl_FragCoord fragCoord\n"
    "#define u_resolution iResolution\n"
    "#define main() mainImage(out vec4 fragColor, vec2 fragCoord)\n"
    "void main() {\n"
    "    vec2 uv = gl_FragCoord.xy / u_resolution.xy;\n"
    "    fragColor = vec4(uv, 0.0, 1.0);\n"
    "}\n"
)


def test_macro_entry_expanded_to_mainimage():
    out = normalize_entry_point(MACRO_ENTRY)
    assert "void mainImage(out vec4 fragColor, vec2 fragCoord)" in out
    # the entry-related defines are consumed
    assert "#define main" not in out
    assert "#define gl_FragCoord" not in out
    # gl_FragCoord occurrences expanded to the parameter
    assert "fragCoord.xy" in out
    assert "gl_FragCoord" not in out
    # unrelated defines are left for the normal preprocessor path
    assert "#define u_resolution iResolution" in out


def test_macro_entry_transpiles():
    result = transpile(MACRO_ENTRY)
    kernel = result.get_kernel()
    # spliced-body kernel model: the normalized entry body lands in the kernel
    assert "fragCoord.xy" in kernel
    assert "fragColor = (float4)(uv, 0.0f, 1.0f);" in kernel


# ============================================================================
# 3. Bare void main() (GLSL-Sandbox style)
# ============================================================================

BARE_MAIN = (
    "float vign(vec2 p) { return 1.0 - dot(p, p) * gl_FragCoord.w; }\n"
    "void main(void) {\n"
    "    vec2 uv = gl_FragCoord.xy / iResolution.xy;\n"
    "    gl_FragColor = vec4(uv * vign(uv), 0.0, 1.0);\n"
    "}\n"
)


def test_bare_main_rewritten_to_mainimage():
    out = normalize_entry_point(BARE_MAIN)
    assert "void mainImage(out vec4 fragColor, in vec2 fragCoord)" in out
    assert "void main" not in out.replace("void mainImage", "")
    # gl_FragColor becomes the out param
    assert "fragColor = vec4(uv" in out
    assert "gl_FragColor" not in out
    # gl_FragCoord stays referenceable from helpers: global + entry-body init
    assert "vec4 gl_FragCoord;" in out
    assert "gl_FragCoord = vec4(fragCoord, 0.0, 1.0);" in out


def test_bare_main_transpiles():
    result = transpile(BARE_MAIN)
    header = result.get_header()
    kernel = result.get_kernel()
    # the entry body (spliced into the kernel) sets the gl_FragCoord global
    assert "gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f);" in kernel
    # the helper still sees gl_FragCoord (as a global declared in the header)
    assert "gl_FragCoord" in header


def test_bare_main_without_gl_fragcoord_gets_no_global():
    glsl = (
        "void main() {\n"
        "    gl_FragColor = vec4(1.0);\n"
        "}\n"
    )
    out = normalize_entry_point(glsl)
    assert "void mainImage(out vec4 fragColor, in vec2 fragCoord)" in out
    assert "gl_FragCoord" not in out
