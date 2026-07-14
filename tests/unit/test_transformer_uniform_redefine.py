"""
Unit tests for category AG cluster 1 — user #define of a read-only Shadertoy
uniform.

Shadertoy uniforms (iTime, iFrame, ...) are read-only, so shaders legally
remap reads with an object-like macro, e.g.

    #define iTime mod(iTime, 20.0)          // loop-every-20s hack (XdyBW1)
    #define iFrame (int)(texelFetch(...).w)  // MtXBDf

In our split output the user macro lands in the emitted header while the
`SHADERTOY_INPUTS` assignment block (`iTime = AT_Time; ...`) lives in the
kernel prefix. With the macro still active there the LHS is rewritten to a
non-lvalue (`mod(iTime, 20.0) = AT_Time;`) -> clang "expression is not
assignable".

Fix: after all user code in the header, emit `#undef <U>` for every
OBJECT-LIKE redefine of a uniform. That confines the macro to the region it
covered on Shadertoy (header helper functions still see the remap) while the
SHADERTOY_INPUTS assignments (which never existed in GLSL) bind the real
global.

Gating: a BARE-IDENTIFIER body (`#define iTime myGlobal`) currently compiles
(the assignment poisons to `myGlobal = AT_Time;`, which is assignable) and the
shader renders through `myGlobal`. Injecting `#undef` there would silently
change runtime semantics, so bare-identifier bodies are exempt. Function-like
redefines are exempt too (they only expand before a `(`).
"""

import sys
from pathlib import Path

import pytest

# tests/transpile.py is the campaign transpiler (header/kernel split path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def header_of(glsl):
    return transpile(glsl).get_header()


def header_kernel_of(glsl):
    r = transpile(glsl)
    return r.get_header(), r.get_kernel()


MAIN = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
    "    fragColor = vec4(0.0);\n"
    "}\n"
)


def test_object_like_uniform_redefine_appends_undef():
    """XdyBW1-style loop hack: header keeps the #define AND ends with #undef."""
    glsl = "#define iTime mod(iTime, 20.0)\n" + MAIN
    header = header_of(glsl)
    # The user's #define is still present (header helpers get the remap)...
    assert "#define iTime" in header
    # ...and a scoping #undef is emitted after all user code.
    assert "#undef iTime" in header
    # The #undef must come AFTER the #define (confinement, not removal).
    assert header.index("#undef iTime") > header.index("#define iTime")


def test_iframe_cast_body_appends_undef():
    """MtXBDf-style: object-like body starting with a cast is still remapped."""
    glsl = "#define iFrame (int)(iFrame + 1)\n" + MAIN
    header = header_of(glsl)
    assert "#undef iFrame" in header


def test_function_like_redefine_is_exempt():
    """`#define iFrame(x)` is function-like — never poisons `iFrame = ...`."""
    glsl = "#define iFrame(x) ((x) + 1)\n" + MAIN
    header = header_of(glsl)
    assert "#undef iFrame" not in header


def test_non_uniform_define_is_exempt():
    """Ordinary macros (`#define PI 3.14`) are left completely alone."""
    glsl = "#define PI 3.14159\n" + MAIN
    header = header_of(glsl)
    assert "#undef PI" not in header


def test_bare_identifier_body_is_gated_out():
    """`#define iTime myGlobal` compiles today; #undef would break it -> skip."""
    glsl = "float myGlobal;\n#define iTime myGlobal\n" + MAIN
    header = header_of(glsl)
    assert "#undef iTime" not in header


def test_duplicate_user_undef_still_emits_ours():
    """A user #undef is legal; a trailing duplicate #undef is a no-op in C.

    Pin the chosen behavior: we still emit our confinement #undef regardless of
    any user #undef (undef of an undefined macro is legal and harmless).
    """
    glsl = "#define iTime mod(iTime, 20.0)\n#undef iTime\n" + MAIN
    header = header_of(glsl)
    assert "#undef iTime" in header


def test_multiple_uniform_redefines():
    glsl = (
        "#define iTime mod(iTime, 20.0)\n"
        "#define iFrame (int)(iFrame + 1)\n" + MAIN
    )
    header = header_of(glsl)
    assert "#undef iTime" in header
    assert "#undef iFrame" in header


# ============================================================================
# Push-pop refinement: the suppressed macro is RE-EMITTED at the top of the
# kernel glue (after SHADERTOY_INPUTS has expanded, before the inlined
# mainImage body) so body-level reads keep the user's remap.
# ============================================================================

def test_kernel_starts_with_reemitted_define():
    """XdyBW1's `iTime * K` inside mainImage must still see the remap."""
    glsl = (
        "#define iTime mod(iTime, 20.0)\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(iTime * 0.5);\n"
        "}\n"
    )
    header, kernel = header_kernel_of(glsl)
    assert "#undef iTime" in header
    # The re-emitted define must be present in the kernel glue...
    assert "#define iTime GLSL_mod(iTime, 20.0f)" in kernel
    # ...and precede the inlined body (the first use of iTime).
    assert kernel.index("#define iTime") < kernel.index("iTime * 0.5f")


def test_kernel_redefine_last_definition_wins():
    """Multiple user redefines: re-emit only the FINAL one."""
    glsl = (
        "#define iTime mod(iTime, 10.0)\n"
        "#undef iTime\n"
        "#define iTime mod(iTime, 20.0)\n" + MAIN
    )
    _, kernel = header_kernel_of(glsl)
    assert "#define iTime GLSL_mod(iTime, 20.0f)" in kernel
    assert "GLSL_mod(iTime, 10.0f)" not in kernel


def test_kernel_no_redefine_when_user_left_it_undefined():
    """User #define then #undef with no re-define: nothing to re-emit."""
    glsl = "#define iTime mod(iTime, 20.0)\n#undef iTime\n" + MAIN
    header, kernel = header_kernel_of(glsl)
    # Header confinement #undef still emitted (harmless no-op)...
    assert "#undef iTime" in header
    # ...but the kernel re-emits nothing (macro was dead at end of user code).
    assert "#define iTime" not in kernel


def test_kernel_no_redefine_for_function_like():
    glsl = "#define iFrame(x) ((x) + 1)\n" + MAIN
    _, kernel = header_kernel_of(glsl)
    assert "#define iFrame" not in kernel


def test_kernel_no_redefine_for_bare_identifier():
    glsl = "float myGlobal;\n#define iTime myGlobal\n" + MAIN
    _, kernel = header_kernel_of(glsl)
    assert "#define iTime" not in kernel


def test_kernel_no_redefine_for_non_uniform():
    glsl = "#define PI 3.14159\n" + MAIN
    _, kernel = header_kernel_of(glsl)
    assert "#define PI" not in kernel
