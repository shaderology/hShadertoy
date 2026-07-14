"""
Unit tests for category AC — user identifier collides with a *predefined OpenCL
macro* (Session 43).

A GLSL shader may legally declare its own `M_PI`, `FLT_MAX`, `INFINITY`, ... as a
C-level constant or `#define` — none of those names are predefined by
GLSL/Shadertoy WebGL, so the shader is supplying a value it needs. But OpenCL C
*predefines* all of them (`<math.h>`/`<float.h>` macros in `cl_kernel.h`). At
compile the predefined macro expands the *declared name*:

    const float M_PI = 3.14159265359f;
    -> const float 3.14159265359f = 3.14159265359f;   // "expected identifier"

The fix is the same whole-source pre-parse rename used for reserved words
(`_rename_reserved_identifiers` in `_normalize_array_syntax`): a predefined-macro
name used as an identifier is suffixed with `_` (`M_PI` -> `M_PI_`) everywhere it
appears, so the declaration and every use stay consistent. Because these names
are never predefined in GLSL, any occurrence in GLSL source is necessarily a user
identifier, and any shader that uses one is already failing downstream — so the
rename can only fix, never regress. This is the direct sibling of category U.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.parser.glsl_parser import _normalize_array_syntax


@pytest.fixture
def parser():
    return GLSLParser()


# ---------------------------------------------------------------------------
# The rewrite itself.
# ---------------------------------------------------------------------------

def test_normalize_renames_m_pi_const_decl():
    # lsycWW: `const float M_PI = 3.14159265359;`
    assert _normalize_array_syntax("const float M_PI = 3.14159265359;") == \
        "const float M_PI_ = 3.14159265359;"


def test_normalize_renames_m_pi_plain_decl():
    # tsfGW4: `float M_PI = 3.1415972;`
    assert _normalize_array_syntax("float M_PI = 3.1415972;") == \
        "float M_PI_ = 3.1415972;"


def test_normalize_renames_flt_max_and_min():
    # MsVBzW / Wsf3D2: `const float FLT_MAX = 1e30;` / `const float FLT_MIN = 1e-30;`
    assert _normalize_array_syntax("const float FLT_MAX = 1e30;") == \
        "const float FLT_MAX_ = 1e30;"
    assert _normalize_array_syntax("const float FLT_MIN = 1e-30;") == \
        "const float FLT_MIN_ = 1e-30;"


def test_normalize_renames_decl_and_uses_consistently():
    src = "const float M_PI = 3.14159;\nfloat hue = theta / (2.0 * M_PI);"
    out = _normalize_array_syntax(src)
    assert "const float M_PI_ = 3.14159;" in out
    assert "(2.0 * M_PI_)" in out


def test_normalize_leaves_macro_substring_untouched():
    # `M_PI` inside a longer identifier must not be renamed.
    src = "float M_PICK = M_PIXELS + user_M_PI;"
    assert _normalize_array_syntax(src) == src


def test_normalize_ignores_macro_in_comment():
    src = "float x = 1.0; // M_PI is predefined in OpenCL\nfloat y = 2.0;"
    assert _normalize_array_syntax(src) == src


def test_normalize_preserves_line_numbers():
    src = "void f() {\n  const float M_PI = 3.14;\n  float k = 2.0 * M_PI;\n}"
    out = _normalize_array_syntax(src)
    assert out.count("\n") == src.count("\n")


# ---------------------------------------------------------------------------
# End-to-end: a shader declaring M_PI parses without the macro clobbering it.
# ---------------------------------------------------------------------------

def test_shader_declaring_m_pi_parses(parser):
    parser.parse(
        "const float M_PI = 3.14159265359;\n"
        "void mainImage(out vec4 c, vec2 f) {\n"
        "    float hue = f.x / (2.0 * M_PI);\n"
        "    c = vec4(hue);\n"
        "}"
    )
