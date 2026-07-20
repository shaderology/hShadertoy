"""
Category P — entry point trapped inside a program-scope preprocessor
conditional.

Some shaders guard their ONLY `mainImage` definition with an `#ifdef`/`#ifndef`
whose condition is statically known (the macro is `#define`d — or left
undefined — in the same translation unit). tree-sitter parses the whole
`#ifdef ... #endif` block as one opaque program-scope node, so the transformer
keeps it as a raw-text passthrough and the entry never becomes a top-level
declaration. partition_translation_unit() then reports
"Could not find mainImage()".

The fix evaluates the constant conditional (conditional_eval.strip_conditionals)
and re-runs the pipeline, but ONLY when the normal partition already failed —
so shaders that already expose a top-level mainImage are byte-for-byte
unchanged. Real corpus shapes: lljGDm (#ifdef SIMPLE_VERSION, defined) and
wssBz2 (#ifndef CFG_NO_POSTPROD, undefined).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(glsl, common=""):
    return transpile(glsl, common=common).get_kernel()


def test_ifdef_defined_entry_is_found():
    """#ifdef whose macro IS defined -> the taken branch's mainImage is used."""
    glsl = """
#define SIMPLE_VERSION

#ifdef SIMPLE_VERSION
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
#else
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = vec4(0.0, 1.0, 0.0, 1.0);
}
#endif
"""
    kernel = _kernel(glsl)
    # The taken (#ifdef) branch: red.
    assert "(float4)(1.0f, 0.0f, 0.0f, 1.0f)" in kernel
    # The dead #else branch must not leak in as a second entry body.
    assert "(float4)(0.0f, 1.0f, 0.0f, 1.0f)" not in kernel


def test_ifndef_undefined_entry_is_found():
    """#ifndef whose macro is NOT defined -> the #ifndef branch is taken."""
    glsl = """
#ifndef CFG_NO_POSTPROD
void mainImage(out vec4 fragColor, vec2 fragCoord)
{
    fragColor = vec4(0.25, 0.5, 0.75, 1.0);
}
#else
void mainImage(out vec4 fragColor, vec2 fragCoord)
{
    fragColor = vec4(0.0);
}
#endif
"""
    kernel = _kernel(glsl)
    assert "(float4)(0.25f, 0.5f, 0.75f, 1.0f)" in kernel


def test_commented_entry_does_not_count_as_top_level():
    """A block-commented mainImage before the real (guarded) one must not
    fool detection — the real entry is still inside the #ifdef."""
    glsl = """
/*void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = vec4(0.0);
}*/

#define SIMPLE_VERSION
#ifdef SIMPLE_VERSION
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = vec4(0.9, 0.1, 0.2, 1.0);
}
#endif
"""
    kernel = _kernel(glsl)
    assert "(float4)(0.9f, 0.1f, 0.2f, 1.0f)" in kernel
