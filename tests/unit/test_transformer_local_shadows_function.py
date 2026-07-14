"""
Unit tests for category AE — a local variable whose name shadows a same-named
user function (Session 40).

GLSL allows a local declaration whose name matches a user function and whose
initializer calls that function:

    float ao(vec3 p, ...) { ... }
    ...
    float ao = ao(p, n);   // legal GLSL: the call resolves to the function

OpenCL C resolves the bare name `ao` inside the body to the LOCAL (a value),
so the same `ao(...)` becomes a call on a non-callable:

    error: called object type 'float' is not a function or function pointer

Corpus casualties (sole-blockers): 4d3BDM (normal/shadow), 4sVcz3 (light),
ltcBzN (sampleShip), tdBXWw (ao).

Fix (transformer-only): when a NON-global declaration's name shadows a known
user function, rename the LOCAL variable (e.g. `ao` -> `ao_v`) at the
declaration and every later identifier read in the same scope. The function
call keeps the original name (call callees are not routed through the identifier
path), so it still resolves to the function. Registered AFTER the initializer is
transformed, so the call in the initializer itself is unaffected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _full(src: str) -> str:
    return transpile(src).full


def test_local_shadows_function_in_initializer():
    # The tdBXWw shape: `float ao = ao(...)` inside the body.
    src = """
    float ao(vec3 p, vec3 n) { return dot(p, n); }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 p = vec3(U, 1.0);
        float ao = ao(p, p);
        O = vec4(ao);
    }
    """
    full = _full(src)
    # The local is renamed at the declaration...
    assert "float ao_v = ao(" in full
    # ...and its later read is renamed too...
    assert "(float4)(ao_v)" in full or "ao_v)" in full
    # ...but the function call keeps the original name.
    assert "ao(p, p)" in full
    # No bare `float ao =` survives (that would be the shadowing local).
    assert "float ao =" not in full


def test_shadowed_call_keeps_function_name():
    # The 4sVcz3 shape: `float2 light = light(iTime);`
    src = """
    vec2 light(float t) { return vec2(cos(t), sin(t)); }
    void mainImage(out vec4 O, in vec2 U) {
        vec2 light = light(1.0);
        O = vec4(light, 0.0, 1.0);
    }
    """
    full = _full(src)
    assert "light_v = light(1.0f)" in full
    # later read renamed
    assert "light_v" in full.split("light_v = light(1.0f)")[1]


def test_non_shadowing_local_unchanged():
    # A local whose name is NOT a user function keeps its name.
    src = """
    float helper(float x) { return x * 2.0; }
    void mainImage(out vec4 O, in vec2 U) {
        float value = helper(U.x);
        O = vec4(value);
    }
    """
    full = _full(src)
    assert "float value = helper(U.x)" in full
    assert "value_v" not in full


def test_shadow_without_call_in_initializer_not_renamed():
    # The WsBGRW regression shape: a local shadows a function but its
    # initializer does NOT call that function (a legal, compiling shadow).
    # Renaming here would be unsafe — reads inside a textual #ifdef/#define
    # block bypass the AST rename and would be left dangling. Leave it alone.
    src = """
    vec3 color(vec2 uv) { return texture(iChannel0, uv).rgb; }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 color = texture(iChannel0, U).rgb;
        O = vec4(color, 1.0);
    }
    """
    full = _full(src)
    assert "float3 color = texture(iChannel0, U).rgb" in full
    assert "color_v" not in full


def test_shadow_local_used_multiple_times():
    # The local is read several times after the shadowing declaration.
    src = """
    float shadow(vec3 ro) { return ro.y; }
    void mainImage(out vec4 O, in vec2 U) {
        vec3 ro = vec3(U, 1.0);
        float shadow = shadow(ro);
        float lit = 1.0 - shadow;
        O = vec4(shadow * lit);
    }
    """
    full = _full(src)
    assert "float shadow_v = shadow(ro)" in full
    assert "1.0f - shadow_v" in full
    assert "shadow_v * lit" in full
    # the call resolves to the function, unchanged
    assert "shadow(ro)" in full
    assert "float shadow =" not in full
