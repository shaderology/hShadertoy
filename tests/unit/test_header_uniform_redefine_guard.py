"""Guard test for category AG cluster 1 — uniform #define collision.

Shadertoy shaders may ``#define`` a read-only uniform, e.g.::

    #define iTime GLSL_mod(iTime, 20.0f)

The transpiled user code carries that ``#define`` verbatim. It sits BETWEEN
``main_header.cl`` (which DEFINES the ``SHADERTOY_INPUTS`` / ``DO_CUBEMAP``
macros) and ``main_kernel.cl`` (which EXPANDS ``SHADERTOY_INPUTS``). So any
bare uniform name-token inside those macro bodies expands AFTER the user
``#define`` and gets rewritten — e.g. ``iTime = AT_Time;`` becomes
``GLSL_mod(iTime, 20.0f) = AT_Time;`` -> clang "expression is not assignable".

The fix moves every uniform assignment into header-defined setter functions
(``shadertoy_bind_inputs`` / ``shadertoy_cubemap_bind``) that are COMPILED
before the user ``#define`` can reach them. This test locks that invariant in:
NO bare Shadertoy uniform name-token may appear in the ``SHADERTOY_INPUTS`` or
``DO_CUBEMAP`` macro bodies.
"""

import re
from pathlib import Path

import pytest

MAIN_HEADER = Path(__file__).parent.parent / "ocl" / "main_header.cl"

# Read-only Shadertoy uniform globals that get poisoned by a user #define.
# fragCoord / fragColor are kernel-scope locals declared by SHADERTOY_INPUTS
# itself (the transpiled glue reads them) and are intentionally allowed.
UNIFORM_TOKENS = [
    "iResolution",
    "iTime",
    "iTimeDelta",
    "iFrameRate",
    "iFrame",
    "iMouse",
    "iDate",
    "iChannel0",
    "iChannel1",
    "iChannel2",
    "iChannel3",
    "iChannelTime",
    "iChannelResolution",
]


def _extract_macro_body(text: str, macro_name: str) -> str:
    """Return the full body of ``#define <macro_name> ...`` following
    backslash-newline line continuations. Excludes the ``#define <name>``
    token itself so we only inspect what the macro EXPANDS to."""
    lines = text.splitlines()
    body_lines = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("#define " + macro_name):
            # first line: drop the "#define <name>" prefix
            first = stripped[len("#define " + macro_name):]
            collecting = [first]
            cont = lines[i].rstrip().endswith("\\")
            while cont:
                i += 1
                collecting.append(lines[i])
                cont = lines[i].rstrip().endswith("\\")
            body_lines = collecting
            break
        i += 1
    if not body_lines:
        pytest.fail(f"macro {macro_name!r} not found in {MAIN_HEADER}")
    # strip trailing line-continuation backslashes
    return "\n".join(l.rstrip().rstrip("\\") for l in body_lines)


@pytest.fixture(scope="module")
def header_text() -> str:
    return MAIN_HEADER.read_text(encoding="utf-8", errors="replace")


def test_shadertoy_inputs_has_no_bare_uniform_token(header_text):
    body = _extract_macro_body(header_text, "SHADERTOY_INPUTS")
    offenders = [t for t in UNIFORM_TOKENS if re.search(rf"\b{t}\b", body)]
    assert not offenders, (
        "SHADERTOY_INPUTS macro body still contains bare uniform tokens that a "
        f"user #define can poison: {offenders}\nbody:\n{body}"
    )


def test_do_cubemap_has_no_bare_uniform_token(header_text):
    body = _extract_macro_body(header_text, "DO_CUBEMAP")
    offenders = [t for t in UNIFORM_TOKENS if re.search(rf"\b{t}\b", body)]
    assert not offenders, (
        "DO_CUBEMAP macro body still contains bare uniform tokens that a user "
        f"#define can poison: {offenders}\nbody:\n{body}"
    )


def test_setter_functions_defined_before_macros(header_text):
    """The setter functions must be DEFINED (so they compile before user
    #defines) and the macros must call them."""
    assert "shadertoy_bind_inputs" in header_text
    inputs_body = _extract_macro_body(header_text, "SHADERTOY_INPUTS")
    assert "shadertoy_bind_inputs(" in inputs_body, (
        "SHADERTOY_INPUTS must delegate to shadertoy_bind_inputs()"
    )
