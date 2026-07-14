"""
Category AG cluster 1 — confine user #define of a read-only Shadertoy uniform.

Shadertoy uniforms (iTime, iFrame, ...) are read-only, so shaders legally
remap *reads* of a uniform with an object-like macro:

    #define iTime mod(iTime, 20.0)           // loop-every-20s hack (XdyBW1)
    #define iFrame (int)(texelFetch(...).w)   // MtXBDf (4 buffer passes)

On Shadertoy the macro only ever rewrites reads; the driver assigns the real
uniform for us. In hShadertoy's split output the user macro lands in the
emitted header, while the `SHADERTOY_INPUTS` assignment block
(`iTime = AT_Time; ...`, injected by the HDA / tests/ocl/main_kernel.cl) sits
in the kernel prefix that follows the header. With the macro still active
there, the assignment LHS is rewritten to a non-lvalue
(`mod(iTime, 20.0) = AT_Time;`) and clang rejects it with
"expression is not assignable".

Fix (shared across both transpiler hosts so live Houdini benefits with no HDA
change): after all user code in the emitted header, append `#undef <U>` for
every OBJECT-LIKE redefine of a uniform. This confines the macro to exactly
the region it covered on Shadertoy — every header-level use (helper functions,
globals) already expanded textually before the `#undef`, while the
SHADERTOY_INPUTS assignments (which never existed in GLSL) bind the real
global.

Push-pop refinement (render correctness): the entry function's BODY is inlined
into the kernel section AFTER the header, so with only the trailing `#undef`
a body-level read (`iTime * CAMERA_SPEED` in XdyBW1's mainImage) would lose
the remap — compile-correct but render-wrong. Therefore each suppressed macro
is RE-EMITTED (its FINAL user definition, last-one-wins; nothing if the user
themselves left it `#undef`'d) at the very top of the kernel glue: after
SHADERTOY_INPUTS has already expanded, before the inlined entry body. Nothing
downstream of the entry body assigns a uniform (campaign: compilecl appends
only `AT_fragColor_set(fragColor);}`; Houdini: `@fragColor.set(fragColor);`),
so the re-defined macro cannot poison anything.

Root-cause alternative (header-side): a corpus-proven redesign moves every
uniform assignment out of SHADERTOY_INPUTS into a `shadertoy_bind_inputs()`
setter defined before user code, making this whole module a no-op safety net.
Full implementation + adoption steps: bottom of
`houdini/ocl/include/shadertoyInputs.h` (branch `fix/header-ag1-setter-main`).

Precision:
  * FUNCTION-LIKE redefines (`#define iFrame(x) ...`) are exempt — they only
    expand when followed by `(`, so `iFrame = AT_iFrame;` is never poisoned.
  * BARE-IDENTIFIER bodies (`#define iTime myGlobal`) are exempt — such a
    shader compiles today (the assignment poisons to `myGlobal = AT_Time;`,
    which is assignable) and renders through `myGlobal`; a trailing `#undef`
    would silently rebind the assignment to the real `iTime` and leave
    `myGlobal` uninitialised. Gating on non-bare bodies keeps this a strict
    compile-fix with no runtime-semantics change to already-passing shaders.
  * A `#define` guarded by `#if`/`#ifdef` is fine for the trailing `#undef`
    (unconditional undef of a maybe-undefined macro is a legal no-op), but the
    kernel-side RE-EMIT is unconditional too — a conditionally-defined uniform
    remap would be re-applied regardless of the condition. Residual edge; no
    corpus shader hits it.
"""

import re
from typing import Dict, List, Tuple

# The read-only Shadertoy uniforms (input uniform set). A user #define of any
# of these names is a read-remap hack, not an ordinary macro.
SHADERTOY_UNIFORMS = frozenset({
    "iResolution", "iTime", "iTimeDelta", "iFrameRate", "iFrame", "iMouse",
    "iDate", "iSampleRate",
    "iChannel0", "iChannel1", "iChannel2", "iChannel3",
    "iChannelTime", "iChannelResolution",
})

# `#define NAME<rest>` — NAME captured, the remainder (params/body/comment)
# left in group 2 so we can distinguish object-like from function-like and
# inspect the body.
_DEFINE_RE = re.compile(
    r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_][A-Za-z0-9_]*)(.*)$'
)

_BARE_IDENT_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')

_UNDEF_RE = re.compile(r'^[ \t]*#[ \t]*undef[ \t]+([A-Za-z_][A-Za-z0-9_]*)')


def _strip_line_comment(text: str) -> str:
    pos = text.find('//')
    if pos != -1:
        text = text[:pos]
    # crude single-line /* ... */ strip (define bodies rarely nest these)
    text = re.sub(r'/\*.*?\*/', ' ', text)
    return text


def scan_uniform_redefines(source: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Scan for OBJECT-LIKE, non-bare-identifier #defines of Shadertoy uniforms.

    ``source`` should be the PREPROCESSED GLSL source (post
    PreprocessorTransformer): its #define bodies are already transformed
    (GLSL_mod, f-suffixes, ...) so the captured line is valid OpenCL, and —
    unlike the emitted header, where the emitter DROPS #undef directives —
    user `#undef` lines are still visible, so end-of-user-code state is exact.

    Returns:
        (names, final_defs)
        names       — ordered, de-duplicated list of every uniform name that
                      was EVER redefined this way (drives the header #undef
                      block; a trailing #undef is a no-op even if the user
                      later #undef'd the macro themselves).
        final_defs  — insertion-ordered dict name -> the exact `#define` line
                      that is IN EFFECT at end-of-user-code (last definition
                      wins; a name the user left #undef'd is absent). Drives
                      the kernel-side re-emit block.

    Function-like redefines, bare-identifier bodies, empty bodies and
    non-uniform names are all excluded.
    """
    names: List[str] = []
    seen = set()
    final_defs: Dict[str, str] = {}
    for line in source.splitlines():
        um = _UNDEF_RE.match(line)
        if um:
            final_defs.pop(um.group(1), None)
            continue
        m = _DEFINE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name not in SHADERTOY_UNIFORMS:
            continue
        rest = m.group(2)
        # Function-like: '(' immediately follows the name (no separating
        # whitespace). `#define iFrame(x) ...` -> exempt.
        if rest[:1] == '(':
            continue
        body = _strip_line_comment(rest).strip()
        if not body:
            # Degenerate `#define iTime` (empty body) — not a read-remap; leave
            # it alone (it was already broken/no-op and injecting #undef would
            # not make the empty expansion assignable anyway).
            continue
        # Bare-identifier body gate (see module docstring).
        if _BARE_IDENT_RE.match(body):
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
        final_defs[name] = line.strip()
    return names, final_defs


def find_redefined_uniforms(source: str) -> List[str]:
    """Ordered list of uniform names ever object-like-redefined in ``source``."""
    return scan_uniform_redefines(source)[0]


def collect_uniform_redefines(processed: str, raw: str = None) -> Dict[str, str]:
    """
    name -> `#define` line in effect at end-of-user-code (last-one-wins;
    absent when the user left the macro #undef'd). Feed the result to
    uniform_redefine_prefix() for the kernel-side re-emit block.

    ``processed`` is the PreprocessorTransformer output: its #define bodies
    are already valid OpenCL (GLSL_mod, f-suffixes, ...) so its lines are what
    get re-emitted. But that pass DROPS user `#undef` lines, so liveness at
    end-of-user-code is decided by scanning ``raw`` (the original GLSL, where
    the #undefs are still visible) and filtering to names still defined there.
    """
    final_defs = scan_uniform_redefines(processed)[1]
    if raw is not None:
        live = set(scan_uniform_redefines(raw)[1])
        final_defs = {k: v for k, v in final_defs.items() if k in live}
    return final_defs


def uniform_redefine_prefix(final_defs: Dict[str, str]) -> str:
    """
    The kernel-glue prefix that re-applies the user's uniform remaps for the
    inlined entry body (push-pop counterpart of append_uniform_undefs).
    Empty string when there is nothing to re-emit.
    """
    if not final_defs:
        return ""
    return (
        "// ---- category AG: re-apply user uniform remaps for the entry body ----\n"
        + "\n".join(final_defs.values())
        + "\n"
    )


def append_uniform_undefs(header: str) -> str:
    """
    Append a `#undef <U>` for every object-like uniform redefine detected in
    ``header``, at the very end (after all user code). Returns ``header``
    unchanged when there is nothing to confine.
    """
    names = find_redefined_uniforms(header)
    if not names:
        return header
    block = (
        "// ---- category AG: confine user #define of read-only uniforms ----\n"
        + "\n".join(f"#undef {n}" for n in names)
    )
    sep = "" if header.endswith("\n") else "\n"
    return f"{header}{sep}{block}\n"
