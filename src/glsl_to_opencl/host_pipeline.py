"""
Shared GLSL→OpenCL host pipeline — the single source of truth for both hosts.

Two host wrappers used to each carry a full PRIVATE COPY of this pipeline:

  * Host A  ``tests/transpile.py``            — campaign / compilecl / unit tests
  * Host B  ``houdini/.../transpile_glsl.py`` — the real Houdini HDA deployment

Because every semantic step lived in two places, a fix that landed in one copy
but not the other silently diverged the two hosts. Every documented drift bug
came from exactly this: the S63b ``matrix_macros`` seed, the S59
entry-trapped-in-conditional rescue, category-A global-init hoisting (once lost
in Host B), and the tsKXR3 Common-tab inout signatures. A shader could pass the
campaign (Host A) yet crash the real Houdini cook (Host B).

This module ends that. It owns EVERY semantic step — normalize, merge/harvest
Common, preprocess, parse, transform (+ all seeds), partition entry vs header,
S59 rescue, header/body emit, ``#ifdef`` post-processing, category-A hoisting,
category-AG uniform push-pop. The two hosts keep ONLY the two things that
genuinely differ between them:

  1. **Output format** — Host A returns split ``header``/``kernel``/``full`` for
     compilecl; Host B wraps the body in Houdini's ``@KERNEL { ... }``.
  2. **Common-tab strategy** — Host A MERGES Common into the pass translation
     unit (compilecl compiles one file); Host B keeps Common as a separate
     ``code_common`` node and HARVESTS its signatures only (``merge_common``).

Everything else is shared here. ``transpile_pass`` returns a ``TranspiledPass``
of neutral pieces; each host formats them. The ``hoist_indent`` / ``bridge_indent``
knobs exist solely so each host reproduces its historical whitespace verbatim
(byte-for-byte identical output — see ``tests/unit/test_host_parity.py``).
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

from .parser import GLSLParser
from .analyzer import TypeChecker, create_builtin_symbol_table
from .transformer.ast_transformer import ASTTransformer, HoistedArrayInit
from .transformer import transformed_ast as IR
from .codegen.opencl_emitter import OpenCLEmitter
from .preprocessor import (
    PreprocessorTransformer,
    append_uniform_undefs,
    collect_uniform_redefines,
    uniform_redefine_prefix,
)
from .preprocessor.conditional_eval import strip_conditionals


class TranspileError(Exception):
    """Raised when transpilation fails. Both hosts alias their public error
    type to this (Host A: ``TranspileError``; Host B: ``TranspilationError``)."""
    pass


# Shadertoy renderpass entry points. A pass's own entry is split out for the
# kernel body; the OTHER entries are excluded when they appear AFTER it (they
# were never part of the emitted header), but kept BEFORE it — some shaders
# define mainVR first and CALL it from mainImage.
ENTRY_POINTS = {"mainImage", "mainCubemap", "mainSound", "mainVR"}


# ---------------------------------------------------------------------------
# Shared helpers (were duplicated verbatim in both hosts)
# ---------------------------------------------------------------------------

def post_process_ifdef_blocks(opencl_code: str) -> str:
    """
    Fix GLSL types and functions inside ``#ifdef`` blocks.

    Workaround for the Session 9 limitation where code inside ``#ifdef`` blocks
    is not transformed by the AST transformer (tree-sitter treats them as opaque
    preprocessor nodes), so a raw-text pass rewrites types, GLSL_-prefixes
    builtins, and adds ``f`` float suffixes.
    """
    type_map = {
        r'\bvec2\b': 'float2',
        r'\bvec3\b': 'float3',
        r'\bvec4\b': 'float4',
        r'\bivec2\b': 'int2',
        r'\bivec3\b': 'int3',
        r'\bivec4\b': 'int4',
        r'\buvec2\b': 'uint2',
        r'\buvec3\b': 'uint3',
        r'\buvec4\b': 'uint4',
    }
    for glsl_type, opencl_type in type_map.items():
        opencl_code = re.sub(glsl_type, opencl_type, opencl_code)

    glsl_functions = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
        'abs', 'sign', 'floor', 'ceil', 'trunc', 'fract', 'mod', 'modf',
        'min', 'max', 'clamp', 'mix', 'step', 'smoothstep',
        'length', 'distance', 'dot', 'cross', 'normalize',
        'faceforward', 'reflect', 'refract',
        'radians', 'degrees',
        'transpose', 'inverse', 'determinant',
    ]
    for func_name in glsl_functions:
        pattern = r'(?<!GLSL_)\b' + func_name + r'\s*\('
        opencl_code = re.sub(pattern, f'GLSL_{func_name}(', opencl_code)

    # Float literal 'f' suffixes.
    opencl_code = re.sub(r'(?<!\w)(\d+\.\d*)(?![fF\w])', r'\1f', opencl_code)
    opencl_code = re.sub(r'(?<!\w)(\d+[eE][+-]?\d+)(?![fF\w])', r'\1f', opencl_code)
    opencl_code = re.sub(r'(?<!\w)(\.\d+)(?![fF\w])', r'\1f', opencl_code)
    return opencl_code


def normalize_entry_point(glsl_source: str) -> str:
    """
    Rewrite unconventional Shadertoy entry idioms into a standard
    ``void mainImage(out vec4 fragColor, in vec2 fragCoord)`` shader (S2 of
    docs/handover/ENTRYPOINT_REDESIGN.md).

    (a) macro-entry: ``#define main() mainImage(...)`` (+ ``gl_*`` object macros)
        — expand and drop (our preprocessor pass never expands macros);
    (b) bare ``void main(void)`` using gl_FragColor/gl_FragCoord — rewrite the
        signature, map gl_FragColor to the out param, provide gl_FragCoord as a
        file-scope global set at the top of the entry body.

    Sources already defining a conventional entry are returned unchanged.
    """
    if re.search(r'\bvoid\s+mainImage\s*\(', glsl_source):
        return glsl_source

    src = glsl_source

    # --- idiom (a): #define main() <replacement containing mainImage> ------
    m = re.search(
        r'^[ \t]*#[ \t]*define[ \t]+main[ \t]*\([ \t]*\)[ \t]+(.*\bmainImage\b.*)$',
        src, re.MULTILINE)
    if m:
        replacement = m.group(1).strip()
        src = src[:m.start()] + src[m.end():]          # drop the define
        src = re.sub(r'\bmain[ \t]*\([ \t]*\)', replacement, src)
        for gl_name in ('gl_FragCoord', 'gl_FragColor'):
            dm = re.search(
                r'^[ \t]*#[ \t]*define[ \t]+' + gl_name + r'[ \t]+(\S+)[ \t]*$',
                src, re.MULTILINE)
            if dm:
                alias = dm.group(1)
                src = src[:dm.start()] + src[dm.end():]
                src = re.sub(r'\b' + gl_name + r'\b', alias, src)
        return src

    # --- idiom (b): bare void main(void?) -----------------------------------
    sig = re.search(r'\bvoid\s+main\s*\(\s*(?:void)?\s*\)', src)
    if not sig:
        return glsl_source  # nothing we recognize; let transpile() report

    src = (src[:sig.start()]
           + "void mainImage(out vec4 fragColor, in vec2 fragCoord)"
           + src[sig.end():])
    src = re.sub(r'\bgl_FragColor\b', 'fragColor', src)

    if re.search(r'\bgl_FragCoord\b', src):
        body_open = re.search(r'void\s+mainImage\s*\([^)]*\)\s*\{', src)
        insert_at = body_open.end()
        src = ("vec4 gl_FragCoord;\n"
               + src[:insert_at]
               + "\n    gl_FragCoord = vec4(fragCoord, 0.0, 1.0);"
               + src[insert_at:])
    return src


def detect_renderpass_type(glsl_source: str) -> str:
    """Detect the renderpass entry from source: ``mainImage`` / ``mainCubemap``
    / ``mainSound``, or ``Common`` (no entry = header-only)."""
    if "void mainImage" in glsl_source:
        return "mainImage"
    elif "void mainCubemap" in glsl_source:
        return "mainCubemap"
    elif "vec2 mainSound" in glsl_source:
        return "mainSound"
    else:
        return "Common"


def partition_translation_unit(
    ir: "IR.TranslationUnit",
    entry_name: str = "mainImage",
) -> Tuple[Optional["IR.FunctionDefinition"], list]:
    """
    Split the transformed translation unit into the entry function and the
    header declarations (single-TU entry-point model, ENTRYPOINT_REDESIGN.md).

    Code AFTER the entry stays in the header (category S: prototype-style
    shaders define helpers at the bottom). The OTHER Shadertoy entry points are
    excluded only when they come after the entry. When several entry definitions
    exist, the LAST one wins. Returns ``(None, all_decls)`` when no entry is
    found — the caller decides whether that is an error (Host A: mainImage
    required) or a header-only pass (Host B: Common / no-entry).
    """
    def is_definition(decl):
        return (isinstance(decl, IR.FunctionDefinition)
                and not getattr(decl, 'is_prototype', False))

    entry_indices = [i for i, d in enumerate(ir.declarations)
                     if is_definition(d) and d.name == entry_name]
    if not entry_indices:
        return None, list(ir.declarations)
    first_entry, last_entry = entry_indices[0], entry_indices[-1]
    entry_ir = ir.declarations[last_entry]

    other_entries = ENTRY_POINTS - {entry_name}
    header_declarations = []
    for i, decl in enumerate(ir.declarations):
        if is_definition(decl) and decl.name == entry_name:
            continue  # last definition wins; all copies stay out of header
        if (is_definition(decl) and i > first_entry
                and decl.name in other_entries):
            continue
        header_declarations.append(decl)

    return entry_ir, header_declarations


# A `void mainImage(` definition sitting inside an un-evaluated program-scope
# conditional blob (see entry_trapped_in_conditional). Permissive by design —
# it only gates a rescue attempt that is re-checked by re-partitioning.
_TRAPPED_ENTRY_RE = re.compile(r'\bvoid\s+mainImage\s*\(')


def entry_trapped_in_conditional(ir: "IR.TranslationUnit") -> bool:
    """True when a mainImage definition is hidden inside a raw preprocessor
    blob — a program-scope ``#ifdef``/``#ifndef`` that tree-sitter kept verbatim
    (so the entry never became a top-level FunctionDefinition and partition
    found none). Comment-safe: a block-commented mainImage is a separate Comment
    node, never a PreprocessorDirective."""
    for decl in ir.declarations:
        if (isinstance(decl, IR.PreprocessorDirective)
                and _TRAPPED_ENTRY_RE.search(decl.text or "")):
            return True
    return False


def entry_param_names(entry_ir: "IR.FunctionDefinition") -> Tuple[str, str]:
    """
    The user's entry parameter names. Shadertoy permits custom names (golf
    shaders use ``out vec4 O, vec2 U``), but the Houdini kernel always exposes
    ``fragColor``/``fragCoord``; the gap is bridged with alias declarations
    rather than renaming identifiers in the body.
    """
    out_name, in_name = "fragColor", "fragCoord"
    params = entry_ir.parameters or []
    if len(params) >= 1 and params[0].name:
        out_name = params[0].name
    if len(params) >= 2 and params[1].name:
        in_name = params[1].name
    return out_name, in_name


def harvest_common_signatures(common_glsl: str):
    """Transform the Common tab in isolation to learn the out/inout parameter
    signatures (and return types) of the helper functions it defines.

    Houdini transpiles the Common tab as a SEPARATE node (code_common) injected
    before @KERNEL, so a renderpass that CALLS a Common-defined helper is
    transpiled WITHOUT seeing that helper's definition. When the helper has an
    out/inout parameter the emitter turns it into a ``T*`` pointer; the call
    site must then take the argument's address (``&x``) or clang rejects the
    kernel ("passing 'float3' to parameter of incompatible type
    '__generic float3 *'"). Host A merges the Common tab into the pass TU and
    discovers these signatures for free; a ``merge_common=False`` host restores
    parity WITHOUT duplicating Common's declarations into the pass header (they
    already live in code_common).

    Returns ``(function_signatures, user_function_return_types)`` harvested from
    Common, or ``({}, {})`` when Common is empty or fails to transform
    (best-effort: a Common tab that cannot transform here fails on its own
    code_common node with a clearer error — never break the renderpass over it).
    """
    if not common_glsl or not common_glsl.strip():
        return {}, {}
    try:
        preprocessor = PreprocessorTransformer()
        processed = preprocessor.transform(common_glsl)
        ast = GLSLParser().parse(processed)
        transformer = ASTTransformer(TypeChecker(create_builtin_symbol_table()))
        transformer.user_function_return_types.update(preprocessor.matrix_macros)
        transformer.transform(ast)
        return (dict(transformer.function_signatures),
                dict(transformer.user_function_return_types))
    except Exception:
        return {}, {}


def bridge_custom_params(body: str, out_name: str, in_name: str,
                         indent: str) -> str:
    """Bridge custom entry parameter names (golf ``out vec4 O, vec2 U``) to the
    kernel's ``fragColor``/``fragCoord`` with alias declarations + a final write,
    rather than renaming identifiers in the body. ``indent`` is the per-host
    leading whitespace for the alias/finalize lines (kept so each host's output
    stays byte-identical)."""
    out_alias = in_alias = out_finalize = ""
    if out_name != "fragColor":
        out_alias = f"{indent}float4 {out_name} = (float4)(0.0f, 0.0f, 0.0f, 1.0f);\n"
        out_finalize = f"\n{indent}fragColor = {out_name};"
    if in_name != "fragCoord":
        in_alias = f"{indent}float2 {in_name} = fragCoord;\n"
    if out_alias or in_alias or out_finalize:
        body = f"{out_alias}{in_alias}    {body}{out_finalize}".strip()
    return body


def render_hoisted_global_inits(entries: list, emitter: "OpenCLEmitter",
                                indent: str) -> str:
    """Render category-A hoisted global initializers as kernel-body statements.

    Program-scope globals whose initializer is not a compile-time constant are
    declared bare in the header; their real initializers run here at the top of
    the kernel body (declaration order, so inter-global dependencies hold), where
    runtime uniforms and helper calls are legal. A2 arrays use a temp-local
    aggregate + element copy loop. ``indent`` is per-host leading whitespace.
    """
    lines = [f"{indent}// ---- hoisted global initializers (category A) ----"]
    for entry in entries:
        if isinstance(entry, HoistedArrayInit):
            b, et, sz, init_ir = entry
            init_txt = emitter.emit_initializer(init_ir)
            lines.append(
                f"{indent}{{ {et} __init_{b}[{sz}] = {init_txt};"
                f" for (int __hi = 0; __hi < {sz}; ++__hi)"
                f" {b}[__hi] = __init_{b}[__hi]; }}"
            )
        else:
            name, init_ir = entry
            lines.append(f"{indent}{name} = {emitter.emit(init_ir)};")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one shared pipeline
# ---------------------------------------------------------------------------

@dataclass
class TranspiledPass:
    """Neutral, format-independent result of transpiling one renderpass.

    Each host formats these pieces for its target. ``body`` is the kernel body
    (custom-param bridged + ``#ifdef`` post-processed + category-A hoisting
    applied) WITHOUT the category-AG ``ag_prefix`` — kept separate because Host A
    reuses the body in both its ``kernel`` and ``full`` outputs, each with its
    own wrapper around ``ag_prefix``.
    """
    mode: str
    entry_ir: Optional["IR.FunctionDefinition"]
    header_opencl: str
    body: str
    ag_prefix: str
    out_name: str
    in_name: str
    hoisted_global_inits: list = field(default_factory=list)
    ag_redefines: dict = field(default_factory=dict)


def transpile_pass(
    glsl_source: str,
    *,
    mode: Optional[str] = None,
    common: str = "",
    merge_common: bool = False,
    require_entry: bool = False,
    hoist_indent: str = "",
    bridge_indent: str = "",
) -> TranspiledPass:
    """Run the full shared GLSL→OpenCL pipeline for one renderpass.

    Args:
        glsl_source: the renderpass GLSL.
        mode: entry type (``mainImage``/``mainCubemap``/``mainSound``/``Common``);
            ``None`` auto-detects from the source.
        common: optional Common-tab GLSL.
        merge_common: ``True`` prepends Common into the pass TU (Host A, compilecl
            single-file); ``False`` keeps Common separate and only harvests its
            signatures so call sites pointerize inout/out args (Host B).
        require_entry: ``True`` raises :class:`TranspileError` when no entry is
            found (Host A: mainImage is mandatory). ``False`` returns a
            header-only pass (Host B: Common / no-entry).
        hoist_indent, bridge_indent: per-host leading whitespace so each host's
            output stays byte-for-byte identical to its pre-unification form.
    """
    if not glsl_source or not isinstance(glsl_source, str):
        raise TranspileError("Invalid GLSL source: must be a non-empty string")

    # Stage -1: merge the Common tab ahead of the pass (Host A strategy).
    if merge_common and common and common.strip():
        glsl_source = common.rstrip() + "\n\n" + glsl_source

    # Stage -0.5: normalize unconventional entry idioms.
    glsl_source = normalize_entry_point(glsl_source)

    if mode is None:
        mode = detect_renderpass_type(glsl_source)

    # Stage 0: preprocess. glsl_raw (pre-preprocessor, post-normalize) keeps
    # user #undef lines for AG liveness and feeds the S59 strip_conditionals.
    preprocessor = PreprocessorTransformer()
    glsl_raw = glsl_source
    glsl_processed = preprocessor.transform(glsl_source)

    # Stage 1: parse — ONE parse of the whole (merged) source.
    parser = GLSLParser()
    try:
        ast = parser.parse(glsl_processed)
    except Exception as e:
        raise TranspileError(f"Failed to parse GLSL: {e}")

    def _make_transformer(pp: "PreprocessorTransformer") -> "ASTTransformer":
        t = ASTTransformer(TypeChecker(create_builtin_symbol_table()))
        # The renderpass entry's out-param (fragColor) is a host @KERNEL local,
        # not a deref'able pointer — mark it so the transformer excludes it.
        t.entry_function = mode or "mainImage"
        # Category J — seed matrix-returning #define macros so `p *= rot(a)`
        # dispatches through the matmul helper instead of a raw matrix `*=`.
        t.user_function_return_types.update(pp.matrix_macros)
        # Cross-tab parity (merge_common=False): seed Common helper signatures
        # so call sites take the address of inout/out arguments.
        if not merge_common and common and mode != "Common":
            csig, crets = harvest_common_signatures(common)
            for fname, arity_map in csig.items():
                t.function_signatures.setdefault(fname, {}).update(arity_map)
            for fname, ret in crets.items():
                t.user_function_return_types.setdefault(fname, ret)
        return t

    # Stage 2-4: transform the whole translation unit in source order.
    transformer = _make_transformer(preprocessor)
    try:
        ir = transformer.transform(ast)
    except Exception as e:
        raise TranspileError(f"Failed to transform GLSL: {e}")
    hoisted_global_inits = list(transformer.hoisted_global_inits)

    # Common renderpass: header only, no entry.
    if mode == "Common":
        try:
            emitter = OpenCLEmitter(indent_size=4)
            header_opencl = post_process_ifdef_blocks(emitter.emit(ir))
        except Exception as e:
            raise TranspileError(f"Failed to emit header: {e}")
        return TranspiledPass(
            mode=mode, entry_ir=None, header_opencl=header_opencl, body="",
            ag_prefix="", out_name="fragColor", in_name="fragCoord",
            hoisted_global_inits=[], ag_redefines={})

    # Stage 5: split entry vs header on IR.
    entry_ir, header_decls = partition_translation_unit(ir, mode)

    # S59 rescue — the entry may be trapped inside a program-scope
    # #ifdef/#ifndef that tree-sitter kept as one opaque raw blob, so it never
    # became a top-level FunctionDefinition and partition found none. If the
    # guarding conditional is statically decidable, evaluate it on the
    # pre-preprocessor source and rebuild the IR once. Only reached AFTER a
    # failed partition, so shaders that transpile today are untouched.
    if entry_ir is None and entry_trapped_in_conditional(ir):
        stripped = strip_conditionals(glsl_raw)
        if stripped.balanced and stripped.source != glsl_raw:
            preprocessor = PreprocessorTransformer()
            glsl_processed = preprocessor.transform(stripped.source)
            ast = parser.parse(glsl_processed)
            transformer = _make_transformer(preprocessor)
            ir = transformer.transform(ast)
            hoisted_global_inits = list(transformer.hoisted_global_inits)
            entry_ir, header_decls = partition_translation_unit(ir, mode)

    if entry_ir is None and require_entry:
        raise TranspileError(
            "Could not find mainImage() function in GLSL source")

    # Header emit + #ifdef post-process + category-AG uniform undefs.
    header_opencl = ""
    ag_redefines = {}
    if header_decls:
        try:
            emitter = OpenCLEmitter(indent_size=4)
            header_opencl = emitter.emit(
                IR.TranslationUnit(declarations=header_decls))
            header_opencl = post_process_ifdef_blocks(header_opencl)
            # Category AG cluster 1 — confine any user #define of a read-only
            # uniform (iTime/iFrame/...) with a trailing #undef so the kernel's
            # SHADERTOY_INPUTS assignments are not rewritten to non-lvalues. The
            # suppressed macros' FINAL definitions are captured FIRST (before the
            # #undef block lands) and re-emitted at the top of the kernel body
            # (push-pop). Definitions from the PREPROCESSED source (bodies valid
            # OpenCL); liveness from the RAW source (both the emitter and the
            # preprocessor drop user #undef lines, which last-one-wins needs).
            ag_redefines = collect_uniform_redefines(glsl_processed, glsl_raw)
            header_opencl = append_uniform_undefs(header_opencl)
        except Exception as e:
            raise TranspileError(f"Failed to emit header: {e}")

    if entry_ir is None:
        # No entry, header-only (Host B: non-Common pass without an entry).
        return TranspiledPass(
            mode=mode, entry_ir=None, header_opencl=header_opencl, body="",
            ag_prefix="", out_name="fragColor", in_name="fragCoord",
            hoisted_global_inits=hoisted_global_inits,
            ag_redefines=ag_redefines)

    out_name, in_name = entry_param_names(entry_ir)

    # Stage 6: emit the kernel body from the entry function's IR. Entry params
    # were never pointerized, so the body references fragColor/fragCoord as the
    # plain @KERNEL locals they are (no '*fragColor' surgery, no re-parse).
    try:
        body_emitter = OpenCLEmitter(indent_size=4)
        body_emitter.indent_level = 1  # body statements at function-body depth
        body_stmts = entry_ir.body.statements if entry_ir.body else []
        body = "".join(body_emitter.emit(s) for s in body_stmts).strip()
    except Exception as e:
        raise TranspileError(f"Failed to emit kernel body: {e}")

    # Custom entry param names — bridge with alias declarations (mainImage only,
    # matching both hosts; other entries always use fragColor/fragCoord).
    if mode == "mainImage":
        body = bridge_custom_params(body, out_name, in_name, bridge_indent)

    body = post_process_ifdef_blocks(body)

    # Category A — assign hoisted global initializers at the top of the body.
    if hoisted_global_inits:
        hoist_emitter = OpenCLEmitter(indent_size=4)
        body = (render_hoisted_global_inits(
                    hoisted_global_inits, hoist_emitter, hoist_indent)
                + "\n" + body)

    # Category AG push-pop — re-apply the user's uniform remaps at the top of
    # the body (SHADERTOY_INPUTS has already expanded above it, and nothing
    # after the body assigns a uniform). Kept separate from `body` so Host A can
    # place it inside both its kernel and full wrappers.
    ag_prefix = uniform_redefine_prefix(ag_redefines)

    return TranspiledPass(
        mode=mode, entry_ir=entry_ir, header_opencl=header_opencl, body=body,
        ag_prefix=ag_prefix, out_name=out_name, in_name=in_name,
        hoisted_global_inits=hoisted_global_inits, ag_redefines=ag_redefines)
