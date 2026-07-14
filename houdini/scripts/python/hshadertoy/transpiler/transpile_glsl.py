"""
Production GLSL to OpenCL Transpiler for Houdini

Wraps the core transpiler (src/glsl_to_opencl) and formats output for
Houdini's @KERNEL structure.

Single-translation-unit entry-point model (docs/handover/
ENTRYPOINT_REDESIGN.md): one parse + one transform of the whole source;
the header/body split happens on transformed IR (no brace-counting over
emitted text, no '*fragColor' substring surgery). Mirrors Host A
(tests/transpile.py) — keep the two in sync.
"""
import re
import sys
from pathlib import Path

# Ensure transpiler is importable
# This is needed because package PYTHONPATH not automatically applied in headless hython
sys.path.insert(0, 'C:/dev/hShadertoy')

from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer import transformed_ast as IR
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter
from src.glsl_to_opencl.preprocessor import (
    PreprocessorTransformer,
    append_uniform_undefs,
    collect_uniform_redefines,
    uniform_redefine_prefix,
)

# Shadertoy renderpass entry points. A pass's own entry is split out for the
# @KERNEL body; the OTHER entries are excluded when they appear after it
# (they were never part of the emitted header), but kept before it — some
# shaders define mainVR first and CALL it from mainImage.
ENTRY_POINTS = {"mainImage", "mainCubemap", "mainSound", "mainVR"}


class TranspilationError(Exception):
    """Raised when transpilation fails"""
    pass


def _normalize_entry_point(glsl_source: str) -> str:
    """
    Rewrite unconventional Shadertoy entry idioms into a standard
    `void mainImage(out vec4 fragColor, in vec2 fragCoord)` shader
    (entry-point redesign S2; mirror of Host A's normalize_entry_point).

    (a) macro-entry: `#define main() mainImage(...)` (+ gl_* object macros)
        — expand and drop (our preprocessor pass never expands macros);
    (b) bare `void main(void)` with gl_FragColor/gl_FragCoord — rewrite the
        signature, map gl_FragColor to the out param, provide gl_FragCoord
        as a file-scope global set at the top of the entry body.

    Sources already defining any conventional entry are returned unchanged.
    """
    if re.search(r'\b(?:void\s+mainImage|void\s+mainCubemap|vec2\s+mainSound)\s*\(',
                 glsl_source):
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
        return glsl_source  # nothing we recognize

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


def _detect_renderpass_type(glsl_source: str) -> str:
    """
    Detect the renderpass type from GLSL source.

    Returns: "mainImage", "mainCubemap", "Common", or "mainSound"
    """
    # Simple detection based on function signature
    if "void mainImage" in glsl_source:
        return "mainImage"
    elif "void mainCubemap" in glsl_source:
        return "mainCubemap"
    elif "vec2 mainSound" in glsl_source:
        return "mainSound"
    else:
        # No main function = Common renderpass (header only)
        return "Common"


def _partition_translation_unit(ir, entry_name):
    """
    Split the transformed translation unit into the entry function and the
    header declarations (mirror of Host A's partition_translation_unit).

    Code AFTER the entry is kept in the header too (category S: prototype-
    style shaders define helpers at the bottom of the file). The OTHER
    Shadertoy entry points are excluded only when they come after the entry.
    When several entry definitions exist, the LAST one wins.

    Returns:
        (entry_function_ir_or_None, header_declarations)
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


def _entry_param_names(entry_ir):
    """
    The user's mainImage parameter names. Shadertoy permits custom names
    (golf shaders use `out vec4 O, vec2 U`), but the Houdini @KERNEL always
    exposes fragColor/fragCoord; the gap is bridged with alias declarations
    rather than renaming identifiers in the body.
    """
    out_name, in_name = "fragColor", "fragCoord"
    params = entry_ir.parameters or []
    if len(params) >= 1 and params[0].name:
        out_name = params[0].name
    if len(params) >= 2 and params[1].name:
        in_name = params[1].name
    return out_name, in_name


def _post_process_ifdef_blocks(opencl_code: str) -> str:
    """
    Post-process OpenCL code to fix GLSL types and functions inside #ifdef blocks.

    This is a workaround for Session 9's limitation where code inside #ifdef blocks
    is not transformed by the AST transformer because tree-sitter treats them as
    opaque preprocessor nodes.

    Args:
        opencl_code: OpenCL code string

    Returns:
        Fixed OpenCL code with all GLSL types and functions transformed
    """
    # Type transformations
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

    # Function transformations (add GLSL_ prefix)
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
        # Only add GLSL_ prefix if not already present
        # Use negative lookbehind to avoid matching GLSL_func
        pattern = r'(?<!GLSL_)\b' + func_name + r'\s*\('
        replacement = f'GLSL_{func_name}('
        opencl_code = re.sub(pattern, replacement, opencl_code)

    # Float literal transformations (add 'f' suffix if missing)
    # Pattern 1: Numbers with decimal point
    opencl_code = re.sub(r'(?<!\w)(\d+\.\d*)(?![fF\w])', r'\1f', opencl_code)
    # Pattern 2: Numbers with exponent
    opencl_code = re.sub(r'(?<!\w)(\d+[eE][+-]?\d+)(?![fF\w])', r'\1f', opencl_code)
    # Pattern 3: Decimal point at start
    opencl_code = re.sub(r'(?<!\w)(\.\d+)(?![fF\w])', r'\1f', opencl_code)

    return opencl_code


def _format_for_houdini(header: str, body: str, mode: str) -> str:
    """
    Format header and body for Houdini @KERNEL structure.

    Args:
        header: Global declarations, functions, etc.
        body: Main function body (fragColor/fragCoord are @KERNEL locals;
              the transformer never pointerizes the entry's params)
        mode: Renderpass type

    Returns:
        Complete Houdini-formatted OpenCL code
    """
    if mode == "Common":
        # Common renderpass: header only, no @KERNEL
        return header

    # For mainImage, mainCubemap, etc: header + @KERNEL wrapper
    output = []

    # Add header if present
    if header.strip():
        output.append("// ---- HEADER: Global declarations ----")
        output.append(header)
        output.append("")

    # Add @KERNEL block
    output.append("@KERNEL")
    output.append("{")
    output.append("    SHADERTOY_INPUTS  // HDA-defined: iResolution, iTime, etc.")
    output.append("")

    # Add body with proper indentation
    if body.strip():
        # Indent body lines
        body_lines = body.split('\n')
        for line in body_lines:
            if line.strip():
                output.append("    " + line)
            else:
                output.append("")

    output.append("")
    output.append("    @fragColor.set(fragColor);  // Houdini output bind")
    output.append("}")

    return '\n'.join(output)


def transpile(glsl_source: str, mode: str = None) -> str:
    """
    Transpile GLSL shader code to Houdini-compatible OpenCL.

    Args:
        glsl_source: GLSL shader source code
        mode: Renderpass type ("mainImage", "mainCubemap", "Common", "mainSound")
              If None, auto-detect from source.

    Returns:
        Complete OpenCL code formatted for Houdini HDA

    Raises:
        TranspilationError: If transpilation fails
    """
    # Validate input
    if not glsl_source or not isinstance(glsl_source, str):
        raise TranspilationError("Invalid GLSL source: must be a non-empty string")

    try:
        # Normalize unconventional entry idioms (macro-entry, bare void
        # main() + gl_*) before mode detection so they become mainImage.
        glsl_source = _normalize_entry_point(glsl_source)

        # Auto-detect mode if not provided
        if mode is None:
            mode = _detect_renderpass_type(glsl_source)

        # Stage 1: Preprocess
        preprocessor = PreprocessorTransformer()
        glsl_processed = preprocessor.transform(glsl_source)

        # Stage 2: Parse — ONE parse of the whole source
        parser = GLSLParser()
        ast = parser.parse(glsl_processed)

        # Stage 3: Type check (initialize)
        symbol_table = create_builtin_symbol_table()
        type_checker = TypeChecker(symbol_table)

        # Stage 4: Transform the whole translation unit in source order.
        transformer = ASTTransformer(type_checker)
        # The renderpass entry function's out-param (fragColor) is a host
        # @KERNEL local, not a deref'able pointer (see
        # ASTTransformer.entry_function / entry_params_are_locals).
        transformer.entry_function = mode
        ir = transformer.transform(ast)

        # Category A — program-scope globals with non-constant initializers
        # are declared bare; their real initializers are assigned at the top
        # of the @KERNEL body (mirrors Host A; previously LOST in this host —
        # TRANSPILER_REVIEW §0.2).
        hoisted_global_inits = list(transformer.hoisted_global_inits)

        emitter = OpenCLEmitter(indent_size=4)

        if mode == "Common":
            header_opencl = _post_process_ifdef_blocks(emitter.emit(ir))
            return _format_for_houdini(header_opencl, "", mode)

        # Stage 5: Split entry vs header on IR and emit each.
        entry_ir, header_decls = _partition_translation_unit(ir, mode)

        header_opencl = ""
        ag_redefines = {}
        if header_decls:
            header_opencl = emitter.emit(
                IR.TranslationUnit(declarations=header_decls))
            header_opencl = _post_process_ifdef_blocks(header_opencl)
            # Category AG cluster 1 — confine any user #define of a read-only
            # uniform (iTime/iFrame/...) with a trailing #undef so the
            # @KERNEL's SHADERTOY_INPUTS assignments are not rewritten to
            # non-lvalues. The suppressed macros' FINAL definitions are
            # captured FIRST (before our #undef block lands in the text) and
            # re-emitted at the top of the @KERNEL body below (push-pop: the
            # entry body sits AFTER SHADERTOY_INPUTS and must still see the
            # user's remap). Mirror of Host A tests/transpile.py. Fixes live
            # Houdini with no HDA change.
            # Definitions from the PREPROCESSED source (bodies already valid
            # OpenCL); liveness from the RAW source (both the emitter and the
            # preprocessor drop user #undef lines, which last-one-wins needs).
            ag_redefines = collect_uniform_redefines(glsl_processed, glsl_source)
            header_opencl = append_uniform_undefs(header_opencl)

        if entry_ir is None:
            # No entry found (defensive parity with the old line-scan split):
            # treat everything as header.
            return _format_for_houdini(header_opencl, "", mode)

        body_emitter = OpenCLEmitter(indent_size=4)
        body_emitter.indent_level = 1  # statements at function-body depth
        body_stmts = entry_ir.body.statements if entry_ir.body else []
        body = "".join(body_emitter.emit(s) for s in body_stmts).strip()

        # Custom mainImage param names (out vec4 O, vec2 U): bridge with
        # alias declarations + a final write to the @KERNEL fragColor.
        if mode == "mainImage":
            out_name, in_name = _entry_param_names(entry_ir)
            out_alias = ""
            in_alias = ""
            out_finalize = ""
            if out_name != "fragColor":
                out_alias = (f"float4 {out_name} = "
                             f"(float4)(0.0f, 0.0f, 0.0f, 1.0f);\n")
                out_finalize = f"\nfragColor = {out_name};"
            if in_name != "fragCoord":
                in_alias = f"float2 {in_name} = fragCoord;\n"
            if out_alias or in_alias or out_finalize:
                body = f"{out_alias}{in_alias}    {body}{out_finalize}".strip()

        body = _post_process_ifdef_blocks(body)

        # Category A hoisted global initializers run first (declaration
        # order, so inter-global dependencies hold).
        if hoisted_global_inits:
            hoist_emitter = OpenCLEmitter(indent_size=4)
            hoist_lines = ["// ---- hoisted global initializers (category A) ----"]
            for name, init_ir in hoisted_global_inits:
                hoist_lines.append(f"{name} = {hoist_emitter.emit(init_ir)};")
            body = "\n".join(hoist_lines) + "\n" + body

        # Category AG push-pop — re-apply the user's uniform remaps at the
        # very top of the @KERNEL body: SHADERTOY_INPUTS (emitted by
        # _format_for_houdini just above the body) has already expanded, and
        # nothing after the body assigns a uniform (only
        # `@fragColor.set(fragColor);` follows), so the re-defined macro
        # covers exactly the inlined user code (hoisted globals included —
        # they are user code and saw the macro on Shadertoy).
        ag_prefix = uniform_redefine_prefix(ag_redefines)
        if ag_prefix:
            body = ag_prefix + body

        # Format for Houdini
        houdini_code = _format_for_houdini(header_opencl, body, mode)

        return houdini_code

    except TranspilationError:
        # Re-raise our own errors
        raise
    except Exception as e:
        # Wrap unexpected errors with context
        error_msg = f"Transpilation failed: {type(e).__name__}: {e}"
        raise TranspilationError(error_msg) from e
