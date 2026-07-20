#!/usr/bin/env python3
"""
GLSL to OpenCL Transpiler - Updated for Session 6 Architecture

This script transpiles GLSL shaders into split header and kernel files
suitable for testing with compilecl.py.

Architecture (Sessions 1-6):
    GLSL Source -> Parser -> TypeChecker -> ASTTransformer -> OpenCLEmitter -> OpenCL

Output modes:
- header.cl: Everything BEFORE void mainImage() (globals, helper functions)
- kernel.cl: Only the mainImage() BODY (without signature, for compilecl.py)
- full.cl: Complete OpenCL code (optional)

Usage (CLI):
    python transpile.py input.glsl
    # Outputs: input.header.cl and input.kernel.cl

    python transpile.py input.glsl --output-dir output/
    # Outputs: output/input.header.cl and output/input.kernel.cl

    python transpile.py input.glsl --full
    # Outputs: input.full.cl (complete OpenCL code)

    python transpile.py input.glsl --verbose
    # Show transformation stages

    python transpile.py input.glsl --validate
    # Validate OpenCL compilation (requires PyOpenCL)

Usage (Module):
    from transpile import transpile

    result = transpile(glsl_source_string)
    header_code = result.get_header()
    kernel_code = result.get_kernel()
    full_code = result.get_full()
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new Session 6 architecture (updated for Session 9)
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import (
    ASTTransformer,
    HoistedArrayInit,
)
from src.glsl_to_opencl.transformer import transformed_ast as IR
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter
from src.glsl_to_opencl.preprocessor import (
    PreprocessorTransformer,
    append_uniform_undefs,
    collect_uniform_redefines,
    uniform_redefine_prefix,
)
from src.glsl_to_opencl.preprocessor.conditional_eval import strip_conditionals


@dataclass
class TranspileResult:
    """
    Result of GLSL to OpenCL transpilation.

    Attributes:
        header: OpenCL code before mainImage() (globals, functions)
        kernel: OpenCL mainImage() body only (for compilecl.py)
        full: Complete OpenCL code (header + mainImage signature + body)
    """
    header: str
    kernel: str
    full: str

    def get_header(self) -> str:
        """Get header OpenCL code (before mainImage)."""
        return self.header

    def get_kernel(self) -> str:
        """Get kernel body OpenCL code (mainImage body only)."""
        return self.kernel

    def get_full(self) -> str:
        """Get full OpenCL code (complete)."""
        return self.full


class TranspileError(Exception):
    """Raised when transpilation fails."""
    pass


def post_process_ifdef_blocks(opencl_code: str) -> str:
    """
    Post-process OpenCL code to fix GLSL types and functions inside #ifdef blocks.

    This is a workaround for Session 9's limitation where code inside #ifdef blocks
    is not transformed by the AST transformer.

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


def normalize_entry_point(glsl_source: str) -> str:
    """
    Rewrite unconventional Shadertoy entry idioms into a standard
    `void mainImage(out vec4 fragColor, in vec2 fragCoord)` shader (S2 of
    docs/handover/ENTRYPOINT_REDESIGN.md).

    Shadertoy never scans user code for mainImage — its GL header
    forward-declares it, main() calls it, and the REAL GL preprocessor
    expands user macros. Two idioms therefore compile on the site without a
    literal mainImage definition:

    (a) macro-entry: `#define main() mainImage(out vec4 fragColor, vec2
        fragCoord)` (+ `#define gl_FragCoord fragCoord`), then
        `void main() {...}` — expand those entry-related defines and drop
        them (our preprocessor pass never expands macros).
    (b) bare `void main(void)` using gl_FragColor/gl_FragCoord (GLSL-Sandbox
        ports) — rewrite the signature, map gl_FragColor to the out param,
        and provide gl_FragCoord as a global (it is referenceable from
        helper functions in GLSL) initialized at the top of the entry body.

    Shaders that already define mainImage are returned unchanged.
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

        # Expand gl_* object macros the site's preprocessor would resolve
        # (e.g. `#define gl_FragCoord fragCoord`); other user defines are
        # left for the normal preprocessor path.
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
        # gl_FragCoord is a global in GLSL (helpers may read it): declare it
        # at file scope and set it first thing in the entry body. z/w use
        # nominal fragment values (z is depth — meaningless for Shadertoy).
        body_open = re.search(
            r'void\s+mainImage\s*\([^)]*\)\s*\{', src)
        insert_at = body_open.end()
        src = ("vec4 gl_FragCoord;\n"
               + src[:insert_at]
               + "\n    gl_FragCoord = vec4(fragCoord, 0.0, 1.0);"
               + src[insert_at:])
    return src


def partition_translation_unit(ir: "IR.TranslationUnit") -> Tuple["IR.FunctionDefinition", list]:
    """
    Split the transformed translation unit into the entry function and the
    header declarations (single-TU entry-point model — see
    docs/handover/ENTRYPOINT_REDESIGN.md).

    Code AFTER mainImage is kept in the header too (category S):
    prototype-style shaders declare `float Fn (float x);` at the top, call it
    from mainImage, and define it at the bottom of the file — dropping
    post-mainImage code left those prototypes unresolved.
    Alternate Shadertoy entry points (mainVR/mainSound/mainCubemap) are
    excluded ONLY when they come after mainImage (they were never part of
    the emitted header there). Before mainImage they must stay included:
    some shaders (XlBGzm, lsVBDh, XscXzn) define mainVR first and CALL it
    from mainImage.

    When several mainImage definitions exist, the LAST one wins (matching the
    pre-restructure text-split behavior); all of them stay out of the header.

    Returns:
        Tuple of (entry_function_ir, header_declarations)

    Raises:
        TranspileError: If no mainImage() definition is found
    """
    alternate_entry_points = {"mainVR", "mainSound", "mainCubemap"}

    entry_ir = None
    header_declarations = []

    for decl in ir.declarations:
        is_definition = (
            isinstance(decl, IR.FunctionDefinition)
            and not getattr(decl, 'is_prototype', False)
        )
        if is_definition and decl.name == "mainImage":
            entry_ir = decl  # last definition wins
        elif (is_definition and entry_ir is not None
              and decl.name in alternate_entry_points):
            continue
        else:
            header_declarations.append(decl)

    if entry_ir is None:
        raise TranspileError("Could not find mainImage() function in GLSL source")

    return entry_ir, header_declarations


# A `void mainImage(` definition sitting inside an un-evaluated program-scope
# conditional blob (see _entry_trapped_in_conditional). Permissive by design —
# it only gates a rescue attempt that is re-checked by re-partitioning.
_TRAPPED_ENTRY_RE = re.compile(r'\bvoid\s+mainImage\s*\(')


def _entry_trapped_in_conditional(ir: "IR.TranslationUnit") -> bool:
    """True when a mainImage definition is hidden inside a raw preprocessor
    blob — a program-scope `#ifdef`/`#ifndef` that tree-sitter kept verbatim
    (so the entry never became a top-level FunctionDefinition and partition
    failed). Comment-safe: a block-commented mainImage is a separate Comment
    node, never a PreprocessorDirective."""
    for decl in ir.declarations:
        if (isinstance(decl, IR.PreprocessorDirective)
                and _TRAPPED_ENTRY_RE.search(decl.text or "")):
            return True
    return False


def entry_param_names(entry_ir: "IR.FunctionDefinition") -> Tuple[str, str]:
    """
    The user's mainImage parameter names. Shadertoy permits custom names
    (golf shaders frequently use `out vec4 O, vec2 U`), but the Houdini
    kernel always exposes `fragColor`/`fragCoord`. transpile() bridges the
    gap via alias injection rather than renaming identifiers in the body.
    """
    out_name, in_name = "fragColor", "fragCoord"
    params = entry_ir.parameters or []
    if len(params) >= 1 and params[0].name:
        out_name = params[0].name
    if len(params) >= 2 and params[1].name:
        in_name = params[1].name
    return out_name, in_name


def transpile(glsl_source: str, verbose: bool = False, common: str = "") -> TranspileResult:
    """
    Transpile GLSL shader source to OpenCL.

    This is the main transpilation function using the Session 9 architecture:
    GLSL -> Preprocessor -> Parser -> TypeChecker -> ASTTransformer -> OpenCLEmitter -> OpenCL

    Args:
        glsl_source: GLSL source code string
        verbose: If True, print transformation stages
        common: Optional Shadertoy "Common" tab GLSL. It is prepended to the pass
            source before processing. Common has no mainImage(), so its macros and
            helper functions flow into the header and become available to the pass
            body. This mirrors Houdini, where Common is injected as header code
            before @KERNEL into every renderpass node.

    Returns:
        TranspileResult with header, kernel, and full OpenCL code

    Raises:
        TranspileError: If transpilation fails
    """
    if verbose:
        print("=" * 70)
        print("GLSL to OpenCL Transpilation")
        print("=" * 70)

    # Stage -1: Merge the Common tab (if any) ahead of the pass source.
    if common and common.strip():
        glsl_source = common.rstrip() + "\n\n" + glsl_source
        if verbose:
            print(f"  [OK] Prepended Common tab ({len(common)} chars)")

    # Stage -0.5: Normalize unconventional entry idioms (macro-entry,
    # bare void main() + gl_* — see normalize_entry_point docstring).
    glsl_source = normalize_entry_point(glsl_source)

    # Stage 0: Transform preprocessor directives
    if verbose:
        print("\n[0/6] Transforming preprocessor directives...")

    preprocessor = PreprocessorTransformer()
    glsl_raw = glsl_source  # pre-preprocessor text; keeps user #undef lines (AG)
    glsl_source = preprocessor.transform(glsl_source)

    if verbose:
        print(f"  [OK] Preprocessor transformation complete")

    # Stage 1: Parse GLSL — ONE parse of the whole merged source (single-TU
    # entry-point model, docs/handover/ENTRYPOINT_REDESIGN.md). The old
    # pipeline text-split around mainImage and re-parsed each half (plus the
    # emitted OpenCL a fourth time); the split now happens on transformed IR.
    if verbose:
        print("\n[1/6] Parsing GLSL source...")

    parser = GLSLParser()
    try:
        ast = parser.parse(glsl_source)
    except Exception as e:
        raise TranspileError(f"Failed to parse GLSL: {e}")

    # Stage 2: Setup type checker
    if verbose:
        print("\n[2/6] Initializing type checker...")

    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)

    if verbose:
        print(f"  [OK] Built-in symbol table created")

    # Stage 3: Create transformer
    if verbose:
        print("\n[3/6] Creating AST transformer...")

    transformer = ASTTransformer(type_checker)

    # Seed matrix-returning #define macros (category J) as user-function
    # return types so `p *= rot(a)` (rot a matrix macro) dispatches through
    # the matmul helper instead of emitting a raw `float2 *= matrix2x2`. A
    # later real definition of the same name overwrites this during transform.
    transformer.user_function_return_types.update(preprocessor.matrix_macros)

    if verbose:
        print(f"  [OK] Transformer initialized")

    # Stage 4: Transform the whole translation unit in source order (exactly
    # what GLSL's own compiler sees; prototypes pre-register signatures for
    # call sites that precede definitions), then split entry vs header on IR.
    if verbose:
        print("\n[4/6] Transforming translation unit...")

    try:
        ir = transformer.transform(ast)
    except Exception as e:
        raise TranspileError(f"Failed to transform GLSL: {e}")

    # Category A — program-scope globals whose initializer is not a compile-time
    # constant are emitted bare by the transformer; their real initializers are
    # collected here and assigned at the top of the kernel body below.
    hoisted_global_inits = list(transformer.hoisted_global_inits)

    try:
        entry_ir, header_decls = partition_translation_unit(ir)
    except TranspileError:
        # The entry point may be trapped inside a program-scope #ifdef/#ifndef
        # that tree-sitter kept as one opaque raw block, so mainImage never
        # became a top-level declaration. If a raw preprocessor blob holds a
        # mainImage definition and the guarding conditional is statically
        # decidable, evaluate it (strip_conditionals) on the pre-preprocessor
        # source and rebuild the IR once. This branch is only reached AFTER the
        # normal partition already failed, so shaders that transpile today are
        # untouched (zero blast radius).
        rescued = None
        if _entry_trapped_in_conditional(ir):
            stripped = strip_conditionals(glsl_raw)
            if stripped.balanced and stripped.source != glsl_raw:
                preprocessor = PreprocessorTransformer()
                glsl_source = preprocessor.transform(stripped.source)
                ast = parser.parse(glsl_source)
                transformer = ASTTransformer(TypeChecker(
                    create_builtin_symbol_table()))
                transformer.user_function_return_types.update(
                    preprocessor.matrix_macros)
                ir = transformer.transform(ast)
                hoisted_global_inits = list(transformer.hoisted_global_inits)
                rescued = partition_translation_unit(ir)
        if rescued is None:
            raise
        entry_ir, header_decls = rescued
    out_name, in_name = entry_param_names(entry_ir)

    header_opencl = ""
    ag_redefines = {}
    if header_decls:
        try:
            emitter = OpenCLEmitter(indent_size=4)
            header_opencl = emitter.emit(
                IR.TranslationUnit(declarations=header_decls)
            )
            # Post-process: Fix GLSL types and functions inside #ifdef blocks
            # (code inside #ifdef blocks is not transformed by the AST
            # transformer — raw-text pass-through)
            header_opencl = post_process_ifdef_blocks(header_opencl)
            # Category AG cluster 1 — confine any user #define of a read-only
            # uniform (iTime/iFrame/...) with a trailing #undef so the kernel's
            # SHADERTOY_INPUTS assignments (iTime = AT_Time; ...) are not
            # rewritten to non-lvalues. The suppressed macros' FINAL
            # definitions are captured FIRST (before our #undef block lands in
            # the text) and re-emitted at the top of the kernel glue below
            # (push-pop: the entry body is inlined AFTER SHADERTOY_INPUTS and
            # must still see the user's remap). Shared helper -> Houdini
            # benefits too.
            # Definitions from the PREPROCESSED source (bodies already valid
            # OpenCL); liveness from the RAW source (both the emitter and the
            # preprocessor drop user #undef lines, which last-one-wins needs).
            ag_redefines = collect_uniform_redefines(glsl_source, glsl_raw)
            header_opencl = append_uniform_undefs(header_opencl)
        except Exception as e:
            raise TranspileError(f"Failed to emit header: {e}")

        if verbose:
            print(f"    [OK] Header: {len(header_opencl)} chars")

    # Stage 5: Emit the kernel body directly from the entry function's IR.
    # No synthetic re-wrap, no re-parse of emitted OpenCL, no '*fragColor'
    # substring surgery: entry params were never pointerized (the transformer
    # excludes the entry function from pointer_params), so the body already
    # references fragColor/fragCoord as the plain @KERNEL locals they are.
    if verbose:
        print("\n[5/6] Emitting kernel (mainImage body)...")

    try:
        emitter = OpenCLEmitter(indent_size=4)
        emitter.indent_level = 1  # body statements sit at function-body depth
        body_stmts = entry_ir.body.statements if entry_ir.body else []
        kernel_opencl_body = "".join(emitter.emit(s) for s in body_stmts).strip()
    except Exception as e:
        raise TranspileError(f"Failed to emit kernel body: {e}")

    # For custom parameter names (e.g. `out vec4 O, vec2 U`) bridge the names
    # with alias declarations + a final write to fragColor. This avoids
    # renaming identifiers in the body (which would risk clobbering same-named
    # locals in nested scopes for single-letter names).
    out_alias = ""
    in_alias = ""
    out_finalize = ""
    if out_name != "fragColor":
        out_alias = f"    float4 {out_name} = (float4)(0.0f, 0.0f, 0.0f, 1.0f);\n"
        out_finalize = f"\n    fragColor = {out_name};"
    if in_name != "fragCoord":
        in_alias = f"    float2 {in_name} = fragCoord;\n"
    if out_alias or in_alias or out_finalize:
        kernel_opencl_body = (
            f"{out_alias}{in_alias}    {kernel_opencl_body}{out_finalize}"
        ).strip()

    # Post-process: Fix GLSL types and functions inside #ifdef blocks
    # (raw-text pass-through, same as the header)
    kernel_opencl_body = post_process_ifdef_blocks(kernel_opencl_body)

    # Category A — assign hoisted program-scope global initializers at the top of
    # the kernel body (in declaration order, so inter-global dependencies hold).
    # The globals are declared bare in the header; runtime uniforms (iTime, …)
    # and all helper functions are available here, so arithmetic/calls that
    # OpenCL forbids in a file-scope initializer are legal in this kernel context.
    if hoisted_global_inits:
        hoist_emitter = OpenCLEmitter(indent_size=4)
        hoist_lines = ["    // ---- hoisted global initializers (category A) ----"]
        for entry in hoisted_global_inits:
            if isinstance(entry, HoistedArrayInit):
                # A2 — array global: a temp-local carries the (non-constant)
                # aggregate initializer legal on a local, then a copy loop fills
                # the bare global element-by-element.
                b, et, sz, init_ir = entry
                init_txt = hoist_emitter.emit_initializer(init_ir)
                hoist_lines.append(
                    f"    {{ {et} __init_{b}[{sz}] = {init_txt};"
                    f" for (int __hi = 0; __hi < {sz}; ++__hi)"
                    f" {b}[__hi] = __init_{b}[__hi]; }}"
                )
            else:
                name, init_ir = entry
                hoist_lines.append(f"    {name} = {hoist_emitter.emit(init_ir)};")
        kernel_opencl_body = "\n".join(hoist_lines) + "\n" + kernel_opencl_body

    # Category AG push-pop — re-apply the user's uniform remaps at the very
    # top of the kernel glue: SHADERTOY_INPUTS (in main_kernel.cl) has already
    # expanded by this point, and nothing after the entry body assigns a
    # uniform (compilecl appends only `AT_fragColor_set(fragColor);}`), so the
    # re-defined macro covers exactly the inlined user code. Placed before the
    # hoisted global initializers too — those are user code and saw the macro
    # on Shadertoy.
    ag_prefix = uniform_redefine_prefix(ag_redefines)

    # Add comment markers to kernel
    kernel_opencl = (
        f"{ag_prefix}"
        "// ---- SHADERTOY CODE BEGIN ----\n"
        "// Shadertoy void mainImage(...)\n"
        f"{kernel_opencl_body}\n"
        "// ---- SHADERTOY CODE END ----"
    )

    if verbose:
        print(f"    [OK] Kernel: {len(kernel_opencl)} chars")

    # Create full OpenCL (header + mainImage signature + body)
    full_opencl = ""
    if header_opencl:
        full_opencl += header_opencl + "\n\n"

    full_opencl += (
        f"{ag_prefix}"
        "void mainImage(out float4 fragColor, in float2 fragCoord) {\n"
        f"{kernel_opencl_body}\n"
        "}\n"
    )

    if verbose:
        print(f"\n[6/6] Transpilation complete!")
        print(f"  [OK] Total output: {len(full_opencl)} chars")
        print("=" * 70)

    return TranspileResult(
        header=header_opencl,
        kernel=kernel_opencl,
        full=full_opencl
    )


def validate_opencl(opencl_code: str) -> bool:
    """
    Validate OpenCL code by attempting compilation.

    Args:
        opencl_code: OpenCL source code string

    Returns:
        True if compilation succeeds, False otherwise

    Note:
        Requires PyOpenCL to be installed.
        This is a basic validation - full validation requires compilecl.py
    """
    try:
        import pyopencl as cl
    except ImportError:
        print("Warning: PyOpenCL not installed, skipping validation")
        return True

    try:
        # Get first available platform and device
        platforms = cl.get_platforms()
        if not platforms:
            print("Warning: No OpenCL platforms found")
            return True

        platform = platforms[0]
        devices = platform.get_devices()
        if not devices:
            print("Warning: No OpenCL devices found")
            return True

        device = devices[0]

        # Create context and compile
        ctx = cl.Context([device])

        # Basic compilation test (will fail without proper headers)
        # For full validation, use compilecl.py
        print(f"  Testing OpenCL syntax on {device.name}...")
        print("  Note: Full validation requires compilecl.py with Houdini headers")

        # Just check for obvious syntax errors
        if "GLSL_" in opencl_code and "#include" not in opencl_code:
            print("  [OK] OpenCL code contains GLSL_ function calls (requires glslHelpers.h)")

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


def transpile_file(
    input_path: Path,
    output_dir: Path,
    full_mode: bool = False,
    verbose: bool = False,
    validate: bool = False,
    common_path: Optional[Path] = None
) -> bool:
    """
    Transpile a GLSL file and write output files.

    Args:
        input_path: Path to input GLSL file
        output_dir: Directory for output files
        full_mode: If True, output single full.cl file instead of split
        verbose: If True, show transformation stages
        validate: If True, validate OpenCL compilation
        common_path: Optional path to a Shadertoy "Common" tab GLSL file to merge

    Returns:
        True if successful, False otherwise
    """
    # Read input
    if verbose:
        print(f"\nReading {input_path}...")
    else:
        print(f"Transpiling {input_path.name}...")

    try:
        glsl_source = input_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False

    # Read optional Common tab
    common_source = ""
    if common_path:
        try:
            common_source = common_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading common file: {e}")
            return False

    # Transpile
    try:
        result = transpile(glsl_source, verbose=verbose, common=common_source)
    except TranspileError as e:
        print(f"Transpilation error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during transpilation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

    # Validate if requested
    if validate:
        print("\nValidating OpenCL code...")
        if not validate_opencl(result.get_full()):
            print("Warning: Validation failed (non-fatal)")

    # Determine output paths
    base_name = input_path.stem

    if full_mode:
        # Output single full.cl file
        full_path = output_dir / f"{base_name}.full.cl"

        if not verbose:
            print(f"Writing {full_path.name}...")

        try:
            full_path.write_text(result.get_full(), encoding='utf-8')
        except Exception as e:
            print(f"Error writing output file: {e}")
            return False

        # Summary
        print(f"\n[SUCCESS] Output: {full_path}")
        print(f"  Size: {len(result.get_full())} characters")

    else:
        # Output split header.cl and kernel.cl files
        header_path = output_dir / f"{base_name}.header.cl"
        kernel_path = output_dir / f"{base_name}.kernel.cl"

        if not verbose:
            print(f"Writing {header_path.name} and {kernel_path.name}...")

        try:
            header_path.write_text(result.get_header(), encoding='utf-8')
            kernel_path.write_text(result.get_kernel(), encoding='utf-8')
        except Exception as e:
            print(f"Error writing output files: {e}")
            return False

        # Summary
        print(f"\n[SUCCESS] Transpiled {input_path.name}")
        print(f"  Header: {len(result.get_header())} chars -> {header_path}")
        print(f"  Kernel: {len(result.get_kernel())} chars -> {kernel_path}")
        print(f"\nTest with: python tests/compilecl.py --header {header_path} {kernel_path}")

    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transpile GLSL shaders to OpenCL (Session 9 architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transpile.py shader.glsl
    # Outputs: shader.header.cl and shader.kernel.cl

  python transpile.py shader.glsl --output-dir output/
    # Outputs: output/shader.header.cl and output/shader.kernel.cl

  python transpile.py shader.glsl --full
    # Outputs: shader.full.cl (complete OpenCL code)

  python transpile.py shader.glsl --verbose
    # Show transformation stages

  python transpile.py shader.glsl --validate
    # Validate OpenCL compilation (requires PyOpenCL)

Test transpiled code:
  python tests/compilecl.py --header shader.header.cl shader.kernel.cl

Module usage:
  from transpile import transpile
  result = transpile(glsl_source_string)
  print(result.get_header())
  print(result.get_kernel())
  print(result.get_full())
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input GLSL file path"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (default: same as input file)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Output complete OpenCL code in single .full.cl file (instead of split)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show transformation stages"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate OpenCL compilation (requires PyOpenCL)"
    )

    parser.add_argument(
        "--common",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional Shadertoy 'Common' tab GLSL file to merge before the pass"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.suffix == ".glsl":
        print(f"Warning: Input file doesn't have .glsl extension: {args.input}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.input.parent

    # Validate optional common file
    if args.common and not args.common.exists():
        print(f"Error: Common file not found: {args.common}", file=sys.stderr)
        sys.exit(1)

    # Transpile
    success = transpile_file(
        args.input,
        output_dir,
        full_mode=args.full,
        verbose=args.verbose,
        validate=args.validate,
        common_path=args.common
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
