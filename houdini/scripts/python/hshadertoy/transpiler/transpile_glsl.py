"""
Production GLSL to OpenCL Transpiler for Houdini

Wraps the core transpiler (src/glsl_to_opencl) and formats output for
Houdini's @KERNEL structure.

Phase 2 of Houdini Integration
"""
import sys
from pathlib import Path

# Ensure transpiler is importable
# This is needed because package PYTHONPATH not automatically applied in headless hython
sys.path.insert(0, 'C:/dev/hShadertoy')

from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter
from src.glsl_to_opencl.preprocessor import PreprocessorTransformer


class TranspilationError(Exception):
    """Raised when transpilation fails"""
    pass


def _normalize_main_image(glsl_source: str) -> str:
    """
    Rewrite a custom-named mainImage to the standard fragColor/fragCoord form.

    Shadertoy permits custom parameter names (golf shaders often use
    ``void mainImage(out vec4 O, vec2 U)``), but the Houdini kernel always
    exposes ``fragColor``/``fragCoord``. Rather than renaming identifiers in the
    body (which risks clobbering same-named locals in nested scopes), we keep the
    body verbatim and bridge the names with alias declarations plus a final write
    to ``fragColor``::

        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec4 O = vec4(0.0, 0.0, 0.0, 1.0);   // out alias
            vec2 U = fragCoord;                  // in alias
            ...original body...
            fragColor = O;                       // final write
        }

    The downstream transform/split/pointer-fix is then name-agnostic. Standard
    signatures are returned unchanged.
    """
    parser = GLSLParser()
    try:
        ast = parser.parse(glsl_source)
    except Exception:
        return glsl_source

    main = None
    for node in ast.named_children:
        if node.type == "function_definition" and node.name == "mainImage":
            main = node
            break
    if main is None:
        return glsl_source

    params = main.parameters
    out_name, in_name = "fragColor", "fragCoord"
    if len(params) >= 1:
        decl = params[0].child_by_field_name("declarator")
        if decl is not None and decl.type == "identifier" and decl.text:
            out_name = decl.text.strip()
    if len(params) >= 2:
        decl = params[1].child_by_field_name("declarator")
        if decl is not None and decl.type == "identifier" and decl.text:
            in_name = decl.text.strip()

    if out_name == "fragColor" and in_name == "fragCoord":
        return glsl_source

    body = main.body
    if body is None:
        return glsl_source

    inner = body.text.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]

    out_alias = f"    vec4 {out_name} = vec4(0.0, 0.0, 0.0, 1.0);\n" if out_name != "fragColor" else ""
    in_alias = f"    vec2 {in_name} = fragCoord;\n" if in_name != "fragCoord" else ""
    finalize = f"\n    fragColor = {out_name};" if out_name != "fragColor" else ""

    new_fn = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        f"{out_alias}{in_alias}{inner}{finalize}\n"
        "}"
    )

    # Splice by byte offsets so surrounding code (helpers, globals, comments) is
    # preserved exactly.
    src_bytes = glsl_source.encode("utf-8")
    start = main._node.start_byte
    end = main._node.end_byte
    return src_bytes[:start].decode("utf-8") + new_fn + src_bytes[end:].decode("utf-8")


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


def _split_header_and_body(opencl_code: str, mode: str) -> tuple:
    """
    Split OpenCL code into header (declarations) and body (main function).

    Returns:
        (header, body) tuple

    For Common renderpass, body will be empty.
    """
    if mode == "Common":
        # Common renderpass is all header, no body
        return (opencl_code, "")

    # Find the main function
    # Pattern: void mainImage(...) { ... }
    # or: void mainCubemap(...) { ... }

    lines = opencl_code.split('\n')
    main_func_start = -1
    main_func_name = "mainImage" if mode == "mainImage" else mode

    # Find where main function starts
    for i, line in enumerate(lines):
        if f"void {main_func_name}" in line or f"float2 {main_func_name}" in line:
            main_func_start = i
            break

    if main_func_start == -1:
        # No main function found - treat as header only
        return (opencl_code, "")

    # Header: everything before main function
    header = '\n'.join(lines[:main_func_start]).strip()

    # Body: Extract function body (between { and })
    # Start from function signature line
    body_lines = []
    brace_count = 0
    in_body = False
    function_complete = False

    for i in range(main_func_start, len(lines)):
        if function_complete:
            break

        line = lines[i]
        line_to_add = []

        # Process each character to track braces
        for j, char in enumerate(line):
            if char == '{':
                brace_count += 1
                if brace_count == 1:
                    in_body = True
                    # Start capturing after the opening brace
                elif brace_count > 1:
                    # Nested brace - include it
                    line_to_add.append(char)
            elif char == '}':
                if brace_count == 1:
                    # This is the closing brace of the main function
                    in_body = False
                    function_complete = True
                    break
                else:
                    # Nested closing brace - include it
                    line_to_add.append(char)
                    brace_count -= 1
            elif in_body and brace_count > 0:
                # Regular character inside the function body
                line_to_add.append(char)

        # Add the processed line if we collected any content
        if line_to_add:
            body_lines.append(''.join(line_to_add))

    body = '\n'.join(body_lines).strip()
    return (header, body)


def _fix_houdini_pointers(body: str) -> str:
    """
    Replace pointer dereferences with direct variable access for Houdini.

    In Houdini's @KERNEL context, fragColor is a pre-defined variable,
    not a function parameter, so we don't use pointer syntax.

    Args:
        body: Function body with potential pointer dereferences

    Returns:
        Body with pointer syntax removed
    """
    # Replace *fragColor with fragColor
    body = body.replace("*fragColor", "fragColor")

    # May need to handle other out parameters in future
    # (e.g., *outColor, *result, etc.)

    return body


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
    import re

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
        body: Main function body
        mode: Renderpass type

    Returns:
        Complete Houdini-formatted OpenCL code
    """
    if mode == "Common":
        # Common renderpass: header only, no @KERNEL
        return header

    # Fix pointer syntax for Houdini
    body = _fix_houdini_pointers(body)

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
        # Normalize custom mainImage parameter names (e.g. `out vec4 O, vec2 U`)
        # to the standard fragColor/fragCoord via alias injection, so the rest of
        # the pipeline is name-agnostic.
        glsl_source = _normalize_main_image(glsl_source)

        # Auto-detect mode if not provided
        if mode is None:
            mode = _detect_renderpass_type(glsl_source)

        # Stage 1: Preprocess
        preprocessor = PreprocessorTransformer()
        glsl_processed = preprocessor.transform(glsl_source)

        # Stage 2: Parse
        parser = GLSLParser()
        ast = parser.parse(glsl_processed)

        # Stage 3: Type check (initialize)
        symbol_table = create_builtin_symbol_table()
        type_checker = TypeChecker(symbol_table)

        # Stage 4: Transform AST
        transformer = ASTTransformer(type_checker)
        # The renderpass entry function's out-param (fragColor) is a host @KERNEL
        # local, not a deref'able pointer (see ASTTransformer.entry_function).
        transformer.entry_function = mode
        transformed_ast = transformer.transform(ast)

        # Stage 5: Emit OpenCL
        emitter = OpenCLEmitter(indent_size=4)
        opencl_code = emitter.emit(transformed_ast)

        # Stage 6: Post-process to fix code inside #ifdef blocks
        # This is a workaround for Session 9's limitation where code inside #ifdef blocks
        # is not transformed by the AST transformer
        opencl_code = _post_process_ifdef_blocks(opencl_code)

        # Split into header and body
        header, body = _split_header_and_body(opencl_code, mode)

        # Format for Houdini
        houdini_code = _format_for_houdini(header, body, mode)

        return houdini_code

    except TranspilationError:
        # Re-raise our own errors
        raise
    except Exception as e:
        # Wrap unexpected errors with context
        error_msg = f"Transpilation failed: {type(e).__name__}: {e}"
        raise TranspilationError(error_msg) from e
