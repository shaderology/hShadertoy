"""
Production GLSL to OpenCL Transpiler for Houdini — Host B (real HDA deployment).

Thin adapter over the shared pipeline in
``src/glsl_to_opencl/host_pipeline.py`` (the single source of truth shared with
Host A ``tests/transpile.py`` — see that module's docstring for why the pipeline
was unified after repeated Host A/B drift). This host owns only:

  * its output FORMAT — Houdini's ``@KERNEL { SHADERTOY_INPUTS ... }`` wrapper
    (``_format_for_houdini``), and
  * its Common-tab STRATEGY — ``merge_common=False``: Houdini transpiles the
    Common tab as a SEPARATE node (``code_common``) injected before ``@KERNEL``,
    so Common is NOT merged into the pass; instead the pipeline harvests
    Common's out/inout signatures so call sites still take ``&`` of their args.

Everything semantic (normalize, preprocess, parse, transform, partition, S59
rescue, category-A hoisting, category-AG uniform push-pop) lives in the shared
core, so a fix there reaches BOTH hosts at once.
"""
import re
import sys
from pathlib import Path

# Ensure transpiler is importable
# This is needed because package PYTHONPATH not automatically applied in headless hython
sys.path.insert(0, 'C:/dev/hShadertoy')

from src.glsl_to_opencl.host_pipeline import (
    TranspileError,
    ENTRY_POINTS,
    transpile_pass,
    detect_renderpass_type as _detect_renderpass_type,
    harvest_common_signatures as _harvest_common_signatures,
    # Re-exported so existing importers keep working now that these live in the
    # shared core rather than in this host.
    normalize_entry_point as _normalize_entry_point,
    partition_translation_unit as _partition_translation_unit,
    post_process_ifdef_blocks as _post_process_ifdef_blocks,
    entry_param_names as _entry_param_names,
    entry_trapped_in_conditional as _entry_trapped_in_conditional,
)


class TranspilationError(Exception):
    """Raised when transpilation fails (Host B's public error type)."""
    pass


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


def transpile(glsl_source: str, mode: str = None, common: str = "") -> str:
    """
    Transpile GLSL shader code to Houdini-compatible OpenCL.

    Args:
        glsl_source: GLSL shader source code
        mode: Renderpass type ("mainImage", "mainCubemap", "Common", "mainSound")
              If None, auto-detect from source.
        common: Optional Shadertoy "Common" tab GLSL. Houdini transpiles Common
              as a separate node (code_common) and injects it before @KERNEL, so
              a renderpass never sees Common's function definitions. When given,
              the pipeline seeds the out/inout parameter signatures of Common's
              helpers so call sites to them pointerize inout/out arguments (`&x`)
              — WITHOUT re-emitting Common's declarations here.

    Returns:
        Complete OpenCL code formatted for Houdini HDA

    Raises:
        TranspilationError: If transpilation fails
    """
    if not glsl_source or not isinstance(glsl_source, str):
        raise TranspilationError("Invalid GLSL source: must be a non-empty string")

    try:
        tp = transpile_pass(
            glsl_source,
            mode=mode,
            common=common,
            merge_common=False,
            require_entry=False,
            hoist_indent="",
            bridge_indent="",
        )

        if tp.mode == "Common" or tp.entry_ir is None:
            # Common tab, or a pass with no entry: header-only, no @KERNEL body.
            return _format_for_houdini(tp.header_opencl, "", tp.mode)

        # Category-AG uniform remaps prefix the body (SHADERTOY_INPUTS, emitted
        # by _format_for_houdini just above, has already expanded).
        body = tp.ag_prefix + tp.body
        return _format_for_houdini(tp.header_opencl, body, tp.mode)

    except TranspilationError:
        raise
    except TranspileError as e:
        # Shared-core failure — surface with the core's granular message.
        raise TranspilationError(str(e)) from e
    except Exception as e:
        raise TranspilationError(
            f"Transpilation failed: {type(e).__name__}: {e}") from e
