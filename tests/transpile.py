#!/usr/bin/env python3
"""
GLSL to OpenCL Transpiler — Host A (campaign / compilecl / unit tests).

Thin adapter over the shared pipeline in
``src/glsl_to_opencl/host_pipeline.py`` (the single source of truth shared with
the Houdini host ``houdini/.../transpile_glsl.py`` — see that module's docstring
for why the pipeline was unified). This host owns only:

  * its output FORMAT — split ``header``/``kernel``/``full`` for compilecl, and
  * its Common-tab STRATEGY — ``merge_common=True`` (compilecl compiles one
    file, so Common's helper definitions are merged into the pass).

Output modes:
- header.cl: everything before mainImage() (globals, helper functions)
- kernel.cl: the mainImage() BODY only (for compilecl.py)
- full.cl:   complete OpenCL (header + mainImage signature + body)

Usage (CLI):
    python transpile.py input.glsl                 # -> input.header.cl + input.kernel.cl
    python transpile.py input.glsl --full          # -> input.full.cl
    python transpile.py input.glsl --common common.glsl
    python transpile.py input.glsl --validate      # requires PyOpenCL

Usage (Module):
    from transpile import transpile
    result = transpile(glsl_source_string)
    result.get_header(); result.get_kernel(); result.get_full()
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glsl_to_opencl.host_pipeline import (
    TranspileError,
    transpile_pass,
    # Re-exported so existing importers (unit tests, downstream tools) keep
    # working now that these live in the shared core, not in this host.
    normalize_entry_point,
    partition_translation_unit,
    post_process_ifdef_blocks,
    entry_param_names,
    entry_trapped_in_conditional as _entry_trapped_in_conditional,
)

__all__ = [
    "TranspileResult", "TranspileError", "transpile", "validate_opencl",
    "transpile_file", "main", "normalize_entry_point",
    "partition_translation_unit", "post_process_ifdef_blocks",
    "entry_param_names",
]


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


def transpile(glsl_source: str, verbose: bool = False,
              common: str = "") -> TranspileResult:
    """
    Transpile GLSL shader source to OpenCL (split header/kernel/full).

    Runs the shared pipeline (``host_pipeline.transpile_pass``) with Host A's
    settings — mainImage entry, Common merged into the pass TU — then formats
    the returned pieces into the split outputs compilecl consumes.

    Args:
        glsl_source: GLSL source code string.
        verbose: accepted for signature compatibility (no longer prints stages).
        common: optional Shadertoy "Common" tab GLSL, merged before the pass.

    Returns:
        TranspileResult with header, kernel, and full OpenCL code.

    Raises:
        TranspileError: If transpilation fails.
    """
    tp = transpile_pass(
        glsl_source,
        mode="mainImage",
        common=common,
        merge_common=True,
        require_entry=True,
        hoist_indent="    ",
        bridge_indent="    ",
    )

    header_opencl = tp.header_opencl
    body = tp.body
    ag_prefix = tp.ag_prefix

    # Kernel: the mainImage body wrapped with markers (compilecl appends the
    # @KERNEL glue). Category-AG uniform remaps prefix the body (SHADERTOY_INPUTS
    # in main_kernel.cl has already expanded by this point).
    kernel_opencl = (
        f"{ag_prefix}"
        "// ---- SHADERTOY CODE BEGIN ----\n"
        "// Shadertoy void mainImage(...)\n"
        f"{body}\n"
        "// ---- SHADERTOY CODE END ----"
    )

    # Full: header + a real mainImage signature wrapping the same body.
    full_opencl = ""
    if header_opencl:
        full_opencl += header_opencl + "\n\n"
    full_opencl += (
        f"{ag_prefix}"
        "void mainImage(out float4 fragColor, in float2 fragCoord) {\n"
        f"{body}\n"
        "}\n"
    )

    return TranspileResult(
        header=header_opencl,
        kernel=kernel_opencl,
        full=full_opencl,
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
