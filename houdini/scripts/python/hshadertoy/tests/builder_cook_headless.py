"""
Builder + cook test for headless Houdini (hython).

Like builder_test_headless.py, but after building the HDA it force-cooks the
node (`hou.node().cook(force=True)`), which makes Houdini actually compile and
run the OpenCL kernels of every enabled renderpass. Unlike
template_load_headless.py, cook errors are collected, printed, and turn into
a non-zero exit code.

Usage:
    hython builder_cook_headless.py <path_to_shadertoy_api_json> [mode]

    mode: "Transpile" (default here - production transpiler) or "Template"

Example:
    HSHADERTOY_ROOT="C:/dev/hShadertoy" HOUDINI_OCL_PATH="C:/dev/hShadertoy/houdini/ocl;&" \
    hython builder_cook_headless.py \
        "C:/dev/hShadertoy/resources/examples/BuffersAndTextures/BuffersAndTextures_API.json"

Requires (headless mode does not apply the package env automatically):
    HSHADERTOY_ROOT   repo root (drives sys.path + explicit HDA install)
    HOUDINI_OCL_PATH  "<repo>/houdini/ocl;&" - resolves the #include'd runtime
                      headers at cook time. The trailing ';&' KEEPS Houdini's
                      default OCL search path; without it the built-in #imports
                      (imx.h, ...) break and the APEX-implemented COPs inside
                      the HDA SEGFAULT hython instead of erroring.

Prefer running via the wrapper (sets the env correctly, propagates exit code):
    python tests/fixcampaign/houdini_smoke.py
"""

import hou
import json
import os
import sys
from pathlib import Path

# Add hshadertoy paths to sys.path if not already present
# This is needed because package PYTHONPATH not automatically applied in headless mode
hshadertoy_root = hou.getenv('HSHADERTOY_ROOT')
if hshadertoy_root:
    hshadertoy_python = str(Path(hshadertoy_root) / 'houdini' / 'scripts' / 'python')
    if hshadertoy_python not in sys.path:
        sys.path.insert(0, hshadertoy_python)
    if hshadertoy_root not in sys.path:
        sys.path.insert(0, hshadertoy_root)

from hshadertoy.builder import build_shadertoy_hda


def _collect_errors(hda_node):
    """
    Gather cook errors from the HDA node and every node inside it.

    The OpenCL compile errors land on the internal OpenCL COPs, not on the
    HDA node itself, so we must walk allSubChildren().
    """
    errors = []
    for n in (hda_node,) + hda_node.allSubChildren():
        try:
            msgs = n.errors()
        except Exception:
            continue
        if msgs:
            errors.append((n.path(), " | ".join(m.strip() for m in msgs)))
    return errors


def main():
    # Load HDA file explicitly (needed in headless mode)
    if hshadertoy_root:
        hda_path = str(Path(hshadertoy_root) / 'houdini' / 'otls' / 'hShadertoy.hda')
        if Path(hda_path).exists():
            hou.hda.installFile(hda_path)
            print(f"Loaded HDA: {hda_path}\n")
        else:
            print(f"Warning: HDA not found at {hda_path}\n")

    if len(sys.argv) < 2:
        print("Error: No JSON file path provided")
        print("Usage: hython builder_cook_headless.py <path_to_json_file> [mode]")
        sys.exit(1)

    json_file_path = sys.argv[1]
    # Default to the production transpiler - this test exists to prove the
    # transpiled OpenCL actually compiles inside Houdini.
    mode = sys.argv[2] if len(sys.argv) > 2 else "Transpile"

    # The cook needs HOUDINI_OCL_PATH to resolve the runtime-header #includes.
    # Fail fast with a clear message instead of a wall of compile errors
    # (or worse, a segfault - see module docstring on the ';&' suffix).
    ocl_path = os.environ.get('HOUDINI_OCL_PATH') or hou.getenv('HOUDINI_OCL_PATH')
    if not ocl_path:
        print("Error: HOUDINI_OCL_PATH is not set - the OpenCL cook cannot "
              "resolve #include'd headers (set it to '<repo>/houdini/ocl;&')")
        sys.exit(1)
    if '&' not in ocl_path:
        print("Error: HOUDINI_OCL_PATH has no '&' entry - it would REPLACE "
              "Houdini's default OCL search path and the cook would segfault "
              "hython. Append ';&' to keep the defaults.")
        sys.exit(1)

    try:
        with open(json_file_path, 'r') as f:
            shader_json = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Building HDA from: {json_file_path}")
    print(f"Transpiler mode: {mode}")
    print(f"{'='*60}\n")

    try:
        node = build_shadertoy_hda(shader_json, mode=mode)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"BUILD FAILURE!")
        print(f"Error building HDA: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Node created at: {node.path()}")

    # The actual point of this script: force-cook the HDA so Houdini compiles
    # and executes the OpenCL kernels of every enabled renderpass.
    print(f"\nCooking {node.path()} (force=True)...")
    cook_exception = None
    try:
        hou.node(node.path()).cook(force=True)
    except Exception as e:
        cook_exception = e

    errors = _collect_errors(node)

    if cook_exception or errors:
        print(f"\n{'='*60}")
        print(f"COOK FAILURE!")
        if cook_exception:
            print(f"Cook raised: {cook_exception}")
        for path, msg in errors:
            print(f"\n[{path}]\n{msg}")
        print(f"{'='*60}\n")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"COOK SUCCESS!")
    print(f"Node cooked with no errors: {node.path()}")
    print(f"{'='*60}\n")

    shader_name = shader_json["Shader"]["info"]["name"]
    num_renderpasses = len(shader_json["Shader"]["renderpass"])
    print(f"Shader: {shader_name}")
    print(f"Renderpasses: {num_renderpasses}")


if __name__ == "__main__":
    main()
