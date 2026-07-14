"""
Houdini HDA smoke test for the fix campaign - one command, plain python.

Builds the canonical full-renderpass-stack shader wfffRN (BuffersAndTextures:
Image + Buffer A/B/C + Cube A + Common) in the hShadertoy HDA via hython and
force-cooks it (`hou.node().cook(force=True)`), so Houdini actually compiles
and runs the transpiled OpenCL of every renderpass - including the
mainCubemap/Common paths that the mainImage-only mass-test campaign never
exercises.

Usage (from repo root, end of every fix session, after the corpus delta):
    python tests/fixcampaign/houdini_smoke.py

Exit 0 = build + cook clean. Exit != 0 = regression on the Houdini side;
treat it like a corpus regression (stop, root-cause, fix or revert).

Override the hython location with the HYTHON env var if needed.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COOK_SCRIPT = (REPO_ROOT / 'houdini' / 'scripts' / 'python' / 'hshadertoy'
               / 'tests' / 'builder_cook_headless.py')
SHADER_JSON = (REPO_ROOT / 'resources' / 'examples' / 'BuffersAndTextures'
               / 'BuffersAndTextures_API.json')
TIMEOUT_S = 600  # hython startup (license checkout) + full OpenCL compile


def find_hython() -> Path:
    """HYTHON env var wins; otherwise newest Houdini under Side Effects."""
    env = os.environ.get('HYTHON')
    if env:
        return Path(env)
    sfx = Path('C:/Program Files/Side Effects Software')
    candidates = sorted(sfx.glob('Houdini */bin/hython.exe'))
    if not candidates:
        sys.exit(f"ERROR: no hython.exe found under {sfx} (set HYTHON)")
    return candidates[-1]


def main() -> int:
    hython = find_hython()
    env = os.environ.copy()
    # Headless mode does not apply the package env - set it explicitly.
    # The trailing ';&' KEEPS Houdini's default OCL search path ('&' expands
    # to it). Without it every built-in #import (imx.h, ...) breaks and the
    # APEX-implemented COPs then SEGFAULT hython instead of erroring.
    env['HSHADERTOY_ROOT'] = str(REPO_ROOT)
    env['HOUDINI_OCL_PATH'] = str(REPO_ROOT / 'houdini' / 'ocl') + ';&'

    cmd = [str(hython), str(COOK_SCRIPT), str(SHADER_JSON), 'Transpile']
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"ERROR: Houdini smoke test timed out after {TIMEOUT_S}s")
        return 1
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
