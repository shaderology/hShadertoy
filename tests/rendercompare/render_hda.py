"""
Plain-python wrapper that runs hda_render_headless.py under hython with the
right environment (same rules as tests/fixcampaign/houdini_smoke.py).
Imported by rc.py; can also be run standalone:

    python tests/rendercompare/render_hda.py <manifest.json> <results.json>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HYTHON_SCRIPT = Path(__file__).resolve().parent / "hda_render_headless.py"


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


def hython_env() -> dict:
    env = os.environ.copy()
    env['HSHADERTOY_ROOT'] = str(REPO_ROOT)
    # The iChannel file COPs resolve textures via $HSHADERTOY_HOUDINI
    # (see iChannel_interior_ref.json: '$HSHADERTOY_HOUDINI/pic/named/...').
    # Headless mode does not apply the Houdini package env, so without this
    # every texture reads BLACK while the cook still reports success.
    env['HSHADERTOY_HOUDINI'] = str(REPO_ROOT / 'houdini')
    env['MEDIA_JSON_PATH'] = str(REPO_ROOT / 'houdini' / 'pic' / 'media.json')
    # Trailing ';&' KEEPS Houdini's default OCL search path - without it the
    # APEX-implemented COPs segfault hython (see houdini_smoke.py).
    env['HOUDINI_OCL_PATH'] = str(REPO_ROOT / 'houdini' / 'ocl') + ';&'
    return env


def run_chunk(manifest_path: str, results_path: str,
              timeout_s: int = 1800) -> int:
    cmd = [str(find_hython()), str(HYTHON_SCRIPT),
           str(manifest_path), str(results_path)]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=hython_env(), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"ERROR: hython chunk timed out after {timeout_s}s "
              f"(partial results kept in {results_path})")
        return 1
    return result.returncode


if __name__ == '__main__':
    sys.exit(run_chunk(sys.argv[1], sys.argv[2]))
