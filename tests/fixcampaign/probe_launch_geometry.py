#!/usr/bin/env python
"""Launch-geometry probe: is fragCoord == get_global_id() in a real HDA cook?

The category-Q fix (Session 56, gid-derived gl_FragCoord accessor in
glslHelpers.h) rests on this geometry: pixel == get_global_id() + a UNIFORM
offset. The runtime offset seed self-corrects any uniform change, so the only
thing that can silently break rendering is a NON-uniform mapping (e.g. a
Copernicus redesign cooking one work-item per TILE of pixels).

RUN THIS AFTER EVERY HOUDINI VERSION/BUILD CHANGE (and after regenerating
main_header.cl / main_kernel.cl from a new Houdini):

    python tests/fixcampaign/probe_launch_geometry.py     # exit 0 = geometry OK

It cooks a diagnostic kernel through the real hShadertoy HDA at several
resolutions (rop_image, colorconversion=raw) and decodes the PNGs:
  R = 0.5 + (fragCoord.x - get_global_id(0))/16   -> must be FLAT (uniform)
  G = 0.5 + (fragCoord.y - get_global_id(1))/16   -> must be FLAT (uniform)
  B = 1 if get_global_size() == iResolution       -> must be 1 everywhere
Any gradient in R/G, or B != 1, means the gid->pixel mapping changed:
STOP and re-evaluate the Q accessor design (fallback: branch
fix/q-fragcoord-threading, the geometry-independent call-graph threading).

Needs: a free Houdini license, PIL + numpy in the system python.
HYTHON env var overrides hython discovery (same rule as houdini_smoke.py).
First verified: 2026-07-17 on Houdini 22.0.368 (offset 0 exactly, gsize==res
exactly, at 512x288 / 1024x576 / 513x289 / 2048x1152).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_S = 600
RESOLUTIONS = [(512, 288), (1024, 576), (513, 289), (2048, 1152)]

HYTHON_STAGE = r'''
import os
import sys
from pathlib import Path

import hou

ROOT = Path(os.environ['HSHADERTOY_ROOT'])
OUT_DIR = Path(sys.argv[1])
RESOLUTIONS = [(512, 288), (1024, 576), (513, 289), (2048, 1152)]

PROBE_FN = r"""
// ---- LAUNCH GEOMETRY PROBE ----
static float4 probe_diag(float2 fc, float3 res) {
    float offx = fc.x - (float)get_global_id(0);
    float offy = fc.y - (float)get_global_id(1);
    return (float4)(
        clamp(0.5f + offx / 16.0f, 0.0f, 1.0f),
        clamp(0.5f + offy / 16.0f, 0.0f, 1.0f),
        ((fabs((float)get_global_size(0) - res.x) < 0.5f) &&
         (fabs((float)get_global_size(1) - res.y) < 0.5f)) ? 1.0f : 0.0f,
        1.0f);
}
"""

PROBE_KERNEL = """
@KERNEL
{
    SHADERTOY_INPUTS
    fragColor = probe_diag(fragCoord, iResolution);
    @fragColor.set(fragColor);
}
"""

hou.hda.installFile(str(ROOT / 'houdini' / 'otls' / 'hShadertoy.hda'))
copnet = hou.node('/obj').createNode('copnet', 'probe_net')
node = copnet.createNode('hShadertoy::shadertoy', 'probe')
hdr = node.parm('code_header')
hdr.set(hdr.evalAsString() + PROBE_FN)
node.parm('code_rp0').set(PROBE_KERNEL)

rop = copnet.createNode('rop_image', 'probe_rop')
rop.parm('coppath').set(node.path())
cc = rop.parm('colorconversion')
cc.set('raw' if 'raw' in cc.menuItems() else 'ocio')
rop.parm('mkpath').set(True)

failures = 0
for (xr, yr) in RESOLUTIONS:
    node.parmTuple('init_iResolution').set((xr, yr, 0))
    out = OUT_DIR / f'probe_{xr}x{yr}.png'
    rop.parm('copoutput').set(str(out))
    err = ''
    try:
        rop.render(frame_range=(1, 1))
    except Exception as e:
        err = f'render raised: {e}'
    for n in (node,) + node.allSubChildren() + (rop,):
        try:
            msgs = n.errors()
        except Exception:
            continue
        if msgs:
            err += f' | [{n.path()}] ' + ' ; '.join(m.strip() for m in msgs)
    if err or not out.exists():
        failures += 1
        print(f'FAIL {xr}x{yr}: {err or "no output written"}')
    else:
        print(f'OK   {xr}x{yr}')
sys.exit(1 if failures else 0)
'''


def find_hython() -> Path:
    env = os.environ.get('HYTHON')
    if env:
        return Path(env)
    sfx = Path('C:/Program Files/Side Effects Software')
    candidates = sorted(sfx.glob('Houdini */bin/hython.exe'))
    if not candidates:
        sys.exit(f"ERROR: no hython.exe found under {sfx} (set HYTHON)")
    return candidates[-1]


def decode(out_dir: Path) -> int:
    import numpy as np
    from PIL import Image

    bad = 0
    for (xr, yr) in RESOLUTIONS:
        png = out_dir / f'probe_{xr}x{yr}.png'
        a = np.asarray(Image.open(png)).astype(np.float32) / 255.0
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        # raw-mode PNG write linearizes: flat 0.5 lands near 0.214-0.216.
        uniform = r.std() < 1e-4 and g.std() < 1e-4
        sizes_ok = b.min() > 0.99
        verdict = 'OK' if (uniform and sizes_ok) else 'GEOMETRY CHANGED'
        print(f'{png.name}: R std={r.std():.6f} G std={g.std():.6f} '
              f'B min={b.min():.3f} -> {verdict}')
        if not (uniform and sizes_ok):
            bad += 1
    return bad


def main() -> None:
    hython = find_hython()
    out_dir = Path(tempfile.mkdtemp(prefix='hst_geom_probe_'))
    stage = out_dir / 'probe_stage.py'
    stage.write_text(HYTHON_STAGE, encoding='utf-8')

    env = os.environ.copy()
    env['HSHADERTOY_ROOT'] = str(REPO)
    env['HOUDINI_OCL_PATH'] = f'{REPO}/houdini/ocl;&'  # ;& or hython segfaults
    print(f'probe: {hython}')
    proc = subprocess.run(
        [str(hython), str(stage), str(out_dir)],
        env=env, timeout=TIMEOUT_S, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(f'probe render stage failed (exit {proc.returncode})')

    bad = decode(out_dir)
    if bad:
        sys.exit(f'{bad} resolution(s) show a NON-1:1 gid->pixel mapping. '
                 'The gid-derived gl_FragCoord accessor is unsafe on this '
                 'Houdini build - see module docstring.')
    print('launch geometry OK: pixel == get_global_id() + uniform offset')


if __name__ == '__main__':
    main()
