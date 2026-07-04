---
name: render-compare
description: Build/run the automated image-comparison harness that renders original Shadertoy GLSL via wgpu-shadertoy and hShadertoy output via Houdini, then diffs the images to prove render CORRECTNESS (not just compile success). Use when validating that transpiled shaders render the same as the originals, building the comparison pipeline, or investigating "compiles but looks wrong" bugs.
---

# Render-correctness comparison (wgpu-shadertoy ⇄ hShadertoy)

> **STATUS: SPEC — not yet built.** The compile campaigns prove shaders
> *compile*; nothing yet proves they *render correctly*. This skill is the
> implementation spec, written by the departing lead. Build it as described,
> in phases, each phase proven before the next. When it exists, replace this
> STATUS line with usage docs.

## Why this exists

`tests/campaign/` measures transpile+compile only (see its README "scope"
note). A shader can compile and still render garbage due to *semantic*
transpile bugs. The reference renderer is **wgpu-shadertoy** — a local,
already-working checkout at `C:\dev\wgpu_shadertoy` that renders original
Shadertoy GLSL headlessly. Its environment notes and gotchas are in
`docs/WGPU-SHADERTOY.md` (owner-verified 2026-07, treat as current).

## Reference side (works today — verified recipe)

```python
import os
os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "true"   # BEFORE importing
from wgpu_shadertoy import Shadertoy
shader = Shadertoy(code, resolution=(800, 450), offscreen=True)
frame = shader.snapshot(time_float=2.0, frame=120)     # HxWx4 numpy uint8
```

- Install: `pip install -e C:\dev\wgpu_shadertoy` (editable; the PyPI 0.2.0
  release is broken against wgpu 0.31.1 — `wgpu.gui` was removed upstream).
- `snapshot()` requires `offscreen=True`.
- Check `shader._format`: on this machine it's `rgba8unorm` (no swap); if it
  contains `bgra`, swap channels `arr[..., [2,1,0,3]]`.
- `Shadertoy.from_id("wtcSzN", ...)` downloads by id (uses `SHADERTOY_KEY` env
  var; Cloudflare rules in the shadertoy-api skill apply). Prefer feeding GLSL
  from `tests/campaign/cache/<id>.json` — zero API cost, same corpus the
  campaigns use.
- Limits: sampler3D/samplerCube/iChannelTime/iSampleRate unsupported; multipass
  renders the image pass but flags `complete=False`. **Media (`/media/`) cannot be
  downloaded** — source iChannel textures from the HDA's bundled assets
  (`houdini/scripts/python/hshadertoy/builder/hda/assets.json`).

## Test side (Houdini render via hython)

The houdini-testing skill documents the two headless scripts. **Phase-1 task:
extend `template_load_headless.py`** — it ALREADY force-cooks the OpenCL COPs
(`--cook opencl`), which is the hard part; it just doesn't write images and it
swallows cook errors (known bug, fix that first — see houdini-testing skill).
Do NOT build on `builder_test_headless.py`: that one deliberately never cooks
(the cook call is commented out in `builder.py` by design). Add: save the
cooked COP output to PNG + non-zero exit on cook errors. Verify with the
owner-blessed canonical shader `wfffRN` (BuffersAndTextures,
`resources/examples/BuffersAndTextures/`, template shape `..._HDA.json`) which
exercises buffers+cubemap+common.

## Comparison design (build exactly this, phased)

Home: `tests/rendercompare/` — mirror the campaign pattern: a `ledger.json`
source of truth, per-shader artifacts dir, idempotent CLI stages
(`render-ref`, `render-hda`, `compare`, `report`), resumable after any crash.

**Determinism contract (both sides identical):** resolution 800×450; fixed
`iTime=2.0`, `iFrame=120`; `iMouse=(0,0,0,0)`; `iDate` pinned; skip shaders
using true randomness/webcam/mic/video/keyboard.

**Metrics per shader:** MAE, PSNR, and SSIM over RGB (ignore alpha); store all
three in the ledger + a diff-heatmap PNG artifact. Suggested initial gates
(tune on data): PASS ≤2 MAE / ≥30 dB PSNR; WARN below that; FAIL <20 dB.
Always eyeball the contact sheet (a thumbnail grid of ref/test/diff triplets)
before trusting gates — a uniform 1-pixel y-shift can pass MAE while being a
real bug.

**Existing helper — partially fake, handle with care:**
`tests/helpers/image_comparison.py` already has real `compute_mse`/
`compute_psnr`, BUT its `compute_ssim` is a placeholder (`1/(1+mse)` — NOT
SSIM) and `save_comparison_image` raises `NotImplementedError`. Reuse the real
parts; replace the SSIM with `skimage.metrics.structural_similarity` (or drop
SSIM) — never let the placeholder write into the ledger.

**Known pitfalls to design for (check these FIRST when images differ):**
1. **Y-flip**: Shadertoy `fragCoord` origin is bottom-left; numpy arrays are
   top-down; Houdini COPs have their own convention. Calibrate once with a
   gradient shader (e.g. `fragColor = vec4(uv, 0, 1)`), bake the flip into the
   harness, and add that gradient as a permanent self-test.
2. **Color space**: wgpu snapshot is sRGB-encoded uint8; a Houdini COP render
   may be linear float. Convert both to linear float before metrics.
3. **Known semantic divergences** (real transpiler-bug hunting ground): GLSL
   `mod` vs OpenCL `fmod` on negatives; vector relational true = 1 (GLSL) vs
   -1 (OpenCL masks); derivative fns (`dFdx`/`fwidth`) in a non-raster
   context; texture filtering/wrap defaults; `precision`/fast-math flags.
4. **Time-dependence**: two frames (t=2.0 and t=7.3) per shader cheaply
   catches "correct at t=0 only" bugs.

**Corpus rollout order:** (1) the calibration gradient + `wfffRN`; (2) all
campaign-PASS shaders with a single image pass and NO iChannels (fully
deterministic, no assets); (3) texture-using shaders via bundled HDA assets;
(4) multipass — needs wgpu-shadertoy multipass support to mature first,
deprioritize.

## Phased build plan (one session each, TDD per project rules)

- **P1**: hython render-to-PNG for one shader + the y-flip/color calibration
  pair. Exit: `wfffRN` and the gradient render from both sides, aligned.
- **P2**: `tests/rendercompare/compare.py` metrics + unit tests on synthetic
  arrays (known-MAE fixtures). Exit: pytest green.
- **P3**: CLI + ledger + contact-sheet report over corpus tier 2 (~100
  shaders). Exit: ranked mismatch report, reviewed with owner.
- **P4**: triage loop — each systematic mismatch becomes a classified bug
  category (extend the campaign taxonomy letter scheme, two-letter codes) and
  feeds the fix-campaign workflow.

## Related

- `.claude/skills/houdini-testing/SKILL.md` — hython invocation details
- `.claude/skills/shadertoy-api/SKILL.md` — Cloudflare/API rules
- `.claude/skills/mass-test-campaign/SKILL.md` — the ledger/stage pattern to copy
- `docs/WGPU-SHADERTOY.md` — full wgpu-shadertoy environment notes
