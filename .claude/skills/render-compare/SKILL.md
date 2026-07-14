---
name: render-compare
description: Build/run the automated image-comparison harness that renders original Shadertoy GLSL via wgpu-shadertoy and hShadertoy output via Houdini, then diffs the images to prove render CORRECTNESS (not just compile success). Use when validating that transpiled shaders render the same as the originals, building the comparison pipeline, or investigating "compiles but looks wrong" bugs.
---

# Render-correctness comparison (wgpu-shadertoy ⇄ hShadertoy)

> **STATUS: BUILT & CALIBRATED (2026-07-11).** Home: `tests/rendercompare/`
> (its README is the usage doc of record). The compile campaigns prove
> shaders *compile*; this proves they *render the same* — perceptually, not
> pixel-exact.

## Usage

```bash
# fixed 3-shader end-to-end gate (gradient + london + digits); exit!=0 on fail
python tests/rendercompare/rc.py smoke

# mass campaign, resumable at every stage (state: tests/rendercompare/ledger.json)
python tests/rendercompare/rc.py run --tier 2 --limit 50
# or stage-by-stage: select / render-ref / render-hda / compare / report
```

Selection reads `tests/campaign/ledger.json` (overall==PASS only) and code
comes from `tests/campaign/cache/` — zero API calls. Tier 2 = single image
pass, no iChannels (408 candidates); tier 3 = + texture iChannels served
from the local media mirror `houdini/pic/media/` (233). Multipass is out of
scope until wgpu-shadertoy buffer support matures.

**Always eyeball `tests/rendercompare/contact_sheet.html`** (ref | hda | diff
triplets, worst first) before trusting gates. `ledger.json`, `REPORT.md`,
`contact_sheet.html`, `artifacts/` are generated — never hand-edit.

## The contract (both sides identical — details in `tests/rendercompare/common.py`)

800×450, FPS 60, probe frames **601** (iTime 10.0 s) and **151** (iTime
2.5 s) — frame ≥100 so sims have played out and fade-from-black intros are
over. `iFrame` = the **Houdini frame number** (HDA binds `$FF`), so the ref
gets 601, not 600. Verdict gates (in `compare.py`): SSIM + dMAE (MAE after
~10× box downsample — integrates away half-pixel shifts and filtering
noise). PASS ≥0.85 SSIM & ≤4 dMAE; RGB only (site forces alpha=1 on Image).

## Hard-won calibration facts (do not rediscover)

- **Orientation & color already match** — no y-flip, no transfer curve
  difference. wgpu snapshot rows are top-down display order; the `rop_image`
  PNG (OCIO "Automatic", HDA output "sRGB - Texture") matches it. The
  gradient shader in `rc.py smoke` guards this permanently.
- **Headless Houdini textures read BLACK without `HSHADERTOY_HOUDINI`.** The
  iChannel file COPs resolve `$HSHADERTOY_HOUDINI/pic/named/...`; headless
  mode doesn't apply the package env, and the cook still reports success.
  `render_hda.py` sets it (plus `HOUDINI_OCL_PATH='<repo>/houdini/ocl;&'` —
  the `;&` segfault rule from houdini_smoke.py applies).
- **The builder now forwards API sampler settings** (vflip/wrap/filter) to
  the HDA channel parms (`builder.py`, found by this harness — HDA menu
  tokens: wrap Clamp='1' Wrap='3', filter 0/1/2=nearest/linear/mipmap, vflip
  is site-semantics direct). Unit tests: `tests/unit/test_builder_sampler_params.py`.
- **hython one-shot probes** (parm menus etc.): create a tiny script, run it
  with `HSHADERTOY_ROOT` + `HOUDINI_OCL_PATH` set — see the parm_menu_probe
  pattern in the 2026-07-11 session.

## Known accepted divergences (flagged per shader in the ledger)

`uses_iDate` (both sides datetime.now(), seconds drift), `uses_iTimeDelta`
(HDA 0.0 vs ref 1/60), `uses_iMouse` (HDA has a baked ~0..1 px demo anim),
`uses_textureLod`/`uses_derivatives`, `input_*` (webcam/video/volume/…).
Site-contract divergences (half-pixel fragCoord, alpha, `vec4(1e20)`
unwritten fragColor, …): `docs/handover/SHADERTOY_SITE_NOTES.md` §2 — when a
mismatch reproduces one of those signatures, attribute it there before
hunting new transpiler bugs.

## Fix-campaign hookup

`rc.py smoke` is the render-correctness complement to
`tests/fixcampaign/houdini_smoke.py` (which cooks the full multipass wfffRN
stack but writes no images). Run both at end of a fix session; a smoke FAIL
is a regression — stop, root-cause, fix or revert. Systematic mismatch
categories found by mass runs feed the fix-campaign taxonomy (two-letter
codes extending the campaign scheme).

## Reference-side notes (wgpu-shadertoy)

Local editable checkout `C:\dev\wgpu_shadertoy` (PyPI 0.2.0 is broken vs
wgpu 0.31.1); env notes in `docs/WGPU-SHADERTOY.md`. Offscreen recipe:
`RENDERCANVAS_FORCE_OFFSCREEN=true` before import, `snapshot()` needs
`offscreen=True`, check `shader._format` for bgra. sampler3D/samplerCube/
iChannelTime unsupported; `/media/` is Cloudflare-blocked — textures come
from the repo mirror via `common.media_path_for_src()`.

## Related

- `tests/rendercompare/README.md` — usage doc of record
- `.claude/skills/houdini-testing/SKILL.md` — hython invocation details
- `.claude/skills/mass-test-campaign/SKILL.md` — the ledger/stage pattern this copies
- `docs/WGPU-SHADERTOY.md` — wgpu-shadertoy environment notes
