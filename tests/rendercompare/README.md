# Render-compare campaign (`tests/rendercompare/`)

Proves that transpiled shaders **render the same** as the originals — the
compile campaigns (`tests/campaign/`, `tests/fixcampaign/`) only prove they
compile. Reference = original GLSL rendered headlessly by **wgpu-shadertoy**
(`C:\dev\wgpu_shadertoy`, editable install); test = transpiled OpenCL rendered
by the **hShadertoy HDA** via hython + a `rop_image` (Copernicus) ROP.

Runs standalone and in parallel with the fix campaign: it reads the mass-test
ledger (`tests/campaign/ledger.json`) for selection and the local shader
cache (`tests/campaign/cache/`) for code — zero Shadertoy API calls.

## Usage

```bash
# fixed 3-shader end-to-end check (gradient + london + digits); exit!=0 on fail
python tests/rendercompare/rc.py smoke

# mass campaign, resumable at every stage
python tests/rendercompare/rc.py select --tier 2 --limit 50
python tests/rendercompare/rc.py render-ref          # wgpu, chunked subprocesses
python tests/rendercompare/rc.py render-hda          # hython, chunked, slow
python tests/rendercompare/rc.py compare
python tests/rendercompare/rc.py report              # REPORT.md + contact_sheet.html

# or everything at once
python tests/rendercompare/rc.py run --tier 2 --limit 50
```

Open `contact_sheet.html` in a browser and **eyeball it** before trusting the
gates — ref | hda | diff triplets, worst first.

Tiers: `2` = campaign-PASS, single image pass, no iChannels.
`3` = same but with texture iChannels (served from the local media mirror
`houdini/pic/media/`; shadertoy.com `/media/` is Cloudflare-blocked).
Multipass is out of scope until wgpu-shadertoy's buffer support matures.

## Determinism contract

Both renderers get identical uniforms (see `common.py`):
resolution **800×450**, FPS **60**, probe frames **601** (iTime 10.0 s) and
**151** (iTime 2.5 s). Frame ≥100 on purpose: sims have played out and
fade-from-black intros are over. `iFrame` is the **Houdini frame number**
(the HDA binds it to `$FF`), so the ref side is fed 601 — not 600.

Known, accepted divergences (flagged per shader in the ledger, not "fixed"):

| flag | why it diverges |
|---|---|
| `uses_iDate` | both sides use `datetime.now()` at render time — seconds drift |
| `uses_iTimeDelta` | HDA leaves it 0.0; ref side gets 1/60 |
| `uses_iMouse` | HDA has a baked ~0..1 px demo animation; ref uses (0,0,0,0) |
| `uses_textureLod` / `uses_derivatives` | unsupported/approximated in OpenCL |
| `input_*` (webcam, video, volume…) | not reproducible headlessly |

## Verdict = perceptual, not pixel-exact

Texture filtering, mip levels, half-pixel fragCoord and color-pipeline
differences make per-pixel diffs noisy even for perfect transpiles. Gates
(in `compare.py`) use **SSIM** (structure) + **dMAE** (mean abs error after
~10× box downsample, 0–255 units): PASS ≥0.85 / ≤4.0, WARN ≥0.60 / ≤12.0,
else FAIL. Plain MAE/PSNR are stored for forensics only. RGB only — alpha is
a known site-contract divergence (site forces alpha=1 on Image passes).

## Files

- `rc.py` — stage driver (this is the only entry point you need)
- `common.py` — contract constants, ledger IO, shader flags
- `compare.py` — metrics + diff heatmap
- `render_ref.py` — wgpu renderer (subprocess per chunk)
- `render_hda.py` + `hda_render_headless.py` — hython renderer (subprocess per chunk)
- `ledger.json` — generated state, **never hand-edit**
- `artifacts/<id>/{ref,hda,diff}_f<frame>.png` — generated images
- `REPORT.md`, `contact_sheet.html` — generated reports

## Calibration

`rc.py smoke` renders `fragColor = vec4(uv, 0, 1)`: if orientation or color
transfer of either renderer changes, the gradient fails loudly. The baked
flip constants live in `common.py` (`FLIP_HDA_Y`, `FLIP_REF_Y`).

## Fix-campaign hookup

`python tests/rendercompare/rc.py smoke` is the render-correctness
complement to `tests/fixcampaign/houdini_smoke.py` (which cooks the full
multipass stack wfffRN but writes no images). Run both at the end of a fix
session; a smoke FAIL is treated like a corpus regression.
