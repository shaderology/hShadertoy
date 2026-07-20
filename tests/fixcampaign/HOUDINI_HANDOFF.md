# Houdini "handoff" — mostly NOT needed (runtime headers are LIVE)

**Key fact (verified 2026-06-25):** the runtime headers in
`houdini/ocl/include/` (`glslHelpers.h`, `textureHelpers.h`, `matrix_ops.h`,
`matrix_types.h`) are **live-`#include`d** by BOTH:
- the **Houdini HDA** — its `code_header` does `#include "textureHelpers.h"`,
  resolved via the package env `HOUDINI_OCL_PATH = C:/dev/hShadertoy/houdini/ocl`
  (Houdini searches `<path>/include`); and
- the **campaign** — `tests/ocl/main_header.cl` does `#include "textureHelpers.h"`
  / `glslHelpers.h`, resolved via `tests/build_options.json`'s
  `-I C:/dev/hShadertoy/houdini/ocl/include`.

Neither flattens/embeds a copy (verified: 0 helper bodies inside `main_header.cl`).
**So editing one of those four headers takes effect immediately in both
the campaign AND live Houdini renders — NO HDA regeneration, NO handoff.**

**Correction (verified 2026-07-04):** `shadertoyInputs.h` is NOT one of them —
nothing `#include`s it (0 hits in the HDA binary and repo-wide). It is a
read-only documentation MIRROR of the HDA `code_header`; editing it changes
nothing at runtime. Treat its content as owner-handoff territory.

The ONLY things that DO require the owner to regenerate `tests/ocl/main_header.cl`
from the HDA (and live in the OTL): the HDA `code_header` **structure** itself —
the `#bind` lines, the `static` global decls (`iResolution`, `iTime`, …), the
`SHADERTOY_INPUTS` macro, `shadertoy_cubemap`, `DO_CUBEMAP`. A fix touching THOSE
is a real handoff; a fix touching the included `.h` helpers is not.

---

## Log

### code_header restructure + Houdini 22 recapture RUNBOOK (Design C) — 2026-07-17, **REAL HANDOFF, MERGED to main**
Design + proof: `tests/fixcampaign/DESIGN_C_header_restructure.md` (merged;
branch `fix/header-bindinputs-fragcoord`). Owner plan: adopt in **Houdini 22**
and regenerate `main_header.cl` + `main_kernel.cl` from H22 (big Copernicus
update — the generated kernel code may have been redesigned). Full sequence:

**Part 1 — update the HDA `code_header` (H22 GUI, one regeneration)**
1. In H22, edit the `hShadertoy::shadertoy` asset (Asset Manager → Type
   Properties, or your usual HDA edit flow) and replace the `code_header`
   parameter DEFAULT with the contents of `houdini/ocl/include/shadertoyInputs.h`
   from merged main, **verbatim** — it is the exact `@`-form source; the VEX
   backtick initializers must survive the paste (spot-check `` `fpadzero ``,
   `` `rint ``, `ch('init_` appear in the parm afterwards). What changes:
   - uniform assignments move into `shadertoy_bind_inputs(...)` (AG root-cause);
   - `SHADERTOY_INPUTS` = one `shadertoy_bind_inputs(..., (int2)(@ix, @iy));`
     call (the trailing `int2 in_pix_base` seeds `GLSL_glFragCoord_off`,
     making the category-Q accessor transpiler-independent) + the unchanged
     `fragCoord`/`fragColor`/`DO_CUBEMAP` lines;
   - `DO_CUBEMAP` → `shadertoy_cubemap_bind(...)` wrapper; the wrapper function
     itself sits after `shadertoy_cubemap(...)`.
   `code_rp{N}`/`@KERNEL` need NO change. Prerequisite already on main:
   `glslHelpers.h` carries `GLSL_glFragCoord_off`/`GLSL_glFragCoord()` (live
   include — H22 picks it up via `HOUDINI_OCL_PATH`, no HDA action).
2. Sanity in the GUI: default node cooks (gradient), then build `wfffRN`
   (BuffersAndTextures — exercises the cubemap wrapper + Common + buffers).

**Part 2 — recapture `main_header.cl` / `main_kernel.cl` from H22**
3. Set `HOUDINI_OCL_REPORT_BUILD_LOGS = 2` in `houdini.env`, cook the HDA —
   Houdini logs the full assembled kernel source + the exact build options
   (this is how the current snapshots were made; see `tests/COMPILECL.md`).
4. Split the logged source at the repo's convention: everything through the
   expanded code_header (includes, `AT_*`/`_bound_*` `#bind` plumbing, statics,
   setter, macros, cubemap functions) → `tests/ocl/main_header.cl`; the
   `kernel void generickernel(...)` wrapper with the transpiled-code region
   replaced by the `// ---- SHADERTOY CODE BEGIN ----` marker comments →
   `tests/ocl/main_kernel.cl` (capture with the DEFAULT gradient shader so no
   real shader code needs stripping). Keep the closing-footer convention —
   `compilecl.py construct_kernel_source()` appends
   `AT_fragColor_set(fragColor);}`; if H22 changed that call's spelling,
   update `construct_kernel_source()` to match.
5. **Diff the fresh capture against the old one before committing.** The two
   Copernicus-redesign hotspots to inspect:
   - the work-item preamble — today
     `int _bound_gidx = get_global_id(0) * _bound_tilesize.x;` (tilesize was
     (1,1) on 22.0.368). If the formula changed or a tile LOOP appeared, the
     category-Q accessor geometry may be broken → run step 7 before anything
     else and see its docstring for the fallback;
   - renamed/new `_bound_*`/`AT_*` macros or kernel params (grep the transpiler
     emissions for `AT_ix`/`AT_iy` uses — `ast_transformer.py` emits them in
     the Q offset seed).
6. Update `tests/build_options.json` from the step-3 logged options: the
   Houdini include must point at H22 (**the current file still points at
   `C:/PROGRA~1/SIDEEF~1/HOUDIN~1.440` = Houdini 21.0.440!**). Use the 8.3
   short path (spaces break the option string):
   `cmd /c for %A in ("C:\Program Files\Side Effects Software\Houdini 22.0.368") do @echo %~sA`
   Refresh the `-D` defines from the log too (H22 may add/rename some).
   Then: `python tests/compilecl.py` (defaults) must compile the gradient.
7. **`python tests/fixcampaign/probe_launch_geometry.py`** — exit 0 required.
   This is the standing guard for the Q accessor's geometry assumption
   (pixel == get_global_id() + uniform offset). Run it after EVERY Houdini
   version change. If it fails: the gid-derived design is unsafe on that
   build; the proven geometry-independent fallback is branch
   `fix/q-fragcoord-threading` (78d01832).
8. Full validation of the recaptured snapshots (a header/kernel recapture
   changes the compile environment of every campaign shader):
   - `python -m pytest tests/unit/ -q` (guard test
     `test_header_uniform_redefine_guard.py` locks the setter invariant);
   - sample re-test: ~50 diverse PASS ids + the 6 Q winners + AG history ids
     (`XdyBW1 MtXBDf ldsBWl MsXfD7 lslXW8`), `--force`, ≤10-id batches, ledger
     backed up first, set-diff = 0 PASS→FAIL;
   - `houdini_smoke.py` + `rc.py smoke`, both exit 0;
   - then a FULL PASS-set re-test (background, hours — this is the real proof;
     start a fix-campaign session and it can babysit it).

**Part 3 — make `shadertoyInputs.h` LIVE (recommended after Parts 1-2 green)**
9. Set `HSHADERTOY_LIVE_HEADER=1` (package env or houdini.env): `builder.py`
   (merged) then populates `code_header` from `shadertoyInputs.h` at build
   time — the file becomes the single source of truth, future header fixes
   need no HDA regeneration. Existing scenes keep their baked parm until
   rebuilt.
10. Re-run step 8's quick gates once with the flag on; then consider making it
    the default and adding the drift unit test (HDA-ref default ==
    shadertoyInputs.h, whitespace-normalized) per DESIGN_C doc §6.

**Post-adoption cleanup (transpiler, any later session):** once the HDA
code_header carries the setter seed, the transpiler's gated entry-body
`GLSL_glFragCoord_off = ...` emission is redundant → retire it (keep the
helper-local `float4 gl_FragCoord = GLSL_glFragCoord();` injection). Nothing
breaks if this is deferred — the two seeds write identical values.

### texture(sampler, P, bias) 3-arg overloads (category M) — Session 6, LIVE
Added two additive overloads to `houdini/ocl/include/textureHelpers.h`
(float2+bias, float3+bias; bias ignored — no mipmaps in COPs). **+11 PASS, 0
regressed; campaign 451→462, M 94→74.** Merged to main (5b3bb49).
**Owner action: NONE.** Confirmed live in Houdini by rendering `4dXBW2`
(a 3-arg-texture shader) — works with no re-sync. (This file previously listed
re-sync steps; they were unnecessary — see the key fact above.)
