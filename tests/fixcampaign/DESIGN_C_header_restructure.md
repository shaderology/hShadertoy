# Design C — Header restructure: `shadertoy_bind_inputs()` setter + `gl_FragCoord` offset carrier

Status: **prototype, corpus-proven, ready for owner adoption into the HDA `code_header`.**
Branch: `fix/header-bindinputs-fragcoord` (cut from main @ b97284f3; merges
`fix/header-ag1-setter-main` and is reconciled with Design B's landed commit 801c2243).
Scope: infrastructure + adoption plan. Design C does **not** by itself flip any
category-Q shader (that needs the transpiler-side rewrite — Design B has landed one);
it carries the proven category-AG root-cause fix forward and makes the gl_FragCoord
offset seeding **transpiler-independent**.

---

## 1. What this changes and why

Two header-scoped problems share one root cause: **a value that must be visible to
program-scope helper functions, or an assignment that must survive a user `#define`,
cannot live in a macro that expands *after* the transpiled user code.**

- **Category AG** — a Shadertoy shader may `#define iTime ...` (legal there; uniforms
  are read-only). The HDA `SHADERTOY_INPUTS` macro expands *after* that `#define`, so
  `iTime = @Time;` becomes `<macro-body> = @Time;` → clang *"expression is not
  assignable"*. Root-cause fix = move every uniform assignment into a function
  (`shadertoy_bind_inputs`) **defined in the header, before any user code**, so the
  bare uniform tokens are compiled before a user `#define` can reach them. Values
  arrive as parameters from the `@`-bindings at the call site. (Proven on
  `fix/header-ag1-setter-main`, 5c379d93: full 862-shader PASS-set re-test, 0
  regressions, +2.)

- **Category Q** — `gl_FragCoord` read inside a *helper* function. `gl_FragCoord.xy`
  == `fragCoord` == this work-item's pixel coordinate, but `fragCoord` and `@ix/@iy`
  are **kernel-body locals**, invisible to helpers. A per-work-item value can't live in
  a program-scope global directly (last-writer-wins race). The orchestrator **proved**
  (diagnostic kernel cooked through the real HDA in Houdini 22.0.368 at 4 resolutions
  incl. odd + 2048×1152): `fragCoord == (float2)(get_global_id(0), get_global_id(1))`
  **exactly**, and `get_global_size() == iResolution` exactly. So we store the
  **offset** `pixel_base − global_id`, which is *uniform* across work-items (identical
  value → benign race, the same trick the header's existing statics rely on), and
  reconstruct each reader's own pixel in any function via `get_global_id() + offset`.

---

## 2. The restructured header (key diff hunks)

Applied to `tests/ocl/main_header.cl` (`AT_*` forms, HSHADERTOY BEGIN section only)
and mirrored to `houdini/ocl/include/shadertoyInputs.h` (`@` forms — the authoritative
code_header mirror).

### 2a. Setter — AG fix + Q carrier (defined before any user code)

```c
static void shadertoy_bind_inputs(
    float3 in_iResolution, float in_iTime, float in_iFrameRate, int in_iFrame,
    float4 in_iMouse, float4 in_iDate,
    const IMX_Layer* in_iChannel0, const IMX_Layer* in_iChannel1,
    const IMX_Layer* in_iChannel2, const IMX_Layer* in_iChannel3,
    float3 in_iChannelResolution0, float3 in_iChannelResolution1,
    float3 in_iChannelResolution2, float3 in_iChannelResolution3,
    int2 in_pix_base)                       // <- NEW: pixel base = (@ix, @iy)
{
    iResolution = in_iResolution;
    iTime = in_iTime;
    /* ... all uniform assignments, identical values/order to the old macro ... */
    iChannelResolution[3] = in_iChannelResolution3;
    // Category Q carrier: seed the uniform gid->pixel offset DECLARED IN
    // glslHelpers.h (included above; it also defines GLSL_glFragCoord()).
    GLSL_glFragCoord_off = in_pix_base - (int2)(get_global_id(0), get_global_id(1));
}
```

**Design decision — offset computed *in the setter*, not at the call site.** The
setter is itself a program-scope function, so it can call `get_global_id()`. The macro
only passes the pixel base `(int2)(@ix, @iy)` — no `get_global_id()` token in the
macro body, one place owns the arithmetic, and the AG "no poisonable token in the
macro" invariant is preserved.

### 2b. `SHADERTOY_INPUTS` — delegates, passes the pixel base

```c
#define SHADERTOY_INPUTS \
    shadertoy_bind_inputs( \
        (float3)(@xres, @yres, 0.0f), @Time, @iFrameRate, @iFrame, @iMouse, @iDate, \
        @iChannel0.layer, @iChannel1.layer, @iChannel2.layer, @iChannel3.layer, \
        (float3)(@iChannel0.res, 0.0f), (float3)(@iChannel1.res, 0.0f), \
        (float3)(@iChannel2.res, 0.0f), (float3)(@iChannel3.res, 0.0f), \
        (int2)(@ix, @iy)); \
    float2 fragCoord = @fragCoord; \
    if (!@fragCoord.bound) { fragCoord = (float2)(@ix, @iy); } \
    float4 fragColor = (float4)(0.0f, 0.0f, 0.0f, 1.0f); \
    DO_CUBEMAP
```

`DO_CUBEMAP` likewise delegates to a header-defined
`shadertoy_cubemap_bind(@ix,@iy,@xres,@yres,&rayDir)` wrapper so the poisonable
`&iResolution` token stays out of the macro body (AG fix, unchanged from the proven
setter branch).

### 2c. Accessor + offset — live in `glslHelpers.h` (Design B's landed hunk, adopted)

```c
static int2 GLSL_glFragCoord_off;

static float4 GLSL_glFragCoord(void) {
  return (float4)((float)((int)get_global_id(0) + GLSL_glFragCoord_off.x),
                  (float)((int)get_global_id(1) + GLSL_glFragCoord_off.y),
                  0.0f, 1.0f);
}
```

This branch cherry-picks that hunk verbatim (`git checkout 801c2243 --
houdini/ocl/include/glslHelpers.h`) so it is self-consistent standalone AND merges
byte-identically with Design B.

---

## 3. Design question answered — where does the accessor live?

**Original Design-C position:** the code_header, because the accessor is coupled to the
offset static the setter writes, and `glslHelpers.h` is `#include`d *above* the header's
statics (an accessor there could not see an offset declared below it).

**Final position (reconciled): `glslHelpers.h` — Design B's home wins.** Mid-prototype,
Design B **landed** (commit 801c2243, +6 corpus, 0 regressed) with
`GLSL_glFragCoord_off` + `GLSL_glFragCoord()` in `glslHelpers.h` — the coupling
objection dissolves because B declares the offset *itself* in glslHelpers.h, making the
include self-contained; the setter (later in the translation unit) can see and write
it. The collision was observed empirically, not hypothetically: this prototype's first
corpus run raced B's landing and shader ldsczf failed
`redefinition of 'GLSL_glFragCoord'` (my header duplicate vs glslHelpers.h line 80) the
moment 801c2243 hit the shared include dir. Duplicates removed; single definition
restored; ldsczf re-tested PASS.

What each design contributes after reconciliation:
- **glslHelpers.h (B):** the symbols. Live-editable, no HDA regeneration, zero-init
  default already correct under today's geometry.
- **code_header setter (C):** the **seed**. B seeds the offset via transpiler-emitted
  entry-body glue, *gated* on a helper actually using gl_FragCoord; C's setter seeds it
  unconditionally for **every kernel with no transpiler emission required**. Both
  writes are identical values (benign). Once C's header is adopted in the HDA, B's
  emitted seed becomes redundant and can be retired from the transpiler (keep the
  helper-local `float4 gl_FragCoord = GLSL_glFragCoord();` injection — that part stays
  necessary).

---

## 4. Proof (this worktree, branch state)

- **Unit suite:** `python -m pytest tests/unit/ -q` → **2137 passed, 6 skipped**
  (baseline 2134+6; +3 from the setter guard test
  `tests/unit/test_header_uniform_redefine_guard.py`, which asserts no bare uniform
  token survives in `SHADERTOY_INPUTS`/`DO_CUBEMAP` and that the macro delegates to the
  setter).
- **Default gradient kernel:** `compilecl.py` against the worktree header+kernel+includes
  → compiled successfully.
- **Category-Q helper-accessor demo**
  (`tests/ocl/q_fragcoord_helper_test_{header,kernel}.cl`): a program-scope helper
  `q_demo_shade()` calls `GLSL_glFragCoord()` and reads `iResolution`/`iTime`; the
  kernel body calls the helper after `SHADERTOY_INPUTS` (which runs the setter). →
  **compiled successfully** — the accessor is reachable from helpers with the
  setter-seeded offset. (Capability proof; flipping real Q shaders additionally needs
  B's transpiler-side injection.)
- **Corpus sample re-test** (worktree ledger copy, backed up first; `--force`; batches
  of ≤10; includes pinned to the worktree's own `houdini/ocl/include` for determinism):
  **59 shaders re-tested** —
  - 50 diverse currently-PASSING ids (15 with Common pass, 20 with iChannels, 15
    plain) → all still PASS.
  - AG-history ids `XdyBW1 MtXBDf ldsBWl MsXfD7 lslXW8` → all still PASS (the two
    AG cluster-1 fixes and the bare-identifier-#define pattern shaders survive the
    setter header).
  - `ltKSRG 4dsfRn` (AG-adjacent, failing on other blockers) → still FAIL, unchanged.
  - Q ids `3dK3zR Mt3GDl 3t2GRD` → still COMPILE_FAIL with the **identical** error
    (`use of undeclared identifier 'gl_FragCoord'`) at identical columns; only line
    numbers shifted by the header's added lines. Failure semantically unchanged.
  - **Ledger set-diff: PASS 1354 → 1354, REGRESSED = 0, FIXED = 0.** Exactly the
    expected null delta for an infrastructure-only header change.
  - The ledger/artifacts were then restored to HEAD — this branch ships **no
    generated-file churn** (the proof lives here; B's branch owns the Q ledger flips).

---

## 5. The tiling caveat (honest limitation)

The offset is uniform across work-items **iff** `pixel_base − global_id` is constant.
With `AT_ix == _bound_gidx == get_global_id(0) * tilesize.x`, the offset is
`get_global_id(0) * (tilesize.x − 1)` — constant **only when `tilesize == 1`**, which
is the proven cook geometry (`fragCoord == get_global_id()` ⇒ tilesize 1 ⇒ offset 0).
Under a hypothetical `tilesize > 1` runover the offset would be non-uniform and the
accessor would return wrong pixels — that regime needs per-tile call-graph threading,
not this carrier. The carrier self-corrects any *uniform* launch-offset shift (a global
NDRange offset applied equally to every work-item). `rc.py smoke` render-compare
remains the gate before any Q shader is declared render-correct via this path; a
multi-tile cook should be spot-checked if Houdini's COP runover is ever configured with
tiling. (Same caveat applies identically to Design B's seeding — the mechanism is
shared.)

---

## 6. Live `code_header` wiring recommendation

The setter branch ships an **opt-in** prototype (carried forward here) in
`houdini/scripts/python/hshadertoy/builder/builder.py`: when `HSHADERTOY_LIVE_HEADER=1`,
`_load_live_code_header()` reads `houdini/ocl/include/shadertoyInputs.h` (resolution:
`$HSHADERTOY_ROOT` first, then relative to builder.py) and assigns it to
`all_params["code_header"]` at build time — making the header **live-editable from the
repo file** without an HDA regeneration. Default off → behavior byte-identical to today.

**Can the builder populate `code_header` from `shadertoyInputs.h`? Yes — it is a
faithful source.** The HDA `code_header` default (per `hShadertoy_hda_ref.json`) and
`shadertoyInputs.h` are the same dialect: `@` bindings (`@xres`, `@Time`,
`@iChannel0.layer`) and VEX backtick expressions in the static initializers
(`` static float iTime = `fpadzero(1,4,ch('init_iTime'))+'f'`; ``). The file preserves
the backticks verbatim.

Risks and mitigations:
1. **VEX backticks must survive.** They do — `code_header` is a string parm Houdini
   backtick-evaluates at cook time; assigning the file text verbatim keeps them intact.
   *Mitigation:* build-time assert that the populated parm still contains
   `` `fpadzero ``/`` `rint ``/`ch('init_` markers.
2. **Drift between file and HDA default.** With live-wiring on, the HDA's embedded
   default becomes dead weight and can silently diverge. *Mitigation:* declare
   `shadertoyInputs.h` the single source of truth; add a unit test that the HDA ref's
   `code_header` default equals `shadertoyInputs.h` (whitespace-normalized) so drift
   trips CI.
3. **Existing scenes** keep their baked `code_header` until rebuilt — unaffected; the
   setter form is semantically identical to the old macro form. Forward-only, opt-in.

**Rollout:**
1. Land this prototype (this branch) — `shadertoyInputs.h`/`main_header.cl` restructured.
2. Owner pastes the section-7 text into the HDA `code_header` parm once (one
   regeneration) so the *default* is correct even with live-wiring off.
3. Flip `HSHADERTOY_LIVE_HEADER=1` in a dev build; verify a normal shader cooks;
   run `houdini_smoke` + `rc.py smoke` (orchestrator/owner — not from this worktree).
4. Add the drift unit test. Once green in CI, make live-wiring the default and demote
   the HDA embedded default to a fallback.

---

## 7. Copy-paste-ready HDA `code_header` replacement

**Use `houdini/ocl/include/shadertoyInputs.h` from this branch verbatim as the parm
value** — it is the authoritative `@`-form mirror (`#bind` lines, backtick statics,
setter, macros, cubemap glue). Relative to today's `code_header`, the owner is adding:

1. The `shadertoy_bind_inputs(...)` function (15 params incl. trailing
   `int2 in_pix_base`, ending with
   `GLSL_glFragCoord_off = in_pix_base - (int2)(get_global_id(0), get_global_id(1));`),
   placed **after** the static uniform globals.
2. `SHADERTOY_INPUTS` rewritten to a single
   `shadertoy_bind_inputs(..., (int2)(@ix, @iy));` call + the unchanged
   `fragCoord`/`fragColor`/`DO_CUBEMAP` lines.
3. `DO_CUBEMAP` calling `shadertoy_cubemap_bind(@ix,@iy,@xres,@yres,&rayDir);`.
4. The `shadertoy_cubemap_bind(...)` wrapper after `shadertoy_cubemap(...)`.

Prerequisite: `glslHelpers.h` must contain the `GLSL_glFragCoord_off`/`GLSL_glFragCoord`
block (Design B's commit 801c2243 — live include, no HDA action needed; already present
once B merges, and also carried on this branch).

---

## 8. The `@KERNEL` question (owner asked) — answer: **No, it enables nothing new**

The HDA `code_rp{N}` parms hold `@KERNEL { SHADERTOY_INPUTS ... }`; Houdini expands
`@KERNEL` into the full `kernel void generickernel(...)` signature + `_bound_*` locals
+ the `get_global_id()`-derived `_bound_gidx/gidy` (this is what produces
`tests/ocl/main_kernel.cl`). **Hand-expanding `@KERNEL` enables nothing for
`gl_FragCoord` that the setter approach does not**, because everything `@KERNEL`
produces is **kernel-body scope**:

- The pixel coordinate (`_bound_gidx`, `fragCoord`) is a kernel-body local. Helpers are
  defined at **program scope, before the kernel**; OpenCL C has **no nested functions**,
  so a helper can never close over a kernel local no matter how the kernel is spelled.
- The only scope bridges that exist are: (a) parameters (call-graph threading), (b) a
  program-scope global — only sound as the *uniform offset* (per-work-item values race),
  or (c) `get_global_id()` called from the helper itself — which is exactly what
  `GLSL_glFragCoord()` does and needs no kernel change at all.

Expanding `@KERNEL` merely inlines the signature text; it creates no new scope bridge.
**Verified, not merely asserted:** the section-4 helper-accessor demo compiles with the
*unmodified* `@KERNEL`-derived `main_kernel.cl` — the accessor already reaches helpers
without touching the kernel side.

---

## 9. Coordination with Designs A and B

- **Design B has landed** (801c2243): glslHelpers.h symbols + transpiler injection
  (entry seeds offset gated on helper usage; matching helpers get a body-local
  `float4 gl_FragCoord = GLSL_glFragCoord();`), +6 corpus. Design C is **fully
  reconciled** with it on this branch: symbols single-homed in glslHelpers.h
  (cherry-picked, merges clean), duplicates removed, setter carries a second identical
  seed. **Post-adoption cleanup for B's owner:** once the HDA code_header carries the
  setter, the transpiler's entry-body seed emission is redundant and can be retired
  (one fewer emission path); the helper-local injection stays.
- **If Design A (call-graph threading) had won instead:** Design C degrades gracefully —
  the AG setter is untouched by A; the `in_pix_base` param + seed line and the
  glslHelpers.h block would be dormant infrastructure (dead-code-eliminated statics) or
  could be dropped by removing the last setter line + the 15th param.
- **The AG setter lands regardless** of A vs B: it is a header change with no
  transpiler dependency; the transpiler's `uniform_redefine.py` push-pop stays as a
  harmless belt-and-braces safety net.
- **Race lesson (recorded for the campaign):** two agents editing the shared live
  include dir race each other's corpus runs — the ldsczf "regression" in this
  prototype's first batch 3 was B's landing, not a header bug. Sample re-runs against
  worktree-pinned includes disambiguated it in one retry.

---

## 10. Files changed on this branch

- `tests/ocl/main_header.cl` — setter (15-param, offset seed) + `SHADERTOY_INPUTS`
  delegation + `shadertoy_cubemap_bind` wrapper (`AT_*` forms; HSHADERTOY BEGIN
  section only; lines 1-1401 untouched).
- `houdini/ocl/include/shadertoyInputs.h` — same, in `@` form (code_header mirror =
  the owner's copy-paste source).
- `houdini/ocl/include/glslHelpers.h` — Design B's accessor hunk, cherry-picked
  verbatim from 801c2243 (merge-clean with B).
- `houdini/scripts/python/hshadertoy/builder/builder.py` — opt-in
  `HSHADERTOY_LIVE_HEADER=1` live-wiring (carried from `fix/header-ag1-setter-main`).
- `tests/unit/test_header_uniform_redefine_guard.py` — guard test (carried from the
  setter branch).
- `tests/ocl/q_fragcoord_helper_test_{header,kernel}.cl` — the Q capability demo.
- `tests/fixcampaign/DESIGN_C_header_restructure.md` (this file),
  `tests/fixcampaign/HOUDINI_HANDOFF.md` — owner adoption steps.
