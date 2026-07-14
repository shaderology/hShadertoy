# NEXT SESSION → Session 49: **category AI — member/swizzle on a scalar** (1 sole-blocker, localized transformer fix)

**READ FIRST, in this order, then execute:**
1. `tests/fixcampaign/README.md` — workflow & guardrails (TDD, full unit suite
   green, **0 PASS→FAIL regressions**, commit rules, Houdini smoke +
   `python tests/rendercompare/rc.py smoke` render-correctness gate — run BOTH).
2. this file.
3. The AI row in `tests/fixcampaign/BACKLOG.md` (Wave-3 table).

## What Session 48 finished (do NOT redo)
**AG cluster 1 is DONE and merged** (+2: XdyBW1, MtXBDf; 0 regressed) via the
transpiler-side undef/re-define push-pop (`preprocessor/uniform_redefine.py`,
both hosts — live Houdini needs no HDA change). The rejected-but-proven
header-side alternative (`shadertoy_bind_inputs()` setter) is embedded with
adoption steps at the bottom of `houdini/ocl/include/shadertoyInputs.h` and
lives on branch `fix/header-ag1-setter-main` — do not re-implement; point the
owner there if a SHADERTOY_INPUTS-poisoning bug ever reappears.
**AG residual = cluster 2 only** (4dsfRn: S24 macro-expander pastes `- -1` as
`--1` on `T(-, )` — design in the AG BACKLOG row; gated S24 path, needs
per-shader proof).

## THIS session: **AI** (1 sole — MlySRh)
`corpus.py list AI` — 3 tagged, **1 sole (MlySRh)**. Confirmed localized during
the S48 investigation:

`float3 colorMG = linearstep(0.f, 2.f, grids.y).xyz;` — `grids.y` is a `float`,
inference *correctly* resolves the `float linearstep(float,float,float)` overload
(3 overloads exist: float/float2/float4 in glslHelpers.h), so the call returns
`float`. The source then **swizzles a scalar with `.xyz`** — legal on
Shadertoy/lenient GLSL drivers (scalar→vector broadcast), invalid in OpenCL
→ clang: *"member reference base type 'float' is not a structure or union"*.

**Fix (recommend, don't ask):** detect a swizzle/member access whose base
expression infers to a **scalar** (`float`/`int`/`uint`/`bool`), and lower it to
a broadcast — e.g. `expr.xyz` → `((float3)(expr)).xyz` (or directly
`(float3)(expr)` for a pure-broadcast swizzle like `.xyz`/`.xxx`). Guard on the
base being scalar so vector swizzles are untouched. Prefer the transformer
(`ast_transformer.py`, swizzle/`field`/member-access path) where the scalar-base
type is known from inference; if it lands in emission instead, **mirror BOTH
emitters** (`codegen/opencl_emitter.py` production + `transformer/code_emitter.py`
dead-but-tested). Add a unit test: scalar-with-`.xyz` → broadcast substring.

**Watch:** swizzle length must match the cast width (`.xyz`→float3,
`.xy`→float2); a single-component `.x` on a scalar is a no-op — emit the bare
scalar, don't wrap.

The other two AI tags (**Md2fzV, ldKcz3**) are the *PrintState font idiom* and
also carry **B** — NOT sole, do not chase them under AI. MlySRh is the only
clean +1.

## Gates & proof recipe (same as always)
- Unit suite: `python -m pytest tests/unit/ -q` — **baseline 2090 passed + 6
  skipped** (S48 added 13). Must stay green. Add the AI repro test.
- Corpus: back up `tests/campaign/ledger.json` to scratch FIRST. Transformer/
  emitter change → **hash blast-radius rig is meaningful** (recipe below). Get
  the changed-id set, re-test those `--force`, then `report`.
- Delta: diff PASS-sets between the ledger backup and live ledger directly (dict
  keyed by id, `overall=='PASS'`). **REGRESSED must be 0.** Do NOT trust
  `corpus.py delta`'s FIXED/REGRESSED lists.
- Houdini: `python tests/fixcampaign/houdini_smoke.py` (exit 0) AND
  `python tests/rendercompare/rc.py smoke` (exit 0). BOTH mandatory. **Run them
  from the MAIN tree with your fix branch checked out** — they always test
  `C:\dev\hShadertoy` (the Houdini package pins HSHADERTOY_ROOT), so running
  them from a worktree silently mixes trees. Capture exit codes directly
  (`cmd > log 2>&1; echo $?`), NOT after a pipe through `tail`.
- Re-test cmd: `python tests/campaign/campaign.py test --ids MlySRh --force`.

## `hash_outputs.py` blast-radius rig (recreate in scratch each session)
The rig does NOT persist — recreate it (~50 lines). It iterates `cache/*.json`,
transpiles each testable pass (image/buffer/cubemap; common merged via
`transpile(glsl, common=common_src)`), hashes `get_header()+"\x00"+get_kernel()`.
Usage: `--tree <repo_root> --cache <repo>/tests/campaign/cache --out <json>`.
Baseline worktree: `git worktree add --detach <scratch>/main-wt HEAD`; hash both
trees; diff → changed-id set; `git worktree remove --force <scratch>/main-wt`.
**WINDOWS TRAP:** with `os.chdir(tree)` pass `--out` Windows-style
(`C:/Users/...`), not `/c/...`; and `sys.modules` must be purged of
`transpile`/`src.glsl_to_opencl*` between the two tree imports.
**NEW TRAP (S48):** if a subagent/worktree is involved, `git log` the worktree
FIRST — agent worktrees have been cut from a stale (S40-era) base; cherry-pick
onto a fresh main-based branch for the decisive proof.

## Baselines (after Session 48)
- Unit suite: **2090 passed + 6 skipped, 0 failed**.
- Corpus: **864 / 999 PASS**; by-failing-pass top cats (`campaign.py report`):
  N=28 B=24 C=18 A=17 E=14 X=13 (run `corpus.py summary`/`list` for by-SHADER
  live counts — `report` counts failing *passes*, reads higher).
- HEAD = main after the `fix/transpiler-ag1-undef-main` merge + `0ade4bc3`
  (in-code docs for the header-setter design).
- The owner keeps uncommitted WIP in the tree (README.md, the HDA, ref jsons,
  `tests/ocl/main_header.cl` restructuring) — never stage files you didn't
  change; stage bookkeeping hunks surgically.

## Progress this arc (context — do NOT redo)
S38 **+29** (G race), S39 **+4** + UNKNOWN 27→0, S40 **+5** (AE), S41 **+9** (U),
S42 **+6** (W), S43 **+4** (AC), S44 **+2** (AB), S45 **+2** (AD), S46 **+1**
(AF), S47 **+1** (AH), S48 **+2** (AG cluster 1, two-branch experiment).
**PASS 795 → 864.**
