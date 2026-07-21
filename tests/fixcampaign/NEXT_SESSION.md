# NEXT SESSION → Session 64

**READ FIRST, in this order, then execute:**
1. `tests/fixcampaign/README.md` — workflow & guardrails (TDD, full unit suite
   green, **0 PASS→FAIL regressions**, commit rules, Houdini smoke +
   `python tests/rendercompare/rc.py smoke` render-correctness gate — run BOTH).
2. this file.
3. The relevant BACKLOG row for whichever category you pick.

## State after Session 63 (2026-07-21) — category E program-scope residual MERGED
- **E residual — program-scope preproc-block AST routing (fix `625604fe`, merge
  `f35f3f5b`).** Program-scope `#if DFx / #elif DFy` chains wrapping whole
  `function_definition`s now route through the AST (out/inout params, matrix
  `*=`, vec ctors lower correctly); entry-point defs in a conditional stay raw
  (S59 owns them). Trigger was `tests/shaders/complex/cubes.glsl` (shadertoy
  mslfR2) — now compiles. **Blast radius 35 PASS shaders, 0 regressed.**
  Corpus stays **1393/1499** (cubes is a manual test shader, NOT in the corpus,
  left untracked). Collateral: corpus shader **MdVcRK** improved `[G,K,T]→[G]`
  (its `out`-param blocker cleared; residual is an independent category-G
  `invalid token at start of a preprocessor expression`).
- **Unit baseline: 2196 passed + 6 skipped** (+3 in
  `test_transformer_preproc_matrix.py` Part 3). All four gates green on merged
  main. See PROGRESS.md S63 + BACKLOG E "Program-scope residual".
- Top failing categories now: **B=25, X=16, D=12, K=12, T=10, J=10**
  (`corpus.py summary` / `report` is live truth; these are snapshots).

## State after Session 62 (2026-07-21) — UN-PAUSED, +28 merged
- **Category N + P design competition, both branches merged to main (+28, 0
  regressed, 1365→1393):**
  - **N — overloadable ctor dispatcher (+21)** (`glslHelpers.h`
    `GLSL_{vec,ivec,uvec,bvec}{2,3,4}` + scalar dispatchers; single-arg arity
    scan in the textual macro/`#if` path + AST untypeable-arg fallback). **N is
    off the top of the board** — residual is multi-blocker buffer passes only.
  - **P/G — gated macro-expander extension (+7)** (wrapping-object macros,
    comment-blanking, multi-line call sites, entry-macro-only gate). Bonus: it
    retired the two category-G "`#define` splits a statement" shaders (3d23Dc,
    wsByWz) **without** the approval-gated G redesign.
- Corpus: **1393/1499 PASS.** Top failing categories now: **B=25, X=16, K=13,
  D=12, T=11, J=10** (`corpus.py summary` is live truth; these are snapshots).
- Unit baseline: **2193 passed + 6 skipped.**
- H22 migration remains fully validated; `HSHADERTOY_LIVE_HEADER=1`,
  `shadertoyInputs.h` LIVE. Treat header edits with campaign-proof discipline.

## PRIORITY — pick the next localized win from the top buckets
No approval-gated redesign is queued. Per scope policy (fix if common+localized;
skip if edge/rewrite), spawn a read-only investigation agent to cluster the top
bucket's error logs FIRST, then fix the localized slice. Candidates, in order:
- **B (25)** — pointer/address-space param model (`__generic T *` fix-it hints,
  out-param `&`, global-pointer deref). The full B model is a redesign (needs
  owner approval — see the B BACKLOG rows), but individual localized shapes have
  flipped before; cluster before committing to scope.
- **X (16)** — bitcast / `as_int`/`as_float` family + `uintBitsToFloat` cousins;
  historically transformer-only, localized (`fix/transpiler-x-bitcast`).
- **K (13)** — array/struct ctor residual (mostly mis-tagged macro-abuse
  ParseErrors per the K row — verify before chasing).
- **D (12)**, **T (11)**, **J (10)** — overloadable-user-fn / param-qualifier /
  macro-body-matmul residuals; each row records what's left.

## Residuals worth grabbing if blocked (from S62)
- **3t2XzW** (ex-P) now PARSES; residual is category **B** — `pmod`
  inout-pointer called inside an object-macro body.
- **ldfyRn** (ex-P) Image parses; Buf A residual is category **X**.
- **Mty3zh** (ex-Q): AF ctor-overflow `vec2(hashRace(...), gl_FragCoord.xy/
  iResolution.xy)`; precedent = `_truncate_overflow_ctor_args` (S45 family).

## ⚠ TWO TRANSPILER HOSTS — a fix can pass the campaign yet fail in Houdini (S63b)
There are **two host wrappers** around the shared core `src/glsl_to_opencl`:
- **Host A** `tests/transpile.py` — what the campaign, `compilecl.py`, the
  ledger, and every "proof" exercise. Corpus PASS ≠ Houdini PASS.
- **Host B** `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py`
  — the **real HDA/Houdini deployment** wrapper. Only a real cook runs it.
They have **diverged before** (S63b: Host B was missing the `matrix_macros`
seed, so cubes.glsl compiled in the campaign but a raw `float2 *= matrix2x2`
crashed the real Houdini cook). **Any fix that lives in a host WRAPPER pass
(pre/post-processing, not the shared transformer/emitter core) MUST be mirrored
into BOTH files.** When a shader "passes the campaign but fails in Houdini",
suspect Host A/B drift first. The only Host-B gates are real cooks:
`houdini_smoke.py`, `rc.py smoke`, or a per-shader
`hython builder_cook_headless.py <api.json> Transpile` (env `HSHADERTOY_ROOT`
+ `HOUDINI_OCL_PATH=<repo>/houdini/ocl;&`).

## Houdini / environment notes
- Gates green unpinned on **22.0.368** through S62, with the live header.
- **After any Houdini version/build change: run
  `python tests/fixcampaign/probe_launch_geometry.py` FIRST** (exit 0 required).
- Harness background tasks die at ~2 h / 10-min-per-call cap — long corpus runs
  must be chunk-resumable or owner-run. `campaign.py test --ids` wants a
  **comma-separated** list (space-separated silently no-ops).
- Campaign build `-I` is **absolute to `C:\dev\hShadertoy`** — worktree header
  edits need a local (uncommitted) `tests/build_options.json` repoint, and
  Houdini gates MUST run in the main tree.

## Gates & proof recipe (same as always)
- Unit suite: `python -m pytest tests/unit/ -q` — **baseline 2193 passed + 6
  skipped.** Add failing repro tests FIRST.
- Corpus: back up `tests/campaign/ledger.json` to scratch FIRST; re-test
  changed ids `--force` (≤25-id batches); delta = direct Python set-diff on
  `overall=='PASS'`; **REGRESSED must be 0.** For a pre-parse/emission change,
  enumerate the exact changed-id set with the Stage-0 hash rig
  (`scratchpad .../hash_rig.py` pattern) and re-test all changed currently-PASS
  ids as the regression proof.
- Houdini: `houdini_smoke.py` AND `rc.py smoke`, both exit 0, main tree.
- The owner keeps uncommitted WIP in the tree — never stage files you didn't
  change; stage your files by name.
