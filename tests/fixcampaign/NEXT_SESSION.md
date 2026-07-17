# NEXT SESSION → Session 56: **category Q (gl_FragCoord in helpers) + P (parse residual)** — OWNER IS DIGGING FIRST

**READ FIRST, in this order, then execute:**
1. `tests/fixcampaign/README.md` — workflow & guardrails (TDD, full unit suite
   green, **0 PASS→FAIL regressions**, commit rules, Houdini smoke +
   `python tests/rendercompare/rc.py smoke` render-correctness gate — run BOTH).
2. this file — the **Q and P findings** below were gathered in Session 55; the
   owner asked to do their own digging on Q/P before we implement, so treat
   these as a briefing, not a locked plan. **Q's fix is a MAJOR REDESIGN —
   confirm the approach with the owner before coding (guardrail: major redesigns
   need approval).**
3. The **Q and B rows in `tests/fixcampaign/BACKLOG.md`** (grep `### Q`, `### B`).

## Houdini 22 note (still current)
Houdini **22.0.368** is the default gate target; both gates run **exit 0
unpinned** (verified S53/S54/S55). If a gate ever fails with
`ModuleNotFoundError: tree_sitter`, that's a stale hython env — re-pin
`$env:HYTHON = '...Houdini 21.0.440\bin\hython.exe'` and tell the owner.

---

## CATEGORY Q — `gl_FragCoord` used in HELPER functions  (14 fails / 7 shaders, all sole-blockers)

**Every Q sole-blocker fails on the identical error:** `use of undeclared
identifier 'gl_FragCoord'`. The ENTRY-body case is already handled (Session-Q
injected a `float4 gl_FragCoord = (float4)(fragCoord, 0.0f, 1.0f);` local at the
top of `mainImage` — see `_transform_function_definition` ~line 3611). The 7
remaining shaders all reference `gl_FragCoord` **outside the entry body**, where
that injected local is out of scope.

**Where each shader uses it (verified from cache source, Session 55):**
- `3t2GRD` — helper `rand` (+ a custom `main`)
- `Mt3GDl` — helper `map` (image + Buf A)
- `Mty3zh` — helper `sdlineRoundTile`
- `XsfyDl` — helper `draw_char`
- `XtSGRV` — `mainImage` + helpers `map`, `softshadow`
- `XlSBRW` — via `#define F gl_FragCoord`, `F` used in helpers (image + 2 buffers)
- `3dK3zR` — via `#define F gl_FragCoord` in the **Common** pass (merged into all)

**Why there's no cheap fix (Session 55 dug this out):**
`gl_FragCoord.xy` == Shadertoy `fragCoord` == the pixel-center coordinate. But:
- `fragCoord` is a **kernel-body local** (`SHADERTOY_INPUTS` macro,
  `tests/ocl/main_header.cl` ~line 1492: `float2 fragCoord = AT_fragCoord;`
  with fallback `(float2)(AT_ix, AT_iy)`). Not visible in helpers.
- `AT_ix`/`AT_iy` = `_bound_gidx`/`_bound_gidy` (main_header.cl lines 7-8) —
  **kernel `#bind` params**, also only in scope inside the kernel body.
- OpenCL **cannot** hold a per-work-item value in a program-scope global
  (program-scope is `__global`/`__constant`, shared across work-items → race).
  So "make gl_FragCoord a global" does NOT work.
- `get_global_id(0/1)` IS callable from any function, BUT Houdini's COP runover
  is not guaranteed to be a raw 1:1 pixel grid (tiling/offset via `_bound_*`),
  so using it would risk a "compiles but renders wrong" bug — exactly what
  `rc.py smoke` guards against. **Do not go this route without a render-compare
  proof on a multi-tile cook.**

**⇒ The correct fix is to THREAD the coordinate through the call graph** (the
existing code comment calls this "a larger redesign" and deliberately left it
unfixed). Proposed design (**for owner approval before implementing**):
1. **Reachability pre-scan:** mark every user function whose body references
   `gl_FragCoord` (directly, or via an object-macro alias like `#define F
   gl_FragCoord` — resolve those first), OR that CALLS a marked function
   (transitive closure over the call graph; handle recursion/forward calls like
   `_collect_function_renames` does).
2. **Signature rewrite:** append a synthetic `float2 _fragCoord` (or `float4
   gl_FragCoord`) param to each marked function. **WATCH the arity collision:**
   `function_signatures` is keyed by parameter count to disambiguate overloads
   (`_transform_function_definition` ~line 3571, and the call-site resolver) —
   adding a param shifts arity and can alias a real overload. The threaded param
   must be tracked so the call-site resolver counts the ORIGINAL arity.
3. **Call-site rewrite:** at every call to a marked function, pass the in-scope
   coordinate (the entry passes its `fragCoord`; a marked helper passes its own
   threaded param through).
4. **Read resolution:** inside a marked function, `gl_FragCoord` reads resolve
   to the threaded param (build the `.xy`/`float4` shape at the read site or as
   a one-line local alias at function top, mirroring the entry injection).
5. Interactions to test: the object-macro alias (`#define F gl_FragCoord`), the
   Common-pass merge (3dK3zR), D2 function renames, the out-param `&`/deref
   machinery, and multi-pass buffers. Blast radius = only shaders using
   gl_FragCoord in helpers (all currently FAILING → low PASS-set risk) but the
   transform touches the shared signature/call-site paths → **hash rig +
   full-corpus re-test mandatory.**

**Value:** up to +7 shaders / 14 passes — the single most homogeneous bucket
after N. **Owner is investigating Q first; align with them before coding.**

---

## CATEGORY P — parse-stage residual  (15 fails / 14 shaders)  — HETEROGENEOUS, no single slice

Session 55 spot-checked the "Could not find mainImage()" sub-cluster; it is
**at least 3 different root causes**, not one fix:
- **mainImage defined as a MACRO** — `4djfDR` (`#define mainImage(C,U) C.xy=...`)
  and `tlsSDs` (`#define mainImage(z,u) \ ...`). The entry detector looks for a
  `void mainImage(...)` FUNCTION; a macro-defined entry is invisible. Would need
  the macro expander to materialize it (N-adjacent).
- **cubemap false-positive** — `3tVSRG`: the failing pass is the CUBEMAP pass
  (has `mainCubemap`, not `mainImage`); the mainImage-only campaign harness
  can't test it. NOT a real transpiler bug — a harness limitation. Its image +
  buffer passes are fine.
- **genuinely odd** — `wssBz2` and `lljGDm` both contain a normal
  `void mainImage(out vec4 fragColor, vec2 fragCoord)` yet still report "not
  found". Root-cause these two FRESH (likely a parse error earlier in the file
  aborts detection, or a commented-out `/*void mainImage...*/` decoy in lljGDm).
The rest of P is assorted one-off `ParseError(line N, col M)` at distinct
constructs (3d23Dc, 3t2XzW, 4sjcz1, ldfXzB, ldfyRn, tdjfWc, tsSyWG, wsByWz) +
`FloatLiteral must end with 'f': 0.95100F` (3lX3Rr — an uppercase-`F` float
suffix the normalizer misses; **this one looks genuinely cheap** — extend the
float-suffix handling to accept `F`). **Recommendation: if picking P, start with
3lX3Rr (uppercase F suffix) as a clean single, then the wssBz2/lljGDm pair.**

---

## Gates & proof recipe (same as always)
- Unit suite: `python -m pytest tests/unit/ -q` — **baseline 2134 passed + 6
  skipped** (S55 added 2 in `test_transformer_pointer_param_shadow.py`). Add
  failing repro tests FIRST.
- Corpus: back up `tests/campaign/ledger.json` to scratch FIRST. Q's threading
  touches shared signature/call-site paths → **hash blast-radius rig mandatory**
  (recipe below) + full-corpus re-test. Re-test changed ids `--force`;
  currently-FAILING ids are slow — ≤10-id batches.
- Delta: diff PASS-sets between ledger backup and live ledger directly (Python
  set diff on `overall=='PASS'`). **REGRESSED must be 0.** Don't trust
  `corpus.py delta`'s lists.
- Houdini: `houdini_smoke.py` AND `rc.py smoke`, both exit 0, main tree on the
  fix branch. **For Q especially, `rc.py smoke` is the critical gate** (a wrong
  coordinate compiles fine but renders wrong).

## `hash_outputs.py` blast-radius rig (recreate in scratch each session)
Does NOT persist. ~30 lines: iterate `cache/*.json`, for each testable pass
(image/buffer/cubemap) `transpile(code, common=common)`, sha1 of
`get_header()+get_kernel()`. Run once per tree:
`--tree <repo_root> --cache <repo>/tests/campaign/cache --out <win-path.json>`.
Baseline: `git worktree add --detach <scratch>/main-wt HEAD`; diff the two hash
maps → changed-id set; `git worktree remove --force` after. WINDOWS TRAP: the
rig `os.chdir`s into the tree — pass `--out` as a full Windows path (use
`cygpath -w`). ~2-3 min/tree. S55 used it to prove the pointer-shadow fix
changed **exactly 1 pass** (tsXBzs) corpus-wide.

## Baselines (after Session 55)
- Unit suite: **2134 passed + 6 skipped, 0 failed.**
- Corpus: **1354 / 1499 PASS.** Top failing passes: N=75 B (macro-residual) P=15 X≈15 Q=14 G=13.
- HEAD after merge = the S55 merge commit of `fix/transpiler-b-pointer-shadow`.
- The owner keeps uncommitted WIP in the tree — never stage files you didn't
  change; stage your files by name.

## Progress this arc (context — do NOT redo)
S51 **+12** (A3+A1), S52 **+10** (A2 — A closed), S53 **+35** (E — preproc
routing, campaign record), S54 **+5** (X square-matrix spellings), S55 **+1** (B
pointer-param scope-shadow; B residual now all macro-textual). Remaining big
buckets: **N** (macro-expander, STRUCTURAL / NEEDS-APPROVAL), **Q** (gl_FragCoord
threading — redesign, owner digging), and the P/G/X tails.
