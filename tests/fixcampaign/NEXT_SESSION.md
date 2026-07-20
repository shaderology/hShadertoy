# NEXT SESSION → Session 62: **CAMPAIGN PAUSED (owner, 2026-07-20)** — awaiting a direction decision

> **PAUSED at 1365/1499.** Session 61 exhausted the cheap, no-approval P
> singles: every remaining P and N pass now needs macro-expander (category N) or
> category-G (`#define`-splits-statement) STRUCTURAL work, which is the owner's
> approval gate — and the owner chose to pause rather than authorize it (or a
> pivot) at the end of S61. **Do NOT start category N / G / P-macro work without
> the owner re-authorizing a direction.** When the owner resumes, the live
> options (from the S61 ask) are:
> 1. **Gated P-macro extension** — extend the S24 function-macro expander for the
>    parse-FAILING P shapes (ldfXzB object→function macro + `#undef`/redefine;
>    3t2XzW token-paste; ldfyRn macro-DSL). Gated on parse-failure ⇒ 0-regression
>    on passing shaders by construction. ~3-4 P passes; hard preprocessor
>    semantics but bounded blast radius. *(was the recommended path.)*
> 2. **Category N compile-stage** — the 75-pass bucket (`ivec2(U)` in `#define`
>    bodies); highest payoff, highest regression risk (expander must run on
>    already-parsing shaders).
> 3. **Pivot to another top bucket** with possible localized no-approval wins:
>    B=24, X=15, K=13, D=12.
>
> Once the owner picks, rewrite this file for that session and follow the normal
> protocol below.

**READ FIRST, in this order, then execute (when un-paused):**
1. `tests/fixcampaign/README.md` — workflow & guardrails (TDD, full unit suite
   green, **0 PASS→FAIL regressions**, commit rules, Houdini smoke +
   `python tests/rendercompare/rc.py smoke` render-correctness gate — run BOTH).
2. this file.
3. The relevant BACKLOG rows (the `P` table row; `### N`/the `N` table row for
   the approval item).

## State after Session 61 (2026-07-20) — PAUSED
- Session 61 was **triage-only, no code change** (see PROGRESS.md S61). The last
  two un-triaged P singles (ldfXzB, 3t2XzW) were root-caused as category-N
  macro-expander work. Corpus unchanged: **1365/1499 PASS.** Unit **2154 + 6**.

## State after Session 60 (2026-07-20)
- **More P singles (+2, 1363→1365, 0 regressed):** expression-size type-first
  array decl (tdjfWc `vec3[SZ*3]`) — broadened `_TYPE_FIRST_ARRAY_DECL`'s size
  group AND pinned it single-line (a multi-line match fused a bracketed comment
  range with the next line's identifier); `(bool(x) ? …)` cast-ambiguity
  (4sjcz1) — new `_PAREN_BOOL_CTOR` inserts identity `!!`. Both in
  `_normalize_array_syntax` (`glsl_parser.py`), same family as S22/S23.
- Corpus: **1365/1499 PASS.** Top failing: N=75, B=24, X=15, K=13, G=12, D=12.
- Unit baseline: **2154 passed + 6 skipped.**
- H22 migration remains fully validated; `HSHADERTOY_LIVE_HEADER=1`,
  `shadertoyInputs.h` LIVE. Treat header edits with campaign-proof discipline.
- If the owner hasn't committed the H22 changeset yet (HDA, main_header.cl,
  main_kernel.cl, build_options.json, `*_h21/_h22` archives, ledger.json), it
  may appear as uncommitted WIP — **do not stage or revert it.**

## PRIORITY 1 — the LAST cheap P parse-error singles (no approval needed)
`corpus.py list P` is live truth. The remaining sole-blocker P singles are each
their own root cause and the cheap ones are getting thin — root-cause FRESH from
`tests/campaign/cache/<id>.json` and check the pattern isn't already handled by
`_normalize_array_syntax` (`glsl_parser.py`). Triaged in S60:
- **ldfXzB** (`#undef PRIM` cascade at L615) and **3t2XzW** (error at
  post-Common line ~290) — NOT yet root-caused; look first, they *may* be
  localized.
- **3d23Dc**, **wsByWz** — a `#define` appears mid-expression, splitting a
  statement across a preprocessor directive → **category G** (preprocessor
  splits statements; HIGH-risk/redesign — needs owner approval, don't attempt
  as a "single").
- **ldfyRn** — macro-DSL (`c(...) C(...) path(style(...))`) → **category N**.
- **tsSyWG** — golf shader calling `mainSound(in int samp, …)` INSIDE mainImage
  (a param declaration in call-arg position); genuinely malformed idiom →
  **edge case, skip** per scope policy.
If no cheap P single remains after looking at ldfXzB/3t2XzW, go to Priority 2.

## PRIORITY 2 (owner approval required) — category N (75 passes, biggest bucket)
Macro-expander structural work — write the design into the BACKLOG N row and
ASK before implementing. Stop and ask the owner for this one. Note the P
residual (4djfDR, tlsSDs mainImage-as-macro; ldfyRn macro-DSL; the `#define`-
splits-statement pair) largely funnels into the N/G preprocessor work, so an
N session may retire several P passes too.

## Deferred follow-ups (grab if blocked)
- **Mty3zh** (ex-Q): AF ctor-overflow `vec2(hashRace(...), gl_FragCoord.xy/
  iResolution.xy)`; precedent = `_truncate_overflow_ctor_args` (S45 family).
- **rendercompare:** README caveat — wgpu-shadertoy's raw `gl_FragCoord.y` is
  the FLIP of its own `fragCoord.y` (proven S56; HDA side is faithful) →
  Q-shader compares vs wgpu mis-verdict. Consider a reference-free
  self-checking smoke shader (green iff helper gl_FragCoord == entry
  fragCoord) so `rc.py smoke` guards the geometry assumption end-to-end.

## Houdini / environment notes
- Gates green unpinned on **22.0.368** through S60, with the live header.
- **After any Houdini version/build change: run
  `python tests/fixcampaign/probe_launch_geometry.py` FIRST** (exit 0
  required). If it fails, the gid-derived gl_FragCoord accessor is unsafe on
  that build → fallback is branch `fix/q-fragcoord-threading` @ 78d01832. The
  H22 setter is the SOLE seeder of `GLSL_glFragCoord_off` (S58); the probe
  exercises that path.
- Harness background tasks die at ~2 h — long corpus runs must be
  chunk-resumable (`full_retest_s57.py <start-offset>` pattern) or owner-run.

## Gates & proof recipe (same as always)
- Unit suite: `python -m pytest tests/unit/ -q` — **baseline 2154 passed + 6
  skipped.** Add failing repro tests FIRST.
- Corpus: back up `tests/campaign/ledger.json` to scratch FIRST; re-test
  changed ids `--force` (≤10-id batches for failing ids); delta = direct
  Python set-diff on `overall=='PASS'`; **REGRESSED must be 0.**
- Blast radius: for a pre-parse normalizer change, enumerate the EXACT set of
  changed ids by diffing OLD.sub vs NEW.sub over the corpus cache (as in S60) —
  and diff over the currently-PASS shaders specifically to catch comment/false-
  positive rewrites before they regress anything.
- Houdini: `houdini_smoke.py` AND `rc.py smoke`, both exit 0, main tree.
- The owner keeps uncommitted WIP in the tree — never stage files you didn't
  change; stage your files by name.
