# Bug-Fixing Campaign — GLSL→OpenCL transpiler

Systematic, resumable workflow for **fixing** the transpiler bugs that the
mass-test campaign (`tests/campaign/`) catalogued. The mass-test campaign is
*intelligence gathering* (no fixes); this campaign is *the fixing*, and the two
are designed to interlock.

> **You are the engineer.** Each session you pick the next backlog item, fix it
> test-first, prove it with the real shaders, and hand off. Context is cleared
> between sessions, so the repo files below — not your memory — are the state.

---

## The two campaigns, and how they interlock

```
tests/campaign/   (mass-test)   tests/fixcampaign/  (bug-fix, THIS)
  ledger.json  ◀──────────────────  measures fixes via `test --ids .. --force`
  failures.csv ◀── report ──────────  regenerated from ledger after every fix
  classify.py                         BACKLOG.md  ranked fix targets
  REPORT.md                           PROGRESS.md append-only session journal
                                      NEXT_SESSION.md  prompt for next session
                                      baseline_ledger.json  frozen start state
                                      corpus.py   query targets / measure delta
```

`ledger.json` is the **single measurement source of truth** for both campaigns.
A fix is *proven* by re-running the mass-test `test` stage on the affected
shaders (`--force`), which rewrites their ledger entries with the new transpiler
output. `report` then regenerates `failures.csv`/`REPORT.md`. **No file is edited
by hand to record a fix** — the numbers move because the transpiler got better.

`baseline_ledger.json` is a frozen copy of the ledger at campaign start (413/999
PASS). `corpus.py delta` compares live-vs-baseline to show exactly which shaders
were **FIXED** (FAIL→PASS) and, critically, any that **REGRESSED** (PASS→FAIL).

---

## Resume protocol (start every session here)

```bash
# 1. read the state (durable across context clears)
cat tests/fixcampaign/NEXT_SESSION.md     # what to do this session
cat tests/fixcampaign/BACKLOG.md          # full ranked plan + root causes
# 2. pull this category's concrete targets (ids, errors, re-test cmd)
python tests/fixcampaign/corpus.py list <CAT>
# 3. ... do the fix (see Session protocol) ...
# 4. prove + measure
python tests/fixcampaign/corpus.py delta
```

---

## Session protocol (mandatory order — TDD per `docs/transpiler/TESTING.md`)

1. **Pick** the category from `NEXT_SESSION.md` (one category / sub-cluster per
   session — keeps context bounded and attribution clean).
2. **Gather** targets: `corpus.py list <CAT>`. Rows marked `*` are *sole-blocker*
   shaders that flip to PASS once this category is fixed; un-marked rows carry
   other categories too. Read 3-5 representative `*` shaders' error excerpts and
   the relevant transpiler code. (For an un-root-caused category — e.g. the P
   catch-all — this is where a read-only investigation subagent earns its keep;
   see Orchestration.)
3. **Write failing unit test(s) FIRST** in `tests/unit/test_*.py` — a *minimal*
   GLSL→OpenCL repro of the bug (string in, expected OpenCL substring out). Run
   it; confirm it fails. New transpiler function ⇒ its own new unit test.
4. **Implement** the minimal fix in `src/glsl_to_opencl/...` (or `tests/transpile.py`
   post-processing). Match surrounding style.
5. **Green the unit test**, then run the **unit suite**:
   `python -m pytest tests/unit/ -q` (~1670+ tests; the gate is `tests/unit/`
   ONLY — `tests/integration/` is pre-existing-broken on clean HEAD, do not
   gate on `pytest tests/`). It MUST stay green — a fix that breaks existing
   tests is a regression, not a fix.
6. **Re-test the real shaders** (proof on production input):
   `python tests/campaign/campaign.py test --ids <ids> --force`
   then `python tests/campaign/campaign.py report`.
7. **Measure**: `python tests/fixcampaign/corpus.py delta`.
   - Net PASS must be **≥ 0**. Any **REGRESSED** id ⇒ stop, root-cause, fix or
     revert before continuing.
   - Expect *transpile-stage* fixes (R, G, K, S, T, P) to **unmask** new
     downstream compile errors rather than flip straight to PASS — that is
     normal and improves later data. *Compile-stage* fixes (D, B, A, N, …) flip
     sole-blocker shaders to PASS directly.
8. **Houdini smoke test** (real HDA, real cook — catches what the
   mainImage-only campaign can't, e.g. mainCubemap/Common regressions):
   `python tests/fixcampaign/houdini_smoke.py`
   Builds wfffRN (BuffersAndTextures) in the hShadertoy HDA via hython and
   force-cooks it (`cook(force=True)` ⇒ Houdini compiles + runs every
   renderpass's OpenCL). Exit 0 required — a non-zero exit is a regression:
   stop, root-cause, fix or revert. Needs a free Houdini license; budget
   ~2-5 min (timeout 600 s built in).
   **Render-correctness complement** (proves pixels, not just cooks —
   catches semantic regressions the cook test can't):
   `python tests/rendercompare/rc.py smoke`
   Renders gradient+london+digits through wgpu-shadertoy AND the HDA and
   gates on perceptual similarity. Exit 0 required, same regression rule.
   Budget ~2-4 min. Docs: `tests/rendercompare/README.md`.

   > ⚠ **TWO transpiler hosts — the campaign only tests one.** Steps 6-7
   > (`campaign.py test`, `compilecl.py`) run **Host A** `tests/transpile.py`.
   > The shipping Houdini path is **Host B**
   > `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py`, a
   > drifting near-duplicate. The ONLY steps that exercise Host B are the real
   > cooks in step 8 (`houdini_smoke.py` / `rc.py smoke`, or a per-shader
   > `hython builder_cook_headless.py <api.json> Transpile`). **So a fix can be
   > green in the corpus yet crash in Houdini from pure host drift** (S63b:
   > `mp *= ROT(...)` compiled in the campaign but emitted a raw
   > `float2 *= matrix2x2` in Houdini). If your fix lives in a host WRAPPER pass
   > (pre/post-processing, not the shared `src/` core) you MUST mirror it into
   > BOTH hosts and prove it with a real cook — the must-mirror checklist is in
   > the transpiler-dev skill ("campaign only exercises Host A" box). To cook an
   > arbitrary shader: wrap its GLSL in an API JSON
   > (`{"Shader":{"info":{...},"renderpass":[{"code":<glsl>,"type":"image","name":"Image","inputs":[],"outputs":[{"channel":0}]}]}}`)
   > and run `builder_cook_headless.py` with `HSHADERTOY_ROOT` +
   > `HOUDINI_OCL_PATH=<repo>/houdini/ocl;&`.
9. **Record**: append a dated entry to `PROGRESS.md` (what changed, files,
   tests added, unit-suite result, delta: fixed/regressed/net, any unmasked
   categories, commit hash). Update the item's status in `BACKLOG.md`.
10. **Hand off**: overwrite `NEXT_SESSION.md` with the next category + any
   gotchas you discovered. **Keep the "READ FIRST, in this order" header** (point
   at README → this file → the BACKLOG item) so the owner can start the next
   session by simply pointing at `NEXT_SESSION.md`. Update the campaign memory
   pointer if the plan shifted materially.
11. **Commit** (only when unit suite is green and no regressions). Branch off
    `main` first if on `main`. Conventional message, e.g.
    `fix(transpiler): category D — overloadable user functions (+31 shaders)`.

---

## Orchestration & subagents (context efficiency)

Sanctioned by the campaign owner for this workflow. Use them to keep noisy
output out of the orchestrator's context — but prefer inline work for the
tightly-coupled transformer edits themselves (a subagent would re-derive the
whole AST model, and you must hold the diff in context to reason about
regressions).

- **Investigation (read-only `Explore`/`general-purpose`):** for an
  *un-root-caused* category (notably **P**, and deep Wave-3 items). Brief:
  "from `corpus.py list <CAT>`, read these N error logs + the relevant
  transformer/emitter region; confirm the root cause; give the exact edit
  location (file:function) + 2-3 minimal repro snippets; report back compactly."
  You keep only the conclusion, not the logs. **Already-root-caused Wave-1/2
  items in BACKLOG.md don't need this.**
- **Verification (`fork`):** offload the noisy proof step — "run `pytest tests/ -v`
  and `campaign.py test --ids ... --force` + `report` + `corpus.py delta`;
  report suite result, net, fixed/regressed ids." The fork inherits context,
  runs in background, and keeps the huge test logs out of yours. Optional: a
  fresh, context-rich session can just run these inline.
- **Never** parallelize two sessions editing the transpiler at once — fixes
  interact (e.g. D↔W, G↔L). One category in flight at a time.

---

## Houdini "handoff" — runtime headers are LIVE (mostly no handoff)

The runtime headers in `houdini/ocl/include/` (`glslHelpers.h`, `matrix_ops.h`,
`textureHelpers.h`, `matrix_types.h` — but NOT `shadertoyInputs.h`, which is an
un-included documentation mirror of the HDA code_header) are **live-`#include`d**
by BOTH the campaign (`tests/ocl/main_header.cl` `#include`s them; build_options
adds `-I houdini/ocl/include`) AND live Houdini (HDA `code_header` `#include`s
them, resolved via the package env `HOUDINI_OCL_PATH = houdini/ocl`). Nothing is
flattened/embedded — **so editing one of these `.h` files takes effect
immediately in both the campaign and live Houdini renders. NO regeneration, NO
handoff.** (Verified 2026-06-25 — texture-bias overloads went live with zero
owner action; see `HOUDINI_HANDOFF.md`.)

This means **categories needing only a runtime-header helper are directly doable
and measured by the campaign** — e.g. category M's `texture(ch,uv,bias)`
overloads (DONE, +11). Z (sampler threading) and parts of Q may be a transformer
change + a header helper, both doable.

The only true handoff: a fix that must change the HDA `code_header` **structure**
(the `#bind` lines, `static` global decls, the `SHADERTOY_INPUTS` macro,
`shadertoy_cubemap`/`DO_CUBEMAP`) — those live in `main_header.cl` directly (not
`#include`d) and in the OTL, so the owner must regenerate. For those: write the
exact diff + steps to `HOUDINI_HANDOFF.md`, mark the BACKLOG item
`BLOCKED-ON-HOUDINI`, move on. Still prefer a transformer-only alternative when
one exists (e.g. category X mapped `uintBitsToFloat`→`as_float` in the
transformer).

---

## Guardrails

- **No fix without a unit test.** TDD is mandatory (`TESTING.md`).
- **Full unit suite green every session.** No commit otherwise.
- **No PASS→FAIL regressions.** `corpus.py delta` must show none.
- **Major redesigns need owner approval.** If a category needs more than a
  localized change (e.g. B's pointer-param model, A's global-init hoisting, G's
  preprocessor handling may qualify), write the proposed design to the BACKLOG
  item and ASK before implementing.
- **Don't hand-edit `ledger.json`/`failures.csv`.** They are generated.

---

## Files

```
README.md             this workflow
BACKLOG.md            ranked fix plan: root cause, fix location, risk, status
PROGRESS.md           append-only journal of completed sessions (+ baseline)
NEXT_SESSION.md       prompt for the next session (overwritten each time)
baseline_ledger.json  frozen ledger at campaign start (413/999 PASS)
corpus.py             list <CAT> | summary | delta | snapshot
HOUDINI_HANDOFF.md    (created on demand) runtime-header changes for the owner
```
