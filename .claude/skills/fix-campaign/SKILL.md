---
name: fix-campaign
description: Run one transpiler bug-fix session end-to-end (tests/fixcampaign/) — TDD-first fix workflow, regression gates, delta measurement, backlog/handoff bookkeeping, and the accumulated traps from 11+ sessions. Use when fixing a transpiler failure category, resuming the fix campaign, proving a fix against the shader corpus, or deciding whether a fix needs owner approval.
---

# Fix-campaign session

**Authoritative protocol: `tests/fixcampaign/README.md` (the 10-step session
protocol + guardrails). The per-session brief is `tests/fixcampaign/NEXT_SESSION.md`
(rewritten each session).** Read order every session:

1. `tests/fixcampaign/NEXT_SESSION.md` — what to do THIS session
2. `tests/fixcampaign/README.md` — the protocol you must follow
3. Your category's row in `tests/fixcampaign/BACKLOG.md`
4. This skill's traps section (durable knowledge that outlives session files)

One category per session. Never two transpiler-editing sessions in parallel —
fixes interact.

## The gates (all mandatory, in this order)

1. **Failing unit test first** in `tests/unit/test_*.py` (minimal GLSL-in →
   OpenCL-substring-out repro). No fix without one. The TDD *policy* comes from
   `docs/transpiler/TESTING.md`, but its mechanics are stale (its `transpiler`
   fixture is a permanent skip-stub; its `pytest tests/` commands predate the
   unit-only gate) — copy the import pattern from a sibling
   `tests/unit/test_transformer_*.py` file instead.
2. **Unit suite green:** `pytest tests/unit/ -q`. The gate is `tests/unit/`
   ONLY — `tests/integration` is pre-existing-broken on clean HEAD, and ~47
   OpenCL `BUILD_PROGRAM_FAILURE` tests fail on baseline via the plain pytest
   path (no GPU build target there); the campaign's own pyopencl path is what
   works. If `test_dummy.py::test_test_directories_exist` fails, recreate the
   empty untracked dirs `tests/fixtures/{simple_shaders,complex_shaders,reference_images}`.
3. **Corpus proof:** back up `tests/campaign/ledger.json` to a scratch dir
   FIRST. Then targeted `python tests/campaign/campaign.py test --ids <targets>
   --force`, then a **full-corpus re-test if the change touches shared emission
   paths** (anything in emitters/declaration handling does), then
   `campaign.py report`. Full re-test = sessions 1-5,7,8,10 whole (`--session N
   --force`, timeout 600000) + sessions 6 and 9 in 25-id `--ids` batches — see
   `.claude/skills/mass-test-campaign/SKILL.md` for why.
4. **Delta: compute it from the ledgers directly** — diff the set of ids with
   `overall == "PASS"` between your backup and the live ledger. **Do NOT trust
   `corpus.py delta`'s FIXED/REGRESSED lists** (it string-compares `=="FAIL"`
   but the ledger stores `COMPILE_FAIL`/`TRANSPILE_FAIL`); its NET count is
   fine. **REGRESSED must be exactly 0** — any PASS→FAIL: stop, root-cause,
   fix or revert.
5. **Record + hand off:** dated entry in `PROGRESS.md`; update the BACKLOG
   row *in place* — the convention is a bold `**DONE (Session N, date, +X
   PASS, 0 regressed).**` prefix plus root-cause/residual notes, matching the
   N/K rows (there is no separate Status column in the Wave-3 table); overwrite
   `NEXT_SESSION.md` (keep its "READ FIRST, in this order" header); commit on
   `fix/transpiler-<cat>` off main, message like
   `fix(transpiler): category S — <summary> (+N shaders)`; merge to main only
   if all gates passed (owner-approved gate: branch → full-test → merge-if-green).

## Scope policy (owner-directed, 2026-06-25)

Optimize for **real artist shaders, not corpus completeness**. Decision rule:
fix it if (common pattern) AND (localized fix); skip if (edge case) OR
(requires rewrite). Rare failures are acceptable. **Major redesigns (e.g.
category G preprocessor handling) need owner approval BEFORE implementing** —
write the design into the BACKLOG row and ask. Don't ask the owner to pick
between technical alternatives — recommend one, then branch→full-test→merge.

## Traps that have each cost a real session (do not relearn these)

- **`local_types` name split (bit twice):** it holds GLSL names (`vec3`) for
  declarations but OpenCL names (`float3`) for parameters. Any
  `TYPE_NAME_MAP.get(inferred)` must normalize via the module-level
  `OPENCL_TO_GLSL_NAME` map in `ast_transformer.py`.
- **Two emitters must mirror:** `codegen/opencl_emitter.py` is production;
  `transformer/code_emitter.py` is dead in production but the unit suite still
  exercises it. An emission change goes in BOTH or the suite lies to you
  (until refactor R4 in `docs/handover/TRANSPILER_REVIEW.md` deletes it).
- **Entry points are special:** `mainImage`, `mainCubemap`, `mainSound`,
  `mainVR` must be excluded from function-level transforms like
  `__attribute__((overloadable))` (`_transform_function_definition`); their
  `fragColor` out-param is a host `@KERNEL` local excluded via
  `transformer.entry_function`. The campaign harness is **mainImage-only** —
  cubemap/sound regressions are invisible to it; validate those by rendering
  shader `wfffRN` in Houdini (`.claude/skills/houdini-testing/SKILL.md`).
- **Probe the real compiler before designing around spec limits.** Campaign
  builds pass NO `-cl-std` → permissive CL2.0-ish mode: compound literals and
  program-scope globals just work (Session 11 nearly built an unnecessary
  hoisting subsystem). But `__constant` arrays can't be passed to
  private-pointer params — avoid emitting them.
- **Vector comparisons are typed `vecN` (float!)** by `_infer_binary_op_type`;
  detect bool masks structurally via `_is_bool_mask`, never by element type.
  OpenCL vector relationals are **-1-for-true** vs GLSL's 1 (`&1` normalizes).
- **`tests/ocl/main_header.cl` is HDA-generated** from ~line 1404 on (`#bind`
  lines, static globals, `SHADERTOY_INPUTS`, cubemap glue; review copy:
  `houdini/ocl/include/shadertoyInputs.h`). Structural changes there → ASK the
  owner. The runtime headers `glslHelpers.h`/`textureHelpers.h`/`matrix_ops.h`
  in `houdini/ocl/include/` ARE live-editable and take effect in both campaign
  and Houdini immediately (no handoff — see houdini-testing skill).
- **Never stage files you didn't change** — the owner keeps uncommitted local
  edits in the working tree (e.g. `shadertoyInputs.h`). Stage your files by
  name; never `git add -A`.
- Transpile-stage fixes **unmask** downstream errors (counts of other
  categories grow — expected, judge by net PASS). Compile-stage fixes flip
  sole-blockers directly.
- Classifier `*` sole-blocker stars are unreliable both ways; read the actual
  error logs (`tests/campaign/artifacts/<id>/`) before believing attribution.
- `BACKLOG.md` file:line pointers drift as the transformer grows — trust the
  function names (grep them), never the cited line numbers. Same for counts in
  session briefs: `corpus.py list <CAT>` is the live truth, briefs are
  snapshots.

## Orchestration (sanctioned)

- Un-root-caused category (e.g. the P catch-all): spawn a read-only
  investigation subagent to cluster error logs and locate the edit site; keep
  only its conclusions in context.
- The noisy proof step (full pytest + corpus re-test + report + delta) can go
  to a fork/subagent that reports back suite result, net, fixed/regressed ids.
- The transformer edit itself: do it inline, holding the diff in your context.
