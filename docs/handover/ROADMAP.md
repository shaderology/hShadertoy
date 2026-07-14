# hShadertoy development roadmap (handover, 2026-07-04)

Written by the departing lead for whoever carries the project forward —
including AI-driven sessions. Each phase is sized so a single focused session
(human or model) can complete a meaningful unit. Read
`.claude/skills/onboarding/SKILL.md` first if you're new.

R-numbers refer to the refactoring catalogue in `TRANSPILER_REVIEW.md`.

---

## Phase 0 — finish the campaigns (ACTIVE, keep the cadence)

The two campaigns (`tests/campaign/`, `tests/fixcampaign/`) are the current
main work. Continue per their skills, one fix session at a time
(`NEXT_SESSION.md` is always the brief; category S is next as of writing).

**Owner's scope policy applies:** fix (common pattern) AND (localized change);
skip (edge case) OR (rewrite-sized) — rare failures are acceptable.

**Completion criteria** (when to declare Phase 0 done):
- every remaining category is either FIXED, dropped-by-policy (recorded in
  `BACKLOG.md` with reasons), or `BLOCKED-ON-HOUDINI`/`NEEDS-APPROVAL`;
- categories P (parse catch-all) and G (preprocessor) have at least been
  *clustered* and their fixable sub-shapes peeled off;
- optionally: mass-test sessions 11–15 fetched+tested (API budget allowing) to
  confirm the pass-rate on fresh data.

Realistic end state: ~70–80% corpus pass rate; the rest is G/P long tail
whose proper fix is R14/R15 (Phase 3), not more session-sized patches.

## Phase 1 — silent-correctness quick wins (do EARLY, can interleave with Phase 0)

These fix bugs that render wrong while compiling fine — invisible to all
current metrics. See `TRANSPILER_REVIEW.md §0`. Run each as a normal
fix-campaign-style session (TDD, corpus gate):

1. **R7 — Houdini host consumes `hoisted_global_inits`** (~6 lines + test).
   The shipping path silently drops global initializers today. Most urgent.
2. **R1 — fail loudly on unknown AST nodes** (+ new taxonomy category for
   what it surfaces). Converts silent deletions into visible failures.
3. **R2 — `switch` support** (currently silently deleted).
4. **R13 — postfix `++/--` fidelity.**
5. **Houdini test script**: `template_load_headless.py` collects cook errors
   but never prints them and exits 0 regardless — make cook failures fail
   loudly (see houdini-testing skill).
6. Quantify exposure: grep the 999 cached shader sources for `switch` and
   expression-position `++`/`--` to size how many "passing" shaders are
   affected; record in `BACKLOG.md`.

## Phase 2 — render-correctness harness

Build the wgpu-shadertoy ⇄ Houdini image-comparison pipeline. The full
implementation spec (phases P1–P4, metrics, calibration, corpus rollout,
pitfalls) is in `.claude/skills/render-compare/SKILL.md`. This is the tool
that would have caught every Phase-1 bug automatically; after it exists, "0
regressions" can mean *pixels*, not just compiles.

Deliverable: `tests/rendercompare/` mirroring the campaign pattern, plus a
ranked render-mismatch report reviewed with the owner. Mismatch clusters
become new fix-campaign categories (two-letter codes).

## Phase 3 — consolidation & refactor program (the "clean core")

Only start after Phase 1; run steps as gated sessions in this order:

1. **R3, R5, R6** — dispatch-dict hoist, one name-map module, one builtin
   registry. Mechanical de-duplication.
2. **R4** — delete dead code (~1,600 lines: broken `codegen/code_generator.py`
   + `formatting.py`, dead `parser/preprocessor.py`, `analyzer/metadata.py`,
   `validator/`, and the legacy emitter after porting its for-init fix and
   repointing its unit tests). Also repair-or-delete `tests/integration/`
   (broken on clean HEAD; imports dead code).
3. **R8** — one shared host-postprocessing module (ends Host A/Host B drift).
4. **R11 + R12** — real package API + IR-based header/kernel emission
   (removes parse #4 and the substring surgery). Golden-file diff against all
   passing corpus artifacts is the gate.
5. **R9, R10** — scoped `local_types`, arity-keyed registries.
6. **R15 (owner sign-off first)** — resurrect the type system as an
   annotation pass, strangler-pattern alongside `local_types`.
7. **R14, R16 (owner sign-off first)** — structural preprocessor (this is the
   real fix for category G), then pass decomposition.

Every step keeps the pipeline runnable and the corpus green — never a
big-bang rewrite (`TRANSPILER_REVIEW.md §4`).

## Phase 4 — repo & docs cleanup

Execute `DOCS_TRIAGE.md` (it has the exact per-file actions and the inbound
references that must be repointed first). Highlights:
- consolidate the duplicated `GLSL_TO_OPENCL_SPEC.md` (live copy is in `src/`)
  and repoint `CLAUDE.md`; update the spec with the ~10 merged 2026 categories;
- rewrite `CLAUDE.md` to point at the skills + campaign state instead of
  frozen Nov-2025 docs;
- archive the ~19 historical docs into the existing `.archive/` convention;
- root clutter: delete/move the six root `test_*.py` debug scripts (committed
  in a commit literally named "backup"), `testsintegration__init__.py`, and
  sweep `tests/` legacy (`phase1_harness.py`, `phase2_*.py` can archive, but
  **keep `tests/shaders/api/json{,_sp}/` caches** — the campaign reads them
  and the budget seeder counts them);
- refresh `README.md` (its "custom mainImage params fail" limitation is no
  longer true).

## Phase 5 — maintenance & deployment posture

- **Bug intake loop:** a reported broken shader → fetch by id (shadertoy-api
  skill) → `campaign.py test --ids <id> --force` → classify → if render-wrong
  but compile-OK, run it through the Phase-2 harness → minimal repro → TDD fix
  session. This is just the fix-campaign skill applied to single reports.
- **Portability debt:** `tests/compilecl.py` defaults and
  `tests/build_options.json` hardcode `C:/dev/hShadertoy` and Houdini
  21.0.440 paths; the API key is hardcoded at `campaign.py:75`. Move to a
  small config (env vars or a git-ignored `local_config.json`) when a second
  machine or contributor appears — not before.
- **Dependency watch:** `tree-sitter`/`tree-sitter-glsl` upgrades can change
  parse trees — pin as in `requirements.txt`, and re-run the full corpus as
  the acceptance test for any bump. Same for pyopencl and Houdini upgrades
  (new Houdini = re-verify `build_options.json` include path + re-capture
  `main_header.cl` with the owner).
- **Public sync:** `syncpublic.ps1` mirrors to the public repo — review what
  it excludes before adding secrets/keys anywhere.
- Transpiler runtime performance is a non-goal: ~ms per shader, dwarfed by
  OpenCL compile times. Don't spend sessions optimizing Python.

## Standing rules (apply to every phase)

All eight rules in `.claude/skills/onboarding/SKILL.md` §"non-negotiable
rules" — TDD, unit gate `tests/unit/`, zero corpus regressions, generated
files untouched, HDA untouched, stage-by-name only, branch→green→merge,
owner approval for rewrite-class work.
