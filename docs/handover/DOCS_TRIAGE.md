# Documentation triage — audited 2026-07-04

Per-file audit of `docs/` and `.agent/`, spot-checked against code (verdicts
verified, not guessed). Owner's standing rule: **code is ground truth; most
docs are stale.** This table says exactly *which* ones, and what to do.

Part of the handover set: see `ROADMAP.md` (when to execute this cleanup) and
`.claude/skills/onboarding/SKILL.md` (what newcomers should read instead).

## Executive summary

- ~19 of 35 audited files are Nov-2025 historical artifacts, safe to archive.
- **6 docs are load-bearing for live workflows** — fix inbound references
  before moving anything: `docs/transpiler/TESTING.md` (mandated by the fix
  campaign), `.agent/PROGRESS.md`, `.agent/PHASE1/2_FAILURE_ANALYSIS.md`
  (referenced by `tests/campaign/classify.py:6` + campaign README),
  `docs/WGPU-SHADERTOY.md`, and the *src* copy of the transpiler spec.
- **Single biggest win:** `GLSL_TO_OPENCL_SPEC.md` exists in TWO diverging
  copies. The LIVE one is `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md`
  (1024 lines, newer); `docs/transpiler/GLSL_TO_OPENCL_SPEC.md` (982 lines) is
  a stale duplicate. `CLAUDE.md` was repointed during the handover
  (2026-07-04), but `tests/TRANSPILE.md:158,200` and
  `examples/QUICK_REFERENCE.md:203` still reference the stale copy.
  Neither copy documents the ~10 fix-campaign transformation categories merged
  in 2026 (sampler, texture-bias, global hoisting, vector-conversion ctors,
  array ctors, …) — the src copy needs a content update too.
- An archive convention already exists: `.archive/docs/` — use it.

## Triage table

| path | last commit | verdict | action |
|---|---|---|---|
| `README.md` | 2026-06-18 | PARTIALLY-OUTDATED — "custom mainImage param names will fail" is now false (`transpile_glsl.py:30-57` rewrites them); dFdx-passthrough claim still true | update |
| `CLAUDE.md` | 2026-07-04 | UPDATED during handover — now points at skills, campaign state, live spec | keep current |
| `docs/PROJECT_SPEC.md` | 2026-06-12 | PARTIALLY-OUTDATED — transpiler still "[ ]" unchecked; wrong API script path (`houdini/scripts/python/api/…`) | update |
| `docs/RULES.md` | 2025-11-25 | CURRENT (minor drift: shader counts) | keep |
| `docs/WGPU-SHADERTOY.md` | 2026-07-04 | CURRENT (owner-written; describes sibling repo `C:\dev\wgpu_shadertoy`) | keep |
| `docs/api/SHADERTOY_API.md` | 2025-10-25 | CURRENT reference; missing the Cloudflare TLS-block caveat | keep + add caveat |
| `docs/builder/BUILDER_HDA_SPEC.md` | 2025-10-25 | PARTIALLY-OUTDATED — references nonexistent `shadertoy_hda_params.ds/.json`; stale G:-drive/WSL paths; wrong wfffRN id/URL | update |
| `docs/editor/EDITOR_SPEC.md` | 2025-11-24 | CURRENT design record (editor.py implements it) | keep |
| `docs/transpiler/GLSL_TO_OPENCL_SPEC.md` | 2025-11-24 | OUTDATED — stale duplicate of the src copy | delete; repoint `CLAUDE.md:4`, `tests/TRANSPILE.md:158,200`, `examples/QUICK_REFERENCE.md:203` to `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md` |
| `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md` | 2025-11-27 | LIVE copy but pre-campaign content (still lists sampler support as "future") | update with 2026 categories |
| `docs/transpiler/TRANSPILER_SPEC.md` | 2025-11-24 | PARTIALLY-OUTDATED — presents long-decided proposals as open; empty Testing section | update or archive |
| `docs/transpiler/TESTING.md` | 2025-10-28 | PARTIALLY-OUTDATED — TDD protocol is LIVE (mandated by `tests/fixcampaign/README.md:54,151`) but Week-N plan + `pytest tests/ -v` gate are stale (gate is `tests/unit/` only) | update, do NOT delete |
| `docs/transpiler/ROADMAP.md` | 2025-11-11 | HISTORICAL — superseded (older rev already in `.archive/docs/`) | archive |
| `docs/transpiler/THE_MATRIX_PROBLEM.md` | 2025-11-04 | HISTORICAL — cites deleted `transformer.py`; problem solved | archive |
| `docs/transpiler/shadertoy.md` | 2025-10-01 | REFERENCE-EXTERNAL (Shadertoy help copy) | keep |
| `docs/transpiler/GLSLangSpec.md` | 2025-11-18 | REFERENCE-EXTERNAL (Khronos GLSL 4.40 spec, 11k lines; PDF alongside) | keep |
| `docs/transpiler/GLSLangSpecMatrix.md` | 2025-11-18 | REFERENCE-EXTERNAL (matrix chapter extract) | keep |
| `docs/transpiler/SPEC_OPTIMIZATION_REPORT.md` | 2025-11-09 | HISTORICAL — numbers meaningless now | archive or delete |
| `docs/transpiler/research/PARSER-RESEARCH.md` | 2025-10-28 | HISTORICAL decision record (tree-sitter choice — adopted); dead link at line 588 | keep as ADR |
| `.agent/PROGRESS.md` | 2025-11-25 | PARTIALLY-OUTDATED — frozen pre-campaign; live-referenced by `CLAUDE.md:5`, `transformed_ast.py:512`, RULES.md | update: add pointer to campaign state; do NOT delete |
| `.agent/PHASE1_FAILURE_ANALYSIS.md` | 2026-06-23 | HISTORICAL but live-referenced (`classify.py:6`) — taxonomy provenance | keep |
| `.agent/PHASE2_FAILURE_ANALYSIS.md` | 2026-06-23 | same | keep |
| `.agent/PHASE2_PIPELINE_GAPS_INVESTIGATION.md` | 2026-06-23 | HISTORICAL — both gaps fixed | archive |
| `.agent/HOUDINI_INTEGRATION*.md` (5 files) | 2025-11-13/14 | HISTORICAL — integration shipped | archive |
| `.agent/PROJECT_COMPLETION_REPORT.md` | 2025-11-13 | HISTORICAL — "PROJECT COMPLETE" claim superseded by campaign reality | archive |
| `.agent/MATRIX_STRUCT_REFACTOR_PLAN.md` | 2025-11-18 | HISTORICAL — refactor shipped (`matrix_types.h`/`matrix_ops.h`) | archive |
| `.agent/PHASE_3/5/6/7_PROMPT.md` | 2025-11-18 | HISTORICAL one-shot prompts | delete or archive |
| `.agent/SESSION_*.md` (4 files) | 2025-11-12/15 | HISTORICAL session logs (mostly for abandoned approaches) | archive |

## Cross-reference map (fix these BEFORE moving/deleting)

| doc | referenced from |
|---|---|
| `docs/transpiler/GLSL_TO_OPENCL_SPEC.md` (stale) | `CLAUDE.md:4`; `tests/TRANSPILE.md:158,200`; `examples/QUICK_REFERENCE.md:203` |
| `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md` (live) | `README.md:80`; `docs/RULES.md:7`; `.agent/PROGRESS.md:14` |
| `.agent/PROGRESS.md` | `CLAUDE.md:5`; `src/.../transformer/transformed_ast.py:512` (docstring!); `docs/RULES.md:6,23` |
| `docs/transpiler/TESTING.md` | `tests/fixcampaign/README.md` (§Session protocol step 3 + §Guardrails — mandatory TDD protocol) |
| `.agent/PHASE1/2_FAILURE_ANALYSIS.md` | `tests/campaign/classify.py:6`; `tests/campaign/README.md:8-9` |
| `docs/WGPU-SHADERTOY.md` | `.claude/skills/render-compare/SKILL.md` |

Not referenced anywhere live (safe to move): `docs/transpiler/ROADMAP.md`,
`THE_MATRIX_PROBLEM.md`, `SPEC_OPTIMIZATION_REPORT.md`, `shadertoy.md`,
`GLSLangSpec*.md`, `PARSER-RESEARCH.md`, and all `.agent/HOUDINI_*`,
`PHASE_*_PROMPT`, `SESSION_*`, `PROJECT_COMPLETION_REPORT`,
`MATRIX_STRUCT_REFACTOR_PLAN`.

## Gaps the skill library now fills (previously missing entirely)

1. Accurate architecture overview → `.claude/skills/transpiler-dev/SKILL.md`
2. Campaign workflow discoverability → `.claude/skills/{mass-test-campaign,fix-campaign}/SKILL.md`
3. HDA/runtime contract (main_header.cl anatomy, live-header boundary) → `.claude/skills/houdini-testing/SKILL.md`
4. Environment setup + Cloudflare workaround → `.claude/skills/{onboarding,shadertoy-api}/SKILL.md`
5. One canonical spec: still a TODO — consolidate the two spec copies (ROADMAP quick-win #1).
