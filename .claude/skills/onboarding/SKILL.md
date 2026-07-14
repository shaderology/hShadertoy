---
name: onboarding
description: START HERE for anyone new to hShadertoy — what the project is, repo map, environment, the non-negotiable rules, current campaign state, and which skill/doc to use for which job. Use at the start of any session when unfamiliar with the project, when unsure which docs are trustworthy, or when deciding where a piece of work belongs.
---

# hShadertoy onboarding

## What this project is

hShadertoy runs **Shadertoy shaders inside Houdini**. A Shadertoy shader
(GLSL, possibly multi-pass: image/buffers/cubemap/common/sound) is fetched
from the Shadertoy API, **transpiled from GLSL to OpenCL C**, and executed by
Houdini's Copernicus (COP) OpenCL nodes, orchestrated by an HDA.

Components, by importance:

1. **The transpiler** — `src/glsl_to_opencl/` (~7,900 lines Python,
   tree-sitter-glsl parser). The most complicated part; nearly all ongoing
   work is here. See `.claude/skills/transpiler-dev/SKILL.md`.
2. **Runtime OpenCL headers** — `houdini/ocl/include/` (`glslHelpers.h`,
   `textureHelpers.h`, `matrix_types.h`, `matrix_ops.h`): GLSL semantics
   emulation the transpiled code calls into. Live-editable (see
   houdini-testing skill for the boundary). A fifth file there,
   `shadertoyInputs.h`, is a read-only documentation mirror of the HDA code
   header — nothing includes it, and the owner may keep uncommitted edits in
   it.
3. **The HDA** — `houdini/otls/hShadertoy.hda`, binary, **owner-maintained,
   never edit**. Houdini-side Python (builder/editor/api) lives in
   `houdini/scripts/python/hshadertoy/`.
4. **Test infrastructure** — unit suite (`tests/unit/`, the TDD gate), dev
   entry points (`tests/transpile.py`, `tests/compilecl.py`), and two
   interlocking campaigns in `tests/campaign/` + `tests/fixcampaign/`.

## Current state (written 2026-07-04 — verify against the state files)

Two campaigns are ACTIVE and are the main ongoing work:

- **Mass-test campaign** (`tests/campaign/`) measures the transpiler against
  real Shadertoy shaders. 999 tested so far; **564/999 pass** (baseline was
  413). Skill: `mass-test-campaign`.
- **Fix campaign** (`tests/fixcampaign/`) fixes the failure categories the
  mass-test found, one category per session, TDD-first, zero-regression gate.
  11 sessions done, all merged. Next session's brief is ALWAYS
  `tests/fixcampaign/NEXT_SESSION.md`. Skill: `fix-campaign`.

After the campaigns wind down, the plan is in `docs/handover/ROADMAP.md`
(optimization, cleanup, render-correctness validation, maintenance).

## Ground truth hierarchy (owner's rule)

1. **Code** — always wins.
2. **Campaign state files** — `tests/campaign/ledger.json` (generated, never
   hand-edit), `tests/fixcampaign/{BACKLOG,PROGRESS,NEXT_SESSION}.md`.
3. **Skills** (`.claude/skills/`) and **handover docs** (`docs/handover/`) —
   maintained; fix them when they drift.
4. **`docs/` and `.agent/`** — mostly stale Nov-2025 artifacts. Trust only
   after checking `docs/handover/DOCS_TRIAGE.md`. Notably: `.agent/PROGRESS.md`
   is frozen pre-campaign (the project did NOT "complete" in Nov 2025), and
   the live transpiler spec copy is `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md`,
   not the one in `docs/transpiler/`.

## Environment (this machine)

- Windows 11; **system Python 3.11.7** (`python` on PATH), **no venv**;
  deps in `requirements.txt` (pyopencl, tree-sitter, tree-sitter-glsl, pytest).
- OpenCL device: pyopencl platform[0]/device[0] = NVIDIA RTX 2070. Some paths
  are machine-hardcoded (`tests/compilecl.py` defaults, `tests/build_options.json`
  include paths) — porting to a new machine means updating those.
- Houdini **21.0.440** at the default install path; hython for headless HDA
  tests (houdini-testing skill).
- Run everything from the repo root: `C:\dev\hShadertoy`.

## The non-negotiable rules

1. **TDD.** No transpiler fix without a failing unit test first
   (`docs/transpiler/TESTING.md` protocol section is live policy).
2. **The unit gate is `python -m pytest tests/unit/ -q`** — NOT `tests/`
   (integration is pre-existing-broken; ~6 skips are permanent placeholders).
   It must be green before any commit.
3. **Zero PASS→FAIL regressions** on the shader corpus (fix-campaign skill has
   the measurement procedure).
4. **Never hand-edit generated files**: `ledger.json`, `failures.csv`,
   `REPORT.md`, and the generated region of `tests/ocl/main_header.cl`.
5. **Never edit the HDA** or the HDA-generated/code-header structure — write a
   handoff note instead (houdini-testing skill).
6. **Never stage files you didn't change** (`git add` by name, never `-A`) —
   the owner keeps uncommitted local edits in the tree.
7. **Branch → full test → merge-if-green.** Don't commit on main directly.
   Conventional messages, e.g. `fix(transpiler): category S — ... (+N shaders)`.
8. **Major redesigns need owner approval first**; for normal decisions,
   recommend one option and proceed — don't ask the owner to pick.

## Which skill for which job

| Job | Skill |
|---|---|
| Understand/modify the transpiler | `transpiler-dev` |
| Fix a failure category end-to-end | `fix-campaign` |
| Measure pass rates / test shader ids | `mass-test-campaign` |
| Anything Houdini/hython/HDA/runtime-header | `houdini-testing` |
| Fetch shaders / API 403s / budget | `shadertoy-api` |
| Prove render correctness (build the image diff) | `render-compare` |
| Plan post-campaign work / refactoring | `docs/handover/ROADMAP.md` + `TRANSPILER_REVIEW.md` |
| What Shadertoy ITSELF does (headers, main(), alpha, fragCoord) | `docs/handover/SHADERTOY_SITE_NOTES.md` |
| Entry-point model (single-TU pipeline) + its open F-items | `docs/handover/ENTRYPOINT_REDESIGN.md` |
