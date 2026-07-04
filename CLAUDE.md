# hShadertoy — GLSL to OpenCL transpiler for Houdini

**New here (human or model)? Read `.claude/skills/onboarding/SKILL.md` first.**
It maps the project, the rules, and which skill covers which job.

Skills (`.claude/skills/`): `onboarding` · `transpiler-dev` · `fix-campaign` ·
`mass-test-campaign` · `houdini-testing` · `shadertoy-api` · `render-compare`

Ground truth & state:
- Code wins over all docs. The live transpiler spec is
  `src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md` (the copy under
  `docs/transpiler/` is a stale duplicate — see `docs/handover/DOCS_TRIAGE.md`).
- Current work = the two campaigns: `tests/campaign/` (measure) +
  `tests/fixcampaign/` (fix; next-session brief in
  `tests/fixcampaign/NEXT_SESSION.md`).
- Future work & refactoring plan: `docs/handover/ROADMAP.md` +
  `docs/handover/TRANSPILER_REVIEW.md`.
- `.agent/PROGRESS.md` is a frozen Nov-2025 archive — do NOT treat it as
  current status.

Hard rules (details in onboarding skill): TDD; gate = `python -m pytest
tests/unit/ -q`; zero corpus regressions; never hand-edit generated files
(`ledger.json`, `failures.csv`, `REPORT.md`); never edit the HDA; never stage
files you didn't change; branch → green → merge.
