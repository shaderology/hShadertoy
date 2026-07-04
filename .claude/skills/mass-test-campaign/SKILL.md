---
name: mass-test-campaign
description: Run the Shadertoy mass-test campaign (tests/campaign/) — fetch/test/report stages, ledger.json rules, API budget, failure-taxonomy classifier maintenance, and the long-compile batching workaround. Use when measuring transpiler pass rate, testing shader ids, resuming a campaign session, re-testing after a fix, regenerating REPORT.md/failures.csv, or refining classify.py.
---

# Mass-test campaign operations

**Purpose: measurement only.** This campaign transpiles+compiles real Shadertoy
shaders and catalogues failures. It NEVER fixes the transpiler — fixes happen in
the fix-campaign (see `.claude/skills/fix-campaign/SKILL.md`), which *uses* this
campaign's `test --force` + `report` to prove its fixes.

**Authoritative workflow doc: `tests/campaign/README.md`. Read it first.**
This skill adds the operational knowledge that isn't (or can't stay) in that
README.

## The iron rules

1. `tests/campaign/ledger.json` is the single source of truth for BOTH
   campaigns. **Never hand-edit it**, `failures.csv`, or `REPORT.md` — they
   change only because `campaign.py test`/`report` reran.
2. `fetch` is the ONLY stage that spends Shadertoy API budget (1500
   calls/month, self-tracked in `api_budget.json`; the API itself signals
   exhaustion with `{"Error":"Too many requests"}` — fetch detects it and stops
   safely). Note there are TWO distinct stop messages: `MONTHLY BUDGET
   EXHAUSTED` is the *local pre-emptive* check (no API call was made);
   `RATE-LIMITED` is the real server-side signal (which also pins the local
   tally to the cap). Both are safe to just re-run next month. `test` and
   `report` are offline and idempotent — re-run freely.
3. Every stage writes state after each shader — crashes/timeouts never lose
   progress; just re-run.

## Command crib (Windows, plain `python`, repo root)

```bash
python tests/campaign/campaign.py status              # always start here
python tests/campaign/campaign.py select              # (re)build selection.json — one-time bootstrap, safe to re-run
python tests/campaign/campaign.py fetch --session N   # ONLY stage costing API
python tests/campaign/campaign.py fetch --ids a,b,c   # cache-first, targeted
python tests/campaign/campaign.py test  --session N   # offline
python tests/campaign/campaign.py test  --ids a,b --force   # re-test after a fix
python tests/campaign/campaign.py report              # regen REPORT.md + failures.csv
python tests/campaign/campaign.py reclassify          # re-bucket stored logs (no recompile)
```

`--force` re-tests shaders already marked tested; without it, tested ids are
skipped. After ANY `test` that changed results, run `report`.

## State as of 2026-07-04 (verify with `status`, don't trust this line)

Sessions 1–10 fetched AND tested (999 shaders; do not re-fetch). Pass count
moves as the fix-campaign lands (baseline 413 → 564 at last count). Next
unfetched session: **11**. The budget tally (`api_budget.json`) re-seeds each
calendar month — always read the remaining budget from `status`, never from
notes like this one.

**Two unrelated "session" numberings exist** — mass-test sessions are batches
of 100 shader ids (1…42, from `selection.json`); fix-campaign "Sessions" are
numbered work sessions in `tests/fixcampaign/PROGRESS.md`. Disambiguation
rule: `campaign.py --session N` only ever means a `selection.json` batch; a
session number quoted from `PROGRESS.md`/`BACKLOG.md`/`NEXT_SESSION.md`
context is a fix-campaign work session.

## Long-compile batching (learned the hard way — sessions 6 and 9)

Some texture-family shaders compile for **minutes** on a cold NVIDIA compiler
cache (`MlfcW4`, `MttcWr`). A full `test --session 6|9 --force` exceeds the
10-minute Bash tool cap and gets killed mid-run (per-shader ledger writes
survive, but you can't tell where it died — don't use artifact mtimes to infer
run progress: `.header.cl`/`.kernel.cl` are written for every transpile-OK
pass, only `*_err.txt` files are failure-only). A forced 100-shader session
takes ~4–5 min warm; all ten ≈ 45 min. Run one session per foreground call —
backgrounded loops over sessions have been killed before.

- Sessions 1–5, 7, 8, 10: one `test --session N --force` per call,
  `timeout: 600000`.
- Sessions 6 and 9: split into **25-id `--ids` batches** (pull each session's
  ids from `ledger.json`'s `session` field).
- `MttcWr` may not finish in 10 min even alone. If your change can't affect its
  output, prove the transpiled output byte-identical (stash → transpile → diff)
  and let its PASS entry stand; otherwise give it a solo slot, and escalate to
  the owner rather than leaving it silently untested.

## What `test` actually does (scope)

Per shader, per renderpass: prepend the `common` pass GLSL (matches Houdini),
transpile via `tests/transpile.py`, compile via `tests/compilecl.py` against
`tests/ocl/main_header.cl` + `-I houdini/ocl/include` on the local GPU
(pyopencl/NVIDIA). `sound` passes are skipped (different signature). It proves
**compile validity only** — never render correctness (see
`.claude/skills/render-compare/SKILL.md`).

## Classifier maintenance (classify.py)

- Regex-based, zero-LLM-cost; taxonomy `A`–`Z` + two-letter `AA`+ codes
  (single letters are exhausted; `I` was deliberately skipped). Definitions +
  difficulty ranks live in `tests/campaign/classify.py`.
- After a `test` run, review the **UNKNOWN** bucket in `REPORT.md` against
  `artifacts/<id>/*_err.txt`; add/refine rules; then `campaign.py reclassify`
  (re-buckets stored logs — no recompile, no API).
- Categories are approximate attribution, not ground truth: tree-sitter reports
  the first ERROR node far from the real cause, so a ParseError pointing at a
  comment/blank line usually means `#define` abuse upstream. The `*`
  sole-blocker stars in fix-campaign listings are unreliable in both
  directions.

## Traps

- Cache layout: `tests/campaign/cache/<id>.json` (inner shader dict) plus
  legacy `tests/shaders/api/json{,_sp}/` (checked first by fetch).
- "Shader not found" ≠ deleted — also returned for non-Public+API shaders.
  These are recorded `unavailable` in the ledger; that's normal.
- Non-ASCII shader titles once crashed fetch printing; fixed (stdout
  `errors="replace"`), but keep console-encoding in mind for new print paths.
- Expect transpile-stage fixes to *unmask* new downstream compile failures —
  category counts can GROW after a genuine improvement. Judge by net PASS.
- `overall: PASS` requires EVERY testable renderpass to compile — fixing a
  shader's image pass flips nothing if a buffer pass carries another category.
- `fetch` statuses `OK`/`UNAVAILABLE` are permanent (no re-fetch, no --force);
  only `ERROR` is retried. Deleting the legacy cache dirs both loses cache and
  mis-seeds a freshly created budget file (they seed the initial tally).
- `classify.py <log.txt>` CLI spot-check feeds the file as compile_err ONLY —
  transpile-stage rules won't fire that way.
