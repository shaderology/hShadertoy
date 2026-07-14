# Phase 5 — Systematic Shadertoy Mass-Test Campaign

Intelligence gathering for the GLSL→OpenCL transpiler. **Bug hunting only — no
transpiler fixes here.** Goal: methodically transpile+compile real Shadertoy
shaders (oldest=simplest first) and build a sortable database of failures ranked
by failure type for future fix work.

Builds on Phase 1/2 (`.agent/PHASE1_FAILURE_ANALYSIS.md`,
`.agent/PHASE2_FAILURE_ANALYSIS.md`) and reuses `tests/transpile.py` +
`tests/compilecl.py`.

> **⚙️ Bug-fixing campaign is now ACTIVE** (started 2026-06-23, baseline 413/999
> PASS) — see `tests/fixcampaign/`. That campaign *fixes* the transpiler; this
> one *measures* it. `ledger.json` is the shared source of truth: fixes are
> proven by re-running `test --ids <fixed> --force` + `report`, which rewrites
> the ledger and regenerates `failures.csv`/`REPORT.md` (never hand-edited).
> **Sessions 1-10 are tested; do not re-fetch.** When fixes mature, resume
> mass-testing from **session 11**, and/or re-measure old sessions with
> `test --session N --force`. The campaign numbers will shift downward as the
> transpiler improves — that is expected.

## How sampling works

`shader_all.json` lists all 41,854 shaders newest→oldest. We take **every 10th
from the oldest end** (`reversed[::10]` → `MsfGRn, MsX3zr, MdlGzn, …`), ~4,186
ids, and process them in **sessions of 100** (oldest first). Each shader costs **1
Shadertoy API call**; the API allows **1500/month**, so fetching is the rate
limit, not Claude.

## Four stages (all idempotent + resumable)

```
select   no API   build the every-10th candidate list + session assignment
fetch    API      download+cache a session's shader JSON (THE ONLY API STEP)
test     offline  transpile+compile every pass, auto-classify, write artifacts
report   no API   aggregate ledger -> ranked failure DB
```

`ledger.json` (keyed by shader id) is the single source of truth; every stage
writes after each shader, so a context-clear or daily-limit never loses progress.

## Resume protocol (start here every session)

```bash
python tests/campaign/campaign.py status            # what's done / budget left
python tests/campaign/campaign.py fetch --session N  # only if budget remains
python tests/campaign/campaign.py test  --session N  # safe to re-run anytime
python tests/campaign/campaign.py report             # regenerate REPORT.md + csv
```

`status` shows, per session: ids, fetched, unavailable, tested, pass, fail, and
how many still need fetching — plus remaining monthly API budget. Pick the lowest
session number with `todo-fetch > 0` (or `tested < fetched`) and continue there.

## Commands

| Command | API | Notes |
|---------|-----|-------|
| `select` | no | (re)builds `selection.json`; safe to re-run |
| `fetch --session N [--limit L] [--dry-run]` | yes | caps NEW calls at `L` (default 100); `--dry-run` plans without calling; cache-first (reuses `tests/shaders/api/json{,_sp}/`) |
| `fetch --ids a,b,c` | maybe | targeted; cache hits cost 0 |
| `test --session N [--force]` | no | `--force` re-tests already-tested shaders |
| `test --ids a,b,c` | no | targeted (debugging) |
| `report` | no | writes `REPORT.md` + `failures.csv` |

## Failure taxonomy (auto-assigned by `classify.py`)

**`classify.py` is the authoritative category list** (`CATEGORY_DESC` +
`CATEGORY_DIFFICULTY`) — the taxonomy has grown to `A`–`Z` (no `I`) plus
two-letter codes `AA`+ as sessions discovered new failure shapes; any list
written here goes stale. The generated `REPORT.md` shows the live ranked
counts per category.

The classifier is regex-based (zero Claude cost). After each `test` run, review
the **UNKNOWN bucket** listed in `REPORT.md` against the full logs in
`artifacts/<id>/*.compile_err.txt` and refine `classify.py` rules as needed.

## Files

```
campaign.py     orchestrator CLI
classify.py     error-text -> taxonomy classifier (importable + CLI)
ledger.json     SOURCE OF TRUTH (per-shader fetch + per-pass results)
selection.json  every-10th candidate list + session numbers
api_budget.json monthly API-call tally + cap (we track our own; no quota API)
cache/<id>.json raw shader JSON
artifacts/<id>/ transpiled .header.cl/.kernel.cl + *_err.txt logs
REPORT.md       generated ranked report
failures.csv    generated flat table (sort/pivot anywhere)
```

## Notes / scope
- We test transpile + **compile** validity, not render correctness. Texture
  (`iChannel`) calls compile via `textureHelpers.h`; buffer inputs are empty
  `IMX_Layer*` but still compile.
- `sound` passes are skipped (different signature); `common` is prepended to each
  image/buffer pass (matches Houdini).
- API budget is seeded conservatively from previously-cached fetches and tracked
  locally (no quota endpoint exists). The **authoritative** signal is the API
  itself: when the monthly quota is exhausted it returns `{"Error":"Too many
  requests"}`. `fetch` detects this, stops immediately, leaves the current id
  unfetched (retryable next month), and bumps the local tally to the cap — so a
  rate-limited run is safe to simply re-run later, not a data loss.
