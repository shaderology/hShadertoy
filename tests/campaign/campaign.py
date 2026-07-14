#!/usr/bin/env python3
"""Phase 5 — Systematic oldest-first Shadertoy mass-test campaign orchestrator.

Intelligence gathering only. NO transpiler fixes.

Four idempotent, resumable stages driven by a single ledger (`ledger.json`):

    select   compute the every-10th-oldest candidate list from shader_all.json,
             assign sessions (100 ids each).            [no API, no cost]
    fetch    download+cache each session's shader JSON. THE ONLY API STEP.
             cache-first; hard per-run limit + monthly budget guard. [API]
    test     transpile + compile every cached pass, auto-classify failures,
             write artifacts, update ledger.             [offline, repeatable]
    report   aggregate ledger -> ranked failure DB (REPORT.md + failures.csv).
    status   print per-session progress + remaining API budget.

Resume after a context-clear / daily-limit with just:
    python tests/campaign/campaign.py status
    python tests/campaign/campaign.py fetch  --session N      (if budget remains)
    python tests/campaign/campaign.py test   --session N
    python tests/campaign/campaign.py report

Usage examples:
    python tests/campaign/campaign.py select
    python tests/campaign/campaign.py fetch --session 1 --dry-run
    python tests/campaign/campaign.py fetch --session 1
    python tests/campaign/campaign.py test  --session 1
    python tests/campaign/campaign.py test  --ids MsfGRn,MsX3zr      (targeted)
    python tests/campaign/campaign.py report
"""
import argparse
import csv
import datetime as _dt
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # repo root
TESTS = ROOT / "tests"
CAMP = TESTS / "campaign"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(ROOT / "houdini" / "scripts" / "python"))

import classify  # noqa: E402  (tests/campaign on path)

# Shader names are arbitrary Unicode; printing them to a Windows charmap console
# can raise UnicodeEncodeError. Never let a cosmetic print crash a fetch/test run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

# ---- paths ----------------------------------------------------------------
SHADER_ALL = TESTS / "shaders" / "api" / "shader_all.json"
EXISTING_CACHES = [
    CAMP / "cache",
    TESTS / "shaders" / "api" / "json",
    TESTS / "shaders" / "api" / "json_sp",
]
CACHE = CAMP / "cache"
ARTIFACTS = CAMP / "artifacts"
LEDGER = CAMP / "ledger.json"
SELECTION = CAMP / "selection.json"
BUDGET = CAMP / "api_budget.json"
REPORT = CAMP / "REPORT.md"
FAILURES_CSV = CAMP / "failures.csv"

MAIN_HEADER = TESTS / "ocl" / "main_header.cl"
MAIN_KERNEL = TESTS / "ocl" / "main_kernel.cl"
BUILD_OPTS = TESTS / "build_options.json"

API_KEY = "NdHjRR"
SESSION_SIZE = 100
SAMPLE_STRIDE = 10
MONTHLY_CAP = 1500
TESTABLE_PASSES = {"image", "buffer", "cubemap"}   # skip 'common' (merged) + 'sound'


# ---- small json helpers ---------------------------------------------------
def _load(path, default):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return default


def _save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _today():
    return _dt.date.today().isoformat()


def _month():
    return _dt.date.today().strftime("%Y-%m")


# ---- budget ---------------------------------------------------------------
def load_budget():
    b = _load(BUDGET, None)
    if b is None or b.get("month") != _month():
        # Seed conservatively: assume previously-cached shaders this month each
        # cost ~1 call (json/ + json_sp/), plus a small overhead for the list/query
        # calls. The Shadertoy API exposes no quota endpoint, so we track our own.
        prior = 0
        for d in (TESTS / "shaders" / "api" / "json", TESTS / "shaders" / "api" / "json_sp"):
            if d.exists():
                prior += len(list(d.glob("*.json")))
        seed = (prior + 5) if (b is None) else 0
        b = {"month": _month(), "calls": seed, "monthly_cap": MONTHLY_CAP,
             "seed_note": f"seeded with ~{seed} prior calls (estimate)"}
        _save(BUDGET, b)
    return b


def budget_remaining(b=None):
    b = b or load_budget()
    return b["monthly_cap"] - b["calls"]


# ---- ledger ---------------------------------------------------------------
def load_ledger():
    return _load(LEDGER, {})


def save_ledger(led):
    _save(LEDGER, led)


# ---- stage A: select ------------------------------------------------------
def candidate_ids():
    data = _load(SHADER_ALL, None)
    if data is None:
        sys.exit(f"missing {SHADER_ALL}")
    results = data["Results"] if isinstance(data, dict) else data
    oldest_first = list(reversed(results))
    return oldest_first[::SAMPLE_STRIDE]      # every 10th from the oldest end


def cmd_select(args):
    ids = candidate_ids()
    sel = [{"id": sid, "order_from_oldest": i * SAMPLE_STRIDE,
            "session": i // SESSION_SIZE + 1} for i, sid in enumerate(ids)]
    _save(SELECTION, sel)

    led = load_ledger()
    for rec in sel:
        e = led.setdefault(rec["id"], {})
        e["id"] = rec["id"]
        e["order_from_oldest"] = rec["order_from_oldest"]
        e["session"] = rec["session"]
    save_ledger(led)

    n_sessions = (len(sel) + SESSION_SIZE - 1) // SESSION_SIZE
    print(f"selected {len(sel)} candidate ids (every {SAMPLE_STRIDE}th oldest-first)")
    print(f"  -> {n_sessions} sessions of up to {SESSION_SIZE}")
    print(f"  session 1 first ids: {[r['id'] for r in sel[:3]]}")


def session_ids(session, ids_arg=None):
    if ids_arg:
        return [s.strip() for s in ids_arg.split(",") if s.strip()]
    sel = _load(SELECTION, None)
    if sel is None:
        sys.exit("run `select` first")
    return [r["id"] for r in sel if r["session"] == session]


# ---- stage B: fetch -------------------------------------------------------
def _find_cached(sid):
    for d in EXISTING_CACHES:
        p = Path(d) / f"{sid}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _record_shader_meta(entry, shader):
    passes = [{"type": p.get("type", ""), "name": p.get("name", "")}
              for p in shader.get("renderpass", [])]
    entry["passes"] = passes
    entry["has_common"] = any(p["type"] == "common" for p in passes)
    code_all = "\n".join(p.get("code", "") for p in shader.get("renderpass", []))
    entry["uses_ichannel"] = "iChannel" in code_all
    entry["name"] = shader.get("info", {}).get("name", "")


def cmd_fetch(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    led = load_ledger()
    ids = session_ids(args.session, args.ids)
    budget = load_budget()

    todo = []
    for sid in ids:
        e = led.get(sid, {})
        st = e.get("fetch", {}).get("status")
        if st in ("OK", "UNAVAILABLE"):
            continue              # stable result, never re-fetch
        todo.append(sid)

    print(f"session {args.session}: {len(ids)} ids, {len(todo)} need fetch")
    print(f"API budget: {budget['calls']}/{budget['monthly_cap']} used "
          f"({budget_remaining(budget)} remaining this month)")

    app = None
    new_calls = 0
    from_cache = 0
    for sid in todo:
        if new_calls >= args.limit:
            print(f"reached --limit {args.limit}; stopping (resume later)")
            break
        entry = led.setdefault(sid, {"id": sid})

        cached = _find_cached(sid)
        if cached is not None:
            _save(CACHE / f"{sid}.json", cached)
            entry["fetch"] = {"status": "OK", "api_call": False, "date": _today()}
            _record_shader_meta(entry, cached)
            from_cache += 1
            save_ledger(led)
            print(f"  {sid:10} OK (cache)   {entry.get('name','')[:40]}")
            continue

        if budget_remaining(budget) - new_calls <= 0:
            print("MONTHLY BUDGET EXHAUSTED — stopping. Resume next month or raise cap.")
            break
        if args.dry_run:
            print(f"  {sid:10} would fetch via API")
            new_calls += 1
            continue

        if app is None:
            from hshadertoy.api.shadertoy import App
            app = App(API_KEY)
        from hshadertoy.api.shadertoy import ShadertoyAPIError
        try:
            shader = app.get_shader(sid)
            _save(CACHE / f"{sid}.json", shader)
            entry["fetch"] = {"status": "OK", "api_call": True, "date": _today()}
            _record_shader_meta(entry, shader)
            print(f"  {sid:10} OK           {entry.get('name','')[:40]}")
        except ShadertoyAPIError as ex:
            msg = str(ex)
            # Real monthly-quota signal: Shadertoy returns {"Error":"Too many
            # requests"}. Stop at once (don't burn the rest of the session
            # hammering a rate-limited API) and leave this id unfetched so it
            # retries next month. Sync our local tally up to the cap.
            if "too many requests" in msg.lower():
                print(f"  {sid:10} RATE-LIMITED — Shadertoy monthly quota hit; stopping run.")
                budget["calls"] = max(budget["calls"], budget["monthly_cap"])
                _save(BUDGET, budget)
                break
            unavailable = "not found" in msg.lower() or "public" in msg.lower()
            entry["fetch"] = {"status": "UNAVAILABLE" if unavailable else "ERROR",
                              "api_call": True, "date": _today(), "error": msg}
            print(f"  {sid:10} {'UNAVAILABLE' if unavailable else 'ERROR'}  {msg[:50]}")
        except Exception as ex:  # network etc — retryable
            entry["fetch"] = {"status": "ERROR", "api_call": True, "date": _today(),
                              "error": f"{type(ex).__name__}: {ex}"}
            print(f"  {sid:10} ERROR        {ex}")
        new_calls += 1
        budget["calls"] += 1
        _save(BUDGET, budget)
        save_ledger(led)
        time.sleep(0.4)

    save_ledger(led)
    print(f"\nfetched: {new_calls} via API, {from_cache} from cache")
    print(f"API budget now: {budget['calls']}/{budget['monthly_cap']}")


# ---- stage C: test --------------------------------------------------------
class _Builder:
    """Reuse a single OpenCL context + build options across all compiles."""
    def __init__(self):
        import pyopencl as cl
        self.cl = cl
        import tests.compilecl as compilecl
        self.compilecl = compilecl
        self.opts = compilecl.load_build_options(str(BUILD_OPTS))
        device = cl.get_platforms()[0].get_devices()[0]
        self.ctx = cl.Context([device])
        self.device = device

    def compile(self, header_path, kernel_path):
        src = self.compilecl.construct_kernel_source(
            str(MAIN_HEADER), str(header_path), str(MAIN_KERNEL), str(kernel_path))
        try:
            self.cl.Program(self.ctx, src).build(options=self.opts)
            return True, ""
        except Exception as e:
            return False, str(e)


def _excerpt(text, n=4):
    """First few `error:`-bearing lines (fallback: first n non-empty lines)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    errs = [ln for ln in lines if "error" in ln.lower()]
    pick = (errs or lines)[:n]
    return " | ".join(pick)[:600]


def cmd_test(args):
    from tests.transpile import transpile
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    led = load_ledger()
    ids = session_ids(args.session, args.ids)

    builder = _Builder()
    print(f"testing on {builder.device.name}")

    tested = 0
    for sid in ids:
        entry = led.get(sid, {})
        if entry.get("fetch", {}).get("status") != "OK":
            continue
        if entry.get("overall") and not args.force:
            continue  # already tested

        shader = _find_cached(sid)
        if shader is None:
            entry["overall"] = "ERROR"
            entry["notes"] = "cache missing at test time"
            save_ledger(led)
            continue

        passes = shader.get("renderpass", [])
        common_src = next((p.get("code", "") for p in passes if p.get("type") == "common"), "")
        out_dir = ARTIFACTS / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for p in passes:
            ptype = p.get("type", "")
            if ptype not in TESTABLE_PASSES:
                continue
            pname = p.get("name", ptype)
            safe = pname.replace(" ", "_").replace("/", "_") or ptype
            glsl = p.get("code", "")
            res = {"pass": ptype, "name": pname, "transpile": None, "compile": None,
                   "categories": [], "difficulty": "", "error_excerpt": "",
                   "artifact": f"artifacts/{sid}/{safe}"}
            try:
                tr = transpile(glsl, common=common_src)
                hp = out_dir / f"{safe}.header.cl"
                kp = out_dir / f"{safe}.kernel.cl"
                hp.write_text(tr.get_header(), encoding="utf-8")
                kp.write_text(tr.get_kernel(), encoding="utf-8")
                res["transpile"] = "OK"
            except Exception as e:
                res["transpile"] = "FAIL"
                tb = traceback.format_exc()
                (out_dir / f"{safe}.transpile_err.txt").write_text(tb, encoding="utf-8")
                cats, diff = classify.classify(transpile_err=f"{type(e).__name__}: {e}",
                                               source=common_src + "\n" + glsl)
                res["categories"], res["difficulty"] = cats, diff
                res["error_excerpt"] = f"{type(e).__name__}: {e}"[:600]
                results.append(res)
                print(f"  {sid:8} {ptype:7} TRANSPILE-FAIL {cats}")
                continue

            ok, err = builder.compile(hp, kp)
            if ok:
                res["compile"] = "OK"
                print(f"  {sid:8} {ptype:7} OK")
            else:
                res["compile"] = "FAIL"
                (out_dir / f"{safe}.compile_err.txt").write_text(err, encoding="utf-8")
                cats, diff = classify.classify(compile_err=err,
                                               source=common_src + "\n" + glsl)
                res["categories"], res["difficulty"] = cats, diff
                res["error_excerpt"] = _excerpt(err)
                print(f"  {sid:8} {ptype:7} COMPILE-FAIL {cats}")
            results.append(res)

        entry["results"] = results
        if not results:
            entry["overall"] = "NO_TESTABLE_PASS"
        elif any(r["transpile"] == "FAIL" for r in results):
            entry["overall"] = "TRANSPILE_FAIL"
        elif any(r["compile"] == "FAIL" for r in results):
            entry["overall"] = "COMPILE_FAIL"
        else:
            entry["overall"] = "PASS"
        led[sid] = entry
        save_ledger(led)
        tested += 1

    save_ledger(led)
    print(f"\ntested {tested} shaders this run")


# ---- stage D: report ------------------------------------------------------
def cmd_report(args):
    led = load_ledger()
    rows = []          # one row per failing pass
    overall = Counter()
    cat_counter = Counter()
    cat_shaders = defaultdict(set)
    avail = Counter()

    for sid, e in led.items():
        st = e.get("fetch", {}).get("status")
        if st:
            avail[st] += 1
        ov = e.get("overall")
        if ov:
            overall[ov] += 1
        for r in e.get("results", []):
            if r.get("compile") == "OK":
                continue
            if r.get("transpile") is None and r.get("compile") is None:
                continue
            stage = "transpile" if r.get("transpile") == "FAIL" else "compile"
            cats = r.get("categories") or ["UNKNOWN"]
            for c in cats:
                cat_counter[c] += 1
                cat_shaders[c].add(sid)
            rows.append({
                "id": sid, "name": e.get("name", ""), "pass": r.get("pass", ""),
                "stage": stage, "categories": "+".join(cats),
                "difficulty": r.get("difficulty", ""),
                "error_excerpt": r.get("error_excerpt", ""),
            })

    # failures.csv
    with open(FAILURES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "pass", "stage",
                                          "categories", "difficulty", "error_excerpt"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["categories"], r["id"])))

    # REPORT.md
    tested = sum(overall.values())
    passed = overall.get("PASS", 0)
    L = []
    L.append("# Phase 5 — Mass-Test Campaign Report\n")
    L.append(f"_Generated {_today()}_\n")
    b = load_budget()
    L.append(f"**API budget:** {b['calls']}/{b['monthly_cap']} calls used this "
             f"month ({budget_remaining(b)} remaining).\n")
    L.append("## Availability (fetch)\n")
    for k in ("OK", "UNAVAILABLE", "ERROR"):
        if avail.get(k):
            L.append(f"- {k}: {avail[k]}")
    L.append("")
    L.append("## Pipeline outcome (tested shaders)\n")
    L.append(f"- Tested: {tested}")
    if tested:
        L.append(f"- **PASS (transpile+compile): {passed} ({100*passed//tested}%)**")
    for k in ("COMPILE_FAIL", "TRANSPILE_FAIL", "NO_TESTABLE_PASS", "ERROR"):
        if overall.get(k):
            L.append(f"- {k}: {overall[k]}")
    L.append("")
    L.append("## Failures ranked by category\n")
    L.append("| Cat | Difficulty | Fails | Shaders | Description |")
    L.append("|-----|-----------|-------|---------|-------------|")

    def _rank(c):
        return (classify._DIFF_RANK.get(classify.CATEGORY_DIFFICULTY.get(c, "unknown"), 0),
                cat_counter[c])
    for c in sorted(cat_counter, key=_rank, reverse=True):
        L.append(f"| {c} | {classify.CATEGORY_DIFFICULTY.get(c,'?')} | "
                 f"{cat_counter[c]} | {len(cat_shaders[c])} | "
                 f"{classify.CATEGORY_DESC.get(c,'')} |")
    L.append("")
    unknown = [r for r in rows if "UNKNOWN" in r["categories"]]
    if unknown:
        L.append(f"## UNKNOWN bucket ({len(unknown)}) — needs manual review\n")
        for r in unknown[:40]:
            L.append(f"- `{r['id']}` [{r['pass']}] {r['error_excerpt'][:160]}")
        L.append("")
    L.append("## Artifacts\n")
    L.append("- Per-shader transpiled `.cl` + error logs: `tests/campaign/artifacts/<id>/`")
    L.append("- Flat sortable table: `tests/campaign/failures.csv`")
    L.append("- Source of truth: `tests/campaign/ledger.json`")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"wrote {REPORT}")
    print(f"wrote {FAILURES_CSV} ({len(rows)} failing passes)")
    if tested:
        print(f"PASS {passed}/{tested}  |  top categories: "
              f"{', '.join(f'{c}={cat_counter[c]}' for c,_ in cat_counter.most_common(6))}")


# ---- reclassify (refine taxonomy without recompiling) ---------------------
def cmd_reclassify(args):
    """Re-run classify.py over already-captured artifact logs + cached source.

    Lets us refine classify.py rules and re-bucket failures with no API and no
    OpenCL recompile. Updates each failing result's categories/difficulty/excerpt.
    """
    import importlib
    importlib.reload(classify)
    led = load_ledger()
    changed = 0
    for sid, e in led.items():
        results = e.get("results")
        if not results:
            continue
        shader = _find_cached(sid)
        passes = shader.get("renderpass", []) if shader else []
        common_src = next((p.get("code", "") for p in passes if p.get("type") == "common"), "")
        by_name = {p.get("name", p.get("type", "")): p.get("code", "") for p in passes}
        for r in results:
            if r.get("compile") == "OK" or (r.get("transpile") is None):
                continue
            src = common_src + "\n" + by_name.get(r.get("name", ""), "")
            base = CAMP / r.get("artifact", "")
            if r.get("transpile") == "FAIL":
                log = (base.parent / (base.name + ".transpile_err.txt"))
                txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else r.get("error_excerpt", "")
                cats, diff = classify.classify(transpile_err=txt, source=src)
            else:
                log = (base.parent / (base.name + ".compile_err.txt"))
                txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else r.get("error_excerpt", "")
                cats, diff = classify.classify(compile_err=txt, source=src)
            if cats != r.get("categories"):
                changed += 1
            r["categories"], r["difficulty"] = cats, diff
        led[sid] = e
    save_ledger(led)
    print(f"reclassified; {changed} pass-results changed category")


# ---- status ---------------------------------------------------------------
def cmd_status(args):
    sel = _load(SELECTION, None)
    led = load_ledger()
    b = load_budget()
    if sel is None:
        print("no selection yet — run `select`")
        return
    by_session = defaultdict(list)
    for r in sel:
        by_session[r["session"]].append(r["id"])
    print(f"candidates: {len(sel)}  sessions: {len(by_session)}  "
          f"(size {SESSION_SIZE})")
    print(f"API budget: {b['calls']}/{b['monthly_cap']} used "
          f"({budget_remaining(b)} remaining this month)")
    print(f"\n{'sess':>4} {'ids':>4} {'fetched':>8} {'unavail':>8} "
          f"{'tested':>7} {'pass':>5} {'fail':>5} {'todo-fetch':>11}")
    for s in sorted(by_session):
        ids = by_session[s]
        fetched = sum(1 for i in ids if led.get(i, {}).get("fetch", {}).get("status") == "OK")
        unavail = sum(1 for i in ids if led.get(i, {}).get("fetch", {}).get("status") == "UNAVAILABLE")
        attempted = sum(1 for i in ids if led.get(i, {}).get("fetch", {}).get("status"))
        tested = sum(1 for i in ids if led.get(i, {}).get("overall"))
        passed = sum(1 for i in ids if led.get(i, {}).get("overall") == "PASS")
        failed = sum(1 for i in ids if led.get(i, {}).get("overall") in
                     ("COMPILE_FAIL", "TRANSPILE_FAIL"))
        todo = len(ids) - attempted
        print(f"{s:>4} {len(ids):>4} {fetched:>8} {unavail:>8} "
              f"{tested:>7} {passed:>5} {failed:>5} {todo:>11}")


# ---- cli ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 5 mass-test campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("select")
    sub.add_parser("status")

    f = sub.add_parser("fetch")
    f.add_argument("--session", type=int, default=1)
    f.add_argument("--ids", default=None, help="comma-separated ids (overrides --session)")
    f.add_argument("--limit", type=int, default=SESSION_SIZE, help="max NEW API calls")
    f.add_argument("--dry-run", action="store_true")

    t = sub.add_parser("test")
    t.add_argument("--session", type=int, default=1)
    t.add_argument("--ids", default=None, help="comma-separated ids (overrides --session)")
    t.add_argument("--force", action="store_true", help="re-test already-tested shaders")

    sub.add_parser("report")
    sub.add_parser("reclassify")

    args = ap.parse_args()
    {"select": cmd_select, "status": cmd_status, "fetch": cmd_fetch,
     "test": cmd_test, "report": cmd_report, "reclassify": cmd_reclassify}[args.cmd](args)


if __name__ == "__main__":
    main()
