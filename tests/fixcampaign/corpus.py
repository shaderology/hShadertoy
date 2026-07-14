#!/usr/bin/env python3
"""Bug-fix campaign helper — query the mass-test ledger for fix targets & deltas.

The mass-test campaign (`tests/campaign/`) is the measurement source of truth.
This tool reads its generated `failures.csv` and live `ledger.json` so a fix
session can, with one command and minimal context:

  * `list <CAT>`  -> every failing pass in taxonomy category CAT (id, pass,
                     stage, name, error) + a ready-to-paste `campaign.py test
                     --ids ...` re-test command for those shaders.
  * `summary`     -> remaining distinct-shader count per category (live).
  * `delta`       -> what changed vs the frozen baseline: FAIL->PASS (fixed) and
                     PASS->FAIL (regressed), with net totals. Run after a fix to
                     prove progress and catch regressions.
  * `snapshot`    -> (re)freeze baseline_ledger.json from the live ledger.
                     Done once at campaign start; use --force to re-freeze.

Zero Claude cost, no API. Categories are the `+`-joined codes in failures.csv
(e.g. a row tagged `B+N` belongs to both B and N).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CAMPAIGN = HERE.parent / "campaign"
FAILURES = CAMPAIGN / "failures.csv"
LEDGER = CAMPAIGN / "ledger.json"
BASELINE = HERE / "baseline_ledger.json"


def _rows():
    if not FAILURES.exists():
        sys.exit(f"missing {FAILURES} — run `campaign.py report` first")
    with open(FAILURES, encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f)


def _cats(row):
    return [c.strip() for c in (row.get("categories") or "").split("+") if c.strip()]


def cmd_list(args):
    cat = args.category
    hits = [r for r in _rows() if cat in _cats(r)]
    if not hits:
        print(f"no failing passes tagged '{cat}'")
        return
    ids, clean = [], []  # clean = passes blocked ONLY by this category
    for r in hits:
        if r["id"] not in ids:
            ids.append(r["id"])
        sole = _cats(r) == [cat]
        if sole and r["id"] not in clean:
            clean.append(r["id"])
        err = (r.get("error_excerpt") or "").strip()
        if len(err) > 150:
            err = err[:150] + " ..."
        mark = "*" if sole else " "  # '*' = sole blocker -> flips to PASS when fixed
        print(f" {mark}{r['id']:8} {r['pass']:7} {r['stage']:9} {r['name'][:30]:30} {err}")
    print(f"\n{len(hits)} failing passes across {len(ids)} shaders tagged '{cat}'")
    print(f"'*' = {cat} is the ONLY blocker ({len(clean)} shaders) -> these flip to PASS once fixed;")
    print("     the rest also carry other categories and need those fixed too.")
    print("\nre-test (clean wins first) after your fix:")
    print(f"  python tests/campaign/campaign.py test --ids {','.join(clean or ids)} --force")
    print("  python tests/campaign/campaign.py report")


def cmd_summary(args):
    counts = {}
    for r in _rows():
        for c in _cats(r):
            counts.setdefault(c, set()).add(r["id"])
    rows = sorted(counts.items(), key=lambda kv: -len(kv[1]))
    print(f"{'cat':5} {'shaders':>7}")
    for c, ids in rows:
        print(f"{c:5} {len(ids):>7}")


def _overall_map(path):
    data = json.load(open(path, encoding="utf-8"))
    return {k: v.get("overall") for k, v in data.items() if v.get("overall")}


def cmd_delta(args):
    if not BASELINE.exists():
        sys.exit("no baseline — run `corpus.py snapshot` first")
    base = _overall_map(BASELINE)
    live = _overall_map(LEDGER)
    fixed = sorted(i for i in live if base.get(i) == "FAIL" and live[i] == "PASS")
    regressed = sorted(i for i in live if base.get(i) == "PASS" and live[i] == "FAIL")
    base_pass = sum(1 for v in base.values() if v == "PASS")
    live_pass = sum(1 for v in live.values() if v == "PASS")
    print(f"baseline PASS: {base_pass}    live PASS: {live_pass}    net: {live_pass - base_pass:+d}")
    print(f"\nFIXED (FAIL->PASS): {len(fixed)}")
    for i in fixed:
        print(f"  + {i}")
    print(f"\nREGRESSED (PASS->FAIL): {len(regressed)}")
    for i in regressed:
        print(f"  ! {i}")
    if regressed:
        print("\n*** REGRESSIONS PRESENT — investigate before committing ***")


def cmd_snapshot(args):
    if BASELINE.exists() and not args.force:
        sys.exit(f"{BASELINE.name} exists — pass --force to re-freeze (loses the reference point)")
    data = json.load(open(LEDGER, encoding="utf-8"))
    json.dump(data, open(BASELINE, "w", encoding="utf-8"), indent=0)
    npass = sum(1 for v in data.values() if v.get("overall") == "PASS")
    print(f"froze baseline: {len(data)} shaders, {npass} currently PASS")


def main():
    ap = argparse.ArgumentParser(description="bug-fix campaign ledger helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="failing passes for a taxonomy category")
    p.add_argument("category")
    p.set_defaults(func=cmd_list)
    sub.add_parser("summary", help="remaining distinct shaders per category").set_defaults(func=cmd_summary)
    sub.add_parser("delta", help="changes vs baseline (fixed / regressed)").set_defaults(func=cmd_delta)
    s = sub.add_parser("snapshot", help="freeze baseline_ledger.json")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_snapshot)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
