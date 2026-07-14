"""
Render-compare campaign driver.

Proves render CORRECTNESS (perceptual), not just compile success: renders the
original GLSL with wgpu-shadertoy and the transpiled OpenCL with the
hShadertoy HDA (hython + rop_image), then compares SSIM + downsampled-MAE.

Stages (all idempotent/resumable - state lives in ledger.json):

    python tests/rendercompare/rc.py select --tier 2 [--limit N] [--ids a b c]
    python tests/rendercompare/rc.py render-ref  [--force] [--chunk 25]
    python tests/rendercompare/rc.py render-hda  [--force] [--chunk 10]
    python tests/rendercompare/rc.py compare     [--force]
    python tests/rendercompare/rc.py report
    python tests/rendercompare/rc.py run --tier 2 --limit N    # all stages
    python tests/rendercompare/rc.py smoke                     # gradient+london+digits

Tiers (from tests/campaign/ledger.json, overall == PASS only):
    2 = single image pass, no iChannels (fully deterministic)
    3 = single image pass, texture iChannels (local media mirror)

Verdict gates live in compare.py. Never hand-edit ledger.json / REPORT.md /
contact_sheet.html - they are generated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import compare as cmp_mod
import render_hda

TMP = common.RC_ROOT / "tmp"

GRADIENT_SHADER = {"Shader": {
    "info": {"id": "_gradient", "name": "RC calibration gradient"},
    "renderpass": [{
        "name": "Image", "type": "image", "inputs": [], "outputs": [],
        "code": ("void mainImage( out vec4 fragColor, in vec2 fragCoord )\n"
                 "{\n    vec2 uv = fragCoord/iResolution.xy;\n"
                 "    fragColor = vec4(uv.x, uv.y, 0.0, 1.0);\n}"),
    }],
}}

SMOKE_EXAMPLES = {
    "_london": common.REPO_ROOT / "resources/examples/london/london_API.json",
    "_digits": common.REPO_ROOT / "resources/examples/digits/digits_API.json",
}


# ------------------------------------------------------------------ selection

def stage_select(args) -> None:
    ledger = common.load_ledger()
    with open(common.CAMPAIGN_LEDGER, encoding="utf-8") as f:
        campaign = json.load(f)

    if args.ids:
        candidates = [campaign[i] for i in args.ids if i in campaign]
    else:
        candidates = [r for r in campaign.values()
                      if r.get("overall") == "PASS"
                      and len(r.get("passes", [])) == 1
                      and r["passes"][0].get("type") == "image"]
        if args.tier == 2:
            candidates = [r for r in candidates if not r.get("uses_ichannel")]
        elif args.tier == 3:
            candidates = [r for r in candidates if r.get("uses_ichannel")]
        candidates.sort(key=lambda r: r.get("order_from_oldest", 1 << 30))

    added = 0
    for rec in candidates:
        sid = rec["id"]
        if sid in ledger and not args.force:
            continue
        if args.limit and added >= args.limit:
            break
        try:
            shader = common.load_cache_shader(sid)
        except FileNotFoundError:
            print(f"skip {sid}: not in campaign cache")
            continue
        flags = common.shader_flags(shader)
        if args.tier == 3 and any(f.startswith("input_") for f in flags):
            continue  # texture-only in tier 3
        ledger[sid] = {
            "id": sid, "name": rec.get("name", ""), "tier": args.tier or 0,
            "flags": flags, "frames": list(common.FRAMES),
            "ref": {}, "hda": {}, "compare": {}, "verdict": "",
        }
        added += 1
    common.save_ledger(ledger)
    print(f"select: {added} shader(s) added, ledger now {len(ledger)}")


# ------------------------------------------------------------------ rendering

def _jobs_for(ledger: dict, side: str, force: bool) -> list[dict]:
    """Build manifest jobs for entries whose <side> renders aren't OK yet."""
    jobs = []
    for sid, e in sorted(ledger.items()):
        outs = {}
        for frame in e["frames"]:
            ref_png, hda_png, _ = common.png_names(frame)
            png = ref_png if side == "ref" else hda_png
            state = e[side].get(str(frame), {})
            if state.get("status") == "OK" and not force:
                continue
            outs[str(frame)] = str(common.artifact_dir(sid) / png)
        if outs:
            job = {"id": sid, "outs": outs}
            if e.get("shader_inline"):
                job["shader_inline"] = e["shader_inline"]
            jobs.append(job)
    return jobs


def _merge_results(ledger: dict, side: str, results: list) -> None:
    for r in results:
        e = ledger.get(r["id"])
        if e is None:
            continue
        e[side][str(r["frame"])] = {
            "status": r["status"], "error": r.get("error", ""),
            "png": r["out_png"], "notes": r.get("notes", []),
        }


def _run_chunks(ledger: dict, side: str, jobs: list, chunk: int,
                timeout_per_job: int) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(jobs), chunk):
        part = jobs[i:i + chunk]
        stamp = f"{side}_{int(time.time())}_{i}"
        manifest = TMP / f"manifest_{stamp}.json"
        results = TMP / f"results_{stamp}.json"
        manifest.write_text(json.dumps({"jobs": part}, indent=1),
                            encoding="utf-8")
        print(f"[{side}] chunk {i // chunk + 1}: {len(part)} shader(s)")
        if side == "ref":
            import subprocess
            script = common.RC_ROOT / "render_ref.py"
            subprocess.run([sys.executable, str(script), str(manifest),
                            str(results)],
                           timeout=120 + timeout_per_job * len(part))
        else:
            render_hda.run_chunk(str(manifest), str(results),
                                 timeout_s=300 + timeout_per_job * len(part))
        if results.exists():
            data = json.loads(results.read_text(encoding="utf-8"))
            _merge_results(ledger, side, data.get("results", []))
        else:
            for job in part:
                for frame_str, png in job["outs"].items():
                    ledger[job["id"]][side][frame_str] = {
                        "status": "FAIL", "png": png, "notes": [],
                        "error": "renderer produced no results file "
                                 "(crash/timeout)"}
        common.save_ledger(ledger)  # resumable after every chunk


def stage_render_ref(args) -> None:
    ledger = common.load_ledger()
    jobs = _jobs_for(ledger, "ref", args.force)
    print(f"render-ref: {len(jobs)} shader(s) to render")
    _run_chunks(ledger, "ref", jobs, args.chunk, timeout_per_job=30)


def stage_render_hda(args) -> None:
    ledger = common.load_ledger()
    jobs = _jobs_for(ledger, "hda", args.force)
    print(f"render-hda: {len(jobs)} shader(s) to render")
    _run_chunks(ledger, "hda", jobs, args.chunk, timeout_per_job=120)


# ------------------------------------------------------------------ compare

def stage_compare(args) -> None:
    ledger = common.load_ledger()
    done = 0
    for sid, e in sorted(ledger.items()):
        verdicts = []
        for frame in e["frames"]:
            fs = str(frame)
            if e["compare"].get(fs) and not args.force:
                verdicts.append(e["compare"][fs].get("verdict", "ERROR"))
                continue
            ref = e["ref"].get(fs, {})
            hda = e["hda"].get(fs, {})
            if ref.get("status") != "OK" or hda.get("status") != "OK":
                continue
            _, _, diff_png = common.png_names(frame)
            m = cmp_mod.compare_files(
                ref["png"], hda["png"],
                diff_path=str(common.artifact_dir(sid) / diff_png))
            e["compare"][fs] = m
            verdicts.append(m["verdict"])
            done += 1
        if verdicts:
            order = {"FAIL": 0, "ERROR": 0, "WARN": 1, "PASS": 2}
            e["verdict"] = min(verdicts, key=lambda v: order.get(v, 0))
        elif any(s.get("status") == "FAIL" for s in
                 list(e["ref"].values()) + list(e["hda"].values())):
            e["verdict"] = "RENDER_FAIL"
    common.save_ledger(ledger)
    print(f"compare: {done} frame comparison(s) computed")


# ------------------------------------------------------------------ report

def stage_report(args) -> None:
    ledger = common.load_ledger()
    entries = sorted(ledger.values(), key=_report_sort_key)
    counts = {}
    for e in entries:
        counts[e.get("verdict") or "PENDING"] = \
            counts.get(e.get("verdict") or "PENDING", 0) + 1

    lines = ["# Render-compare report", "",
             f"Shaders: {len(entries)}  " +
             "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())), "",
             "| id | name | tier | verdict | ssim | dmae | mae | flags |",
             "|---|---|---|---|---|---|---|---|"]
    rows = []
    for e in entries:
        worst = _worst_compare(e)
        if worst:
            met = (f"{worst['ssim']:.3f} | {worst['dmae']:.2f} "
                   f"| {worst['mae']:.2f}")
        else:
            met = " | | "
        err = ""
        if e.get("verdict") == "RENDER_FAIL":
            for side in ("hda", "ref"):
                for st in e[side].values():
                    if st.get("status") == "FAIL":
                        err = st.get("error", "")[:80].replace("|", "/")
                        break
        lines.append(f"| {e['id']} | {e['name'][:32].replace('|', '/')} "
                     f"| {e['tier']} | {e.get('verdict') or 'PENDING'} "
                     f"| {met} | {','.join(e['flags'])} {err} |")
        rows.append(e)
    (common.RC_ROOT / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    _write_contact_sheet(rows)
    print(f"report: {len(entries)} shaders -> REPORT.md + contact_sheet.html")
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def _worst_compare(e: dict) -> dict:
    worst = None
    for m in e.get("compare", {}).values():
        if "ssim" not in m:
            continue
        if worst is None or m["ssim"] < worst["ssim"]:
            worst = m
    return worst or {}


def _report_sort_key(e: dict):
    order = {"FAIL": 0, "RENDER_FAIL": 1, "WARN": 2, "": 3, "PENDING": 3,
             "PASS": 4}
    worst = _worst_compare(e)
    return (order.get(e.get("verdict") or "PENDING", 3),
            worst.get("ssim", 2.0))


def _write_contact_sheet(entries: list) -> None:
    html = ["<!doctype html><meta charset='utf-8'>",
            "<title>render-compare contact sheet</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee}"
            "img{width:266px;image-rendering:auto;border:1px solid #333}"
            ".row{margin-bottom:14px}.PASS{color:#7c7}.WARN{color:#fc6}"
            ".FAIL{color:#f66}.RENDER_FAIL{color:#c6f}"
            "td{padding:2px 8px;font-size:13px}</style>",
            "<h1>render-compare: ref | hda | diff</h1>"]
    for e in entries:
        worst = _worst_compare(e)
        met = (f"ssim {worst.get('ssim', 0):.3f} dmae {worst.get('dmae', 0):.2f}"
               if worst else "no comparison")
        html.append(
            f"<div class='row'><b class='{e.get('verdict', '')}'>"
            f"{e.get('verdict') or 'PENDING'}</b> <b>{e['id']}</b> "
            f"{e['name'][:48]} &mdash; {met} "
            f"<i>{','.join(e['flags'])}</i><br>")
        for frame in e["frames"]:
            ref_png, hda_png, diff_png = common.png_names(frame)
            adir = f"artifacts/{e['id']}"
            html.append(
                f"<img src='{adir}/{ref_png}' loading='lazy'>"
                f"<img src='{adir}/{hda_png}' loading='lazy'>"
                f"<img src='{adir}/{diff_png}' loading='lazy'> f{frame}<br>")
        html.append("</div>")
    (common.RC_ROOT / "contact_sheet.html").write_text(
        "\n".join(html), encoding="utf-8")


# ------------------------------------------------------------------ smoke

def stage_smoke(args) -> int:
    """Fixed 3-shader end-to-end check (fixcampaign hookup candidate)."""
    ledger = common.load_ledger()
    ledger["_gradient"] = {
        "id": "_gradient", "name": "calibration gradient", "tier": 1,
        "flags": [], "frames": list(common.FRAMES),
        "shader_inline": GRADIENT_SHADER,
        "ref": {}, "hda": {}, "compare": {}, "verdict": "",
    }
    for sid, path in SMOKE_EXAMPLES.items():
        with open(path, encoding="utf-8") as f:
            shader = json.load(f)
        if "Shader" not in shader:
            shader = {"Shader": shader}
        ledger[sid] = {
            "id": sid, "name": shader["Shader"]["info"].get("name", sid),
            "tier": 1, "flags": common.shader_flags(shader),
            "frames": list(common.FRAMES), "shader_inline": shader,
            "ref": {}, "hda": {}, "compare": {}, "verdict": "",
        }
    common.save_ledger(ledger)

    ns = argparse.Namespace(force=args.force, chunk=10)
    stage_render_ref(ns)
    stage_render_hda(ns)
    stage_compare(argparse.Namespace(force=args.force))
    stage_report(argparse.Namespace())

    ledger = common.load_ledger()
    bad = [sid for sid in ("_gradient", "_london", "_digits")
           if ledger[sid].get("verdict") not in ("PASS", "WARN")]
    if bad:
        for sid in bad:
            print(f"SMOKE FAIL: {sid} verdict={ledger[sid].get('verdict')}")
        return 1
    print("SMOKE OK: gradient, london, digits all within perceptual gates")
    return 0


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("select")
    p.add_argument("--tier", type=int, choices=(2, 3), default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--force", action="store_true")

    for name in ("render-ref", "render-hda"):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
        p.add_argument("--chunk", type=int,
                       default=25 if name == "render-ref" else 10)

    p = sub.add_parser("compare")
    p.add_argument("--force", action="store_true")

    sub.add_parser("report")

    p = sub.add_parser("run")
    p.add_argument("--tier", type=int, choices=(2, 3), default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("smoke")
    p.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)

    if args.stage == "select":
        stage_select(args)
    elif args.stage == "render-ref":
        stage_render_ref(args)
    elif args.stage == "render-hda":
        stage_render_hda(args)
    elif args.stage == "compare":
        stage_compare(args)
    elif args.stage == "report":
        stage_report(args)
    elif args.stage == "smoke":
        return stage_smoke(args)
    elif args.stage == "run":
        stage_select(args)
        ns = argparse.Namespace(force=args.force, chunk=25)
        stage_render_ref(ns)
        ns = argparse.Namespace(force=args.force, chunk=10)
        stage_render_hda(ns)
        stage_compare(argparse.Namespace(force=args.force))
        stage_report(argparse.Namespace())
    return 0


if __name__ == "__main__":
    sys.exit(main())
