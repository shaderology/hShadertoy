#!/usr/bin/env python3
"""Fetch the single-pass shader set and analyze:
   1. mainImage() parameter-name variety (vs hardcoded fragColor/fragCoord)
   2. presence of a Common code tab
   3. iChannel / texture usage
Caches JSON to tests/shaders/api/json_sp/.
"""
import sys, json, re, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "houdini" / "scripts" / "python"))
from hshadertoy.api.shadertoy import App, ShadertoyAPIError

API = App("NdHjRR")
API_DIR = ROOT / "tests" / "shaders" / "api"
CACHE = API_DIR / "json_sp"; CACHE.mkdir(exist_ok=True)

ids = json.loads((API_DIR / "shader_singlepass.json").read_text())["Results"]

SIG_RE = re.compile(r"void\s+mainImage\s*\(([^)]*)\)")

def analyze():
    rows = []
    for i, sid in enumerate(ids):
        cf = CACHE / f"{sid}.json"
        if cf.exists():
            shader = json.loads(cf.read_text())
        else:
            try:
                shader = API.get_shader(sid)
                cf.write_text(json.dumps(shader), encoding="utf-8")
                time.sleep(0.4)
            except Exception as e:
                rows.append({"id": sid, "err": str(e)}); continue
        passes = shader["renderpass"]
        ptypes = [p.get("type") for p in passes]
        img = next((p for p in passes if p.get("type") == "image"), None)
        rec = {"id": sid, "ptypes": ptypes, "has_common": "common" in ptypes}
        if img:
            m = SIG_RE.search(img["code"])
            if m:
                params = m.group(1)
                # split two params
                parts = [p.strip() for p in params.split(",")]
                names = []
                for p in parts:
                    toks = p.replace("out", "").replace("in", "").split()
                    names.append(toks[-1] if toks else "?")
                rec["out_name"] = names[0] if len(names) > 0 else "?"
                rec["in_name"] = names[1] if len(names) > 1 else "?"
                rec["sig"] = params.strip()
            rec["uses_ichannel"] = "iChannel" in img["code"]
        rows.append(rec)
    return rows

def main():
    rows = analyze()
    ok = [r for r in rows if "ptypes" in r]
    print(f"Fetched {len(ok)}/{len(ids)} (errors: {len(rows)-len(ok)})")
    # param naming
    std = sum(1 for r in ok if r.get("out_name")=="fragColor" and r.get("in_name")=="fragCoord")
    nonstd = [r for r in ok if not (r.get("out_name")=="fragColor" and r.get("in_name")=="fragCoord")]
    print(f"\nmainImage param names:")
    print(f"  standard (fragColor/fragCoord): {std}")
    print(f"  non-standard                  : {len(nonstd)}")
    print("  examples of non-standard sigs:")
    for r in nonstd[:15]:
        print(f"    {r['id']}: ({r.get('sig')})  -> out={r.get('out_name')} in={r.get('in_name')}")
    # common tab
    common = [r for r in ok if r.get("has_common")]
    print(f"\nCommon code tab present: {len(common)}/{len(ok)}")
    for r in common[:10]:
        print(f"    {r['id']}: passes={r['ptypes']}")
    # ichannel
    ich = sum(1 for r in ok if r.get("uses_ichannel"))
    print(f"\niChannel/texture usage: {ich}/{len(ok)}")
    (API_DIR / "signature_analysis.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
