#!/usr/bin/env python3
"""PHASE 2 downloader: fetch shaders from shader_list.json via Shadertoy API.

Saves raw JSON per shader to tests/shaders/api/json/<id>.json.
Extracts GLSL passes (common/image/buffer) to tests/shaders/api/glsl/.
Reports which shaders are accessible and their pass structure.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "houdini" / "scripts" / "python"))
from hshadertoy.api.shadertoy import App, ShadertoyAPIError

API_KEY = "NdHjRR"
API_DIR = ROOT / "tests" / "shaders" / "api"
JSON_DIR = API_DIR / "json"
GLSL_DIR = API_DIR / "glsl"
JSON_DIR.mkdir(parents=True, exist_ok=True)
GLSL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ids = json.loads((API_DIR / "shader_list.json").read_text())
    app = App(API_KEY)
    report = []
    for sid in ids:
        rec = {"id": sid, "status": None, "name": "", "passes": []}
        try:
            shader = app.get_shader(sid)
        except ShadertoyAPIError as e:
            rec["status"] = "UNAVAILABLE"
            rec["error"] = str(e)
            print(f"{sid:10} UNAVAILABLE")
            report.append(rec); continue
        except Exception as e:
            rec["status"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"{sid:10} ERROR {e}")
            report.append(rec); continue

        (JSON_DIR / f"{sid}.json").write_text(json.dumps(shader, indent=2), encoding="utf-8")
        rec["status"] = "OK"
        rec["name"] = shader["info"]["name"]
        for rp in shader["renderpass"]:
            ptype = rp.get("type", "")        # image, buffer, common, sound, cubemap
            pname = rp.get("name", "")
            outputs = rp.get("outputs", [])
            inputs = [(i.get("ctype"), i.get("channel")) for i in rp.get("inputs", [])]
            rec["passes"].append({"type": ptype, "name": pname, "inputs": inputs})
            # save the raw GLSL of each pass
            safe = pname.replace(" ", "_").replace("/", "_") or ptype
            (GLSL_DIR / f"{sid}__{ptype}__{safe}.glsl").write_text(rp["code"], encoding="utf-8")
        ptypes = [p["type"] for p in rec["passes"]]
        print(f"{sid:10} OK   {rec['name'][:35]:35} passes={ptypes}")
        report.append(rec)

    (API_DIR / "download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = [r for r in report if r["status"] == "OK"]
    print(f"\nAvailable: {len(ok)}/{len(ids)}")
    # input-channel summary (texture dependence)
    print("\nPass-type counts across available shaders:")
    from collections import Counter
    c = Counter(p["type"] for r in ok for p in r["passes"])
    for k, v in c.items():
        print(f"  {k:10} {v}")

if __name__ == "__main__":
    main()
