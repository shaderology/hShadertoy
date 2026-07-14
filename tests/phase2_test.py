#!/usr/bin/env python3
"""PHASE 2 transpile+compile test over downloaded API shaders.

Runs the same transpile -> compile pipeline as PHASE 1 on every .glsl in
tests/shaders/api/glsl/. For multi-pass shaders, any 'common' pass is
prepended to image/buffer passes before transpiling.
"""
import sys, json, traceback
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from tests.transpile import transpile
import tests.compilecl as compilecl
import pyopencl as cl

GLSL_DIR = ROOT / "tests" / "shaders" / "api" / "glsl"
RESULTS = ROOT / "tests" / "phase2_results"
RESULTS.mkdir(exist_ok=True)

def try_compile(header_path, kernel_path):
    main_header = ROOT / "tests" / "ocl" / "main_header.cl"
    main_kernel = ROOT / "tests" / "ocl" / "main_kernel.cl"
    build_opts_file = ROOT / "tests" / "build_options.json"
    src = compilecl.construct_kernel_source(str(main_header), str(header_path),
                                            str(main_kernel), str(kernel_path))
    opts = compilecl.load_build_options(str(build_opts_file))
    device = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([device])
    try:
        cl.Program(ctx, src).build(options=opts)
        return True, ""
    except Exception as e:
        return False, str(e)

def main():
    # group passes by shader id (prefix before "__")
    by_shader = defaultdict(dict)
    for f in sorted(GLSL_DIR.glob("*.glsl")):
        sid, ptype, _ = f.name.split("__", 2)
        by_shader[sid][ptype] = f

    report = []
    for sid, passes in by_shader.items():
        common = passes.get("common")
        common_src = common.read_text(encoding="utf-8") if common else ""
        for ptype, f in passes.items():
            if ptype == "common":
                continue
            rec = {"id": sid, "pass": ptype, "file": f.name,
                   "transpile": None, "compile": None, "err": ""}
            glsl = f.read_text(encoding="utf-8")
            if common_src:
                glsl = common_src + "\n\n" + glsl
            stem = f.stem
            try:
                result = transpile(glsl)
                hp = RESULTS / f"{stem}.header.cl"; kp = RESULTS / f"{stem}.kernel.cl"
                hp.write_text(result.get_header(), encoding="utf-8")
                kp.write_text(result.get_kernel(), encoding="utf-8")
                rec["transpile"] = "OK"
            except Exception as e:
                rec["transpile"] = "FAIL"; rec["err"] = f"{type(e).__name__}: {e}"
                (RESULTS / f"{stem}.transpile_err.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"{sid:8} {ptype:7} TRANSPILE-FAIL  {e}")
                report.append(rec); continue
            ok, err = try_compile(hp, kp)
            if ok:
                rec["compile"] = "OK"; print(f"{sid:8} {ptype:7} OK")
            else:
                rec["compile"] = "FAIL"; rec["err"] = err
                (RESULTS / f"{stem}.compile_err.txt").write_text(err, encoding="utf-8")
                print(f"{sid:8} {ptype:7} COMPILE-FAIL")
            report.append(rec)

    (RESULTS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    n = len(report)
    tp = sum(1 for r in report if r["transpile"] == "OK")
    cp = sum(1 for r in report if r["compile"] == "OK")
    print(f"\n==== PHASE 2 SUMMARY ====\nPasses tested: {n}\nTranspile OK : {tp}/{n}\nCompile OK   : {cp}/{n}")

if __name__ == "__main__":
    main()
