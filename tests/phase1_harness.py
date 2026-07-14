#!/usr/bin/env python3
"""
PHASE 1 mass-test harness.

For every .glsl in tests/shaders/{simple,medium,complex}:
  1. transpile via transpile.transpile()
  2. if transpile ok, compile via compilecl pipeline
Captures per-shader status + error text to phase1_results/.
Prints a summary table + dumps a JSON report.
"""
import sys, json, traceback, io, contextlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tests.transpile import transpile, TranspileError
import importlib.util

# Load compilecl as a module so we can call its compile function directly
import tests.compilecl as compilecl

RESULTS = ROOT / "tests" / "phase1_results"
RESULTS.mkdir(exist_ok=True)

DIRS = ["simple", "medium", "complex"]

def list_shaders():
    out = []
    for d in DIRS:
        p = ROOT / "tests" / "shaders" / d
        if p.exists():
            for f in sorted(p.glob("*.glsl")):
                out.append((d, f))
    return out

def try_compile(header_path, kernel_path):
    """Invoke compilecl machinery and capture success/error."""
    # Build the kernel source the same way compilecl.py does
    import pyopencl as cl
    main_header = ROOT / "tests" / "ocl" / "main_header.cl"
    main_kernel = ROOT / "tests" / "ocl" / "main_kernel.cl"
    build_opts_file = ROOT / "tests" / "build_options.json"

    src = compilecl.construct_kernel_source(
        str(main_header), str(header_path), str(main_kernel), str(kernel_path)
    )
    opts = compilecl.load_build_options(str(build_opts_file))

    platforms = cl.get_platforms()
    device = platforms[0].get_devices()[0]
    ctx = cl.Context([device])
    try:
        prog = cl.Program(ctx, src).build(options=opts)
        return True, ""
    except Exception as e:
        return False, str(e)

def main():
    shaders = list_shaders()
    report = []
    for d, f in shaders:
        rec = {"dir": d, "name": f.name, "transpile": None, "compile": None,
               "transpile_err": "", "compile_err": ""}
        # --- transpile ---
        try:
            glsl = f.read_text(encoding="utf-8")
            result = transpile(glsl)
            header_path = RESULTS / f"{f.stem}.header.cl"
            kernel_path = RESULTS / f"{f.stem}.kernel.cl"
            header_path.write_text(result.get_header(), encoding="utf-8")
            kernel_path.write_text(result.get_kernel(), encoding="utf-8")
            rec["transpile"] = "OK"
        except Exception as e:
            rec["transpile"] = "FAIL"
            rec["transpile_err"] = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            (RESULTS / f"{f.stem}.transpile_err.txt").write_text(tb, encoding="utf-8")
            report.append(rec)
            print(f"[{d:7}] {f.name:40} TRANSPILE-FAIL")
            continue
        # --- compile ---
        try:
            ok, err = try_compile(header_path, kernel_path)
            if ok:
                rec["compile"] = "OK"
                print(f"[{d:7}] {f.name:40} OK")
            else:
                rec["compile"] = "FAIL"
                rec["compile_err"] = err
                (RESULTS / f"{f.stem}.compile_err.txt").write_text(err, encoding="utf-8")
                print(f"[{d:7}] {f.name:40} COMPILE-FAIL")
        except Exception as e:
            rec["compile"] = "ERROR"
            rec["compile_err"] = f"{type(e).__name__}: {e}"
            (RESULTS / f"{f.stem}.compile_err.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{d:7}] {f.name:40} COMPILE-ERROR")
        report.append(rec)

    (RESULTS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # summary
    n = len(report)
    tp_ok = sum(1 for r in report if r["transpile"] == "OK")
    cp_ok = sum(1 for r in report if r["compile"] == "OK")
    print("\n==== SUMMARY ====")
    print(f"Total shaders     : {n}")
    print(f"Transpile OK      : {tp_ok}/{n}")
    print(f"Compile OK        : {cp_ok}/{n}")
    print(f"Full pipeline OK  : {cp_ok}/{n}")

if __name__ == "__main__":
    main()
