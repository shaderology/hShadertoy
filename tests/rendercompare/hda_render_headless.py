"""
HDA renderer (runs under hython): transpiled shader -> PNG via rop_image.

Invoked by render_hda.py / rc.py - do not run with plain python:

    hython tests/rendercompare/hda_render_headless.py <manifest.json> <results.json>

Same manifest/results shape as render_ref.py. For every job it:
  1. builds the hShadertoy HDA from the shader JSON (mode="Transpile"),
  2. sets init_iResolution to the contract resolution,
  3. drops a `rop_image` (Copernicus ROP) next to it - the shape saved in
     resources/examples/london/hShadertoy_output_template.json - pointed at
     the HDA, OCIO "Automatic" (the HDA's own output transform is
     "sRGB - Texture", matching its input),
  4. renders each requested Houdini frame to PNG (frame drives iTime via
     @Time and iFrame via the HDA's $FF binding; FPS is set to 60),
  5. collects cook/render errors from the whole node tree and tears the
     network down before the next job.

Requires HSHADERTOY_ROOT and HOUDINI_OCL_PATH='<repo>/houdini/ocl;&' in the
environment (the wrapper sets both; the ';&' rule is the segfault guard
documented in builder_cook_headless.py).
"""

import json
import os
import sys
import traceback
from pathlib import Path

import hou

hshadertoy_root = hou.getenv('HSHADERTOY_ROOT')
if hshadertoy_root:
    py_root = str(Path(hshadertoy_root) / 'houdini' / 'scripts' / 'python')
    for p in (py_root, hshadertoy_root):
        if p not in sys.path:
            sys.path.insert(0, p)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from hshadertoy.builder import build_shadertoy_hda  # noqa: E402


def _collect_errors(node):
    """Cook errors land on internal COPs - walk the whole subtree."""
    errors = []
    for n in (node,) + node.allSubChildren():
        try:
            msgs = n.errors()
        except Exception:
            continue
        if msgs:
            errors.append((n.path(), " | ".join(m.strip() for m in msgs)))
    return errors


def _preflight():
    ocl_path = os.environ.get('HOUDINI_OCL_PATH') or hou.getenv('HOUDINI_OCL_PATH')
    if not ocl_path or '&' not in ocl_path:
        print("Error: HOUDINI_OCL_PATH must be set and contain '&' "
              "(e.g. '<repo>/houdini/ocl;&') - see builder_cook_headless.py")
        sys.exit(1)
    if hshadertoy_root:
        hda_file = Path(hshadertoy_root) / 'houdini' / 'otls' / 'hShadertoy.hda'
        if hda_file.exists():
            hou.hda.installFile(str(hda_file))
        else:
            print(f"Warning: HDA not found at {hda_file}")
    try:
        hou.setFps(common.FPS)
    except AttributeError:
        hou.hscript(f"fps {common.FPS}")


def _pin_uniform_bindings(hda_node):
    """
    Make the HDA render deterministic: its internal OpenCL nodes ship an
    animated iMouse demo binding (sin($F) - even ~1 px flips 'has the mouse
    moved' branches) and an iDate binding evaluating datetime.now(). Unlock
    the instance and pin both to the contract values (common.py).
    """
    try:
        hda_node.allowEditingOfContents(propagate=True)
    except Exception:
        pass
    pinned = {"iMouse": (0.0, 0.0, 0.0, 0.0), "iDate": common.PINNED_IDATE}
    for n in hda_node.allSubChildren():
        if n.type().name() != "opencl":
            continue
        for p in n.parms():
            name = p.name()
            if not (name.startswith("bindings") and name.endswith("_name")):
                continue
            try:
                bname = p.evalAsString()
            except Exception:
                continue
            if bname not in pinned:
                continue
            idx = name[len("bindings"):-len("_name")]
            for row in ("1", "2", "3", "4"):
                pt = n.parmTuple(f"bindings{idx}_v4val{row}")
                if pt is None:
                    continue
                for comp, val in zip(pt, pinned[bname]):
                    comp.deleteAllKeyframes()  # drops the expressions
                    comp.set(val)


def _make_rop(copnet, hda_node):
    rop = copnet.createNode('rop_image', 'rc_rop')
    rop.parm('coppath').set(hda_node.path())
    rop.parm('colorconversion').set('ocio')
    rop.parm('ociocolorspace').set('Automatic')
    rop.parm('mkpath').set(True)
    return rop


def render_job(job):
    results = []
    if job.get("shader_inline"):
        shader_data = job["shader_inline"]
        if "Shader" not in shader_data:
            shader_data = {"Shader": shader_data}
    else:
        shader_data = common.load_cache_shader(job["id"])

    node = build_shadertoy_hda(shader_data, mode="Transpile")
    copnet = node.parent()
    try:
        pt = node.parmTuple('init_iResolution')
        if pt is not None:
            pt.set((common.RESOLUTION[0], common.RESOLUTION[1], 0))
        # iTimeDelta is a plain init parm (never @-bound) - match the ref side
        p_dt = node.parm('init_iTimeDelta')
        if p_dt is not None:
            p_dt.set(1.0 / common.FPS)
        _pin_uniform_bindings(node)
        rop = _make_rop(copnet, node)

        for frame_str, out_png in job["outs"].items():
            frame = int(frame_str)
            out_png_abs = str(Path(out_png).resolve())
            Path(out_png_abs).parent.mkdir(parents=True, exist_ok=True)
            rop.parm('copoutput').set(out_png_abs)
            hou.setFrame(frame)
            err = ""
            try:
                rop.render(frame_range=(frame, frame))
            except Exception as e:
                err = f"render raised: {type(e).__name__}: {e}"
            node_errors = _collect_errors(node) + _collect_errors(rop)
            if node_errors:
                err = (err + " | " if err else "") + "; ".join(
                    f"[{p}] {m}" for p, m in node_errors)
            if not err and not Path(out_png_abs).exists():
                err = "render reported no error but wrote no file"
            results.append({
                "id": job["id"], "frame": frame,
                "status": "FAIL" if err else "OK",
                "error": err[:2000], "out_png": out_png, "notes": []})
            if err:
                break  # same shader will fail at every frame - move on
    finally:
        try:
            copnet.destroy()
        except Exception:
            pass
    return results


def main():
    manifest_path, results_path = sys.argv[1], sys.argv[2]
    _preflight()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    all_results = []
    for job in manifest["jobs"]:
        print(f"\n=== {job['id']} ===")
        try:
            all_results.extend(render_job(job))
        except Exception as e:
            traceback.print_exc()
            for frame_str, out_png in job.get("outs", {}).items():
                all_results.append({
                    "id": job["id"], "frame": int(frame_str),
                    "status": "FAIL",
                    "error": f"build: {type(e).__name__}: {e}"[:2000],
                    "out_png": out_png, "notes": []})
        # write-through after every job: a crash/timeout keeps partial results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"results": all_results}, f, indent=1)

    ok = sum(r['status'] == 'OK' for r in all_results)
    print(f"\nhda_render: {ok}/{len(all_results)} renders OK")


if __name__ == "__main__":
    main()
