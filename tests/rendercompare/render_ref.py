"""
Reference renderer: original Shadertoy GLSL -> PNG via wgpu-shadertoy.

Run as a subprocess by rc.py (one process per chunk - GPU resources are
released with the process):

    python tests/rendercompare/render_ref.py <manifest.json> <results.json>

Manifest format (shared shape with hda_render_headless.py):

    {"jobs": [{"id": "abc123",
               "cache_json": "tests/campaign/cache/abc123.json",  # OR
               "shader_inline": {"Shader": {...}},
               "outs": {"601": "artifacts/abc123/ref_f601.png", ...}}]}

Results: {"results": [{"id", "frame", "status": "OK|FAIL", "error",
                       "out_png", "notes": [...]}]}

Uniform contract: see common.py. iChannel textures are sourced from the
repo's local media mirror (houdini/pic/media/...) - shadertoy.com /media/
cannot be downloaded (Cloudflare). Unsupported input ctypes render as the
wgpu default black texture and are recorded in "notes".
"""

import os
os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "true"  # must precede wgpu import

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def build_inputs(image_pass: dict, notes: list):
    """Map API inputs to wgpu ShadertoyChannel objects."""
    from wgpu_shadertoy.inputs import ShadertoyChannelTexture

    channels = []
    for inp in image_pass.get("inputs", []):
        ctype = inp.get("ctype", "")
        chan = inp.get("channel", 0)
        sampler = inp.get("sampler", {})
        if ctype != "texture":
            notes.append(f"ch{chan}: unsupported ctype '{ctype}' -> black")
            continue
        local = common.media_path_for_src(inp.get("src", ""))
        if local is None:
            notes.append(f"ch{chan}: media not in local mirror -> black")
            continue
        img = Image.open(local)
        img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
        data = np.asarray(img, dtype=np.uint8)
        channels.append(ShadertoyChannelTexture(
            data,
            channel_idx=chan,
            wrap=sampler.get("wrap", "clamp-to-edge"),
            vflip=sampler.get("vflip", "true"),
            filter=sampler.get("filter", "linear"),
        ))
    return channels


def render_job(job: dict) -> list:
    from wgpu_shadertoy import Shadertoy

    notes = []
    results = []
    if job.get("shader_inline"):
        shader_data = job["shader_inline"]
        if "Shader" not in shader_data:
            shader_data = {"Shader": shader_data}
    else:
        shader_data = common.load_cache_shader(job["id"])

    image, common_pass, others = common.split_passes(shader_data)
    if image is None:
        raise RuntimeError("no image pass")
    if others:
        notes.append(f"{len(others)} non-image pass(es) ignored by ref renderer")

    inputs = build_inputs(image, notes)
    shader = Shadertoy(
        image["code"],
        common=(common_pass or {}).get("code", ""),
        resolution=common.RESOLUTION,
        offscreen=True,
        inputs=inputs,
    )
    swap_bgra = "bgra" in str(getattr(shader, "_format", "")).lower()
    date = common.PINNED_IDATE

    for frame_str, out_png in job["outs"].items():
        frame = int(frame_str)
        arr = np.asarray(shader.snapshot(
            time_float=common.itime_for_frame(frame),
            time_delta=1.0 / common.FPS,
            frame=frame,
            framerate=common.FPS,
            mouse_pos=(0.0, 0.0, 0.0, 0.0),
            date=date,
        ))
        if swap_bgra:
            arr = arr[..., [2, 1, 0, 3]]
        if common.FLIP_REF_Y:
            arr = arr[::-1]
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(arr[..., :3])).save(out_png)
        results.append({"id": job["id"], "frame": frame, "status": "OK",
                        "error": "", "out_png": out_png, "notes": notes})
    return results


def main() -> int:
    manifest_path, results_path = sys.argv[1], sys.argv[2]
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    all_results = []
    for job in manifest["jobs"]:
        try:
            all_results.extend(render_job(job))
        except Exception as e:  # keep the chunk going; record the failure
            traceback.print_exc()
            for frame_str, out_png in job.get("outs", {}).items():
                all_results.append({
                    "id": job["id"], "frame": int(frame_str), "status": "FAIL",
                    "error": f"{type(e).__name__}: {e}", "out_png": out_png,
                    "notes": []})

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=1)
    print(f"render_ref: {sum(r['status'] == 'OK' for r in all_results)}/"
          f"{len(all_results)} renders OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
