"""
Shared contract + ledger IO for the render-compare pipeline.

The determinism contract (BOTH renderers must honour it exactly):

    resolution   800 x 450
    FPS          60
    Houdini frame HOUDINI_FRAME (601)  ->  iTime = (601-1)/60 = 10.0 s
    iFrame       601   (the HDA binds iFrame to $FF, so the reference side
                        is fed the same Houdini frame number, NOT frame-1)
    iTimeDelta   1/60  (wgpu side; the HDA leaves it at its init parm = 0.0 -
                        shaders reading iTimeDelta are flagged, see flags())
    iMouse       (0,0,0,0). The HDA ships a baked sin($F) demo animation on
                 the internal iMouse binding; even the ~1 px offset flips
                 "has the mouse moved" branches (seen on 4dlXzN - the Portal
                 turret opens), so hda_render_headless.py unlocks the HDA
                 instance and zeroes the binding.
    iDate        PINNED_IDATE on both sides (the HDA's datetime.now()
                 binding expressions are overridden the same way).

Frame 601 (iTime 10 s) satisfies the "frame 100+" requirement: stateful
shaders get time to play out and artistic fade-ins from black are over.
A second probe frame catches "correct at one time only" bugs cheaply.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RC_ROOT = REPO_ROOT / "tests" / "rendercompare"
ARTIFACTS = RC_ROOT / "artifacts"
LEDGER_PATH = RC_ROOT / "ledger.json"
CAMPAIGN_LEDGER = REPO_ROOT / "tests" / "campaign" / "ledger.json"
CAMPAIGN_CACHE = REPO_ROOT / "tests" / "campaign" / "cache"
MEDIA_ROOT = REPO_ROOT / "houdini" / "pic"  # local mirror of shadertoy /media/

RESOLUTION = (800, 450)  # (width, height)
FPS = 60.0
HOUDINI_FRAME = 601          # primary probe: iTime = 10.0
HOUDINI_FRAME_B = 151        # secondary probe: iTime = 2.5
FRAMES = (HOUDINI_FRAME, HOUDINI_FRAME_B)

# Orientation/color calibration (baked from the phase-1 gradient shader run;
# see README.md "Calibration"). FLIP_HDA_Y=True means the PNG written by the
# rop_image COP ROP is upside-down relative to the wgpu snapshot array.
FLIP_HDA_Y = False
FLIP_REF_Y = False

# (year, month, day, seconds-of-day) - pinned on BOTH sides.
PINNED_IDATE = (2026.0, 1.0, 1.0, 0.0)


def itime_for_frame(houdini_frame: int) -> float:
    """Houdini time at a frame: frame 1 == t 0."""
    return (houdini_frame - 1) / FPS


def now_idate() -> tuple:
    """iDate the way both shadertoy and the HDA build it."""
    n = datetime.now()
    secs = n.hour * 3600 + n.minute * 60 + n.second
    return (float(n.year), float(n.month), float(n.day), float(secs))


# ---------------------------------------------------------------- shader meta

# ctypes the reference renderer cannot reproduce faithfully
_UNSUPPORTED_CTYPES = {
    "volume", "video", "music", "musicstream", "mic",
    "webcam", "keyboard", "cubemap",
}

_FLAG_PATTERNS = {
    "uses_iDate": re.compile(r"\biDate\b"),
    "uses_iTimeDelta": re.compile(r"\biTimeDelta\b"),
    "uses_iChannelTime": re.compile(r"\biChannelTime\b"),
    "uses_iMouse": re.compile(r"\biMouse\b"),
    "uses_iFrame": re.compile(r"\biFrame\b"),
    "uses_textureLod": re.compile(r"\btextureLod\b"),
    "uses_derivatives": re.compile(r"\b(dFdx|dFdy|fwidth)\b"),
}


def load_cache_shader(shader_id: str) -> dict:
    """Load a campaign-cache shader and normalise to {"Shader": {...}}."""
    path = CAMPAIGN_CACHE / f"{shader_id}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "Shader" not in data:
        data = {"Shader": data}
    return data


def split_passes(shader: dict):
    """Return (image_pass, common_pass, other_passes) from a Shader dict."""
    image = common = None
    others = []
    for rp in shader["Shader"]["renderpass"]:
        t = rp.get("type", "")
        if t == "image" and image is None:
            image = rp
        elif t == "common" and common is None:
            common = rp
        else:
            others.append(rp)
    return image, common, others


def shader_flags(shader: dict) -> list[str]:
    """Determinism / divergence flags for the ledger."""
    image, common, others = split_passes(shader)
    code = (image.get("code", "") if image else "")
    if common:
        code += "\n" + common.get("code", "")
    flags = sorted(name for name, rx in _FLAG_PATTERNS.items() if rx.search(code))
    if others:
        flags.append("multipass")
    for rp in shader["Shader"]["renderpass"]:
        for inp in rp.get("inputs", []):
            ct = inp.get("ctype", "")
            if ct in _UNSUPPORTED_CTYPES:
                flags.append(f"input_{ct}")
    return sorted(set(flags))


def media_path_for_src(src: str) -> Path | None:
    """Map an API '/media/a/<hash>.jpg' src to the bundled local file."""
    p = MEDIA_ROOT / src.lstrip("/")
    return p if p.is_file() else None


# ---------------------------------------------------------------- ledger IO

def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ledger(ledger: dict) -> None:
    """Atomic write - the campaign pattern (crash-resumable)."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(LEDGER_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=1, sort_keys=True)
        os.replace(tmp, LEDGER_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def artifact_dir(shader_id: str) -> Path:
    d = ARTIFACTS / shader_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def png_names(houdini_frame: int) -> tuple[str, str, str]:
    """(ref, hda, diff) artifact basenames for a probe frame."""
    return (
        f"ref_f{houdini_frame}.png",
        f"hda_f{houdini_frame}.png",
        f"diff_f{houdini_frame}.png",
    )
