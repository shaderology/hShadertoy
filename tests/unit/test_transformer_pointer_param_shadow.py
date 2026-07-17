"""
Unit tests for category B residual: a LOCAL variable that shadows an out/inout
pointer parameter inside a nested block (Session 55).

out/inout params become OpenCL pointers, and every read of one is dereferenced
(`*p`). `pointer_params` was a flat per-function set with no block scoping, so a
nested local re-declaring the param's name (legal GLSL shadowing) still got the
pointer deref applied to its reads:

    void trace(inout vec3 r) {          // r is a float3*
        ...
        if (cond) {
            float r = 0.05;             // LOCAL float shadows the param
            float r2 = r * r;           // must be r*r, NOT *r * *r
        }
        ... r.x ...                     // back to the param -> (*r).x
    }

The shadowed reads emitted `*r * *r` -> "indirection requires pointer operand
('float' invalid)". Fix: `_transform_compound_statement` saves/restores
`pointer_params` around each block, and `_transform_declaration` discards a name
from `pointer_params` when a local shadows it — so the deref is suppressed for
the block's extent and restored afterwards. Corpus sole-blocker: tsXBzs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _full(src: str) -> str:
    r = transpile(src)
    return r.get_header() + "\n" + r.get_kernel()


SHADOW_SRC = """
void trace(inout vec3 r) {
    float pre = r.x;
    if (pre > 0.0) {
        float r = 0.05;
        float r2 = r * r;
    }
    float post = r.y;
}
void mainImage(out vec4 O, in vec2 U) {
    vec3 v = vec3(1.0);
    trace(v);
    O = vec4(v, 1.0);
}
"""


def test_shadowed_local_read_is_not_dereferenced():
    out = _full(SHADOW_SRC)
    # The bug: the shadowing local's reads were emitted as pointer derefs.
    assert "*r * *r" not in out
    # The local multiply survives with no deref.
    assert "r * r" in out


def test_param_reads_before_and_after_block_still_deref():
    out = _full(SHADOW_SRC)
    # r.x before the shadow block and r.y after it both reference the param.
    assert "(*r).x" in out
    assert "(*r).y" in out
