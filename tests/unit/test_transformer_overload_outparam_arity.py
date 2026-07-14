"""
Unit tests for the UNKNOWN "spurious `&` on an overloaded by-value call"
sub-cluster (Session 35, shader 4dtGWB "GLSL smallpt").

When two user functions share a name but differ in arity, the out-param
`&`-insertion in `_transform_call_expression` looked up `function_signatures`
by name only. The registry stored just ONE signature per name (last definition
wins), so a call to the by-value overload was matched against the OTHER
overload's pointer parameters and gained a spurious `&`:

    float intersect(Sphere s, Ray r);                 // both by value
    int   intersect(Ray r, out float t, out Sphere s, int avoid);  // out-params

    float d = intersect(S, r);   // 2 args -> by-value overload
    ->  intersect(S, &r)         // WRONG: '&' taken from the 4-param overload
        error: no matching function for call to 'intersect'

Fix: bucket `function_signatures` by arity so a call selects the overload whose
parameter count matches the call's argument count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def test_byvalue_overload_call_not_pointerised():
    # The 4dtGWB shape: a by-value 2-arg overload must NOT gain '&' from the
    # 4-param out-param overload registered under the same name.
    src = """
    struct Sphere { float r; };
    struct Ray { vec3 o; vec3 d; };
    float intersect(Sphere s, Ray r) { return s.r + r.o.x; }
    int intersect(Ray r, out float t, out Sphere s, int avoid) {
        t = 1.0; s.r = 2.0; return avoid;
    }
    void mainImage(out vec4 O, in vec2 U) {
        Sphere S; Ray r;
        float d = intersect(S, r);
        O = vec4(d);
    }
    """
    kernel = _kernel(src)
    assert "intersect(S, r)" in kernel
    assert "intersect(S, &r)" not in kernel


def test_outparam_overload_call_still_pointerised():
    # The matching-arity out-param overload must still get '&' on its out args.
    src = """
    struct Sphere { float r; };
    struct Ray { vec3 o; vec3 d; };
    float intersect(Sphere s, Ray r) { return s.r + r.o.x; }
    int intersect(Ray r, out float t, out Sphere s, int avoid) {
        t = 1.0; s.r = 2.0; return avoid;
    }
    void mainImage(out vec4 O, in vec2 U) {
        Ray r; float t; Sphere obj; int id = 0;
        id = intersect(r, t, obj, id);
        O = vec4(t);
    }
    """
    kernel = _kernel(src)
    assert "intersect(r, &t, &obj, id)" in kernel
