"""
Unit tests for the C-cast disambiguation pass (UNKNOWN sub-cluster, Session 32).

tree-sitter-glsl inherits C's grammar, in which `(ident) <expr>` is ambiguous
between a parenthesised grouping and a C-style type cast. GLSL has NO C casts,
but the GLR parser nonetheless resolves the ambiguity toward `cast_expression`
in certain operator contexts — notably when a division (or another `*`/`/`)
sits adjacent to a `(ident) + term` sub-expression:

    PI*2.0*(rot)+PI/turns   ->  parsed as  PI*2.0*<cast (rot) of (+PI)> / turns
    1./(distlpsp) + 1./(d2) ->  parsed as  1./<cast (distlpsp) of (+1.)> / (d2)

The transformer has no `cast_expression` handler, so the mis-parsed operand
transformed to nothing and the emitter dropped a whole chunk of the
expression, e.g. `PI * 2.0f *  / turns` (note the empty gap). Worse, the cast
mis-parse *re-associates* the surrounding operators, so a local node rewrite
could not restore the correct arithmetic.

Fix: after the array/precision/xor normalisations, `GLSLParser.parse` detects
any `cast_expression` nodes and double-parenthesises the offending type span in
the source (`(rot)` -> `((rot))`), which is semantically identical for every
GLSL type and forces the parser back onto the grouping interpretation. The
re-parse restores the correct precedence tree.

Corpus casualties (sole-blockers): 4d3SWl (Another Mobius), MlXSWX
(Abstract Corridor).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def _kernel(src: str) -> str:
    return transpile(src).kernel


def _wrap(expr: str) -> str:
    return (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord){"
        "float a=1.,b=1.,d=2.,turns=3.,PI=3.14,rot=0.5;"
        f"float x={expr};"
        "fragColor=vec4(x);}"
    )


def test_mul_paren_plus_div_not_dropped():
    """`a*(d)+b/d` must keep every operand (no empty gap)."""
    k = _kernel(_wrap("a*(d)+b/d"))
    assert "a *  /" not in k
    # both additive terms survive
    assert "(d)" in k
    assert "b / d" in k


def test_4d3SWl_shape():
    """The Another Mobius idiom `PI*2.0*(rot)+PI/turns`."""
    k = _kernel(_wrap("PI*2.0*(rot)+PI/turns"))
    assert "*  /" not in k          # the dropped-chunk signature
    assert "(rot)" in k
    assert "PI / turns" in k


def test_MlXSWX_shape():
    """The Abstract Corridor idiom `1./(d) + 1./(a)`."""
    k = _kernel(_wrap("1./(d) + 1./(a)"))
    assert "/  /" not in k          # the dropped-chunk signature
    assert "(d)" in k
    assert "(a)" in k


def test_precedence_preserved():
    """Disambiguation must not change arithmetic grouping.

    `a*(d)+b/d` == `(a*d) + (b/d)`; a naive local rewrite would have produced
    `(a*(d+b))/d`. Emit both additive terms as siblings of a top-level `+`.
    """
    k = _kernel(_wrap("a*(d)+b/d"))
    # the `+` joins the two products, and the `/ d` stays attached to b
    assert "+ b / d" in k.replace("  ", " ")


def test_plain_grouping_unaffected():
    """A grouping paren that never mis-parsed is emitted unchanged."""
    k = _kernel(_wrap("a*(d)+b"))
    assert "a * (d) + b" in k
