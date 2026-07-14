"""
Unit tests for category P cluster 5 — function-like macro expansion (Session 24).

tree-sitter-glsl parses `#define` but does not expand function-like macros, so
their call sites (operator args, juxtaposition, partial-expression bodies) are
syntactically invalid GLSL and the parse fails before OpenCL is reached. The
`expand_function_macros` pass expands function-like macro USES on raw GLSL, before
tree-sitter, dropping the consumed `#define` lines. Object-like macros are left
untouched (the existing pass / OpenCL handle them). `mainImage`-as-a-macro is
synthesized into a real entry function.

See tests/fixcampaign/CLUSTER5_MACRO_DESIGN.md.
"""

import re
import pytest

from src.glsl_to_opencl.preprocessor.macro_expander import (
    expand_function_macros,
    maybe_expand_function_macros,
)


def _norm(s):
    # collapse runs of whitespace for tolerant structural comparison
    return re.sub(r'\s+', ' ', s).strip()


def _code(out):
    # only the non-directive (code) lines — #define bodies are kept and would
    # otherwise contain the very substrings we assert on the expanded call sites
    return "\n".join(l for l in out.split("\n") if not l.lstrip().startswith("#"))


# ---------------------------------------------------------------------------
# Basic expansion
# ---------------------------------------------------------------------------

def test_simple_function_macro_expands():
    src = "#define SQ(x) ((x)*(x))\nfloat y = SQ(a+1);"
    out = expand_function_macros(src)
    assert "SQ(" not in _code(out)
    assert "((a+1)*(a+1))" in _norm(out).replace(" ", "")


def test_define_line_preserved_for_opencl():
    # the #define is kept (OpenCL may still need it for uses inside object-like
    # macro bodies that we don't expand); only the call site is expanded inline.
    src = "#define SQ(x) ((x)*(x))\nfloat y = SQ(a);"
    out = expand_function_macros(src)
    assert "#define SQ(x)" in out


def test_object_like_macro_untouched():
    src = "#define PI 3.14159\nfloat y = PI * r;"
    out = expand_function_macros(src)
    # object-like macros are left for the existing pass / OpenCL
    assert "#define PI 3.14159" in out
    assert "PI * r" in out


def test_valid_call_syntax_still_expands_equivalently():
    # `f(a)` is valid function-call syntax; expanding it inline must be equivalent
    src = "#define f(a) (a + 1.0)\nfloat y = f(x) + f(2.0);"
    out = expand_function_macros(src)
    n = _norm(out)
    assert "(x + 1.0)" in n and "(2.0 + 1.0)" in n


# ---------------------------------------------------------------------------
# The hard call-site forms that break tree-sitter
# ---------------------------------------------------------------------------

def test_operator_as_argument():
    src = "#define S(s,d) sin(mod(-atan(U.y,U.x) s 3.14, v) d)\nx = S(+,-1.);"
    out = expand_function_macros(src)
    n = _norm(out)
    assert "sin(mod(-atan(U.y,U.x) + 3.14, v) -1.)" in n


def test_empty_argument():
    src = "#define S(s,d) a s b d\nx = S(+,);"
    out = expand_function_macros(src)
    assert _norm(out).endswith("x = a + b ;") or "a + b" in _norm(out)


def test_juxtaposed_calls():
    src = "#define T(u,v) + texture(iChannel0, U + vec2(u,v))\nx = 0. T(0,0)T(1,0);"
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert n.count("+ texture(iChannel0, U + vec2(") == 2
    assert "T(" not in n


def test_partial_expression_body_chained():
    src = "#define C(v) v.x<e ||\nbool b = C(a) C(b) false;"
    out = expand_function_macros(src)
    n = _norm(out)
    assert "a.x<e || b.x<e || false" in n


# ---------------------------------------------------------------------------
# Nested / recursive expansion (terminating)
# ---------------------------------------------------------------------------

def test_nested_macro_in_body():
    src = "#define A(x) ((x)+1)\n#define B(y) A(y)*2\nint z = B(k);"
    out = expand_function_macros(src)
    assert "((k)+1)*2" in _norm(out).replace(" ", "")


def test_object_macro_referenced_in_function_body_left_alone():
    src = "#define R iResolution\n#define T(u) texture(ch, (u)/R)\nx = T(p);"
    out = expand_function_macros(src)
    n = _norm(out)
    assert "texture(ch, (p)/R)" in n
    assert "#define R iResolution" in n  # object macro kept


def test_self_reference_terminates():
    # pathological: must not infinite-loop
    src = "#define A(x) A(x)+1\nint z = A(q);"
    out = expand_function_macros(src)  # just needs to return
    assert "A(" in out or "+1" in out


# ---------------------------------------------------------------------------
# Redefinition: source-order + #undef
# ---------------------------------------------------------------------------

def test_undef_then_redefine():
    src = (
        "#define P(x) foo(x)\n"
        "a = P(1);\n"
        "#undef P\n"
        "#define P(x) bar(x)\n"
        "b = P(2);\n"
    )
    out = expand_function_macros(src)
    n = _norm(out)
    assert "a = foo(1)" in n
    assert "b = bar(2)" in n


def test_sequential_redefinition_different_arity():
    src = (
        "#define D(m) one(m)\n"
        "a = D(p);\n"
        "#define D(m,z) two(m,z)\n"
        "b = D(p,q);\n"
    )
    out = expand_function_macros(src)
    n = _norm(out)
    assert "a = one(p)" in n
    assert "b = two(p,q)" in n


# ---------------------------------------------------------------------------
# Conditional macro definitions: the active #if branch wins (Session 34)
# ---------------------------------------------------------------------------

def test_conditional_macro_uses_active_ifdef_branch():
    # DISPERSION is defined, so CHANNEL must expand to the #ifdef body, NOT the
    # #else body — matching what OpenCL does for the sibling object-like macro.
    src = (
        "#define DISPERSION\n"
        "#ifdef DISPERSION\n"
        "\t#define CHANNEL(x) dot(x, channel)\n"
        "#else\n"
        "\t#define CHANNEL(x) x\n"
        "#endif\n"
        "y = CHANNEL(color);\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "y = dot(color, channel)" in n


def test_conditional_macro_uses_else_branch_when_undefined():
    # DISPERSION not defined -> the #else body is the active one.
    src = (
        "#ifdef DISPERSION\n"
        "\t#define CHANNEL(x) dot(x, channel)\n"
        "#else\n"
        "\t#define CHANNEL(x) x\n"
        "#endif\n"
        "y = CHANNEL(color);\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "y = color" in n
    assert "dot(" not in n


def test_conditional_macro_ifndef_branch():
    src = (
        "#ifndef FOO\n"
        "\t#define G(x) a(x)\n"
        "#else\n"
        "\t#define G(x) b(x)\n"
        "#endif\n"
        "z = G(p);\n"
    )
    out = expand_function_macros(src)
    assert "z = a(p)" in _norm(_code(out))


def test_conditional_nested_ifdef():
    src = (
        "#define OUTER\n"
        "#ifdef OUTER\n"
        "\t#ifdef INNER\n"
        "\t\t#define H(x) inner(x)\n"
        "\t#else\n"
        "\t\t#define H(x) outer(x)\n"
        "\t#endif\n"
        "#endif\n"
        "w = H(q);\n"
    )
    out = expand_function_macros(src)
    assert "w = outer(q)" in _norm(_code(out))


# ---------------------------------------------------------------------------
# mainImage-as-a-macro → synthesized entry function
# ---------------------------------------------------------------------------

def test_mainimage_macro_expression_body_synthesized():
    src = "#define mainImage(C,U) C = vec4(U,0,1)"
    out = expand_function_macros(src)
    n = _norm(out)
    assert "void mainImage(out vec4 C, in vec2 U)" in n
    assert "{ C = vec4(U,0,1); }" in n


def test_mainimage_macro_brace_body_synthesized():
    src = "#define mainImage(o,u) { o *= .0; o.y = .6; }"
    out = expand_function_macros(src)
    n = _norm(out)
    assert "void mainImage(out vec4 o, in vec2 u) { o *= .0; o.y = .6; }" in n


def test_mainimage_macro_body_expands_inner_macro():
    src = (
        "#define D(a) (u.x - a)\n"
        "#define mainImage(O,u) O = vec4(D(1.) + D(2.))\n"
    )
    out = expand_function_macros(src)
    n = _norm(out)
    assert "void mainImage(out vec4 O, in vec2 u)" in n
    assert "(u.x - 1.) + (u.x - 2.)" in n


# ---------------------------------------------------------------------------
# Line continuations
# ---------------------------------------------------------------------------

def test_body_trailing_comment_stripped():
    # a trailing // comment in the body must not be inlined (it would comment out
    # the rest of the use line, including its ';')
    src = "#define F(x) vec2(x) // draw helper\nvec2 P = F(0.); float k = 1.;"
    out = expand_function_macros(src)
    code = _code(out)
    assert "// draw helper" not in code
    assert "vec2 P = vec2(0.)" in _norm(code).replace(" ;", ";")
    assert ";" in code and "float k = 1." in code


def test_body_block_comment_stripped():
    src = "#define F(x) (x /* inner */ + 1)\ny = F(a);"
    out = expand_function_macros(src)
    assert "/* inner */" not in _code(out)
    assert "(a + 1)" in _norm(_code(out)).replace("  ", " ")


def test_multiline_define_continuation():
    src = "#define M(a,b) foo(a) \\\n + bar(b)\nx = M(1,2);"
    out = expand_function_macros(src)
    assert "foo(1) + bar(2)" in _norm(out)


# ---------------------------------------------------------------------------
# Gating: expansion is a fallback — a shader that already parses is untouched
# ---------------------------------------------------------------------------

def test_gate_skips_when_source_parses():
    # a valid function-call-syntax macro use parses fine -> leave it for OpenCL
    src = ("#define SQ(x) ((x)*(x))\n"
           "void mainImage(out vec4 o, in vec2 g){ float y = SQ(g.x); o = vec4(y); }")
    assert maybe_expand_function_macros(src) == src


def test_gate_expands_when_source_fails_to_parse():
    # operator-as-argument does NOT parse -> expansion kicks in
    src = ("#define S(s) (1. s 2.)\n"
           "void mainImage(out vec4 o, in vec2 g){ float y = S(+); o = vec4(y); }")
    out = maybe_expand_function_macros(src)
    assert out != src
    assert "(1. + 2.)" in _norm(out)


def test_gate_no_macro_returns_unchanged():
    src = "void mainImage(out vec4 o, in vec2 g){ o = vec4(g, 0.0, 1.0); }"
    assert maybe_expand_function_macros(src) == src
