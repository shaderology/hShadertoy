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
    # a top-level function-macro #define is KEPT (not mid-statement): the
    # scalar-ctor-in-#define pass still needs it, OpenCL re-expands it; the call
    # site is expanded inline too.
    src = "#define SQ(x) ((x)*(x))\nfloat y = SQ(a);"
    out = expand_function_macros(src)
    assert "#define SQ(x)" in out
    assert "((a)*(a))" in _norm(out).replace(" ", "")


def test_function_define_dropped_when_mid_statement():
    # a function-macro (re)defined BETWEEN the sub-expressions of one statement
    # (ldfXzB) must be dropped — tree-sitter cannot parse a directive there. The
    # define at a statement boundary (after `;`) is still kept.
    src = (
        "#define KEEP(x) k(x)\n"           # top-level -> kept
        "float r = KEEP(1);\n"
        "bool ok = true\n"
        "#define MID(x) && q(x)\n"         # mid-expression (after `true`) -> dropped
        "MID(a) MID(b)\n"
        ";\n"
    )
    out = expand_function_macros(src)
    assert "#define KEEP(x)" in out
    assert "#define MID(x)" not in out
    n = _norm(_code(out))
    assert "true && q(a) && q(b)" in n
    assert out.count("\n") == src.count("\n")


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
# Object-like macro that WRAPS function-like macro calls (Session 62, ldfXzB)
# — the object macro is expanded at its use site so the wrapped function-macro
#   calls become ordinary code; a plain object macro (PI) is still left alone.
# ---------------------------------------------------------------------------

def test_object_macro_wrapping_function_macro_expands():
    src = (
        "#define PRIM(x) && f(x)\n"
        "#define LIST PRIM(a) PRIM(b) PRIM(c)\n"
        "bool ok = true LIST ;\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "LIST" not in n and "PRIM(" not in n
    assert "true && f(a) && f(b) && f(c)" in n


def test_object_wrapping_macro_uses_current_undef_redefine():
    # PRIM is redefined between two uses of the same wrapping object macro; each
    # use must expand with the PRIM definition live at that point (real cpp order).
    src = (
        "#define LIST PRIM(a) PRIM(b)\n"
        "#define PRIM(x) && sphere(x)\n"
        "r = true LIST ;\n"
        "#undef PRIM\n"
        "#define PRIM(x) && box(x)\n"
        "s = true LIST ;\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "r = true && sphere(a) && sphere(b)" in n
    assert "s = true && box(a) && box(b)" in n


def test_nested_wrapping_object_macros():
    # one wrapping object macro references another (SPHEREPRIMLISTWITHLIGHTS shape)
    src = (
        "#define PRIM(x) && f(x)\n"
        "#define BASE PRIM(a) PRIM(b)\n"
        "#define ALL BASE PRIM(c)\n"
        "bool ok = true ALL ;\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "true && f(a) && f(b) && f(c)" in n


def test_plain_object_macro_still_untouched_with_wrapping_enabled():
    # a wrapping object macro coexists with a plain one; only the wrapper expands
    src = (
        "#define PI 3.14159\n"
        "#define PRIM(x) g(x)\n"
        "#define LIST PRIM(a)\n"
        "float y = PI; bool z = LIST ;\n"
    )
    out = expand_function_macros(src)
    assert "#define PI 3.14159" in out
    assert "PI" in _norm(_code(out))
    assert "g(a)" in _norm(_code(out))


# ---------------------------------------------------------------------------
# Multi-line macro CALL sites (Session 62, ldfyRn) — a call whose argument list
# spans several physical lines with NO backslash continuation.
# ---------------------------------------------------------------------------

def test_multiline_call_site_no_backslash():
    src = (
        "#define P(cmd) d = 0.; cmd ; draw(d);\n"
        "#define S(a,b) step(a,b)\n"
        "x = 1.;\n"
        "P( S(1.,2.)\n"
        "   S(3.,4.)\n"
        "   S(5.,6.) )\n"
        "y = 2.;\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "P(" not in n and "S(" not in n
    assert "d = 0." in n and "step(1.,2.)" in n and "step(5.,6.)" in n
    assert "draw(d)" in n


def test_multiline_call_preserves_line_count():
    src = (
        "#define P(cmd) start cmd end\n"
        "A( just_a_call )\n"          # unrelated line to keep numbering honest
        "P( a\n"
        "   b\n"
        "   c )\n"
        "Z\n"
    )
    out = expand_function_macros(src)
    assert out.count("\n") == src.count("\n")


# ---------------------------------------------------------------------------
# Comments hide #define directives (Session 62, 3t2XzW) — a #define inside a
# block comment must NOT be registered, so a real function definition whose name
# collides with the commented-out macro is NOT mangled.
# ---------------------------------------------------------------------------

def test_commented_out_define_not_registered():
    src = (
        "/*\n"
        "#define foo(a,b) bar(a,b)\n"
        "*/\n"
        "float foo(float a, float b) { return a + b; }\n"
        "#define G(x) foo(x, 1.)\n"
        "y = G(k);\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    # the real function definition survives verbatim (not expanded as a macro)
    assert "float foo(float a, float b)" in n
    assert "bar(" not in n
    # the genuine function macro still expands
    assert "foo(k, 1.)" in n


def test_block_comment_on_one_line_closes_inline():
    # `/**/` closes the comment opened earlier on the same construct
    src = (
        "/* commented\n"
        "#define M(a) mangle(a) /**/\n"
        "float M(float a) { return a; }\n"
    )
    out = expand_function_macros(src)
    n = _norm(_code(out))
    assert "float M(float a)" in n
    assert "mangle(" not in n


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


# ---------------------------------------------------------------------------
# Gate extension (Session 62, tlsSDs/4djfDR): a source that PARSES but defines
# an entry point ONLY as a macro (no real `void mainImage` outside comments)
# still needs expansion — such a shader can never transpile as-is ("Could not
# find mainImage"), so firing on it cannot regress a passing shader.
# ---------------------------------------------------------------------------

def test_gate_fires_on_entry_macro_without_real_entry():
    # a single spliced #define parses fine, but there is no real mainImage
    src = "#define mainImage(z,u) z = vec4(u, 0., 1.)\n"
    out = maybe_expand_function_macros(src)
    assert out != src
    assert "void mainImage(out vec4 z, in vec2 u)" in _norm(out)


def test_gate_fires_when_real_entry_only_in_comments():
    # 4djfDR shape: the real definitions are all commented out
    src = (
        "#define mainImage(C,U) C = vec4(U, 0., 1.)\n"
        "/*\n"
        "void mainImage( out vec4 C, vec2 U )\n"
        "{ C = vec4(0); }\n"
        "*/\n"
    )
    out = maybe_expand_function_macros(src)
    assert "void mainImage(out vec4 C, in vec2 U)" in _norm(out)
    assert "{ C = vec4(0); }" not in out  # commented-out body stays dead


def test_gate_skips_entry_macro_with_real_entry_present():
    # a real mainImage exists -> current behavior is kept (source untouched)
    src = (
        "#define mainImage(o,u) helper(o,u)\n"
        "void mainImage(out vec4 o, in vec2 u) { o = vec4(u, 0., 1.); }\n"
    )
    assert maybe_expand_function_macros(src) == src
