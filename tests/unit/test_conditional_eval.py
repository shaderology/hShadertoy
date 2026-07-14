"""
Unit tests for category G (Session 38) — constant-conditional evaluation +
bounded object-like macro expansion (`preprocessor/conditional_eval.py`).

tree-sitter-glsl does not understand the C preprocessor: a conditional block
that straddles a statement/declaration/else-if chain, a bare `#undef`, or an
object-like macro used as a statement fragment makes the whole translation
unit fail to parse. `maybe_preprocess_directives` runs BEFORE tree-sitter,
gated on parse failure (a source that already parses is returned
byte-identical), and cascades:

  strip constant conditionals -> parses? done
  -> expand object-like macro uses -> parses? done
  -> give up, but return the STRIPPED source (never worse than stage 1).

Undecidable-`#if` policy under test: strict C semantics — after `defined()`
resolution and macro substitution, any remaining identifier evaluates to 0.
Only a genuinely un-evaluable expression (malformed, division by zero) keeps
its frame verbatim.

Shapes S1-S6 from tests/fixcampaign/G_DESIGN_BRIEF.md are each covered.
"""

import re
import pytest

from src.glsl_to_opencl.preprocessor.conditional_eval import (
    strip_conditionals,
    expand_object_macros,
    maybe_preprocess_directives,
)
from src.glsl_to_opencl.preprocessor.macro_expander import _source_parses
from src.glsl_to_opencl.preprocessor import PreprocessorTransformer


def _norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def _strip(src):
    return strip_conditionals(src).source


# ---------------------------------------------------------------------------
# S1 — conditional straddles an else-if chain
# ---------------------------------------------------------------------------

def test_s1_ifdef_straddles_else_if_chain():
    src = (
        "void mainImage(out vec4 c, in vec2 u){\n"
        "    float x = u.x;\n"
        "    if (x < 1.) c = vec4(1);\n"
        "#ifdef SHOW_SEC\n"
        "    else if (x < 2.) c = vec4(2);\n"
        "#endif\n"
        "    else c = vec4(0);\n"
        "}\n"
    )
    assert not _source_parses(src)  # precondition: this is the G failure
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    assert "vec4(2)" not in out          # dead else-if branch deleted
    assert "vec4(1)" in out and "vec4(0)" in out
    assert "#ifdef" not in out and "#endif" not in out
    assert len(out.split("\n")) == len(src.split("\n"))  # line count kept


# ---------------------------------------------------------------------------
# S2 — #if EXPR over a #define'd constant, branch-variant declarations
# ---------------------------------------------------------------------------

def test_s2_if_expr_over_defined_constant():
    src = (
        "#define AA 2\n"
        "#if AA>1\n"
        "    vec2 p = uv;\n"
        "#else\n"
        "    vec2 p = uv * 2.;\n"
        "#endif\n"
    )
    out = _strip(src)
    assert "vec2 p = uv;" in out
    assert "uv * 2." not in out
    assert "#if" not in out and "#else" not in out and "#endif" not in out
    assert "#define AA 2" in out         # defines are KEPT by the strip stage
    assert len(out.split("\n")) == len(src.split("\n"))


def test_s2_elif_chain_middle_branch_live():
    src = (
        "#define AA 2\n"
        "#if AA==1\n"
        "x1;\n"
        "#elif AA==2\n"
        "x2;\n"
        "#elif AA==3\n"
        "x3;\n"
        "#else\n"
        "x4;\n"
        "#endif\n"
    )
    out = _strip(src)
    assert "x2;" in out
    for dead in ("x1;", "x3;", "x4;"):
        assert dead not in out


# ---------------------------------------------------------------------------
# S3 — conditional straddles a mid-expression continuation
# ---------------------------------------------------------------------------

def test_s3_mid_expression_conditional():
    src = (
        "#define LIGHTING 1\n"
        "float shade(float base){\n"
        "    return base\n"
        "#if LIGHTING < 2\n"
        "    * 2.\n"
        "#endif\n"
        "    ;\n"
        "}\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(shade(u.x)); }\n"
    )
    assert not _source_parses(src)
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    assert "* 2." in out


# ---------------------------------------------------------------------------
# S4 — dead branch contains outright invalid GLSL (only deletion can fix)
# ---------------------------------------------------------------------------

def test_s4_dead_branch_invalid_glsl_deleted():
    src = (
        "//#define TESTING\n"
        "#ifdef TESTING\n"
        "vec2 s = mainSound( in int samp, time);\n"
        "#endif\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(0); }\n"
    )
    assert not _source_parses(src)
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    assert "mainSound" not in out


# ---------------------------------------------------------------------------
# S5 — bare #undef chokes tree-sitter: apply to table, blank the line
# ---------------------------------------------------------------------------

def test_s5_undef_at_file_scope_blanked():
    src = (
        "#define P 4.\n"
        "#undef P\n"
        "void mainImage(out vec4 o, in vec2 g){ o = vec4(0); }\n"
    )
    out = maybe_preprocess_directives(src)
    assert "#undef" not in out
    assert _source_parses(out)
    assert len(out.split("\n")) == len(src.split("\n"))


def test_s5_undef_inside_function_body_blanked():
    src = (
        "float f(){\n"
        "#undef CAM_PATH\n"
        "return 1.; }\n"
        "void mainImage(out vec4 o, in vec2 g){ o = vec4(f()); }\n"
    )
    out = maybe_preprocess_directives(src)
    assert "#undef" not in out
    assert _source_parses(out)


def test_undef_makes_name_undefined_for_ifdef():
    src = (
        "#define FLAG\n"
        "#undef FLAG\n"
        "#ifdef FLAG\n"
        "int dead;\n"
        "#endif\n"
        "int live;\n"
    )
    out = _strip(src)
    assert "int dead;" not in out
    assert "int live;" in out


# ---------------------------------------------------------------------------
# Built-in defines: HW_PERFORMANCE=1, __VERSION__=300, GL_ES=1
# ---------------------------------------------------------------------------

def test_builtin_hw_performance():
    src = "#if HW_PERFORMANCE==0\nint lowq;\n#else\nint hiq;\n#endif\n"
    out = _strip(src)
    assert "int hiq;" in out and "int lowq;" not in out


def test_builtin_version_and_gl_es():
    src = (
        "#if __VERSION__ >= 300\nint v3;\n#endif\n"
        "#ifdef GL_ES\nint es;\n#endif\n"
    )
    out = _strip(src)
    assert "int v3;" in out and "int es;" in out


# ---------------------------------------------------------------------------
# Undecidable policy: strict C — unknown identifier evaluates to 0
# ---------------------------------------------------------------------------

def test_unknown_identifier_is_zero():
    src = "#if MYSTERY\nint dead;\n#endif\n#if !MYSTERY\nint live;\n#endif\n"
    out = _strip(src)
    assert "int dead;" not in out
    assert "int live;" in out


def test_defined_operator_both_forms():
    src = (
        "#define FOO 1\n"
        "#if defined(FOO) && !defined(BAR)\nint a;\n#endif\n"
        "#if defined FOO\nint b;\n#endif\n"
        "#if defined(BAR)\nint c;\n#endif\n"
    )
    out = _strip(src)
    assert "int a;" in out and "int b;" in out and "int c;" not in out


def test_c_integer_semantics():
    # division truncates toward zero (Python // would give -2), hex literals,
    # u-suffix
    src = (
        "#if -3/2 == -1\nint trunc_ok;\n#endif\n"
        "#if 0x10 == 16\nint hex_ok;\n#endif\n"
        "#if 3u > 2\nint suffix_ok;\n#endif\n"
    )
    out = _strip(src)
    assert "int trunc_ok;" in out
    assert "int hex_ok;" in out
    assert "int suffix_ok;" in out


def test_unevaluable_if_keeps_frame_verbatim():
    # division by zero is a genuine preprocessor error -> keep both branches
    # and the directives untouched (the escape hatch)
    src = "#if 1/0\nint a;\n#else\nint b;\n#endif\n"
    out = _strip(src)
    assert "#if 1/0" in out
    assert "int a;" in out and "int b;" in out
    assert "#endif" in out


def test_unevaluable_elif_treated_false():
    src = "#if 0\nint a;\n#elif 1/0\nint b;\n#else\nint c;\n#endif\n"
    out = _strip(src)
    assert "int a;" not in out and "int b;" not in out
    assert "int c;" in out


# ---------------------------------------------------------------------------
# Nesting + dead-branch hygiene
# ---------------------------------------------------------------------------

def test_nested_conditionals():
    src = (
        "#define OUTER 1\n"
        "#if OUTER\n"
        "int a1;\n"
        "#if 0\n"
        "int a2;\n"
        "#else\n"
        "int a3;\n"
        "#endif\n"
        "int a4;\n"
        "#else\n"
        "int a5;\n"
        "#endif\n"
    )
    out = _strip(src)
    for live in ("int a1;", "int a3;", "int a4;"):
        assert live in out
    for dead in ("int a2;", "int a5;"):
        assert dead not in out


def test_define_in_dead_branch_not_registered():
    src = (
        "#if 0\n"
        "#define HIDDEN\n"
        "#endif\n"
        "#ifdef HIDDEN\n"
        "int dead;\n"
        "#endif\n"
        "int live;\n"
    )
    out = _strip(src)
    assert "int dead;" not in out
    assert "int live;" in out
    assert "#define HIDDEN" not in out   # dead-branch define blanked too


def test_version_extension_lines_preserved():
    src = "#version 300 es\n#extension GL_OES_standard_derivatives : enable\nint x;\n"
    out = _strip(src)
    assert "#version 300 es" in out
    assert "#extension" in out


# ---------------------------------------------------------------------------
# S6 — object-like macro as statement/expression fragment (expansion stage)
# ---------------------------------------------------------------------------

def test_s6_object_macro_statement_fragments():
    src = (
        "#define fGDFBegin float d = 0.;\n"
        "#define fGDFEnd return d - r;\n"
        "float fOcta(vec3 p, float r) { fGDFBegin d += p.x; fGDFEnd }\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(fOcta(vec3(u,1.), 1.)); }\n"
    )
    assert not _source_parses(src)
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    n = _norm(out)
    assert "float d = 0.;" in n
    assert "return d - r;" in n
    # fully consumed macros: their #define lines are blanked
    assert "#define fGDFBegin" not in out
    assert "#define fGDFEnd" not in out
    assert len(out.split("\n")) == len(src.split("\n"))


def test_s6_multiline_continuation_define_spliced():
    src = (
        "#define fBegin float d = 0.; \\\n"
        "    float e = 1.;\n"
        "#define fEnd return d + e - r;\n"
        "float fOcta(vec3 p, float r) { fBegin d += p.x; fEnd }\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(fOcta(vec3(u,1.), 1.)); }\n"
    )
    assert not _source_parses(src)
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    n = _norm(out)
    assert "float d = 0.;" in n and "float e = 1.;" in n
    assert len(out.split("\n")) == len(src.split("\n"))


def test_expansion_uses_live_branch_definition():
    # a macro defined in a DEAD branch must not be used for expansion
    src = (
        "#ifdef NOPE\n"
        "#define END return 1.;\n"
        "#else\n"
        "#define END return 0.;\n"
        "#endif\n"
        "float g(float d){ END }\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(g(u.x)); }\n"
    )
    assert not _source_parses(src)
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    assert "return 0." in out
    assert "return 1." not in out


def test_define_kept_when_still_referenced_elsewhere():
    # K is expanded in code, but still referenced from a (kept) function-like
    # macro body -> its #define must be KEPT for OpenCL
    src = (
        "#define K 2.0\n"
        "#define F(x) (K*(x))\n"
        "#define END return K;\n"
        "float g(float d){ END }\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(g(u.x)); }\n"
    )
    assert not _source_parses(src)
    out = maybe_preprocess_directives(src)
    assert _source_parses(out)
    assert "#define K 2.0" in out        # still referenced by F's body
    assert "#define END" not in out      # fully consumed
    assert "return 2.0 " in _norm(out).replace(" ;", " ;")


def test_unused_define_is_not_blanked():
    # an object-like macro that is never expanded is NOT consumed -> keep it
    res = strip_conditionals("#define PI 3.14159\nint x;\n")
    out = expand_object_macros(res.source, res.undefs, res.poisoned)
    assert "#define PI 3.14159" in out


def test_expansion_respects_source_order_redefinition():
    src = (
        "#define W 1\n"
        "int a = W;\n"
        "#define W 2\n"
        "int b = W;\n"
    )
    res = strip_conditionals(src)
    out = expand_object_macros(res.source, res.undefs, res.poisoned)
    n = _norm(out)
    assert "int a = 1 ;" in n
    assert "int b = 2 ;" in n
    assert "#define W" not in out        # both consumed everywhere


def test_expansion_respects_undef():
    src = (
        "#define P 5\n"
        "int a = P;\n"
        "#undef P\n"
        "float P;\n"
    )
    res = strip_conditionals(src)
    out = expand_object_macros(res.source, res.undefs, res.poisoned)
    n = _norm(out)
    assert "int a = 5 ;" in n
    assert "float P;" in out             # post-#undef P is a fresh identifier
    # the #undef line was blanked, so the pre-#undef define MUST go too
    assert "#define P 5" not in out


def test_poisoned_macro_from_kept_frame_not_expanded():
    src = (
        "#if 1/0\n"
        "#define Q 1\n"
        "#endif\n"
        "int a = Q;\n"
    )
    res = strip_conditionals(src)
    assert "Q" in res.poisoned
    out = expand_object_macros(res.source, res.undefs, res.poisoned)
    assert "int a = Q;" in out           # left for OpenCL to resolve
    assert "#define Q 1" in out


def test_combined_object_and_function_macro_rescue():
    # lt2SRt/MtXBDf shape: the parse needs BOTH the object-like expansion
    # (here) and the function-like expansion (macro_expander, next pipeline
    # stage). The cascade must keep the object-expanded state when the
    # PAIRING parses, even though neither stage parses alone.
    src = (
        "#define fBegin float d = 0.;\n"
        "#define fExp(v) d += dot(p, v);\n"
        "#define fEnd return d - r;\n"
        "float f(vec3 p, float r){ fBegin fExp(vec3(1)) fExp(vec3(-1)) fEnd }\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(f(vec3(u,1.), 1.)); }\n"
    )
    assert not _source_parses(src)
    mid = maybe_preprocess_directives(src)
    assert "float d = 0.;" in _norm(mid)      # object expansion kept
    assert "return d - r;" in _norm(mid)
    out = PreprocessorTransformer().transform(src)   # + function-like stage
    assert _source_parses(out)


# ---------------------------------------------------------------------------
# Gate + give-up behavior
# ---------------------------------------------------------------------------

def test_gate_parsing_source_returned_byte_identical():
    src = (
        "#ifdef FOO\n"
        "float extraFn(float x){ return x; }\n"
        "#endif\n"
        "void mainImage(out vec4 o, in vec2 g){ o = vec4(0); }\n"
    )
    assert _source_parses(src)           # precondition
    assert maybe_preprocess_directives(src) is src


def test_gate_no_directives_returned_unchanged():
    src = "this does not parse at all (\n"
    assert maybe_preprocess_directives(src) is src


def test_giveup_returns_stripped_source():
    # conditionals strip fine but the residue is hopeless: the strip must
    # still land (never return a worse state than stage 1)
    src = (
        "#ifdef NOPE\n"
        "dead junk !!!\n"
        "#endif\n"
        "this is (not glsl\n"
    )
    out = maybe_preprocess_directives(src)
    assert out == strip_conditionals(src).source
    assert "dead junk" not in out
    assert "this is (not glsl" in out
    assert "#ifdef" not in out


def test_unbalanced_directives_do_not_crash():
    src = "#endif\n#else\nint x;\n"
    res = strip_conditionals(src)        # unmatched: passed through verbatim
    assert "int x;" in res.source
    assert res.balanced is False


def test_unterminated_conditional_refused():
    # a frame still open at EOF would blank everything to EOF on a dead
    # verdict — the cascade must refuse and return the source unchanged
    src = "#ifdef SOME_FLAG"
    assert strip_conditionals(src).balanced is False
    assert maybe_preprocess_directives(src) is src


# ---------------------------------------------------------------------------
# Integration: wired into PreprocessorTransformer.transform
# ---------------------------------------------------------------------------

def test_transform_strips_straddling_conditional():
    src = (
        "void mainImage(out vec4 c, in vec2 u){\n"
        "    if (u.x < 1.) c = vec4(1);\n"
        "#ifdef SHOW_SEC\n"
        "    else if (u.x < 2.) c = vec4(2);\n"
        "#endif\n"
        "    else c = vec4(0);\n"
        "}\n"
    )
    out = PreprocessorTransformer().transform(src)
    assert _source_parses(out)
    assert "vec4(2)" not in out


def test_transform_expands_statement_macro():
    src = (
        "#define fEnd return d - r;\n"
        "float g(vec3 p, float r) { float d = p.x; fEnd }\n"
        "void mainImage(out vec4 c, in vec2 u){ c = vec4(g(vec3(u,1.), 1.)); }\n"
    )
    out = PreprocessorTransformer().transform(src)
    assert _source_parses(out)
    assert "return d - r" in _norm(out)
