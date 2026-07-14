"""
Unit tests for multi-declarator for-loop initializers (Category AB, Session 44).

The Shadertoy code-golf idiom packs a comma-separated (multi-declarator)
declaration into the for-loop init clause:

    for ( vec2 R = iResolution.xy, U = abs((u+u-R)/R.y);  cond;  incr )

Such an init transforms to an IR.DeclarationList. The PRODUCTION emitter
(codegen/opencl_emitter.py) only special-cased single IR.Declaration inits;
a DeclarationList fell through to the statement path, which appends its own
indent + trailing ';' + newline. The result was a malformed header:

    for (    float2 R = ..., U = ...;
    ; cond; incr) ...

i.e. a spurious second ';' (and a mid-header newline) → compile error
"expected ')'". These tests exercise the production emitter directly.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.codegen.opencl_emitter import OpenCLEmitter


@pytest.fixture
def parser():
    return GLSLParser()


@pytest.fixture
def transformer():
    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)
    return ASTTransformer(type_checker)


def transpile_body(glsl_code, parser, transformer):
    """Parse/transform/emit a snippet through the PRODUCTION emitter."""
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    emitter = OpenCLEmitter(indent_size=4)
    return emitter.emit(transformed)


def _for_header(opencl):
    """Return the text of the first for(...) header, parens balanced."""
    idx = opencl.index('for')
    start = opencl.index('(', idx)
    depth = 0
    for i in range(start, len(opencl)):
        if opencl[i] == '(':
            depth += 1
        elif opencl[i] == ')':
            depth -= 1
            if depth == 0:
                return opencl[start:i + 1]
    raise AssertionError("unbalanced for header")


def test_multi_declarator_for_init_no_spurious_semicolon(parser, transformer):
    glsl = """
    void mainImage(out vec4 O, in vec2 u) {
        for (vec2 R = u, U = R; U.x < 9.0; U += 0.1) O = vec4(U, 0.0, 1.0);
    }
    """
    out = transpile_body(glsl, parser, transformer)
    header = _for_header(out)
    # Exactly two ';' inside the for header (init;cond;update) — not three.
    assert header.count(';') == 2, f"malformed for header: {header!r}"
    # Both declarators must survive, comma-separated, in the init clause.
    assert 'R = u' in header
    assert 'U = R' in header
    # No mid-header newline from a statement-style emission.
    assert '\n' not in header, f"header should be single-line: {header!r}"


def test_multi_declarator_for_init_three_vars(parser, transformer):
    glsl = """
    void mainImage(out vec4 O, in vec2 u) {
        for (float t = 0.0, i = 0.0, n = 0.0; n < 12.0; n += 1.0) O.a += t + i;
    }
    """
    out = transpile_body(glsl, parser, transformer)
    header = _for_header(out)
    assert header.count(';') == 2, f"malformed for header: {header!r}"
    assert 't = 0.0' in header
    assert 'i = 0.0' in header
    assert 'n = 0.0' in header


def test_single_declarator_for_init_unchanged(parser, transformer):
    """Single-declarator init must keep emitting exactly as before."""
    glsl = """
    void mainImage(out vec4 O, in vec2 u) {
        for (int i = 0; i < 4; i++) O = vec4(0.0);
    }
    """
    out = transpile_body(glsl, parser, transformer)
    header = _for_header(out)
    assert header.count(';') == 2, f"malformed for header: {header!r}"
    assert 'int i = 0' in header
