"""
Unit tests for category B: dereference out/inout-parameter READS.

GLSL out/inout params become OpenCL pointers (out vec3 p -> __private float3* p).
Previously only the assignment-target and call-site cases were handled; a pointer
param used as an rvalue was emitted as the bare pointer `p` instead of `*p`,
causing "no matching function for call to 'GLSL_*'" and "member reference base
type 'float3 *' is not a structure".

Rule: a pointer-param identifier always dereferences to *p, EXCEPT when passed as
an argument to another function's out/inout (pointer) parameter, where it passes
through as the bare pointer p.

See tests/fixcampaign/DESIGN_B_pointer_param_read.md.
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


@pytest.fixture
def emitter():
    return OpenCLEmitter()


def t(glsl_code, parser, transformer, emitter):
    ast = parser.parse(glsl_code)
    return emitter.emit(transformer.transform(ast))


def test_scalar_read(parser, transformer, emitter):
    out = t("void f(inout float p) { float x = p + 1.0; }", parser, transformer, emitter)
    assert 'float x = *p + 1.0f' in out


def test_vector_read_into_builtin(parser, transformer, emitter):
    out = t("void f(inout float2 p) { float d = dot(p, p); }", parser, transformer, emitter)
    assert 'GLSL_dot(*p, *p)' in out


def test_member_read(parser, transformer, emitter):
    out = t("float f(inout float3 p) { return p.x; }", parser, transformer, emitter)
    assert 'return (*p).x' in out


def test_swizzle_read(parser, transformer, emitter):
    out = t("float2 f(inout float4 p) { return p.xy; }", parser, transformer, emitter)
    assert '(*p).xy' in out


def test_member_assign(parser, transformer, emitter):
    out = t("void f(out float3 p) { p.x = 1.0; }", parser, transformer, emitter)
    assert '(*p).x = 1.0f' in out


def test_scalar_assign_still_works(parser, transformer, emitter):
    """Regression guard: assignment target deref must still produce *p = ..."""
    out = t("void f(out float p) { p = 1.0; }", parser, transformer, emitter)
    assert '*p = 1.0f' in out
    assert '**p' not in out  # no double-deref


def test_compound_assign(parser, transformer, emitter):
    out = t("void f(inout float p) { p += 2.0; }", parser, transformer, emitter)
    assert '*p += 2.0f' in out or '*p = *p + 2.0f' in out


def test_value_arg_passthrough(parser, transformer, emitter):
    out = t("float f(inout float p) { return sin(p); }", parser, transformer, emitter)
    assert 'GLSL_sin(*p)' in out


def test_pointer_to_pointer_passthrough(parser, transformer, emitter):
    """p (a pointer param) passed to g's out param -> g(p), not g(*p) or g(&p)."""
    glsl = """
    void g(out float x) { x = 1.0; }
    void f(out float p) { g(p); }
    """
    out = t(glsl, parser, transformer, emitter)
    assert 'g(p)' in out
    assert 'g(*p)' not in out
    assert 'g(&p)' not in out


def test_local_to_pointer_address_of_still_works(parser, transformer, emitter):
    """A local passed to an out param still gets &."""
    glsl = """
    void g(out float x) { x = 1.0; }
    void f() { float y; g(y); }
    """
    out = t(glsl, parser, transformer, emitter)
    assert 'g(&y)' in out


def test_non_pointer_param_untouched(parser, transformer, emitter):
    out = t("void f(in float q) { float x = q + 1.0; }", parser, transformer, emitter)
    assert 'q + 1.0f' in out
    assert '*q' not in out


def test_fragcolor_is_pseudo_local_not_auto_dereferenced(parser, transformer, emitter):
    """mainImage's fragColor is a host-rewritten pseudo-local (becomes a plain
    local in the @KERNEL): its reads must stay bare and passing it to a user
    out-param fn must take its address (&), not auto-deref / pass through.
    (Regression guard for lltXRn / lsVBDh.)"""
    glsl = """
    void shade(out vec4 c, vec2 u) { c = vec4(u, 0.0, 1.0); }
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        shade(fragColor, fragCoord);
        float x = fragColor.r;
    }
    """
    out = t(glsl, parser, transformer, emitter)
    assert 'shade(&fragColor' in out      # address-of, NOT passthrough
    assert '(*fragColor)' not in out      # read not auto-deref'd
    assert 'fragColor.r' in out
