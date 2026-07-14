"""
Unit tests for multi-argument vector constructors whose arguments supply MORE
scalar components than the target vector holds (Bug-fix campaign category AF).

GLSL vector constructors TRUNCATE excess components: `vec3(v2, v4)` takes the
first 3 of the 2+4 available (both of `v2`, then 1 of `v4`) and silently drops
the rest. OpenCL's `(float3)(v2, v4)` literal syntax instead flattens ALL 6
components and clang rejects it:
    error: too many elements in vector initialization (expected 3, have 6)

The fix (in `_transform_call_expression`, multi-arg vector ctor path) budgets
the target's component count across the args and swizzles the boundary-crossing
argument down to just the components still needed (`v4` -> `v4.x`), dropping any
fully-excess trailing args. Only fires when the summed width EXCEEDS the target
(never pads an under-filled ctor, never touches an exactly-filled one). Args
with un-inferrable width are left alone (no guess).
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


def _fn(body):
    return ("vec4 f(vec2 uv, vec3 p3, vec4 p4, float s) {\n"
            + body + "\n}\n")


# ---- overflow: trailing arg swizzled to the components still needed ---------

def test_vec3_from_vec2_and_vec4(parser, transformer, emitter):
    """The MlycWR shape: vec3(v2, v4) keeps both of v2 + 1 of v4 -> v4.x."""
    out = t(_fn("vec3 v = vec3(uv, p4); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    assert '(float3)(uv, p4.x)' in out
    assert '(float3)(uv, p4)' not in out


def test_vec3_from_vec2_and_texture_expr(parser, transformer, emitter):
    """The exact MlycWR line: the vec4 is a binary-op expression, so the
    truncating swizzle must parenthesize it: (texture(...) * 0.1f).x."""
    glsl = ("vec4 f(sampler2D iChannel0, vec2 v) {\n"
            "  float3 p = vec3(v, texture(iChannel0, v) * 0.1);\n"
            "  return vec4(p, 1.0);\n}\n")
    out = t(glsl, parser, transformer, emitter)
    assert '(float3)(v, (texture(iChannel0, v) * 0.1f).x)' in out
    assert 'have 6' not in out  # sanity: no untruncated 6-component list


def test_vec4_from_vec3_and_vec2(parser, transformer, emitter):
    """vec4(v3, v2): 3+2=5 -> keep v3, take 1 of v2 -> v2.x."""
    out = t(_fn("vec4 v = vec4(p3, uv); return v;"),
            parser, transformer, emitter)
    assert '(float4)(p3, uv.x)' in out


def test_vec3_from_scalar_and_vec4(parser, transformer, emitter):
    """vec3(s, v4): 1+4=5 -> keep s, take 2 of v4 -> v4.xy."""
    out = t(_fn("vec3 v = vec3(s, p4); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    assert '(float3)(s, p4.xy)' in out


def test_vec2_from_two_vec2s_drops_trailing(parser, transformer, emitter):
    """vec2(v2, v2): the first vec2 fills the target; the second is dropped
    entirely (remaining budget 0)."""
    out = t(_fn("vec2 v = vec2(uv, uv); return vec4(v, 0.0, 1.0);"),
            parser, transformer, emitter)
    assert '(float2)(uv)' in out
    assert '(float2)(uv, uv)' not in out


# ---- exactly-filled / under-filled: MUST be left untouched -----------------

def test_vec4_from_vec3_and_scalar_unchanged(parser, transformer, emitter):
    """3+1=4 exactly — legal OpenCL, no truncation."""
    out = t(_fn("vec4 v = vec4(p3, 1.0); return v;"),
            parser, transformer, emitter)
    assert '(float4)(p3, 1.0f)' in out


def test_vec4_from_two_vec2s_unchanged(parser, transformer, emitter):
    """2+2=4 exactly — legal, no truncation."""
    out = t(_fn("vec4 v = vec4(uv, uv); return v;"),
            parser, transformer, emitter)
    assert '(float4)(uv, uv)' in out


def test_vec3_component_list_unchanged(parser, transformer, emitter):
    """Plain scalar component list, exactly filled."""
    out = t(_fn("vec3 v = vec3(1.0, 2.0, 3.0); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    assert '(float3)(1.0f, 2.0f, 3.0f)' in out


def test_overloaded_user_fn_args_unchanged(parser, transformer, emitter):
    """Regression guard (4tdcD4): an OVERLOADED user function only registers
    one return type, so `logc(a.xy)` (truly vec2) mis-infers as vec4. Trusting
    that width would falsely truncate the legal, exactly-filled
    `vec4(logc(a.xy), logc(a.zw))`. Any arg whose type traces to a user fn must
    disable truncation entirely."""
    glsl = ("vec2 logc(vec2 a){ return a; }\n"
            "vec4 logc(vec4 a){ return vec4(logc(a.xy), logc(a.zw)); }\n")
    out = t(glsl, parser, transformer, emitter)
    assert '(float4)(logc(a.xy), logc(a.zw))' in out
    assert '(float4)(logc(a.xy))' not in out


def test_user_fn_in_binop_arg_unchanged(parser, transformer, emitter):
    """The user-fn taint propagates through a binary op: an overloaded call
    scaled by a scalar (`logc(a.xy) * 0.5`) is still an untrustworthy width."""
    glsl = ("vec2 logc(vec2 a){ return a; }\n"
            "vec4 logc(vec4 a){ return vec4(logc(a.xy) * 0.5, logc(a.zw)); }\n")
    out = t(glsl, parser, transformer, emitter)
    assert '(float4)(logc(a.xy) * 0.5f, logc(a.zw))' in out


def test_unknown_arg_width_unchanged(parser, transformer, emitter):
    """A macro-valued arg has un-inferrable width — do not guess/truncate."""
    out = t(_fn("vec3 v = vec3(uv, SOME_MACRO); return vec4(v, 1.0);"),
            parser, transformer, emitter)
    # unchanged: still the raw 2-arg cast (may or may not compile, but we
    # never silently reshape what we cannot measure)
    assert '(float3)(uv, SOME_MACRO)' in out
