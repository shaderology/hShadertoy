"""
Unit tests for category T — combined parameter qualifiers `const in` /
`const out` / `const inout` (Session 16).

tree-sitter-glsl accepts a single parameter qualifier (`const`, `in`, `out`,
`inout`) but rejects the *combination* `const in` (and `const out` /
`const inout`), raising a ParseError before the transformer ever runs. GLSL
allows these combinations, so the parser normalizes them (like the existing
array-syntax rewrites) by collapsing the redundant token:

    const in  T x  -> const T x   (value param, read-only kept)
    const out T x  -> out   T x   (pointer semantics dominate)
    const inout T x -> inout T x   (pointer semantics dominate)

Guard: `const int` / `const invariant` / a plain `const` must be untouched,
and bare `in`/`out`/`inout` params keep working.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer.ast_transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


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
    return CodeEmitter()


def transform_and_emit(glsl_code, parser, transformer, emitter):
    ast = parser.parse(glsl_code)
    transformed = transformer.transform(ast)
    return emitter.emit(transformed)


# ---------------------------------------------------------------------------
# const in — the dominant shape (value param, read-only)
# ---------------------------------------------------------------------------

def test_const_in_scalar_parses_and_drops_in(parser, transformer, emitter):
    glsl = """
    float hash(const in float n) {
        return fract(sin(n));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float n' in opencl
    # `in` must not survive as a qualifier token
    assert ' in ' not in opencl
    assert 'in float' not in opencl


def test_const_in_vector_parses(parser, transformer, emitter):
    glsl = """
    float sdEllipsoid(const in vec3 p, const in vec3 r) {
        return length(p / r) - 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float3 p' in opencl
    assert 'float3 r' in opencl
    assert ' in ' not in opencl


def test_const_in_double_space(parser, transformer, emitter):
    # lljBzz shape: `const in  vec3 p` (two spaces after in)
    glsl = """
    float sdEllipsoid(const in  vec3 p, const in vec3 r) {
        return length(p / r) - 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float3 p' in opencl
    assert ' in ' not in opencl


def test_const_in_mixed_with_bare_out(parser, transformer, emitter):
    # XlycWh shape: const in value params + a bare out pointer param
    glsl = """
    bool refractRay(const in vec3 v, const in vec3 n,
                    const in float nit, out vec3 refracted) {
        refracted = v + n * nit;
        return true;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float3 v' in opencl
    assert 'float3 n' in opencl
    assert 'float nit' in opencl
    # the out param still drives the pointer machinery
    assert 'float3* refracted' in opencl or \
           'float3 *refracted' in opencl


def test_const_in_prototype(parser, transformer, emitter):
    # XdVSRc shape: a forward declaration with const in params
    glsl = """
    float PrintValue(const in vec2 fragCoord, const in float fValue);
    void mainImage(out vec4 c, in vec2 fragCoord) {
        c = vec4(PrintValue(fragCoord, 1.0));
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float2 fragCoord' in opencl
    assert ' in ' not in opencl


def test_const_in_sampler(parser, transformer, emitter):
    # 4sK3W3 shape: const in sampler2D (Session-9 sampler mapping still applies)
    glsl = """
    vec4 textureLimited(const in sampler2D tex, const in vec2 uv) {
        return texture(tex, uv);
    }
    """
    # must at least parse and drop the leaked `in`
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert ' in ' not in opencl
    assert 'float2 uv' in opencl


# ---------------------------------------------------------------------------
# const out / const inout — pointer semantics dominate
# ---------------------------------------------------------------------------

def test_const_out_becomes_pointer(parser, transformer, emitter):
    glsl = """
    void store(const out float x) {
        x = 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float* x' in opencl or 'float *x' in opencl


def test_const_inout_becomes_pointer(parser, transformer, emitter):
    glsl = """
    void bump(const inout float x) {
        x = x + 1.0;
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'float* x' in opencl or 'float *x' in opencl


# ---------------------------------------------------------------------------
# Guards — the rewrite must not corrupt neighbouring syntax
# ---------------------------------------------------------------------------

def test_const_int_not_corrupted(parser, transformer, emitter):
    # `const int` must NOT be seen as `const in` + stray `t`
    glsl = """
    float pick(const int idx) {
        return float(idx);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'int idx' in opencl


def test_const_in_body_variable_untouched(parser, transformer, emitter):
    # A `const in`-looking token inside an expression context must not break;
    # here `in` is not a keyword so there is nothing to rewrite, but a global
    # `const int` declaration must survive.
    glsl = """
    const int N = 5;
    float f(const in float x) {
        return x + float(N);
    }
    """
    opencl = transform_and_emit(glsl, parser, transformer, emitter)
    assert 'int N' in opencl
    assert 'float x' in opencl
