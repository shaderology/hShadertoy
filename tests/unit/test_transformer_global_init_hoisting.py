"""
Unit tests for category A — program-scope global initializer hoisting.

OpenCL rejects ANY arithmetic or function call in a program-scope (file-scope)
variable initializer with "initializer element is not a compile-time constant",
regardless of address space (verified on the NVIDIA CUDA target: even
`__constant float3 f = (float3)(...) * 0.2f;` fails). Only bare literals and
pure literal vector/aggregate constructors are accepted.

Fix (Session 8): a program-scope global whose initializer is NOT a compile-time
constant is emitted as a bare (mutable, un-initialized) declaration, and its real
initializer is hoisted into a per-invocation assignment that the host
(tests/transpile.py / Houdini @KERNEL) prepends to the top of the kernel body.
This mirrors how main_header.cl already handles runtime uniforms
(`static float iTime = 0.0f;` declared at program scope, assigned in @KERNEL).

Compile-time-constant globals (literals, pure vector literals) are left
untouched — they already compile and may be used as array sizes.
"""

import sys
from pathlib import Path

import pytest

# tests/transpile.py is the campaign transpiler (header/kernel split path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import transpile  # noqa: E402


def tp(glsl):
    """Transpile and return (header, kernel_body)."""
    result = transpile(glsl)
    return result.get_header(), result.get_kernel()


MAIN = (
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
    "    fragColor = vec4(0.0);\n"
    "}\n"
)


# ============================================================================
# 1. Compile-time-constant globals are NOT hoisted (no regression)
# ============================================================================

def test_const_vec_literal_global_not_hoisted():
    """A pure vector-literal const global already compiles; leave it in place."""
    header, kernel = tp("const vec3 c = vec3(1.0);\n" + MAIN)
    assert 'const float3 c = (float3)(1.0f);' in header
    # No hoisted assignment in the kernel body.
    assert 'c =' not in kernel


def test_scalar_literal_global_not_hoisted():
    header, kernel = tp("const float PI = 3.14159;\n" + MAIN)
    assert 'const float PI = 3.14159f;' in header
    assert 'PI =' not in kernel


def test_int_literal_const_global_not_hoisted():
    """Integer const globals (array-size candidates) stay constant."""
    header, kernel = tp("const int N = 5;\n" + MAIN)
    assert 'const int N = 5;' in header
    assert 'N =' not in kernel


# ============================================================================
# 2. Non-constant initializers ARE hoisted
# ============================================================================

def test_call_initializer_global_hoisted():
    """vec3 L = normalize(...) -> bare global + hoisted kernel assignment."""
    header, kernel = tp("vec3 L = normalize(vec3(1.0, 0.9, 0.3));\n" + MAIN)
    # Global is emitted bare (no call initializer at program scope).
    assert 'float3 L;' in header
    assert 'GLSL_normalize' not in header
    # The real initializer is hoisted into the kernel body.
    assert 'L = GLSL_normalize(' in kernel


def test_arithmetic_initializer_global_hoisted():
    """vec3 c = vec3(...) * 0.2 -> hoisted (arithmetic is non-constant)."""
    header, kernel = tp("vec3 fogColor = vec3(0.5, 0.5, 0.3) * 0.2;\n" + MAIN)
    assert 'float3 fogColor;' in header
    assert '0.2f' not in header  # the multiply moved out of the global decl
    assert 'fogColor =' in kernel
    assert '* 0.2f' in kernel


def test_const_dropped_when_hoisted():
    """A hoisted global must drop `const` so the kernel can assign to it."""
    header, kernel = tp("const vec3 LightDir = normalize(vec3(0.0, 1.0, 0.0));\n" + MAIN)
    assert 'float3 LightDir;' in header
    assert 'const float3 LightDir' not in header  # const removed
    assert 'LightDir = GLSL_normalize(' in kernel


def test_uniform_initializer_global_hoisted():
    """float t = iTime; -> hoisted (depends on a runtime uniform)."""
    header, kernel = tp("float t = iTime;\n" + MAIN)
    assert 'float t;' in header
    assert 't = iTime;' in kernel


def test_matrix_call_global_hoisted():
    """mat2 m = mat2(...) -> bare struct global + hoisted assignment.

    The matrix placeholder must NOT be a diagonal-constructor call (that is
    itself non-constant at program scope); a bare declaration is used.
    """
    header, kernel = tp("const mat2 m = mat2(0.8, -0.6, 0.6, 0.8);\n" + MAIN)
    assert 'matrix2x2 m;' in header
    assert 'diagonal' not in header  # no GLSL_matrix2x2_diagonal call in the global
    assert 'm = GLSL_mat2(' in kernel


# ============================================================================
# 3. Locals are not hoisted; ordering is preserved
# ============================================================================

def test_local_nonconst_not_hoisted():
    """Non-const locals inside mainImage stay inline (not hoisted)."""
    glsl = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    float x = sin(iTime);\n"
        "    fragColor = vec4(x);\n"
        "}\n"
    )
    header, kernel = tp(glsl)
    # Declared inline in the body, exactly once, with its initializer intact.
    assert 'float x = GLSL_sin(iTime);' in kernel
    # Not turned into a bare global.
    assert 'float x;' not in header


def test_hoist_order_preserved():
    """Multiple hoisted globals keep declaration order in the kernel body."""
    glsl = (
        "vec3 a = normalize(vec3(1.0));\n"
        "vec3 b = a * 2.0;\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    assert 'float3 a;' in header
    assert 'float3 b;' in header
    assert kernel.index('a = GLSL_normalize(') < kernel.index('b =')
