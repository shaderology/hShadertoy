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


# ============================================================================
# 4. A3 — uninitialized matrix globals (synthesized zero-init is a CALL)
# ============================================================================
# An uninitialized global's zero value is synthesized to match GLSL semantics.
# For scalars/vectors that is a literal (compile-time constant → stays), but for
# a matrix it is `GLSL_matrixNxN_diagonal(0.0f)` — a function CALL, which is
# NOT a compile-time constant at program scope. It must be hoisted like an
# explicit non-const initializer, not emitted as a file-scope initializer.

def test_uninitialized_matrix_global_hoisted():
    """mat3 obj;  -> bare `matrix3x3 obj;` + hoisted diagonal-zero assignment."""
    header, kernel = tp("mat3 obj;\n" + MAIN)
    assert 'matrix3x3 obj;' in header
    # No diagonal-constructor CALL at file scope.
    assert 'obj = GLSL_matrix3x3_diagonal' not in header
    assert 'GLSL_matrix3x3_diagonal(0.0f)' not in header
    # The zero-init is hoisted into the kernel body.
    assert 'obj = GLSL_matrix3x3_diagonal(0.0f);' in kernel


def test_uninitialized_matrix_declaration_list_hoisted():
    """mat2 ma, mb;  -> both matrix globals bare + both zero-inits hoisted."""
    header, kernel = tp("mat2 ma, mb;\n" + MAIN)
    assert 'matrix2x2 ma' in header
    assert 'GLSL_matrix2x2_diagonal(0.0f)' not in header
    assert 'ma = GLSL_matrix2x2_diagonal(0.0f);' in kernel
    assert 'mb = GLSL_matrix2x2_diagonal(0.0f);' in kernel


def test_uninitialized_scalar_global_not_hoisted():
    """float x;  -> a literal zero-init (compile-time constant); leave in place."""
    header, kernel = tp("float x;\n" + MAIN)
    # Scalar zero-init is a literal, stays at file scope (no hoist).
    assert 'x =' not in kernel


# ============================================================================
# 5. A1 — non-constant scalar int/uint globals
# ============================================================================
# The int/uint hoist skip must be selective: an integer constant expression
# OpenCL can fold (a possible array size) stays in place, but an int global
# whose initializer OpenCL cannot fold (cast of a runtime value, vector member
# access, dependence on a hoisted global) must be hoisted.

def test_nonconst_int_cast_of_hoisted_float_hoisted():
    """const int n = int(k); where k is a hoisted non-const -> hoist n."""
    glsl = (
        "const float g = 8.0, k = 24.0 / g;\n"
        "const int n = int(k);\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    assert 'int n;' in header
    assert '(int)(k)' not in header  # not a file-scope initializer
    assert 'n = (int)(k);' in kernel


def test_int_vector_member_product_hoisted():
    """int tot = N.x * N.y; -> vector member access, OpenCL can't fold -> hoist."""
    glsl = (
        "const ivec2 N = ivec2(200, 100);\n"
        "int tot_n = N.x * N.y;\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    assert 'int tot_n;' in header
    assert 'tot_n = N.x * N.y;' in kernel


def test_foldable_int_const_not_hoisted():
    """const int M = N * 2; with N a const-int literal -> OpenCL folds; keep.

    This is the regression guard: a foldable integer constant expression is a
    legal array size and must stay a program-scope constant.
    """
    glsl = (
        "const int N = 5;\n"
        "const int M = N * 2;\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    assert 'const int M = N * 2;' in header
    assert 'M =' not in kernel


def test_literal_int_arithmetic_not_hoisted():
    """const int K = 3 + 4; -> pure literal arithmetic, OpenCL folds; keep."""
    header, kernel = tp("const int K = 3 + 4;\n" + MAIN)
    assert 'const int K = 3 + 4;' in header
    assert 'K =' not in kernel


# ============================================================================
# 6. A2 — array/aggregate globals with non-constant elements (Plan B:
#    temp-local-array + copy-loop hoisting)
# ============================================================================
# OpenCL rejects any arithmetic/call in a file-scope initializer, but a
# whole-array assignment (`arr = {...};`) is illegal in C, so the scalar hoist
# cannot be reused. Instead: emit the array bare and sized, and hoist a
# temp-local-array declaration (a NON-constant aggregate initializer is legal
# on a LOCAL) followed by a copy loop into the bare global.

STRUCT = "struct Mat { vec3 c; float a; float b; };\n"


def test_struct_array_arithmetic_elements_hoisted():
    """Mat mats[2] = Mat[2](Mat(vec3(1)*.9,...), ...) -> bare + temp/copy."""
    glsl = (
        STRUCT
        + "Mat mats[2] = Mat[2](Mat(vec3(1)*.9,.5,.23), Mat(vec3(1)*.1,.2,.3));\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    # Global emitted bare and sized, no aggregate initializer at file scope.
    assert 'Mat mats[2];' in header
    assert '{{' not in header  # the brace initializer moved out of file scope
    # Hoisted temp-local carries the real (non-constant) initializer.
    assert 'Mat __init_mats[2] = {{' in kernel
    # Copy loop writes every element into the bare global.
    assert 'for (int __hi = 0; __hi < 2; ++__hi)' in kernel
    assert 'mats[__hi] = __init_mats[__hi];' in kernel


def test_unsized_array_gets_sized_and_hoisted():
    """const vec3 positions[] = vec3[](...) -> sized bare decl, const dropped."""
    glsl = (
        "const vec3 positions[] = "
        "vec3[](vec3(1.0,2.0,3.0), vec3(4.0)*2.0, vec3(0.5));\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    # Unsized [] rewritten to the element count; const dropped (hoisted global).
    assert 'float3 positions[3];' in header
    assert 'const float3 positions' not in header
    # Temp-local sized to the same count + copy loop over 3.
    assert 'float3 __init_positions[3] = {' in kernel
    assert 'for (int __hi = 0; __hi < 3; ++__hi)' in kernel
    assert 'positions[__hi] = __init_positions[__hi];' in kernel


def test_all_constant_array_left_untouched():
    """const vec3 c[2] = vec3[](pure literals) already compiles; leave in place."""
    glsl = (
        "const vec3 c[2] = vec3[](vec3(0.9,0.9,0.9), vec3(0.1,0.2,0.3));\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    # Constant aggregate initializer stays at file scope, byte-identical.
    assert 'const float3 c[2] = {(float3)(0.9f, 0.9f, 0.9f), ' \
           '(float3)(0.1f, 0.2f, 0.3f)};' in header
    # No hoist artifacts.
    assert '__init_c' not in kernel
    assert 'c[__hi]' not in kernel


def test_constant_scalar_array_zero_init_not_hoisted():
    """float pts[2]; -> `{0.0f}` zero-init is constant; leave the wrap in place."""
    header, kernel = tp("float pts[2];\n" + MAIN)
    assert 'float pts[2] = {0.0f};' in header
    assert '__init_pts' not in kernel


def test_uninitialized_matrix_array_hoisted():
    """mat3 grid[3]; -> bare `matrix3x3 grid[3];` + temp/copy of diagonal-zero.

    The synthesized zero-init wraps a CALL (GLSL_matrix3x3_diagonal), which is
    not a compile-time constant at file scope. The single-element wrap plus C's
    tail zero-fill make the temp-local correct for the whole array.
    """
    header, kernel = tp("mat3 grid[3];\n" + MAIN)
    assert 'matrix3x3 grid[3];' in header
    assert 'GLSL_matrix3x3_diagonal' not in header
    assert 'matrix3x3 __init_grid[3] = {GLSL_matrix3x3_diagonal(0.0f)};' in kernel
    assert 'for (int __hi = 0; __hi < 3; ++__hi)' in kernel
    assert 'grid[__hi] = __init_grid[__hi];' in kernel


def test_symbolic_size_matrix_array_hoisted():
    """mat3 grid[N]; with const int N -> bare `grid[N]` + loop bound N (in scope)."""
    glsl = "const int N = 3;\nmat3 grid[N];\n" + MAIN
    header, kernel = tp(glsl)
    assert 'const int N = 3;' in header
    assert 'matrix3x3 grid[N];' in header
    assert 'matrix3x3 __init_grid[N] = {GLSL_matrix3x3_diagonal(0.0f)};' in kernel
    assert 'for (int __hi = 0; __hi < N; ++__hi)' in kernel
    assert 'grid[__hi] = __init_grid[__hi];' in kernel


def test_multi_declarator_array_and_scalar_matrix_hoisted():
    """mat3 flyerMat[3], flMat; -> both bare; array temp/copy + scalar hoist."""
    header, kernel = tp("mat3 flyerMat[3], flMat;\n" + MAIN)
    # Both declarators emitted bare on one line, siblings undisturbed.
    assert 'matrix3x3 flyerMat[3], flMat;' in header
    assert 'GLSL_matrix3x3_diagonal' not in header
    # Array declarator -> temp/copy; scalar declarator -> plain assignment.
    assert 'matrix3x3 __init_flyerMat[3] = ' \
           '{GLSL_matrix3x3_diagonal(0.0f)};' in kernel
    assert 'flyerMat[__hi] = __init_flyerMat[__hi];' in kernel
    assert 'flMat = GLSL_matrix3x3_diagonal(0.0f);' in kernel


def test_array_hoist_preserves_order_with_scalars():
    """A hoisted scalar declared before an array keeps its earlier position."""
    glsl = (
        "vec3 a = normalize(vec3(1.0));\n"
        "vec3 pal[2] = vec3[](vec3(0,0,0)/255.0, a);\n"
        + MAIN
    )
    header, kernel = tp(glsl)
    assert 'float3 a;' in header
    assert 'float3 pal[2];' in header
    # Scalar 'a' initializes before the array temp/copy that reads it.
    assert kernel.index('a = GLSL_normalize(') < kernel.index('__init_pal')
