"""
Array tests - Array declarations and access patterns.

Tests parsing of array declarations, initializations, and access patterns.
Target: 50 tests
"""

import pytest
from glsl_to_opencl.parser import GLSLParser


class TestArrayDeclarations:
    """Test array declaration syntax."""

    def test_parse_1d_array_int(self):
        """Test 1D int array."""
        source = "void main() { int arr[10]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_1d_array_float(self):
        """Test 1D float array."""
        source = "void main() { float arr[20]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_1d_array_vec3(self):
        """Test 1D vec3 array."""
        source = "void main() { vec3 arr[5]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_1d_array_vec4(self):
        """Test 1D vec4 array."""
        source = "void main() { vec4 arr[8]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_1d_array_mat4(self):
        """Test 1D mat4 array."""
        source = "void main() { mat4 arr[3]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_type_first_array_named_size_param(self):
        """Category P cluster 3d23WK: type-first array param with a named (non-digit) size.

        `in vec2[N] poly` must normalize to `in vec2 poly[N]` (mirrors the
        existing digit-size rewrite) so tree-sitter-glsl accepts it.
        """
        source = "float sdPoly(in vec2[N] poly, in vec2 p) { return 0.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_type_first_array_named_size_local(self):
        """`vec2[N] poly;` local declaration with a named size."""
        source = "void main() { vec2[N] poly; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_type_first_array_struct_named_size_param(self):
        """MdKBDy: user-struct type-first array param `ball[BALLCOUNT] balls`."""
        source = "struct ball { float r; };\nvec4 f(ball[BALLCOUNT] balls) { return vec4(0.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_type_first_array_expression_size(self):
        """Category P tdjfWc: type-first array with an *expression* size.

        `vec3[SZ*3] vertices;` must normalize to `vec3 vertices[SZ*3];`. The
        earlier `\\w*` size group stopped at the `*`, so the type-first form
        leaked through and tree-sitter-glsl rejected it.
        """
        source = "#define SZ 40\nvec3[SZ*3] vertices;\nvoid main() { }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_type_first_array_arithmetic_size_local(self):
        """`float[2*K+1] w;` — expression size in a local declaration."""
        source = "void main() { float[2*K+1] w; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_type_first_array_expr_does_not_touch_bracketed_comment(self):
        """A bracketed range in a comment must NOT fuse with the next line's
        code — the size rewrite is single-line only (regression guard for the
        comment false positives found across the passing corpus)."""
        from src.glsl_to_opencl.parser.glsl_parser import _normalize_array_syntax
        source = "// remap to [0,1]\n    vec3 v0 = vec3(0.0);\n"
        # v0's declaration must survive verbatim (not become ` = vec3(...)`).
        assert "vec3 v0 = vec3(0.0);" in _normalize_array_syntax(source)

    def test_parse_2d_array(self):
        """Test 2D array."""
        source = "void main() { float arr[10][20]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_3d_array(self):
        """Test 3D array."""
        source = "void main() { float arr[5][5][5]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_with_const_qualifier(self):
        """Test const array declaration."""
        source = "void main() { const float arr[3]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_global_array(self):
        """Test global array declaration."""
        source = "float arr[10]; void main() {}"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_uniform_array(self):
        """Test uniform array."""
        source = "uniform float arr[5]; void main() {}"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_of_vec2(self):
        """Test array of vec2."""
        source = "void main() { vec2 arr[10]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_of_ivec3(self):
        """Test array of ivec3."""
        source = "void main() { ivec3 arr[5]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_of_mat2(self):
        """Test array of mat2."""
        source = "void main() { mat2 arr[4]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_of_mat3(self):
        """Test array of mat3."""
        source = "void main() { mat3 arr[3]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_function_parameter_array(self):
        """Test array as function parameter."""
        source = "void func(float arr[10]) {} void main() {}"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_in_parameter_array(self):
        """Test array with in qualifier."""
        source = "void func(in float arr[10]) {} void main() {}"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None


class TestArrayAccess:
    """Test array access patterns."""

    def test_parse_simple_array_access(self):
        """Test simple array access."""
        source = "void main() { float x = arr[0]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_variable_index(self):
        """Test array access with variable index."""
        source = "void main() { float x = arr[i]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_expression_index(self):
        """Test array access with expression index."""
        source = "void main() { float x = arr[i + 1]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_2d_array_access(self):
        """Test 2D array access."""
        source = "void main() { float x = arr[i][j]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_3d_array_access(self):
        """Test 3D array access."""
        source = "void main() { float x = arr[i][j][k]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_with_swizzle(self):
        """Test array access with swizzle."""
        source = "void main() { float x = arr[i].x; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_vec_element(self):
        """Test array of vectors element access."""
        source = "void main() { vec2 v = arr[i].xy; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_write(self):
        """Test writing to array."""
        source = "void main() { arr[0] = 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_write_2d(self):
        """Test writing to 2D array."""
        source = "void main() { arr[i][j] = 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_compound_assignment(self):
        """Test compound assignment to array element."""
        source = "void main() { arr[i] += 1.0; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_increment(self):
        """Test incrementing array element."""
        source = "void main() { arr[i]++; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_in_expression(self):
        """Test array access in expression."""
        source = "void main() { float x = arr[0] + arr[1] * arr[2]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_in_function_call(self):
        """Test array element in function call."""
        source = "void main() { float x = sin(arr[i]); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_comparison(self):
        """Test array element comparison."""
        source = "void main() { bool b = arr[i] > arr[j]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_in_loop_condition(self):
        """Test array in loop condition."""
        source = "void main() { for (int i = 0; i < 10; i++) { arr[i] = float(i); } }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_matrix_access(self):
        """Test accessing matrix from array."""
        source = "void main() { mat4 m = matArr[i]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_matrix_column_access(self):
        """Test accessing matrix column from array."""
        source = "void main() { vec4 v = matArr[i][0]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_nested_array_access(self):
        """Test nested array access expression."""
        source = "void main() { float x = arr[arr2[i]]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_with_multiply(self):
        """Test array access with multiplication in index."""
        source = "void main() { float x = arr[i * 2]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_access_with_modulo(self):
        """Test array access with modulo in index."""
        source = "void main() { float x = arr[i % 10]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_in_ternary(self):
        """Test array in ternary expression."""
        source = "void main() { float x = condition ? arr[0] : arr[1]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_of_vec_swizzle_write(self):
        """Test writing to swizzled array element."""
        source = "void main() { arr[i].xy = vec2(1.0); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_initialization_in_loop(self):
        """Test array initialization in loop."""
        source = """
        void main() {
            float arr[10];
            for (int i = 0; i < 10; i++) {
                arr[i] = float(i);
            }
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_multidimensional_initialization(self):
        """Test multidimensional array initialization."""
        source = """
        void main() {
            float arr[2][2];
            arr[0][0] = 1.0;
            arr[0][1] = 2.0;
            arr[1][0] = 3.0;
            arr[1][1] = 4.0;
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_copy(self):
        """Test array element copy."""
        source = "void main() { arr1[i] = arr2[j]; }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_pass_to_function(self):
        """Test passing array to function."""
        source = """
        void process(float arr[10]) {}
        void main() {
            float data[10];
            process(data);
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_declaration_and_access(self):
        """Test declaring and accessing array."""
        source = """
        void main() {
            float arr[3];
            arr[0] = 1.0;
            arr[1] = 2.0;
            arr[2] = 3.0;
            float x = arr[1];
        }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_length(self):
        """Test array.length()."""
        source = "void main() { int len = arr.length(); }"
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None

    def test_parse_array_uniform_access(self):
        """Test accessing uniform array."""
        source = """
        uniform float data[100];
        void main() { float x = data[gl_VertexID]; }
        """
        parser = GLSLParser()
        ast = parser.parse(source)
        assert ast is not None
