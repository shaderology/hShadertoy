"""
Unit tests for struct transformation.

Tests struct definition and struct constructor transformations.
"""

import pytest
from src.glsl_to_opencl.parser import GLSLParser
from src.glsl_to_opencl.analyzer import (
    TypeChecker,
    create_builtin_symbol_table,
)
from src.glsl_to_opencl.transformer import ASTTransformer
from src.glsl_to_opencl.transformer.code_emitter import CodeEmitter


def transform_and_emit(glsl_code: str) -> str:
    """Helper to parse, transform, and emit OpenCL code."""
    parser = GLSLParser()
    ast = parser.parse(glsl_code)

    symbol_table = create_builtin_symbol_table()
    type_checker = TypeChecker(symbol_table)

    transformer = ASTTransformer(type_checker)
    transformed = transformer.transform(ast)

    emitter = CodeEmitter()
    return emitter.emit(transformed)


class TestStructDefinition:
    """Test struct definition transformation."""

    def test_simple_struct(self):
        """Test basic struct with single fields."""
        glsl = """
struct Point {
    float x;
    float y;
    float z;
};
"""
        opencl = transform_and_emit(glsl)

        assert "typedef struct {" in opencl
        assert "float x;" in opencl
        assert "float y;" in opencl
        assert "float z;" in opencl
        assert "} Point;" in opencl

    def test_struct_with_vectors(self):
        """Test struct with vector types."""
        glsl = """
struct Geo {
    vec3 pos;
    vec3 scale;
    vec3 rotation;
};
"""
        opencl = transform_and_emit(glsl)

        assert "typedef struct {" in opencl
        assert "float3 pos;" in opencl
        assert "float3 scale;" in opencl
        assert "float3 rotation;" in opencl
        assert "} Geo;" in opencl

    def test_struct_comma_separated_fields(self):
        """Test struct with comma-separated field names."""
        glsl = """
struct Ray {
    vec3 o, d;
};
"""
        opencl = transform_and_emit(glsl)

        assert "typedef struct {" in opencl
        assert "float3 o, d;" in opencl
        assert "} Ray;" in opencl

    def test_struct_mixed_fields(self):
        """Test struct with mixed field types."""
        glsl = """
struct Hit {
    vec3 p;
    float t, d;
};
"""
        opencl = transform_and_emit(glsl)

        assert "typedef struct {" in opencl
        assert "float3 p;" in opencl
        assert "float t, d;" in opencl
        assert "} Hit;" in opencl

    def test_struct_one_line(self):
        """Test struct defined on one line."""
        glsl = """
struct Camera { vec3 p, t; };
"""
        opencl = transform_and_emit(glsl)

        assert "typedef struct {" in opencl
        assert "float3 p, t;" in opencl
        assert "} Camera;" in opencl

    def test_multiple_structs(self):
        """Test multiple struct definitions."""
        glsl = """
struct Ray { vec3 o, d; };
struct Camera { vec3 p, t; };
struct Hit { vec3 p; float t, d; };
"""
        opencl = transform_and_emit(glsl)

        # All three structs should be defined
        assert "} Ray;" in opencl
        assert "} Camera;" in opencl
        assert "} Hit;" in opencl


class TestStructConstructor:
    """Test struct constructor transformation."""

    def test_struct_constructor_simple(self):
        """Test simple struct constructor."""
        glsl = """
struct Point { float x, y, z; };
Point p = Point(1.0, 2.0, 3.0);
"""
        opencl = transform_and_emit(glsl)

        # Struct constructor should use compound literal syntax
        assert "Point p = {1.0f, 2.0f, 3.0f};" in opencl

    def test_struct_constructor_with_vectors(self):
        """Test struct constructor with vector arguments."""
        glsl = """
struct Geo { vec3 pos; vec3 scale; vec3 rotation; };
Geo _geo = Geo(vec3(0), vec3(1), vec3(0));
"""
        opencl = transform_and_emit(glsl)

        # Should emit compound literal with vector constructors
        # Note: Integer literals in vec3(0) don't get .0f suffix (compiler will convert)
        assert "Geo _geo = {(float3)(0), (float3)(1), (float3)(0)};" in opencl

    def test_struct_constructor_mixed_args(self):
        """Test struct constructor with mixed argument types."""
        glsl = """
struct Foo { vec3 a; float b, c; };
Foo _bar = Foo(vec3(0), 1.0, 0.5);
"""
        opencl = transform_and_emit(glsl)

        # Should emit compound literal
        assert "Foo _bar = {(float3)(0), 1.0f, 0.5f};" in opencl

    def test_struct_constructor_in_function(self):
        """Test struct constructor inside function."""
        glsl = """
struct Hit { vec3 p; float t, d; };

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    Hit _hit = Hit(vec3(0), 1.0, 0.5);
    fragColor = vec4(_hit.p, 1.0);
}
"""
        opencl = transform_and_emit(glsl)

        # Check struct definition
        assert "typedef struct {" in opencl
        assert "} Hit;" in opencl

        # Check struct constructor in function
        assert "Hit _hit = {(float3)(0), 1.0f, 0.5f};" in opencl


class TestStructMemberAccess:
    """Test struct member access (should already work, just verify)."""

    def test_struct_member_access(self):
        """Test accessing struct members."""
        glsl = """
struct Hit { vec3 p; float t, d; };
Hit _hit = Hit(vec3(0), 1.0, 0.5);

void main() {
    vec3 pos = _hit.p;
    float time = _hit.t;
}
"""
        opencl = transform_and_emit(glsl)

        # Member access should work as-is (no transformation needed)
        assert "float3 pos = _hit.p;" in opencl
        assert "float time = _hit.t;" in opencl


class TestStructAssignment:
    """Test struct assignment."""

    def test_struct_copy_assignment(self):
        """Test struct copy assignment at program scope.

        `Point a = {...}` is a compile-time-constant brace initializer and stays
        in place. `Point b = a;` is NOT a compile-time constant at file scope (a
        global/const is not a constant expression in OpenCL C — it fails with
        "initializer element is not a compile-time constant"), so category-A
        hoisting emits `b` bare and moves `b = a;` into the kernel body. In this
        transformer-only path the hoisted assignment is recorded (not injected),
        so only the bare declaration appears here.
        """
        glsl = """
struct Point { float x, y, z; };
Point a = Point(1.0, 2.0, 3.0);
Point b = a;
"""
        opencl = transform_and_emit(glsl)

        # Constant brace-init global stays; non-constant copy-init is hoisted.
        assert "Point a = {1.0f, 2.0f, 3.0f};" in opencl
        assert "Point b;" in opencl


class TestStructArrays:
    """Test arrays of structs."""

    def test_struct_array_declaration(self):
        """Test declaring array of structs."""
        glsl = """
struct Point { float x, y, z; };
Point points[10];
"""
        opencl = transform_and_emit(glsl)

        assert "Point points[10];" in opencl


class TestNestedStructs:
    """Test nested struct types (struct containing another struct)."""

    def test_nested_struct_field(self):
        """Test struct with another struct as field."""
        glsl = """
struct Point { float x, y, z; };
struct Line { Point start, end; };
"""
        opencl = transform_and_emit(glsl)

        # Both structs should be defined
        assert "} Point;" in opencl
        assert "} Line;" in opencl
        # Line should have Point fields
        assert "Point start, end;" in opencl

    def test_nested_struct_constructor(self):
        """Test constructing struct with nested struct arguments."""
        glsl = """
struct Point { float x, y, z; };
struct Line { Point start, end; };
Line l = Line(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));
"""
        opencl = transform_and_emit(glsl)

        # Both constructors should use compound literal syntax
        assert "Line l = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};" in opencl


class TestStructIntegration:
    """Integration tests with full shader examples."""

    def test_structs_glsl_example(self):
        """Test the example from structs.glsl."""
        glsl = """
struct Geo
{
    vec3 pos;
    vec3 scale;
    vec3 rotation;
};

struct Ray { vec3 o, d; };
struct Camera { vec3 p, t; };
struct Hit { vec3 p; float t, d; };

Geo _geo = Geo(vec3(0),vec3(1),vec3(0));
Ray _ray;
Camera _cam;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    struct Foo { vec3 a; float b, c; };
    Foo _bar =  Foo( vec3(0), 1.0, 0.5);
    Hit _hit = Hit( vec3(0), 1.0, 0.5);
    _bar.a = _hit.p;
    float x = _hit.t;
    vec3 col = _cam.p;
    fragColor = vec4(col,1.0);
}
"""
        opencl = transform_and_emit(glsl)

        # Check struct definitions (global)
        assert "typedef struct {" in opencl
        assert "} Geo;" in opencl
        assert "} Ray;" in opencl
        assert "} Camera;" in opencl
        assert "} Hit;" in opencl

        # Check global struct variable initialization
        assert "Geo _geo = {(float3)(0), (float3)(1), (float3)(0)};" in opencl

        # Check declarations without initialization
        assert "Ray _ray;" in opencl
        assert "Camera _cam;" in opencl

        # Check local struct definition inside function
        assert "} Foo;" in opencl

        # Check struct constructors
        assert "Foo _bar = {(float3)(0), 1.0f, 0.5f};" in opencl
        assert "Hit _hit = {(float3)(0), 1.0f, 0.5f};" in opencl

        # Check struct member access
        assert "_bar.a = _hit.p;" in opencl
        assert "float x = _hit.t;" in opencl
        assert "float3 col = _cam.p;" in opencl
