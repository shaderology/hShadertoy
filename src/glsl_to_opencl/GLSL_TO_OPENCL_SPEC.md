# GLSL to OpenCL Transformation Specification

**Project:** hShadertoy GLSL to OpenCL Transpiler
**Target:** GLSL ES 3.0 -> OpenCL C 1.2
**Last Updated:** 2025-11-20
**Status:** Production - Matrix Struct Refactor Complete + Full Struct Support

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Transformation Pipeline](#transformation-pipeline)
3. [Type System](#type-system)
4. [Transformation Rules](#transformation-rules)
5. [Runtime Library](#runtime-library)
6. [Known Limitations](#known-limitations)

---

## Architecture Overview

### System Components

```
GLSL Source Code
    |
    v
[Parser] tree-sitter-glsl
    |
    v
GLSL AST (tree-sitter nodes)
    |
    v
[Analyzer] TypeChecker + SymbolTable
    |
    v
[Transformer] ASTTransformer (1300 lines)
    |
    v
Transformed AST (IR nodes)
    |
    v
[Code Generator] OpenCLEmitter (640 lines)
    |
    v
OpenCL C Code
```

### Core Modules

**Parser** (`src/glsl_to_opencl/parser/`)
- Tree-sitter integration for GLSL parsing
- AST node wrappers for Python access
- Source location tracking

**Analyzer** (`src/glsl_to_opencl/analyzer/`)
- `TypeChecker`: Type inference and validation
- `SymbolTable`: Variable and function tracking
- Built-in function registry (47 GLSL functions)

**Transformer** (`src/glsl_to_opencl/transformer/`)
- `ASTTransformer`: Main transformation logic (single-pass visitor)
- `transformed_ast.py`: IR node definitions (immutable dataclasses)
- `code_emitter.py`: Legacy emitter (retained for compatibility)

**Code Generator** (`src/glsl_to_opencl/codegen/`)
- `OpenCLEmitter`: Production code generator
- Operator precedence system
- Clean formatting and indentation

**Runtime Library** (`houdini/ocl/include/`)
- `glslHelpers.h`: GLSL built-in functions (47 functions, overloadable)
- `matrix_types.h`: Struct-based matrix types (matrix2x2/3x3/4x4)
- `matrix_ops.h`: Matrix operations (mul, transpose, inverse, determinant)

### Design Principles

1. **Single-pass transformation** - No re-parsing or multi-pass analysis
2. **Type-aware** - Type information propagates through transformation
3. **Immutable IR** - Transformed nodes are immutable dataclasses
4. **Struct-based matrices** - All matrices use struct types (can be returned by value)
5. **Systematic visitor pattern** - Consistent node transformation

---

## Transformation Pipeline

### Phase 1: Parsing
```python
from src.glsl_to_opencl.parser import parse_glsl
ast = parse_glsl(glsl_source)
```
- Input: GLSL source code (string)
- Output: Tree-sitter AST (ASTNode tree)

### Phase 2: Type Analysis
```python
from src.glsl_to_opencl.analyzer import TypeChecker, create_builtin_symbol_table
symbol_table = create_builtin_symbol_table()
type_checker = TypeChecker(symbol_table)
```
- Builds symbol table with built-in functions
- Tracks variable and function types

### Phase 3: Transformation
```python
from src.glsl_to_opencl.transformer import ASTTransformer
transformer = ASTTransformer(type_checker)
transformed_ast = transformer.transform(ast)
```
- Applies all transformation rules
- Generates immutable IR nodes
- Tracks local types for matrix operation detection

### Phase 4: Code Generation
```python
from src.glsl_to_opencl.codegen import OpenCLEmitter
emitter = OpenCLEmitter()
opencl_code = emitter.emit(transformed_ast)
```
- Generates clean OpenCL C code
- Handles operator precedence
- Formats with proper indentation

---

## Type System

### Scalar Types
| GLSL | OpenCL | Notes |
|------|--------|-------|
| float | float | Unchanged |
| int | int | Unchanged |
| uint | uint | Unchanged |
| bool | bool | Unchanged (OpenCL: int internally) |
| void | void | Unchanged |

### Vector Types
| GLSL Type | OpenCL Type | Notes |
|-----------|-------------|-------|
| vec2/3/4 | float2/3/4 | Float vectors |
| ivec2/3/4 | int2/3/4 | Integer vectors |
| uvec2/3/4 | uint2/3/4 | Unsigned vectors |
| bvec2/3/4 | int2/3/4 | Boolean vectors (as ints) |

### Matrix Types (Struct-Based)

**Key Design Decision:** All matrices use struct types that can be returned by value.

| GLSL Type | OpenCL Type | C Definition | Layout |
|-----------|-------------|--------------|--------|
| mat2 | matrix2x2 | `struct { float2 cols[2]; }` | Column-major, 2x2 |
| mat3 | matrix3x3 | `struct { float3 cols[3]; }` | Column-major, 3x3 |
| mat4 | matrix4x4 | `struct { float4 cols[4]; }` | Column-major, 4x4 |

### User-Defined Struct Types

**Supported:** Full GLSL struct support with OpenCL typedef struct transformation.

**Struct Definition:**
```glsl
// GLSL
struct Geo {
    vec3 pos;
    vec3 scale;
    vec3 rotation;
};

// OpenCL
typedef struct {
    float3 pos;
    float3 scale;
    float3 rotation;
} Geo;
```

**Features:**
- Global and local struct definitions
- Comma-separated field declarations (`float t, d;`)
- Nested structs
- Struct arrays
- Primitive types (float, int, uint, bool) and vector types (vec2/3/4)

**Limitation:** Matrix types in struct fields are architecturally supported but not extensively tested (see Known Limitations).

**Memory Layout (Column-Major):**
```c
// mat2 - 2 columns of float2
matrix2x2 M2;
M2.cols[0] = (float2)(m00, m10);  // Column 0: [m00, m10]
M2.cols[1] = (float2)(m01, m11);  // Column 1: [m01, m11]

// mat3 - 3 columns of float3
matrix3x3 M3;
M3.cols[0] = (float3)(m00, m10, m20);
M3.cols[1] = (float3)(m01, m11, m21);
M3.cols[2] = (float3)(m02, m12, m22);

// mat4 - 4 columns of float4
matrix4x4 M4;
M4.cols[0] = (float4)(m00, m10, m20, m30);
M4.cols[1] = (float4)(m01, m11, m21, m31);
M4.cols[2] = (float4)(m02, m12, m22, m32);
M4.cols[3] = (float4)(m03, m13, m23, m33);
```

**Why Structs?**
- Arrays cannot be returned by value in C/OpenCL
- Houdini's legacy `typedef fpreal3 mat3[3]` required complex out-parameter transformations
- Struct-based approach eliminates 580+ lines of special-case code
- Enables clean, GLSL-like syntax: `matrix3x3 result = GLSL_transpose_mat3(M);`

### Precision Qualifiers
All precision qualifiers are **removed** during transformation:
- `highp`, `mediump`, `lowp` -> (stripped)
- OpenCL uses native precision

---

## Transformation Rules

### 1. Literals

**Float Literals** - Add 'f' suffix
```glsl
// GLSL
1.0, 0.5, .5, 3.14159

// OpenCL
1.0f, 0.5f, .5f, 3.14159f
```

**Integer Literals** - Unchanged
```glsl
1, 42, 0xFF -> 1, 42, 0xFF
```

**Boolean Literals** - Unchanged
```glsl
true, false -> true, false
```

### 2. Type Constructors

**Vector Constructors** - Cast syntax
```glsl
// GLSL
vec2(1.0, 2.0)
vec3(0.0)
vec4(v3, 1.0)
vec4(v2a, v2b)

// OpenCL
(float2)(1.0f, 2.0f)
(float3)(0.0f)
(float4)(v3, 1.0f)
(float4)(v2a, v2b)
```

**Matrix Constructors - Diagonal (single scalar)**
```glsl
// GLSL
mat2(1.0)
mat3(1.0)
mat4(1.0)

// OpenCL
GLSL_matrix2x2_diagonal(1.0f)
GLSL_matrix3x3_diagonal(1.0f)
GLSL_matrix4x4_diagonal(1.0f)
```

**Matrix Constructors - Full (all elements, column-major)**
```glsl
// GLSL
mat2(1, 2, 3, 4)
mat3(1,0,0, 0,1,0, 0,0,1)
mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)

// OpenCL
GLSL_mat2(1.0f, 2.0f, 3.0f, 4.0f)
GLSL_mat3(1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f)
GLSL_mat4(1.0f,0.0f,0.0f,0.0f, 0.0f,1.0f,0.0f,0.0f, 0.0f,0.0f,1.0f,0.0f, 0.0f,0.0f,0.0f,1.0f)
```

**Matrix Constructors - From columns**
```glsl
// GLSL
mat2(vec2(1,2), vec2(3,4))
mat3(vec3(1,0,0), vec3(0,1,0), vec3(0,0,1))
mat4(vec4(...), vec4(...), vec4(...), vec4(...))

// OpenCL
GLSL_mat2_cols((float2)(1.0f,2.0f), (float2)(3.0f,4.0f))
GLSL_mat3_cols((float3)(1.0f,0.0f,0.0f), (float3)(0.0f,1.0f,0.0f), (float3)(0.0f,0.0f,1.0f))
GLSL_mat4_cols(...)
```

**Matrix Type Casting**
```glsl
// GLSL
mat4(mat3_var)
mat3(mat4_var)

// OpenCL
GLSL_mat4_from_mat3(mat3_var)
GLSL_mat3_from_mat4(mat4_var)
```

**Struct Constructors** - Compound literal syntax
```glsl
// GLSL
struct Geo { vec3 pos; vec3 scale; vec3 rotation; };
Geo g = Geo(vec3(0), vec3(1), vec3(0));

struct Ray { vec3 o, d; };
Ray r = Ray(vec3(0,0,0), vec3(1,0,0));

// OpenCL
typedef struct { float3 pos; float3 scale; float3 rotation; } Geo;
Geo g = {(float3)(0), (float3)(1), (float3)(0)};

typedef struct { float3 o, d; } Ray;
Ray r = {(float3)(0.0f, 0.0f, 0.0f), (float3)(1.0f, 0.0f, 0.0f)};
```
**Note:** Struct constructors use C99 compound literal syntax `{ arg1, arg2, ... }` instead of cast syntax.

### 3. Built-in Functions

**Function Name Transformation** - Add GLSL_ prefix

All 47 GLSL built-in functions map to `GLSL_*()` overloadable functions:

**Trigonometric:**
```glsl
sin(x)    -> GLSL_sin(x)
cos(x)    -> GLSL_cos(x)
tan(x)    -> GLSL_tan(x)
asin(x)   -> GLSL_asin(x)
acos(x)   -> GLSL_acos(x)
atan(x)   -> GLSL_atan(x)
atan(y,x) -> GLSL_atan(y,x)
```

**Hyperbolic:**
```glsl
sinh(x)  -> GLSL_sinh(x)
cosh(x)  -> GLSL_cosh(x)
tanh(x)  -> GLSL_tanh(x)
asinh(x) -> GLSL_asinh(x)
acosh(x) -> GLSL_acosh(x)
atanh(x) -> GLSL_atanh(x)
```

**Exponential/Power:**
```glsl
pow(x,y)      -> GLSL_pow(x,y)
exp(x)        -> GLSL_exp(x)
log(x)        -> GLSL_log(x)
exp2(x)       -> GLSL_exp2(x)
log2(x)       -> GLSL_log2(x)
sqrt(x)       -> GLSL_sqrt(x)
inversesqrt(x)-> GLSL_inversesqrt(x)
```

**Common/Math:**
```glsl
abs(x)              -> GLSL_abs(x)
sign(x)             -> GLSL_sign(x)
floor(x)            -> GLSL_floor(x)
ceil(x)             -> GLSL_ceil(x)
trunc(x)            -> GLSL_trunc(x)
fract(x)            -> GLSL_fract(x)     // x - floor(x)
mod(x,y)            -> GLSL_mod(x,y)      // x - y*floor(x/y), NOT remainder!
modf(x, out i)      -> GLSL_modf(x, &i)   // out parameter
min(x,y)            -> GLSL_min(x,y)
max(x,y)            -> GLSL_max(x,y)
clamp(x,a,b)        -> GLSL_clamp(x,a,b)
mix(a,b,t)          -> GLSL_mix(a,b,t)
step(edge,x)        -> GLSL_step(edge,x)
smoothstep(a,b,x)   -> GLSL_smoothstep(a,b,x)
```

**Geometric:**
```glsl
length(v)            -> GLSL_length(v)
distance(a,b)        -> GLSL_distance(a,b)
dot(a,b)             -> GLSL_dot(a,b)
cross(a,b)           -> GLSL_cross(a,b)
normalize(v)         -> GLSL_normalize(v)
faceforward(N,I,Nref)-> GLSL_faceforward(N,I,Nref)
reflect(I,N)         -> GLSL_reflect(I,N)
refract(I,N,eta)     -> GLSL_refract(I,N,eta)
```

**Angle Conversion:**
```glsl
radians(deg) -> GLSL_radians(deg)
degrees(rad) -> GLSL_degrees(rad)
```

**Derivatives (Dummy Placeholders):**
```glsl
dFdx(x)   -> GLSL_dFdx(x)    // Returns input unchanged
dFdy(x)   -> GLSL_dFdy(x)    // Returns input unchanged
fwidth(x) -> GLSL_fwidth(x)  // Returns input unchanged
```
*Note: OpenCL has no derivative hardware outside fragment shaders*

**Matrix Functions:**

mat2 uses base function names, mat3/mat4 use suffixes:
```glsl
// mat2
transpose(M2) -> GLSL_transpose(M2)
inverse(M2)   -> GLSL_inverse(M2)
determinant(M2)->GLSL_determinant(M2)

// mat3
transpose(M3) -> GLSL_transpose_mat3(M3)
inverse(M3)   -> GLSL_inverse_mat3(M3)
determinant(M3)->GLSL_determinant_mat3(M3)

// mat4
transpose(M4) -> GLSL_transpose_mat4(M4)
inverse(M4)   -> GLSL_inverse_mat4(M4)
determinant(M4)->GLSL_determinant_mat4(M4)
```

**All functions are overloadable** - Support float, float2, float3, float4 variants automatically.

### 4. Matrix Operations

**Critical:** Matrix multiplication requires type-specific function names.

**Matrix * Vector** - Column vector
```glsl
// GLSL
mat2 M2; vec2 v2;
vec2 result = M2 * v2;

mat3 M3; vec3 v3;
vec3 result = M3 * v3;

mat4 M4; vec4 v4;
vec4 result = M4 * v4;

// OpenCL
matrix2x2 M2; float2 v2;
float2 result = GLSL_mul_mat2_vec2(M2, v2);

matrix3x3 M3; float3 v3;
float3 result = GLSL_mul_mat3_vec3(M3, v3);

matrix4x4 M4; float4 v4;
float4 result = GLSL_mul_mat4_vec4(M4, v4);
```

**Vector * Matrix** - Row vector
```glsl
// GLSL
vec2 v2; mat2 M2;
vec2 result = v2 * M2;

vec3 v3; mat3 M3;
vec3 result = v3 * M3;

vec4 v4; mat4 M4;
vec4 result = v4 * M4;

// OpenCL
float2 v2; matrix2x2 M2;
float2 result = GLSL_mul_vec2_mat2(v2, M2);

float3 v3; matrix3x3 M3;
float3 result = GLSL_mul_vec3_mat3(v3, M3);

float4 v4; matrix4x4 M4;
float4 result = GLSL_mul_vec4_mat4(v4, M4);
```

**Matrix * Matrix**
```glsl
// GLSL
mat2 M1, M2;
mat2 result = M1 * M2;

mat3 M1, M2;
mat3 result = M1 * M2;

mat4 M1, M2;
mat4 result = M1 * M2;

// OpenCL
matrix2x2 M1, M2;
matrix2x2 result = GLSL_mul_mat2_mat2(M1, M2);

matrix3x3 M1, M2;
matrix3x3 result = GLSL_mul_mat3_mat3(M1, M2);

matrix4x4 M1, M2;
matrix4x4 result = GLSL_mul_mat4_mat4(M1, M2);
```

**Compound Assignments**
```glsl
// GLSL
v2 *= M2;
v3 *= M3;
M3 *= M3;

// OpenCL
v2 = GLSL_mul_vec2_mat2(v2, M2);
v3 = GLSL_mul_vec3_mat3(v3, M3);
M3 = GLSL_mul_mat3_mat3(M3, M3);
```

**Chained Operations**
```glsl
// GLSL
vec3 result = v * M1 * M2;

// OpenCL
float3 result = GLSL_mul_vec3_mat3(GLSL_mul_vec3_mat3(v, M1), M2);
```

**Type Inference** - Transformer tracks types through:
- Function parameters
- Array access
- Struct field access
- Chained operations
- User-defined function return types

### 5. Operators

**Standard Operators** - Unchanged
```
Arithmetic: +, -, *, /, %
Comparison: <, >, <=, >=, ==, !=
Logical: &&, ||, !, ^^
Bitwise: &, |, ^, ~, <<, >>
Assignment: =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
Ternary: ? :
```

**Exception:** Matrix multiplication `*` transforms to `GLSL_mul_*` (see Section 4).

### 6. Control Flow

**Unchanged** - All control flow statements identical:
```glsl
if (condition) { ... } else { ... }
for (init; condition; update) { ... }
while (condition) { ... }
do { ... } while (condition);
```

### 7. Variable Declarations

**Undefined Variable Initialization (GLSL Semantics)**

**CRITICAL:** GLSL implicitly zero-initializes undefined variables, while OpenCL leaves them undefined. The transpiler automatically adds zero initializers to match GLSL semantics.

```glsl
// GLSL
float foo;       // Implicitly initialized to 0.0
vec3 bar;        // Implicitly initialized to vec3(0.0)
int x;           // Implicitly initialized to 0

// OpenCL (transpiled)
float foo = 0.0f;           // Explicit zero initialization
float3 bar = (float3)(0.0f); // Explicit zero initialization
int x = 0;                  // Explicit zero initialization
```

**Single Declaration**
```glsl
// GLSL
float x = 1.0;      // Explicit initializer (unchanged)
vec3 p;             // No initializer (gets zero-init)

// OpenCL
float x = 1.0f;          // Explicit initializer preserved
float3 p = (float3)(0.0f); // Zero initializer added
```

**Comma-Separated Declarations**
```glsl
// GLSL
float x, y, z;                    // All undefined
int a = 10, b = 20;               // All explicitly initialized
vec3 p, n, t;                     // All undefined
float a = 1.0, b, c = 3.0;        // Mixed (b undefined)

// OpenCL
float x = 0.0f, y = 0.0f, z = 0.0f;             // Zero-init added
int a = 10, b = 20;                             // Unchanged
float3 p = (float3)(0.0f), n = (float3)(0.0f), t = (float3)(0.0f); // Zero-init added
float a = 1.0f, b = 0.0f, c = 3.0f;             // b gets zero-init
```

**Matrix Types**
```glsl
// GLSL
mat2 M1;
mat3 M2, M3;
mat4 M4 = mat4(1.0);

// OpenCL
matrix2x2 M1 = GLSL_matrix2x2_diagonal(0.0f);  // Zero matrix
matrix3x3 M2 = GLSL_matrix3x3_diagonal(0.0f), M3 = GLSL_matrix3x3_diagonal(0.0f);
matrix4x4 M4 = GLSL_matrix4x4_diagonal(1.0f);  // Explicit initializer preserved
```

**Supported Types for Auto-Initialization:**
- Scalars: `float`, `int`
- Float vectors: `vec2`, `vec3`, `vec4`
- Integer vectors: `ivec2`, `ivec3`, `ivec4`
- Matrices: `mat2`, `mat3`, `mat4`

**Not Auto-Initialized:**
- `bool`, `uint`, `uvec*`, `bvec*` (less common, different semantics)
- User-defined structs (require explicit constructors)
- Arrays (require explicit initialization syntax)

**Const Qualifier**
```glsl
// GLSL
const float PI = 3.14159;
const vec3 UP = vec3(0, 1, 0);
const float x = 1.0, y = 2.0;

// OpenCL
const float PI = 3.14159f;
const float3 UP = (float3)(0.0f, 1.0f, 0.0f);
const float x = 1.0f, y = 2.0f;
```

**Struct Declarations**
```glsl
// GLSL - Global struct definition
struct Geo {
    vec3 pos;
    vec3 scale;
    vec3 rotation;
};

struct Ray { vec3 o, d; };  // Comma-separated fields, one-line definition

Geo _geo = Geo(vec3(0), vec3(1), vec3(0));
Ray _ray;

// OpenCL
typedef struct {
    float3 pos;
    float3 scale;
    float3 rotation;
} Geo;

typedef struct {
    float3 o, d;
} Ray;

Geo _geo = {(float3)(0), (float3)(1), (float3)(0)};  // C99 compound literal
Ray _ray;
```

**Local Struct Definitions** (inside functions)
```glsl
// GLSL
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    struct Foo { vec3 a; float b, c; };
    Foo _bar = Foo(vec3(0), 1.0, 0.5);
}

// OpenCL
void mainImage(__private float4* fragColor, float2 fragCoord) {
    typedef struct {
        float3 a;
        float b, c;
    } Foo;
    Foo _bar = {(float3)(0), 1.0f, 0.5f};
}
```

**Struct Arrays**
```glsl
// GLSL
struct Point { float x, y, z; };
Point points[10];

// OpenCL
typedef struct {
    float x, y, z;
} Point;
Point points[10];
```

**Nested Structs**
```glsl
// GLSL
struct Point { float x, y, z; };
struct Line { Point start, end; };
Line l = Line(Point(0,0,0), Point(1,1,1));

// OpenCL
typedef struct {
    float x, y, z;
} Point;

typedef struct {
    Point start, end;
} Line;

Line l = { {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f} };
```

### 8. Function Parameters

**GLSL Qualifiers:**
- `in`: Read-only (default, removed)
- `out`: Write-only output (becomes pointer)
- `inout`: Read-write (becomes pointer)
- `const`: Constant (unchanged)

**Transformation:**
```glsl
// GLSL
void foo(in float x, out vec2 result, inout vec3 data) {
    result = vec2(x);
    data *= 2.0;
}

// OpenCL
void foo(float x, __private float2* result, __private float3* data) {
    *result = (float2)(x);
    *data *= 2.0f;
}

// Call sites get address-of operator
// GLSL: foo(1.0, outVar, inoutVar);
// OpenCL: foo(1.0f, &outVar, &inoutVar);
```

**All matrix types** (mat2/3/4) use pointers for out/inout (struct behavior).

### 9. Swizzling

**Unchanged** - OpenCL supports identical swizzling:
```glsl
v.xy, v.rgb, v.w -> v.xy, v.rgb, v.w
```

### 10. Preprocessor Directives

**Macro Definitions** - Transform bodies
```glsl
// GLSL
#define PI 3.14159265
#define random(x) fract(sin(x))

// OpenCL
#define PI 3.14159265f
#define random(x) GLSL_fract(GLSL_sin(x))
```

**Conditional Compilation** - Pass through unchanged
```glsl
#if, #ifdef, #ifndef, #else, #elif, #endif
```

### 11. Parenthesized Expressions

**Preserved** - Explicit parentheses maintained for order of operations:
```glsl
// GLSL
1.0 * (2.0 / iResolution.y) * (1.0 / fov)

// OpenCL
1.0f * (2.0f / iResolution.y) * (1.0f / fov)
```

---

## Runtime Library

### glslHelpers.h

**47 GLSL built-in functions** - All overloadable for float/float2/float3/float4

**Implementation Strategy:**
- Use OpenCL built-ins when semantics match
- Custom implementations when GLSL semantics differ

**Key Semantic Differences:**
- `GLSL_mod(x, y)`: Uses `x - y * floor(x/y)` (GLSL semantics) vs OpenCL `remainder`
- `GLSL_fract(x)`: Uses `x - floor(x)` (GLSL semantics)
- `GLSL_inversesqrt(x)`: Maps to OpenCL `rsqrt(x)` (reciprocal square root)

**Macros for Code Generation:**
- `DEFINE_UNARY(NAME, BUILTIN)`: Generate 4 overloads
- `DEFINE_BINARY(NAME, BUILTIN)`: Generate 10 overloads (vec+scalar combinations)
- `DEFINE_TERNARY(NAME, BUILTIN)`: Generate 10 overloads

**Example:**
```c
DEFINE_UNARY(GLSL_sin, sin)
// Expands to:
// float  GLSL_sin(float  x){ return sin(x); }
// float2 GLSL_sin(float2 x){ return sin(x); }
// float3 GLSL_sin(float3 x){ return sin(x); }
// float4 GLSL_sin(float4 x){ return sin(x); }
```

### matrix_types.h

**Struct Definitions:**
```c
typedef struct {
    float2 cols[2];  /* 2 columns of float2 */
} matrix2x2;

typedef struct {
    float3 cols[3];  /* 3 columns of float3 */
} matrix3x3;

typedef struct {
    float4 cols[4];  /* 4 columns of float4 */
} matrix4x4;
```

**Column-major layout** - `M.cols[col][row]` matches GLSL `M[col][row]`

### matrix_ops.h

**450+ lines of matrix operations**

**Constructors:**
- Diagonal: `GLSL_matrix2x2_diagonal(float s)`
- Full: `GLSL_mat2/3/4(float m00, float m10, ...)`
- From columns: `GLSL_mat2/3/4_cols(vec col0, vec col1, ...)`
- Type casting: `GLSL_mat3_from_mat4(matrix4x4)`, `GLSL_mat4_from_mat3(matrix3x3)`

**Operations:**
- Matrix-vector multiply: `GLSL_mul_mat2/3/4_vec2/3/4(M, v)`
- Vector-matrix multiply: `GLSL_mul_vec2/3/4_mat2/3/4(v, M)`
- Matrix-matrix multiply: `GLSL_mul_mat2/3/4_mat2/3/4(M1, M2)`
- Transpose: `GLSL_transpose(M2)`, `GLSL_transpose_mat3/4(M3/4)`
- Inverse: `GLSL_inverse(M2)`, `GLSL_inverse_mat3/4(M3/4)`
- Determinant: `GLSL_determinant(M2)`, `GLSL_determinant_mat3/4(M3/4)`
- Component-wise mult: `GLSL_matrixCompMult(M2)`, `GLSL_matrixCompMult_mat3/4(M3/4)`

**All functions return by value** - Enabled by struct-based types.

---

## Known Limitations

### 1. Derivative Functions
**Issue:** `dFdx`, `dFdy`, `fwidth` return input unchanged (no actual derivatives)
**Reason:** OpenCL has no derivative hardware outside fragment shaders
**Impact:** Shaders using derivatives for screen-space effects will not work correctly
**Future:** Potential Houdini COPs "Write Back Kernel" with convolution matrix

### 2. Const Matrix Initialization
**Issue:** Matrix constructors are function calls, not compile-time constants
**Impact:** Cannot use in switch cases or some global const contexts
**Workaround:** Acceptable - OpenCL compilers optimize runtime initialization
**Example:**
```c
const matrix3x3 M = GLSL_mat3(...);  // Runtime initialization (not compile-time)
```

### 3. Address-Of Insertion for Out Parameters
**Issue:** Only simple identifiers supported
**Limitation:** Cannot pass complex expressions as out parameters
**Example:**
```glsl
foo(x, arr[i]);  // NOT SUPPORTED - arr[i] is not a simple identifier
// Workaround: Use temporary variable
vec2 temp = arr[i];
foo(x, temp);
arr[i] = temp;
```

### 4. Matrix Types in Struct Fields
**Status:** Architecturally supported but not extensively tested
**Description:** Struct fields can be matrix types (`matrix2x2`, `matrix3x3`, `matrix4x4`), but this combination has not been thoroughly tested in complex scenarios
**Example:**
```glsl
// GLSL
struct Transform {
    mat3 rotation;
    vec3 translation;
};

// OpenCL (should work but not extensively tested)
typedef struct {
    matrix3x3 rotation;
    float3 translation;
} Transform;
```
**Impact:** Basic usage should work since matrix types are structs themselves, but edge cases may exist
**Workaround:** Test carefully or avoid matrix fields in structs if encountering issues
**Future:** Will be tested and documented as needed when encountered in real shaders

---

## Development Notes

### Test Coverage
**Total: 1514 tests passing, 6 skipped**
- Unit tests: 1508 (parser, analyzer, transformer, code generator)
- Compilation tests: N/A (compilation tests run separately)
- Integration tests: 16+ (9 Shadertoy shaders + struct shader)

**Key Test Files:**
- `test_ast_transformer_basic.py` - Basic transformations (literals, types, operators)
- `test_transformer_constructors.py` - Vector/matrix constructors
- `test_ast_matrix_ops.py` - Matrix multiplication detection
- `test_transformer_matrix_functions.py` - Matrix functions (transpose, inverse, determinant)
- `test_transformer_qualifiers.py` - Function parameter qualifiers (in/out/inout)
- `test_transformer_const_qualifier.py` - Const variable declarations (13 tests)
- `test_transformer_structs.py` - User-defined struct support (16 tests)

### Compilation Verification
**Shaders that compile successfully:**
- `tests/shaders/test/*.glsl` - Feature tests (comma, macros, oporder, qualifiers, statement, structs)
- `tests/shaders/simple/*.glsl` - 9 Shadertoy shaders (vignette, gradient, hexagonal, ripples, silexars, sand, caustic, colorful, warping)
- `tests/shaders/medium/spec.glsl` - GLSL built-in function spec
- `tests/shaders/medium/ocean.glsl` - Ocean waves (medium complexity)
- `tests/shaders/medium/seascape.glsl` - Seascape (complex)
- `tests/shaders/medium/matrix.glsl` - Matrix operations test

**Blockers:**
- `tests/shaders/complex/adjugate.glsl` - Requires preprocessor directive evaluation (#if AA>1)

### Matrix Refactor History
**Pre-2025-11-15:** Houdini array-based mat3 (`fpreal3[3]`) required 580+ lines of special-case code
**Post-2025-11-16:** Struct-based matrices (`matrix2x2/3x3/4x4`) - clean, simple, consistent

**Code Deleted:**
- 522 lines from `ast_transformer.py`
- 118 lines from `opencl_emitter.py`
- 15 lines from `transformed_ast.py` (Mat3ResultReturn class)
- 22 tests from `test_transformer_mat3_functions.py`

**Benefits:**
- All matrices behave identically (no special cases)
- Functions return matrices by value (clean syntax)
- Easier to maintain and extend
- Better performance (struct returns optimize to registers)

### Next Features (Priority Order)
1. **Preprocessor directive evaluation** - For adjugate.glsl (#if/#ifdef with constant expressions)
2. **Sampler/texture support** - Image operations
3. **Array operations** - Multi-dimensional arrays
4. **Advanced struct features** - Matrix fields in structs (if needed), struct methods (not standard GLSL)

---

## Quick Reference

### Common Transformations Cheat Sheet

```glsl
// LITERALS
1.0             -> 1.0f
vec2(1, 2)      -> (float2)(1.0f, 2.0f)
mat3(1.0)       -> GLSL_matrix3x3_diagonal(1.0f)

// BUILT-INS
sin(x)          -> GLSL_sin(x)
mod(x, y)       -> GLSL_mod(x, y)
normalize(v)    -> GLSL_normalize(v)

// MATRIX OPS
M3 * v3         -> GLSL_mul_mat3_vec3(M3, v3)
v3 * M3         -> GLSL_mul_vec3_mat3(v3, M3)
M3 * M3         -> GLSL_mul_mat3_mat3(M3, M3)
transpose(M3)   -> GLSL_transpose_mat3(M3)

// PARAMETERS
out vec2 r      -> __private float2* r
inout vec3 d    -> __private float3* d
foo(x, outVar)  -> foo(x, &outVar)

// DECLARATIONS
const float PI = 3.14; -> const float PI = 3.14f;
float x, y, z;         -> float x, y, z;

// STRUCTS
struct Geo { vec3 pos; vec3 scale; };  -> typedef struct { float3 pos; float3 scale; } Geo;
Geo g = Geo(vec3(0), vec3(1));         -> Geo g = {(float3)(0), (float3)(1)};
g.pos                                  -> g.pos
```

### Type Mapping Quick Lookup

```
vec2  -> float2      mat2 -> matrix2x2
vec3  -> float3      mat3 -> matrix3x3
vec4  -> float4      mat4 -> matrix4x4
ivec2 -> int2
ivec3 -> int3
ivec4 -> int4

User-defined structs: Name -> Name (typedef struct)
```

---

**End of Specification**

For more details on specific features, see:
- Architecture: [Architecture Overview](#architecture-overview)
- Types: [Type System](#type-system)
- Transformations: [Transformation Rules](#transformation-rules)
- Runtime: [Runtime Library](#runtime-library)

For development workflow and progress tracking:
- `docs/RULES.md` - Development guidelines
- `.agent/PROGRESS.md` - Current status and next steps
