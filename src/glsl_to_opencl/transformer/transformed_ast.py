"""
Transformed AST Nodes for OpenCL IR.

This module defines the intermediate representation (IR) nodes created by
the ASTTransformer. These nodes represent OpenCL-semantics constructs and
are emitted by the CodeGenerator.

Design principles:
- Separate from tree-sitter AST (no byte offset issues)
- Carry type information from TypeChecker
- OpenCL semantics (GLSL constructs already resolved)
- Immutable, tree-structured IR
- Easy to traverse and emit

Architecture:
    GLSL AST -> TypeChecker -> ASTTransformer -> Transformed AST -> CodeEmitter -> OpenCL
"""

from dataclasses import dataclass
from typing import List, Optional
from ..analyzer.type_checker import GLSLType


@dataclass(frozen=True)
class TransformedNode:
    """
    Base class for all transformed AST nodes.

    All transformed nodes are immutable (frozen dataclass) to prevent
    accidental modification during traversal.

    Attributes:
        glsl_type: The GLSL/OpenCL type of this expression/statement
        source_location: (line, column) for error reporting
    """
    glsl_type: Optional[GLSLType] = None
    source_location: Optional[tuple] = None

    def accept(self, visitor):
        """Visitor pattern support."""
        method_name = f'visit_{self.__class__.__name__}'
        method = getattr(visitor, method_name, visitor.generic_visit)
        return method(self)


# ============================================================================
# Literals
# ============================================================================

@dataclass(frozen=True)
class FloatLiteral(TransformedNode):
    """
    Float literal with 'f' suffix.

    Examples:
        1.0f, 3.14f, 1.5e-3f
    """
    value: str = None  # String representation with 'f' suffix

    def __post_init__(self):
        """Validate float literal format."""
        if self.value is None:
            raise ValueError("FloatLiteral value cannot be None")
        if not self.value.endswith('f'):
            raise ValueError(f"FloatLiteral must end with 'f': {self.value}")


@dataclass(frozen=True)
class IntLiteral(TransformedNode):
    """
    Integer literal.

    Examples:
        42, 0, -1
    """
    value: str = None


@dataclass(frozen=True)
class BoolLiteral(TransformedNode):
    """
    Boolean literal.

    Examples:
        true, false
    """
    value: bool = None


# ============================================================================
# Types
# ============================================================================

@dataclass(frozen=True)
class TypeName(TransformedNode):
    """
    OpenCL type name.

    Examples:
        float, float2, float3, mat2, mat3

    Note: GLSL types (vec2, vec3) are transformed to OpenCL equivalents
    (float2, float3) during AST transformation.
    """
    name: str = None  # OpenCL type name


# ============================================================================
# Identifiers and References
# ============================================================================

@dataclass(frozen=True)
class Identifier(TransformedNode):
    """
    Variable or function name reference.

    Examples:
        x, fragColor, mainImage
    """
    name: str = None


# ============================================================================
# Expressions
# ============================================================================

@dataclass(frozen=True)
class BinaryOp(TransformedNode):
    """
    Binary operation.

    Examples:
        a + b, x * y, v < w

    Note: For matrix operations, this is transformed to function calls.
    """
    operator: str = None  # "+", "-", "*", "/", etc.
    left: TransformedNode = None
    right: TransformedNode = None


@dataclass(frozen=True)
class UnaryOp(TransformedNode):
    """
    Unary operation.

    Examples:
        -x, !flag, ~bits
    """
    operator: str = None  # "-", "+", "!", "~"
    operand: TransformedNode = None


@dataclass(frozen=True)
class ParenthesizedExpression(TransformedNode):
    """
    Explicitly parenthesized expression.

    Preserves parentheses from the source code to maintain order of operations.
    Unlike BinaryOp which may have parentheses added by precedence rules,
    this node represents parentheses that were explicitly written by the user.

    Examples:
        (2.0 / iResolution.y)
        (1.0 + E.y)

    Note: This is critical for preserving order of operations when the
    programmer explicitly used parentheses to override default precedence.
    """
    expression: TransformedNode = None


@dataclass(frozen=True)
class CallExpression(TransformedNode):
    """
    Function call expression.

    Examples:
        sin(x), GLSL_mod(a, b), normalize(v)

    Note: GLSL built-in functions are prefixed with GLSL_ during transformation.
    """
    function: str = None  # Function name (already transformed to OpenCL name)
    arguments: List[TransformedNode] = None


@dataclass(frozen=True)
class TypeConstructor(TransformedNode):
    """
    Type constructor (cast syntax in OpenCL).

    Examples:
        (float2)(1.0f, 2.0f)
        (float3)(0.0f)
        (mat2)(1.0f, 0.0f, 0.0f, 1.0f)

    Note: GLSL constructors like vec2(1.0, 2.0) are transformed to
    OpenCL cast syntax (float2)(1.0f, 2.0f).
    """
    type_name: str = None  # OpenCL type name (float2, float3, etc.)
    arguments: List[TransformedNode] = None


@dataclass(frozen=True)
class ArrayInitializer(TransformedNode):
    """
    Array initializer with curly braces.

    Examples:
        {0.0f}
        {(float3)(0.0f)}
        {1.0f, 2.0f, 3.0f}

    Note: In OpenCL, array initializers must be wrapped in curly braces.
    This is used for zero-initializing undefined arrays to match GLSL semantics.
    """
    elements: List[TransformedNode] = None


@dataclass(frozen=True)
class MemberAccess(TransformedNode):
    """
    Member access (swizzling or struct field).

    Examples:
        v.xy, color.rgb, pos.x
    """
    base: TransformedNode = None
    member: str = None  # Member name (x, xy, rgb, etc.)


@dataclass(frozen=True)
class ArrayAccess(TransformedNode):
    """
    Array element access.

    Examples:
        arr[0], matrix[i][j]
    """
    base: TransformedNode = None
    index: TransformedNode = None


@dataclass(frozen=True)
class TernaryOp(TransformedNode):
    """
    Ternary conditional operator.

    Examples:
        condition ? true_val : false_val
    """
    condition: TransformedNode = None
    true_expr: TransformedNode = None
    false_expr: TransformedNode = None


@dataclass(frozen=True)
class AssignmentOp(TransformedNode):
    """
    Assignment operation.

    Examples:
        x = 5, v += w, color *= 0.5f

    Note: Compound assignments with matrices may be transformed to
    separate operations.
    """
    operator: str = None  # "=", "+=", "-=", "*=", "/="
    target: TransformedNode = None
    value: TransformedNode = None


# ============================================================================
# Statements
# ============================================================================

@dataclass(frozen=True)
class ExpressionStatement(TransformedNode):
    """
    Expression used as a statement.

    Examples:
        foo();
        x = 5;
    """
    expression: TransformedNode = None


@dataclass(frozen=True)
class Declaration(TransformedNode):
    """
    Variable declaration.

    Examples:
        float x = 1.0f;
        float3 v = (float3)(0.0f);
        const float foo = 0.5f;
    """
    type_name: str = None  # OpenCL type name
    name: str = None
    initializer: Optional[TransformedNode] = None
    qualifiers: List[str] = None  # ["const"] etc.

    def __post_init__(self):
        # Ensure qualifiers is a list
        if self.qualifiers is None:
            object.__setattr__(self, 'qualifiers', [])


@dataclass(frozen=True)
class DeclarationList(TransformedNode):
    """
    Comma-separated variable declarations of the same type.

    Examples:
        float x, y, z;
        int a = 10, b = 20;
        float3 position, normal, tangent;
        const float x = 1.0f, y = 2.0f;
    """
    type_name: str = None  # OpenCL type name (shared by all declarators)
    declarators: List['Declaration'] = None  # List of Declaration nodes (name + initializer)
    qualifiers: List[str] = None  # ["const"] etc. (shared by all declarators)

    def __post_init__(self):
        # Ensure qualifiers is a list
        if self.qualifiers is None:
            object.__setattr__(self, 'qualifiers', [])


@dataclass(frozen=True)
class ReturnStatement(TransformedNode):
    """
    Return statement.

    Examples:
        return x;
        return;
    """
    value: Optional[TransformedNode] = None


@dataclass(frozen=True)
class IfStatement(TransformedNode):
    """
    If statement (with optional else).

    Examples:
        if (condition) { ... }
        if (x > 0) { ... } else { ... }
    """
    condition: TransformedNode = None
    then_block: 'CompoundStatement' = None
    else_block: Optional['CompoundStatement'] = None


@dataclass(frozen=True)
class ForStatement(TransformedNode):
    """
    For loop.

    Examples:
        for (int i = 0; i < 10; i++) { ... }
    """
    init: Optional[TransformedNode] = None
    condition: Optional[TransformedNode] = None
    update: Optional[TransformedNode] = None
    body: 'CompoundStatement' = None


@dataclass(frozen=True)
class WhileStatement(TransformedNode):
    """
    While loop.

    Examples:
        while (condition) { ... }
    """
    condition: TransformedNode = None
    body: 'CompoundStatement' = None


@dataclass(frozen=True)
class DoWhileStatement(TransformedNode):
    """
    Do-while loop.

    Examples:
        do { ... } while (condition);
    """
    body: 'CompoundStatement' = None
    condition: TransformedNode = None


@dataclass(frozen=True)
class BreakStatement(TransformedNode):
    """
    Break statement (exit loop early).

    Examples:
        break;
    """
    pass


@dataclass(frozen=True)
class ContinueStatement(TransformedNode):
    """
    Continue statement (skip to next loop iteration).

    Examples:
        continue;
    """
    pass


@dataclass(frozen=True)
class CompoundStatement(TransformedNode):
    """
    Block of statements.

    Examples:
        { stmt1; stmt2; stmt3; }
    """
    statements: List[TransformedNode] = None


# ============================================================================
# Functions
# ============================================================================

@dataclass(frozen=True)
class Parameter(TransformedNode):
    """
    Function parameter.

    Examples:
        float x
        __global float* output
        const float3 position
        __private float2* out_param  (for out/inout parameters)
        __private mat3 result  (synthetic result param for mat3-returning functions)
    """
    type_name: str = None  # OpenCL type name
    name: str = None
    qualifiers: List[str] = None  # ["const", "__global", etc.]
    is_pointer: bool = False  # True for out/inout parameters (except mat3)
    is_result_param: bool = False  # True if synthetic out-param for mat3 return

    def __post_init__(self):
        # Ensure qualifiers is a list
        if self.qualifiers is None:
            object.__setattr__(self, 'qualifiers', [])


@dataclass(frozen=True)
class FunctionDefinition(TransformedNode):
    """
    Function definition.

    Examples:
        void compute(float x) { ... }
        float3 transform(float3 v) { ... }
    """
    return_type: str = None  # OpenCL type name
    name: str = None
    parameters: List[Parameter] = None
    body: CompoundStatement = None


# ============================================================================
# Structs
# ============================================================================

@dataclass(frozen=True)
class StructField(TransformedNode):
    """
    Struct field declaration.

    Represents one or more fields of the same type in a struct.
    GLSL allows comma-separated field names: float a, b, c;

    Examples:
        vec3 pos;           (single field)
        float t, d;         (multiple fields of same type)
        matrix3x3 transform; (matrix field - TODO: requires special handling)

    Note: Matrix types in structs are NOT YET IMPLEMENTED.
    See .agent/PROGRESS.md for matrix-in-struct implementation plan.
    """
    type_name: str = None  # OpenCL type name (float, float3, etc.)
    names: List[str] = None  # Field name(s) - supports comma-separated declarations


@dataclass(frozen=True)
class StructDefinition(TransformedNode):
    """
    Struct type definition.

    Transformed to OpenCL typedef struct syntax.

    Examples GLSL:
        struct Geo {
            vec3 pos;
            vec3 scale;
            vec3 rotation;
        };

    Examples OpenCL:
        typedef struct {
            float3 pos;
            float3 scale;
            float3 rotation;
        } Geo;

    Note: Structs containing matrix types require special handling.
    Matrix fields should work but have not been extensively tested.
    See matrix_types.h and matrix_ops.h for matrix struct definitions.
    """
    name: str = None  # Struct type name
    fields: List[StructField] = None  # List of field declarations


# ============================================================================
# Top-Level
# ============================================================================

@dataclass(frozen=True)
class PreprocessorDirective(TransformedNode):
    """
    Preprocessor directive (pass-through).

    Preprocessor directives are already transformed by PreprocessorTransformer
    before AST parsing, so we just pass them through as raw text.

    Examples:
        #define PI 3.14159265f
        #define random(x) GLSL_fract(1e4f*GLSL_sin(x))
        #ifdef SOME_FLAG
        #endif
    """
    text: str = None  # Raw directive text (already transformed)


@dataclass(frozen=True)
class TranslationUnit(TransformedNode):
    """
    Root node representing entire shader/program.

    Contains all top-level declarations (functions, global variables, etc.).
    """
    declarations: List[TransformedNode] = None
