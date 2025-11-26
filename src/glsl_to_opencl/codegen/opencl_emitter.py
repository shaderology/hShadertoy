"""
Production OpenCL Code Generator - Session 6.

Generates production-quality OpenCL C code from Transformed AST.

This is the production code generator that replaces the temporary
code_emitter.py from Sessions 1-5. It generates clean, readable,
correctly formatted OpenCL code.

Key Features:
- Proper operator precedence handling
- Clean indentation and formatting
- Correct prefix/postfix unary operators
- mat3 out-parameter pattern support
- Production-quality code generation

Design:
- Visitor pattern over transformed AST (IR nodes)
- Configurable formatting options
- Maintains indentation state
- Handles all node types from Sessions 1-5
"""

from typing import List, Optional, Set
# Import transformed_ast module directly (not through __init__.py) to avoid circular import
import src.glsl_to_opencl.transformer.transformed_ast as IR


# Operator precedence levels (higher = tighter binding)
PRECEDENCE = {
    # Multiplicative
    '*': 13, '/': 13, '%': 13,
    # Additive
    '+': 12, '-': 12,
    # Shift
    '<<': 11, '>>': 11,
    # Relational
    '<': 10, '<=': 10, '>': 10, '>=': 10,
    # Equality
    '==': 9, '!=': 9,
    # Bitwise AND
    '&': 8,
    # Bitwise XOR
    '^': 7,
    # Bitwise OR
    '|': 6,
    # Logical AND
    '&&': 5,
    # Logical OR
    '||': 4,
    # Ternary (handled separately)
    '?:': 3,
    # Assignment
    '=': 2, '+=': 2, '-=': 2, '*=': 2, '/=': 2, '%=': 2,
    '&=': 2, '^=': 2, '|=': 2, '<<=': 2, '>>=': 2,
    # Comma (lowest precedence)
    ',': 1,
}


class OpenCLEmitter:
    """
    Production OpenCL code generator.

    Generates clean, readable, correctly formatted OpenCL C code from
    the transformed AST (IR nodes).

    Usage:
        emitter = OpenCLEmitter()
        opencl_code = emitter.emit(transformed_ast)

    Configuration:
        indent_size: Number of spaces per indentation level (default: 4)
    """

    def __init__(self, indent_size: int = 4):
        """
        Initialize the code generator.

        Args:
            indent_size: Number of spaces per indentation level
        """
        self.indent_size = indent_size
        self.indent_level = 0

    def indent(self) -> str:
        """Get current indentation string."""
        return ' ' * (self.indent_level * self.indent_size)

    def emit(self, node: IR.TransformedNode, parent_precedence: int = 0) -> str:
        """
        Emit OpenCL code for a transformed AST node.

        Args:
            node: Transformed AST node (IR node)
            parent_precedence: Precedence of parent operator (for parentheses)

        Returns:
            OpenCL C code string
        """
        if node is None:
            return ""

        # Dispatch to appropriate emit method
        method_name = f'emit_{node.__class__.__name__}'
        method = getattr(self, method_name, self.emit_generic)

        # Pass parent_precedence to methods that need it (BinaryOp, UnaryOp, etc.)
        if method_name in ['emit_BinaryOp', 'emit_TernaryOp']:
            return method(node, parent_precedence)
        else:
            return method(node)

    def emit_generic(self, node: IR.TransformedNode) -> str:
        """Fallback for unknown node types."""
        raise NotImplementedError(
            f"No emit method for {node.__class__.__name__}. "
            f"Node: {node}"
        )

    # ========================================================================
    # Literals
    # ========================================================================

    def emit_FloatLiteral(self, node: IR.FloatLiteral) -> str:
        """Emit float literal with 'f' suffix."""
        return node.value

    def emit_IntLiteral(self, node: IR.IntLiteral) -> str:
        """Emit integer literal."""
        return node.value

    def emit_BoolLiteral(self, node: IR.BoolLiteral) -> str:
        """Emit boolean literal."""
        return 'true' if node.value else 'false'

    # ========================================================================
    # Identifiers and Types
    # ========================================================================

    def emit_Identifier(self, node: IR.Identifier) -> str:
        """Emit identifier (variable name)."""
        return node.name

    def emit_TypeName(self, node: IR.TypeName) -> str:
        """Emit type name."""
        return node.name

    # ========================================================================
    # Expressions
    # ========================================================================

    def emit_BinaryOp(self, node: IR.BinaryOp, parent_precedence: int = 0) -> str:
        """
        Emit binary operation with proper precedence handling.

        Adds parentheses only when necessary based on operator precedence.

        Args:
            node: BinaryOp node
            parent_precedence: Precedence of parent operator

        Returns:
            Binary operation code with parentheses if needed
        """
        # Get operator precedence
        op_precedence = PRECEDENCE.get(node.operator, 0)

        # Emit operands with current precedence
        left = self.emit(node.left, op_precedence)
        right = self.emit(node.right, op_precedence)

        # Generate code
        code = f"{left} {node.operator} {right}"

        # Add parentheses if this operator has lower precedence than parent
        if op_precedence < parent_precedence:
            code = f"({code})"

        return code

    def emit_UnaryOp(self, node: IR.UnaryOp) -> str:
        """
        Emit unary operation.

        Handles both prefix and postfix operators correctly.
        Note: The transformer should already have marked prefix vs postfix.
        For now, we assume all are prefix (standard C behavior).

        Args:
            node: UnaryOp node

        Returns:
            Unary operation code
        """
        operand = self.emit(node.operand)

        # Prefix operators: ++x, --x, -x, +x, !x, ~x
        # (Postfix would be: x++, x--, but transformer marks these as prefix)
        return f"{node.operator}{operand}"

    def emit_ParenthesizedExpression(self, node: IR.ParenthesizedExpression) -> str:
        """
        Emit parenthesized expression.

        Always emits parentheses to preserve the explicit grouping from
        the source code. This is critical for maintaining order of operations.

        Args:
            node: ParenthesizedExpression node

        Returns:
            Expression wrapped in parentheses
        """
        # Always emit parentheses, regardless of operator precedence
        # This preserves the programmer's explicit intent
        inner = self.emit(node.expression)
        return f"({inner})"

    def emit_CallExpression(self, node: IR.CallExpression) -> str:
        """
        Emit function call.

        Args:
            node: CallExpression node

        Returns:
            Function call code
        """
        args = ', '.join(self.emit(arg) for arg in node.arguments)
        return f"{node.function}({args})"

    def emit_TypeConstructor(self, node: IR.TypeConstructor) -> str:
        """
        Emit type constructor.

        Special cases:
        - Struct constructors: Use compound literal syntax { arg1, arg2, ... }
        - Vector constructors: Use cast syntax (float2)(arg1, arg2)

        Args:
            node: TypeConstructor node

        Returns:
            Type constructor code
        """
        # Check if this is a struct constructor (type_name not in standard type_map)
        # Struct types don't use cast syntax, they use compound literal syntax
        standard_types = {
            'float', 'float2', 'float3', 'float4',
            'int', 'int2', 'int3', 'int4',
            'uint', 'uint2', 'uint3', 'uint4',
            'matrix2x2', 'matrix3x3', 'matrix4x4'
        }

        if node.type_name not in standard_types:
            # This is likely a struct constructor
            # Emit as compound literal: { arg1, arg2, arg3 }
            args = ', '.join(self.emit(arg) for arg in node.arguments)
            return f"{{{args}}}"

        # Standard cast syntax for vectors and standard types
        args = ', '.join(self.emit(arg) for arg in node.arguments)
        return f"({node.type_name})({args})"

    def emit_ArrayInitializer(self, node: IR.ArrayInitializer) -> str:
        """
        Emit array initializer with curly braces.

        Args:
            node: ArrayInitializer node

        Returns:
            Array initializer code in the form {element1, element2, ...}

        Examples:
            {0.0f}
            {(float3)(0.0f)}
            {1.0f, 2.0f, 3.0f}
        """
        elements = ', '.join(self.emit(elem) for elem in node.elements)
        return f"{{{elements}}}"

    def emit_MemberAccess(self, node: IR.MemberAccess) -> str:
        """
        Emit member access (swizzling, struct fields).

        Args:
            node: MemberAccess node

        Returns:
            Member access code
        """
        base = self.emit(node.base)
        return f"{base}.{node.member}"

    def emit_ArrayAccess(self, node: IR.ArrayAccess) -> str:
        """
        Emit array subscript.

        Args:
            node: ArrayAccess node

        Returns:
            Array access code
        """
        base = self.emit(node.base)
        index = self.emit(node.index)
        return f"{base}[{index}]"

    def emit_TernaryOp(self, node: IR.TernaryOp, parent_precedence: int = 0) -> str:
        """
        Emit ternary conditional operator.

        Args:
            node: TernaryOp node
            parent_precedence: Precedence of parent operator

        Returns:
            Ternary operator code with parentheses if needed
        """
        # Ternary has low precedence
        ternary_precedence = PRECEDENCE['?:']

        condition = self.emit(node.condition, ternary_precedence)
        true_expr = self.emit(node.true_expr, ternary_precedence)
        false_expr = self.emit(node.false_expr, ternary_precedence)

        code = f"{condition} ? {true_expr} : {false_expr}"

        # Add parentheses if needed
        if ternary_precedence < parent_precedence:
            code = f"({code})"

        return code

    def emit_AssignmentOp(self, node: IR.AssignmentOp) -> str:
        """
        Emit assignment expression.

        Args:
            node: AssignmentOp node

        Returns:
            Assignment code
        """
        target = self.emit(node.target)
        value = self.emit(node.value)
        return f"{target} {node.operator} {value}"

    # ========================================================================
    # Statements
    # ========================================================================

    def emit_ExpressionStatement(self, node: IR.ExpressionStatement) -> str:
        """
        Emit expression statement.

        Args:
            node: ExpressionStatement node

        Returns:
            Expression statement with semicolon and newline
        """
        expr = self.emit(node.expression)
        return f"{self.indent()}{expr};\n"

    def emit_Declaration(self, node: IR.Declaration) -> str:
        """
        Emit variable declaration.

        Args:
            node: Declaration node

        Returns:
            Declaration code
        """
        # Standard declaration for all types
        # Emit qualifiers if present
        qualifier_str = ' '.join(node.qualifiers) + ' ' if node.qualifiers else ''
        result = f"{self.indent()}{qualifier_str}{node.type_name} {node.name}"
        if node.initializer:
            init = self.emit(node.initializer)
            result += f" = {init}"
        result += ";\n"
        return result

    def emit_DeclarationList(self, node: IR.DeclarationList) -> str:
        """
        Emit comma-separated variable declarations.

        Examples:
            float x, y, z;
            int a = 10, b = 20;
            float3 position, normal, tangent;

        Note: mat3 declarations with special initializers are NOT
        supported in comma-separated form (they require splitting).

        Args:
            node: DeclarationList node

        Returns:
            Comma-separated declaration statement
        """
        # Check for mat3 with special initializers - not supported in comma form
        if node.type_name == 'mat3':
            for decl in node.declarators:
                if decl.initializer and isinstance(decl.initializer, IR.CallExpression):
                    func_name = decl.initializer.function
                    if (func_name.startswith('GLSL_mat3_diagonal') or
                        func_name.startswith('GLSL_mat3_from_') or
                        func_name == 'GLSL_transpose' or
                        func_name == 'GLSL_inverse' or
                        func_name == 'GLSL_mul'):
                        raise ValueError(
                            f"mat3 with special initializer ({func_name}) "
                            "cannot be used in comma-separated declarations"
                        )

        # Build comma-separated list of declarators
        declarator_parts = []
        for decl in node.declarators:
            part = decl.name
            if decl.initializer:
                init = self.emit(decl.initializer)
                part += f" = {init}"
            declarator_parts.append(part)

        # Emit as single line: type name1, name2, name3;
        # Emit qualifiers if present
        qualifier_str = ' '.join(node.qualifiers) + ' ' if node.qualifiers else ''
        result = f"{self.indent()}{qualifier_str}{node.type_name} {', '.join(declarator_parts)};\n"
        return result

    def _get_node_type(self, node: IR.TransformedNode) -> Optional[str]:
        """
        Get the type name of a node.

        Args:
            node: Transformed AST node

        Returns:
            Type name string (e.g., 'mat3', 'float3') or None
        """
        if not hasattr(node, 'glsl_type') or not node.glsl_type:
            return None

        # Try GLSLType.name first
        if hasattr(node.glsl_type, 'name'):
            if node.glsl_type.name is not None:
                return node.glsl_type.name

        # Fall back to str(glsl_type) for CallExpression nodes
        type_str = str(node.glsl_type)
        if type_str and not type_str.startswith('<'):
            return type_str

        return None

    def emit_ReturnStatement(self, node: IR.ReturnStatement) -> str:
        """
        Emit return statement.

        Args:
            node: ReturnStatement node

        Returns:
            Return statement with semicolon and newline
        """
        if node.value:
            value = self.emit(node.value)
            return f"{self.indent()}return {value};\n"
        else:
            return f"{self.indent()}return;\n"

    def emit_IfStatement(self, node: IR.IfStatement) -> str:
        """
        Emit if statement.

        Args:
            node: IfStatement node

        Returns:
            Complete if/else statement
        """
        condition = self.emit(node.condition)
        result = f"{self.indent()}if ({condition}) "

        # Then block
        then_code = self.emit(node.then_block)
        result += then_code

        # Else block
        if node.else_block:
            # Check if else block is another if statement (for else-if chains)
            if isinstance(node.else_block, IR.IfStatement):
                # Emit "else if" on same line without extra indentation
                result += f"{self.indent()}else "
                # The if statement will add its own condition and blocks
                else_code = self.emit(node.else_block)
                # Remove the indentation that IfStatement adds at the start
                else_code_stripped = else_code.lstrip()
                result += else_code_stripped
            else:
                # Regular else block (compound statement)
                result += f"{self.indent()}else "
                else_code = self.emit(node.else_block)
                result += else_code

        return result

    def emit_ForStatement(self, node: IR.ForStatement) -> str:
        """
        Emit for loop.

        Args:
            node: ForStatement node

        Returns:
            Complete for loop
        """
        # Init
        init = ""
        if node.init:
            if isinstance(node.init, IR.Declaration):
                # Declaration: emit without indentation or newline
                init = f"{node.init.type_name} {node.init.name}"
                if node.init.initializer:
                    init += f" = {self.emit(node.init.initializer)}"
            else:
                # Expression: emit without indentation
                init = self.emit(node.init)

        # Condition
        condition = self.emit(node.condition) if node.condition else ""

        # Update
        update = self.emit(node.update) if node.update else ""

        result = f"{self.indent()}for ({init}; {condition}; {update}) "

        # Body
        body_code = self.emit(node.body)
        result += body_code

        return result

    def emit_WhileStatement(self, node: IR.WhileStatement) -> str:
        """
        Emit while loop.

        Args:
            node: WhileStatement node

        Returns:
            Complete while loop
        """
        condition = self.emit(node.condition)
        result = f"{self.indent()}while ({condition}) "

        # Body
        body_code = self.emit(node.body)
        result += body_code

        return result

    def emit_DoWhileStatement(self, node: IR.DoWhileStatement) -> str:
        """
        Emit do-while loop.

        Args:
            node: DoWhileStatement node

        Returns:
            Complete do-while loop
        """
        result = f"{self.indent()}do "

        # Body
        body_code = self.emit(node.body)
        result += body_code

        # Condition
        condition = self.emit(node.condition)
        result += f" while ({condition});\n"

        return result

    def emit_BreakStatement(self, node: IR.BreakStatement) -> str:
        """
        Emit break statement.

        Args:
            node: BreakStatement node

        Returns:
            Break statement with semicolon and newline
        """
        return f"{self.indent()}break;\n"

    def emit_ContinueStatement(self, node: IR.ContinueStatement) -> str:
        """
        Emit continue statement.

        Args:
            node: ContinueStatement node

        Returns:
            Continue statement with semicolon and newline
        """
        return f"{self.indent()}continue;\n"

    def emit_CompoundStatement(self, node: IR.CompoundStatement) -> str:
        """
        Emit compound statement (block).

        Args:
            node: CompoundStatement node

        Returns:
            Block with proper indentation
        """
        result = "{\n"
        self.indent_level += 1

        for stmt in node.statements:
            result += self.emit(stmt)

        self.indent_level -= 1
        result += f"{self.indent()}}}\n"

        return result

    # ========================================================================
    # Structs
    # ========================================================================

    def emit_StructField(self, node: IR.StructField) -> str:
        """
        Emit struct field declaration.

        Handles comma-separated field names:
        - Single field: float3 pos;
        - Multiple fields: float t, d;

        Args:
            node: StructField node

        Returns:
            Field declaration string with proper indentation
        """
        field_names = ', '.join(node.names)
        return f"{self.indent()}{node.type_name} {field_names};\n"

    def emit_StructDefinition(self, node: IR.StructDefinition) -> str:
        """
        Emit struct definition using OpenCL typedef struct syntax.

        GLSL:
            struct Geo {
                vec3 pos;
                vec3 scale;
                vec3 rotation;
            };

        OpenCL:
            typedef struct {
                float3 pos;
                float3 scale;
                float3 rotation;
            } Geo;

        Args:
            node: StructDefinition node

        Returns:
            Complete struct definition
        """
        result = "typedef struct {\n"
        self.indent_level += 1

        # Emit all fields
        for field in node.fields:
            result += self.emit(field)

        self.indent_level -= 1
        result += f"}} {node.name};\n"

        return result

    # ========================================================================
    # Functions
    # ========================================================================

    def emit_Parameter(self, node: IR.Parameter) -> str:
        """
        Emit function parameter.

        Handles GLSL out/inout parameters by adding pointer syntax:
        - out vec2 v -> __private float2* v
        - out mat3 m -> __private mat3 m (no pointer, mat3 is array type)

        Args:
            node: Parameter node

        Returns:
            Parameter declaration
        """
        parts = []

        # Qualifiers (const, __private, etc.)
        if node.qualifiers:
            parts.extend(node.qualifiers)

        # Type with optional pointer
        if node.is_pointer:
            # Add pointer after type (e.g., float2*)
            parts.append(node.type_name + '*')
        else:
            parts.append(node.type_name)

        # Name
        parts.append(node.name)

        return ' '.join(parts)

    def emit_FunctionDefinition(self, node: IR.FunctionDefinition) -> str:
        """
        Emit function definition.

        Args:
            node: FunctionDefinition node

        Returns:
            Complete function definition
        """
        # Function signature
        params = ', '.join(self.emit(p) for p in node.parameters)
        result = f"{node.return_type} {node.name}({params}) "

        # Function body
        body = self.emit(node.body)
        result += body

        return result

    # ========================================================================
    # Preprocessor Directives (Session 9)
    # ========================================================================

    def emit_PreprocessorDirective(self, node: IR.PreprocessorDirective) -> str:
        """
        Emit preprocessor directive.

        Preprocessor directives are already transformed, so just output the text as-is.

        Args:
            node: PreprocessorDirective node

        Returns:
            Preprocessor directive text with newline
        """
        return f"{node.text}\n"

    # ========================================================================
    # Top-Level
    # ========================================================================

    def emit_TranslationUnit(self, node: IR.TranslationUnit) -> str:
        """
        Emit translation unit (entire program).

        Args:
            node: TranslationUnit node

        Returns:
            Complete OpenCL program
        """
        result = ""

        for decl in node.declarations:
            result += self.emit(decl)
            result += "\n"  # Blank line between top-level declarations

        return result
