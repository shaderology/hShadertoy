"""
Simple Code Emitter - Generates OpenCL C from Transformed AST.

This is a TEMPORARY emitter used for testing during Session 1-5.
The full CodeGenerator will be rewritten in Session 6.

Purpose:
- Enable compilation testing of transformed AST
- Verify transformations produce valid OpenCL
- Simple, straightforward emission (no optimization)

Design:
- Visitor pattern over transformed AST
- Generates formatted OpenCL C code
- Maintains indentation for readability
"""

from typing import List
from . import transformed_ast as IR


# TypeConstructor type names that use OpenCL cast syntax (T)(args). Anything
# else is a user struct constructor (brace list / compound literal) — except
# the legacy GLSL matrix names, which keep their historical brace emission.
STANDARD_CTOR_TYPES = {
    'float', 'float2', 'float3', 'float4',
    'int', 'int2', 'int3', 'int4',
    'uint', 'uint2', 'uint3', 'uint4',
    'matrix2x2', 'matrix3x3', 'matrix4x4'
}
MATRIX_CTOR_TYPES = {'mat2', 'mat3', 'mat4'}


class CodeEmitter:
    """
    Emits OpenCL C code from transformed AST.

    This is a simple emitter for testing. The full CodeGenerator
    in Session 6 will be more sophisticated.
    """

    def __init__(self, indent_size: int = 4):
        """
        Initialize emitter.

        Args:
            indent_size: Number of spaces per indentation level
        """
        self.indent_size = indent_size
        self.indent_level = 0

    def emit(self, node: IR.TransformedNode) -> str:
        """
        Emit OpenCL code for a transformed AST node.

        Args:
            node: Transformed AST node

        Returns:
            OpenCL C code string
        """
        # Dispatch to appropriate emit method
        method_name = f'emit_{node.__class__.__name__}'
        method = getattr(self, method_name, self.emit_generic)
        return method(node)

    def emit_generic(self, node: IR.TransformedNode) -> str:
        """Fallback for unknown node types."""
        raise NotImplementedError(
            f"No emit method for {node.__class__.__name__}"
        )

    def indent(self) -> str:
        """Get current indentation string."""
        return ' ' * (self.indent_level * self.indent_size)

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
    # Identifiers
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

    def emit_BinaryOp(self, node: IR.BinaryOp) -> str:
        """Emit binary operation."""
        left = self.emit(node.left)
        right = self.emit(node.right)
        return f"{left} {node.operator} {right}"

    def emit_UnaryOp(self, node: IR.UnaryOp) -> str:
        """Emit unary operation."""
        operand = self.emit(node.operand)

        # A '+'/'-' operator over an operand whose emitted form starts with
        # '+'/'-' must keep a space, else the two glue into '--'/'++' which C
        # lexes as inc/decrement ("expression is not assignable").
        sep = ' ' if (node.operator[-1] in '+-' and operand[:1] in '+-') else ''
        return f"{node.operator}{sep}{operand}"

    def emit_ParenthesizedExpression(self, node: IR.ParenthesizedExpression) -> str:
        """Emit parenthesized expression - always emit parentheses."""
        inner = self.emit(node.expression)
        return f"({inner})"

    def emit_CommaExpression(self, node: IR.CommaExpression) -> str:
        """Emit a comma (sequence) expression `a, b` (parens come from the
        enclosing ParenthesizedExpression)."""
        left = self.emit(node.left)
        right = self.emit(node.right)
        return f"{left}, {right}"

    def emit_CallExpression(self, node: IR.CallExpression) -> str:
        """Emit function call."""
        args = ', '.join(self.emit(arg) for arg in node.arguments)
        return f"{node.function}({args})"

    def emit_TypeConstructor(self, node: IR.TypeConstructor) -> str:
        """
        Emit type constructor.

        Special cases:
        - Struct constructors: Use compound literal syntax { arg1, arg2, ... }
        - mat3 full constructor: { (float3)(...), (float3)(...), (float3)(...) }
        - Vector constructors: Use cast syntax (float2)(arg1, arg2)
        """
        # Struct constructors: C99 compound literal ((S){args}) — valid in
        # every expression position. Declaration initializers instead go
        # through _emit_initializer, which keeps the plain brace list.
        if self._is_struct_ctor(node):
            return f"(({node.type_name}){self._braced_args(node.arguments)})"

        # Legacy matrix brace path (matrices are struct/array types).
        if node.type_name in MATRIX_CTOR_TYPES:
            args = ', '.join(self.emit(arg) for arg in node.arguments)
            return f"{{{args}}}"

        # Standard cast syntax for vectors and standard types
        args = ', '.join(self.emit(arg) for arg in node.arguments)
        return f"({node.type_name})({args})"

    def _is_struct_ctor(self, node) -> bool:
        """True for TypeConstructors of user struct types (category K)."""
        return (isinstance(node, IR.TypeConstructor)
                and node.type_name not in STANDARD_CTOR_TYPES
                and node.type_name not in MATRIX_CTOR_TYPES)

    def _braced_args(self, args) -> str:
        """Emit constructor arguments as a brace list; nested aggregate
        constructors (arrays / structs) recurse into nested brace lists,
        as C initializer syntax requires."""
        return '{' + ', '.join(self._init_item(a) for a in (args or [])) + '}'

    def _init_item(self, node) -> str:
        if isinstance(node, IR.ArrayConstructor) or self._is_struct_ctor(node):
            return self._braced_args(node.arguments)
        return self.emit(node)

    def _emit_initializer(self, node) -> str:
        """Emit a declaration initializer. Aggregate constructors (array or
        struct) become plain brace lists — the only form valid for array
        declarations in C, and the pre-existing form for struct ones."""
        if isinstance(node, IR.ArrayConstructor) or self._is_struct_ctor(node):
            return self._braced_args(node.arguments)
        return self.emit(node)

    def emit_ArrayConstructor(self, node: IR.ArrayConstructor) -> str:
        """
        Emit a GLSL array constructor in expression position as an unsized
        compound literal: float[4](a, b, c, d) -> ((float[]){a, b, c, d}).
        (Declaration initializers go through _emit_initializer instead.)
        """
        return f"(({node.element_type}[]){self._braced_args(node.arguments)})"

    def emit_ArrayInitializer(self, node: IR.ArrayInitializer) -> str:
        """
        Emit array initializer with curly braces.

        Examples:
            {0.0f}
            {(float3)(0.0f)}
            {1.0f, 2.0f, 3.0f}
        """
        elements = ', '.join(self.emit(elem) for elem in node.elements)
        return f"{{{elements}}}"

    def emit_MemberAccess(self, node: IR.MemberAccess) -> str:
        """Emit member access (swizzling)."""
        base = self.emit(node.base)
        # A unary base (e.g. a dereferenced pointer param *p) needs parens so
        # member/swizzle binds to the deref: (*p).x, not *p.x = *(p.x).
        # (Mirrors codegen/opencl_emitter.py.)
        if isinstance(node.base, IR.UnaryOp):
            base = f"({base})"
        return f"{base}.{node.member}"

    def emit_ArrayAccess(self, node: IR.ArrayAccess) -> str:
        """Emit array subscript."""
        base = self.emit(node.base)
        index = self.emit(node.index)
        return f"{base}[{index}]"

    def emit_TernaryOp(self, node: IR.TernaryOp) -> str:
        """Emit ternary conditional."""
        condition = self.emit(node.condition)
        true_expr = self._emit_ternary_branch(node.true_expr)
        false_expr = self._emit_ternary_branch(node.false_expr)
        return f"{condition} ? {true_expr} : {false_expr}"

    def _emit_ternary_branch(self, node: IR.TransformedNode) -> str:
        """Emit a ternary operand, parenthesizing an assignment branch so
        C/OpenCL keeps GLSL's `cond ? a=b : c=d` grouping (else it reparses as
        `(cond ? a=b : c) = d` — "expression is not assignable")."""
        code = self.emit(node)
        if isinstance(node, IR.AssignmentOp):
            code = f"({code})"
        return code

    def emit_AssignmentOp(self, node: IR.AssignmentOp) -> str:
        """Emit assignment."""
        target = self.emit(node.target)
        value = self.emit(node.value)
        return f"{target} {node.operator} {value}"

    # ========================================================================
    # Statements
    # ========================================================================

    def emit_ExpressionStatement(self, node: IR.ExpressionStatement) -> str:
        """Emit expression statement."""
        expr = self.emit(node.expression)
        return f"{self.indent()}{expr};\n"

    def emit_Declaration(self, node: IR.Declaration) -> str:
        """
        Emit variable declaration.

        Special handling for mat3:
        - mat3 with diagonal/cast/mul initializers must split into declaration + call
        """
        # Check if this is a mat3 with special initializer that needs splitting
        if node.type_name == 'mat3' and node.initializer:
            if isinstance(node.initializer, IR.CallExpression):
                func_name = node.initializer.function
                # Functions that require out-parameter pattern for mat3
                needs_split = (
                    func_name.startswith('GLSL_mat3_diagonal') or
                    func_name.startswith('GLSL_mat3_from_') or
                    func_name == 'GLSL_transpose' or
                    func_name == 'GLSL_inverse' or
                    (func_name == 'GLSL_mul' and self._returns_mat3(node.initializer)) or
                    func_name in self.mat3_return_functions  # User-defined mat3-returning functions
                )

                if needs_split:
                    # Split into declaration + function call with out-param
                    # Emit qualifiers if present
                    qualifier_str = ' '.join(node.qualifiers) + ' ' if node.qualifiers else ''
                    result = f"{self.indent()}{qualifier_str}{node.type_name} {node.name};\n"

                    # Emit function call with out-parameter
                    # Build argument list (original args + result parameter)
                    arg_parts = [self.emit(arg) for arg in node.initializer.arguments]
                    arg_parts.append(node.name)  # mat3 is array type, pass directly
                    all_args = ', '.join(arg_parts)
                    result += f"{self.indent()}{func_name}({all_args});\n"
                    return result

        # Standard declaration (all other cases)
        # Emit qualifiers if present
        qualifier_str = ' '.join(node.qualifiers) + ' ' if node.qualifiers else ''
        result = f"{self.indent()}{qualifier_str}{node.type_name} {node.name}"
        if node.initializer:
            init = self._emit_initializer(node.initializer)
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
            const float x = 1.0f, y = 2.0f;

        Note: mat3 declarations with special initializers are NOT
        supported in comma-separated form (they require splitting).
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
                init = self._emit_initializer(decl.initializer)
                part += f" = {init}"
            declarator_parts.append(part)

        # Emit as single line: type name1, name2, name3;
        # Emit qualifiers if present
        qualifier_str = ' '.join(node.qualifiers) + ' ' if node.qualifiers else ''
        result = f"{self.indent()}{qualifier_str}{node.type_name} {', '.join(declarator_parts)};\n"
        return result

    def _returns_mat3(self, call_node: IR.CallExpression) -> bool:
        """
        Check if a function call returns mat3 (requires out-parameter).

        Args:
            call_node: CallExpression node

        Returns:
            True if the call returns mat3
        """
        result_type = self._get_node_type(call_node)
        return result_type == 'mat3'

    def _is_mat3_mul(self, call_node: IR.CallExpression) -> bool:
        """
        Check if a GLSL_mul call involves mat3 (requires out-parameter).

        Args:
            call_node: CallExpression node for GLSL_mul

        Returns:
            True if this is a mat3*mat3 multiplication
        """
        if len(call_node.arguments) >= 2:
            first_arg = call_node.arguments[0]
            second_arg = call_node.arguments[1]

            # Check if either argument is mat3
            first_type = self._get_node_type(first_arg)
            second_type = self._get_node_type(second_arg)

            # mat3 * mat3 requires out-parameter
            return first_type == 'mat3' and second_type == 'mat3'

        return False

    def _get_node_type(self, node: IR.TransformedNode) -> str:
        """
        Get the type name of a node for emission.

        Args:
            node: Transformed AST node

        Returns:
            Type name string (e.g., 'mat3', 'float3')
        """
        if hasattr(node, 'glsl_type') and node.glsl_type:
            if hasattr(node.glsl_type, 'name'):
                # Only use .name if it's not None
                if node.glsl_type.name is not None:
                    return node.glsl_type.name
                # Fall through to str() if .name is None

            # Use GLSLType.__str__() representation
            type_str = str(node.glsl_type)
            if type_str and not type_str.startswith('<'):
                return type_str

        return None

    def emit_ReturnStatement(self, node: IR.ReturnStatement) -> str:
        """Emit return statement."""
        if node.value:
            value = self.emit(node.value)
            return f"{self.indent()}return {value};\n"
        else:
            return f"{self.indent()}return;\n"

    def emit_IfStatement(self, node: IR.IfStatement) -> str:
        """Emit if statement."""
        condition = self.emit(node.condition)
        result = f"{self.indent()}if ({condition}) "

        # Then block
        then_code = self.emit(node.then_block)
        result += then_code

        # Else block
        if node.else_block:
            result += f"{self.indent()}else "
            else_code = self.emit(node.else_block)
            result += else_code

        return result

    def emit_ForStatement(self, node: IR.ForStatement) -> str:
        """Emit for loop."""
        # Init
        init = ""
        if node.init:
            if isinstance(node.init, IR.Declaration):
                # Declaration: emit without newline
                init = f"{node.init.type_name} {node.init.name}"
                if node.init.initializer:
                    init += f" = {self.emit(node.init.initializer)}"
            elif isinstance(node.init, IR.DeclarationList):
                # DeclarationList: emit comma-separated declarations without newline
                declarator_parts = []
                for decl in node.init.declarators:
                    part = decl.name
                    if decl.initializer:
                        init_expr = self.emit(decl.initializer)
                        part += f" = {init_expr}"
                    declarator_parts.append(part)
                init = f"{node.init.type_name} {', '.join(declarator_parts)}"
            else:
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
        """Emit while loop."""
        condition = self.emit(node.condition)
        result = f"{self.indent()}while ({condition}) "

        # Body
        body_code = self.emit(node.body)
        result += body_code

        return result

    def emit_CompoundStatement(self, node: IR.CompoundStatement) -> str:
        """Emit block statement."""
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
        """
        # Build field declaration with proper indentation
        field_names = ', '.join(node.names)
        return f"{self.indent()}{node.type_name} {field_names};\n"

    def emit_StructDefinition(self, node: IR.StructDefinition) -> str:
        """
        Emit struct definition.

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
        """Emit function parameter with pointer syntax if needed."""
        parts = []

        # Qualifiers
        if node.qualifiers:
            parts.extend(node.qualifiers)

        # Type with optional pointer
        if node.is_pointer:
            parts.append(node.type_name + '*')
        else:
            parts.append(node.type_name)

        # Name (array params carry their bracket suffix: float3 pts[4]).
        # Unnamed prototype params (`float Fn(float3);`) emit the type alone.
        if node.array_suffix:
            parts.append(node.name + node.array_suffix)
        elif node.name:
            parts.append(node.name)

        return ' '.join(parts)

    def emit_FunctionDefinition(self, node: IR.FunctionDefinition) -> str:
        """Emit function definition."""
        # Signature
        params = ', '.join(self.emit(p) for p in node.parameters)

        # Prototype / forward declaration (category S): no body, emit `sig;`.
        if node.is_prototype:
            result = f"{node.return_type} {node.name}({params});"
            if node.overloadable:
                result = "__attribute__((overloadable))\n" + result
            return result

        result = f"{node.return_type} {node.name}({params}) "

        # OpenCL C has no overloading; user functions are marked overloadable so
        # same-named GLSL functions of different signatures compile.
        if node.overloadable:
            result = "__attribute__((overloadable))\n" + result

        # Body
        body = self.emit(node.body)
        result += body

        return result

    # ========================================================================
    # Top-Level
    # ========================================================================

    def emit_Comment(self, node: IR.Comment) -> str:
        """Emit a preserved file-scope comment verbatim (license/attribution)."""
        return f"{node.text}\n"

    def emit_TranslationUnit(self, node: IR.TranslationUnit) -> str:
        """Emit translation unit (entire program)."""
        result = ""

        for decl in node.declarations:
            result += self.emit(decl)
            result += "\n"  # Blank line between top-level declarations

        return result
