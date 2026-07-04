"""
AST Transformer - Converts GLSL AST to OpenCL Transformed AST.

This module implements the core transformation logic that converts
tree-sitter GLSL AST nodes into OpenCL-semantic transformed nodes.

Architecture:
    GLSL Source AST (tree-sitter) -> ASTTransformer -> Transformed AST (IR)

The transformer:
- Queries TypeChecker for type information
- Applies transformation rules (float suffixes, type names, etc.)
- Creates immutable transformed nodes
- Returns transformed tree ready for code emission

Design principles:
- Single-pass transformation (no re-parsing)
- Type-aware (uses TypeChecker)
- No byte offset tracking (works with tree structure)
- Systematic visitor pattern
"""

import re
from typing import Dict, Optional, List
from ..parser.ast_nodes import ASTNode
from ..analyzer.type_checker import TypeChecker, GLSLType, TYPE_NAME_MAP
from ..analyzer.symbol_table import SymbolTable
from . import transformed_ast as IR

# Import TYPE_NAME_MAP at module level for use in _transform_call_expression
# This avoids repeated imports inside the method

# Vector type name -> (element base, width), accepting BOTH name families:
# declarations record GLSL names ('vec3') in local_types while parameters
# record OpenCL names ('float3'). The 'bool' base marks bvec masks, whose
# OpenCL representation is an int vector (-1 for true from vector relational
# operators, where GLSL bvec->number conversion requires 1).
VECTOR_TYPE_INFO = {
    'vec2': ('float', 2), 'vec3': ('float', 3), 'vec4': ('float', 4),
    'float2': ('float', 2), 'float3': ('float', 3), 'float4': ('float', 4),
    'ivec2': ('int', 2), 'ivec3': ('int', 3), 'ivec4': ('int', 4),
    'int2': ('int', 2), 'int3': ('int', 3), 'int4': ('int', 4),
    'uvec2': ('uint', 2), 'uvec3': ('uint', 3), 'uvec4': ('uint', 4),
    'uint2': ('uint', 2), 'uint3': ('uint', 3), 'uint4': ('uint', 4),
    'bvec2': ('bool', 2), 'bvec3': ('bool', 3), 'bvec4': ('bool', 4),
}

# OpenCL vector name -> GLSL name, for TYPE_NAME_MAP lookups on types that
# came from parameter registration (see VECTOR_TYPE_INFO note above).
OPENCL_TO_GLSL_NAME = {
    'float2': 'vec2', 'float3': 'vec3', 'float4': 'vec4',
    'int2': 'ivec2', 'int3': 'ivec3', 'int4': 'ivec4',
    'uint2': 'uvec2', 'uint3': 'uvec3', 'uint4': 'uvec4',
}


class TransformationError(Exception):
    """Raised when transformation fails."""
    def __init__(self, message: str, location: Optional[tuple] = None):
        self.message = message
        self.location = location
        if location:
            line, col = location
            super().__init__(f"{message} at line {line+1}, column {col+1}")
        else:
            super().__init__(message)


class ASTTransformer:
    """
    Transforms GLSL AST to OpenCL Transformed AST.

    The transformer applies systematic transformations:
    1. Float literal suffixes (1.0 -> 1.0f)
    2. Type name conversions (vec2 -> float2)
    3. Function name transformations (mod -> GLSL_mod)
    4. Type constructors (vec2(...) -> (float2)(...))
    5. Matrix operations (M * v -> GLSL_mul(M, v))

    Usage:
        symbol_table = create_builtin_symbol_table()
        type_checker = TypeChecker(symbol_table)
        transformer = ASTTransformer(type_checker)
        transformed_ast = transformer.transform(glsl_ast)
    """

    def __init__(self, type_checker: TypeChecker):
        """
        Initialize transformer.

        Args:
            type_checker: TypeChecker instance for type inference
        """
        self.type_checker = type_checker
        self.symbol_table = type_checker.symbol_table

        # Local type environment for tracking variable types during transformation
        # Maps variable name -> GLSL type name
        self.local_types = {}

        # Track pointer parameters in current function (for out/inout handling)
        # Set of parameter names that are pointers (need * dereference on assignment)
        self.pointer_params = set()

        # Name of the renderpass entry function (mainImage / mainCubemap /
        # mainVR / mainSound). Its out-param (fragColor) is wrapped as a pointer
        # for parsing, but the host (tests/transpile.py + Houdini @KERNEL)
        # provides it as a plain LOCAL in the final kernel, so it must NOT be
        # treated as a deref'able pointer. A HELPER that merely shares the name
        # (e.g. a user mainVR called from mainImage) keeps a real out-param.
        # Hosts may override per renderpass type.
        self.entry_function = 'mainImage'

        # Track function signatures for handling call sites with out parameters
        # Maps function name -> list of parameter info (name, is_pointer)
        self.function_signatures = {}

        # Register GLSL built-in functions with out parameters
        # modf(x, out i) - returns fractional part, stores integer part in i
        self.function_signatures['GLSL_modf'] = [
            ('x', False),  # input value
            ('i', True)    # output pointer (out parameter)
        ]

        # Track user-defined function return types for matrix operation detection
        # Maps function name -> GLSL type name (e.g., 'foo' -> 'mat2')
        # This is populated during transformation when function definitions are encountered
        self.user_function_return_types = {}

        # Track current function's return type during body transformation
        # Used to detect if return statements need special handling
        self.current_function_return_type = None

        # Category A — program-scope global initializer hoisting.
        # True only while transforming top-level (file-scope) declarations;
        # False inside function bodies. A global whose initializer is NOT a
        # compile-time constant (any arithmetic or call) is rejected by OpenCL
        # at program scope, so it is emitted as a bare declaration and its real
        # initializer is recorded here as (name, initializer_ir) for the host to
        # assign per-invocation at the top of the kernel body.
        self._global_scope = False
        self.hoisted_global_inits = []

        # Track user-defined struct types for constructor detection
        # Maps struct name -> dict of field info {'field_name': 'field_type', ...}
        # This enables:
        # 1. Detecting struct constructors (e.g., Geo(...))
        # 2. Type inference for struct member access (e.g., geo.pos -> vec3)
        # Populated during transformation when struct definitions are encountered
        self.struct_types = {}

        # Type name mapping: GLSL -> OpenCL
        self.type_map = {
            # Vectors
            'vec2': 'float2',
            'vec3': 'float3',
            'vec4': 'float4',
            'ivec2': 'int2',
            'ivec3': 'int3',
            'ivec4': 'int4',
            'uvec2': 'uint2',
            'uvec3': 'uint3',
            'uvec4': 'uint4',
            'bvec2': 'int2',  # OpenCL uses int for bool vectors
            'bvec3': 'int3',
            'bvec4': 'int4',
            # Matrices (use struct-based types, can be returned by value)
            'mat2': 'matrix2x2',
            'mat3': 'matrix3x3',
            'mat4': 'matrix4x4',
            # Scalars (unchanged)
            'float': 'float',
            'int': 'int',
            'uint': 'uint',
            'bool': 'bool',
            'void': 'void',
            # Samplers: the Copernicus runtime has no sampler types — channels
            # are layer pointers, the same type every texture builtin consumes
            # (houdini/ocl/include/textureHelpers.h) and the same type the
            # runtime header gives iChannel0..3. Used for user-function sampler
            # params (the triplanar tex3D(sampler2D,...) idiom); NOT marked
            # is_pointer, so the out-param dereference machinery ignores them.
            'sampler2D': 'const IMX_Layer*',
            'sampler3D': 'const IMX_Layer*',
            'samplerCube': 'const IMX_Layer*',
        }

    def transform(self, ast: ASTNode) -> IR.TranslationUnit:
        """
        Transform GLSL AST to OpenCL Transformed AST.

        Args:
            ast: Root GLSL AST node (TranslationUnit)

        Returns:
            Transformed AST root (IR.TranslationUnit)

        Raises:
            TransformationError: If transformation fails
        """
        if ast.type != 'translation_unit':
            raise TransformationError(
                f"Expected translation_unit, got {ast.type}",
                ast.start_point
            )

        # Transform all top-level declarations. Top-level declarations are at
        # program (file) scope; hoisted_global_inits is (re)populated here.
        self.hoisted_global_inits = []
        self._global_scope = True
        declarations = []
        for decl in ast.named_children:
            transformed = self._transform_node(decl)
            if transformed is not None:
                declarations.append(transformed)
        self._global_scope = False

        return IR.TranslationUnit(
            declarations=declarations,
            source_location=ast.start_point
        )

    def _transform_node(self, node: ASTNode) -> Optional[IR.TransformedNode]:
        """
        Transform a single AST node.

        Dispatches to appropriate transformation method based on node type.

        Args:
            node: Source AST node

        Returns:
            Transformed node, or None if node should be skipped
        """
        # Map node types to transformation methods
        transform_methods = {
            'function_definition': self._transform_function_definition,
            'declaration': self._transform_declaration,
            'expression_statement': self._transform_expression_statement,
            'return_statement': self._transform_return_statement,
            'if_statement': self._transform_if_statement,
            'else_clause': self._transform_else_clause,
            'for_statement': self._transform_for_statement,
            'while_statement': self._transform_while_statement,
            'do_statement': self._transform_do_statement,
            'break_statement': self._transform_break_statement,
            'continue_statement': self._transform_continue_statement,
            'compound_statement': self._transform_compound_statement,
            'binary_expression': self._transform_binary_expression,
            'unary_expression': self._transform_unary_expression,
            'call_expression': self._transform_call_expression,
            'identifier': self._transform_identifier,
            'number_literal': self._transform_number_literal,
            'true': self._transform_bool_literal,
            'false': self._transform_bool_literal,
            'field_expression': self._transform_field_expression,
            'subscript_expression': self._transform_subscript_expression,
            'conditional_expression': self._transform_conditional_expression,
            'assignment_expression': self._transform_assignment_expression,
            'update_expression': self._transform_update_expression,
            'parenthesized_expression': self._transform_parenthesized_expression,
            # Struct definitions
            'struct_specifier': self._transform_struct_specifier,
            # Preprocessor directives (Session 9)
            'preproc_def': self._transform_preprocessor,
            'preproc_function_def': self._transform_preprocessor,
            'preproc_if': self._transform_preprocessor,
            'preproc_ifdef': self._transform_preprocessor,
            'preproc_ifndef': self._transform_preprocessor,
            'preproc_else': self._transform_preprocessor,
            'preproc_elif': self._transform_preprocessor,
            'preproc_endif': self._transform_preprocessor,
        }

        method = transform_methods.get(node.type)
        if method:
            return method(node)

        # Unknown node type - this is a warning, not error
        # Some nodes might not need transformation
        return None

    # ========================================================================
    # Literals
    # ========================================================================

    def _transform_number_literal(self, node: ASTNode) -> IR.TransformedNode:
        """Transform numeric literal (float or int)."""
        text = node.text
        location = node.start_point

        # Check if it's a float literal (has decimal point or exponent).
        # Exclude hex literals: a hex int like 0x9e3853U contains the hex digit
        # 'e' and must NOT be treated as a float exponent (that appended a stray
        # 'f' -> invalid 'Uf' suffix or a silently wrong value).
        text_lower = text.lower()
        if not text_lower.startswith('0x') and ('.' in text or 'e' in text_lower):
            # Add 'f' suffix if not present
            if not text.endswith('f') and not text.endswith('F'):
                text = text + 'f'

            return IR.FloatLiteral(
                value=text,
                glsl_type=TYPE_NAME_MAP['float'],
                source_location=location
            )
        else:
            # Integer literal
            return IR.IntLiteral(
                value=text,
                glsl_type=TYPE_NAME_MAP['int'],
                source_location=location
            )

    def _transform_bool_literal(self, node: ASTNode) -> IR.BoolLiteral:
        """Transform boolean literal."""
        value = (node.type == 'true')
        return IR.BoolLiteral(
            value=value,
            glsl_type=TYPE_NAME_MAP['bool'],
            source_location=node.start_point
        )

    # ========================================================================
    # Identifiers and Types
    # ========================================================================

    def _transform_identifier(self, node: ASTNode) -> IR.TransformedNode:
        """Transform identifier (variable reference)."""
        name = node.text

        # Try to infer type from symbol table
        symbol = self.symbol_table.lookup(name)
        glsl_type = symbol.glsl_type if symbol else None

        ident = IR.Identifier(
            name=name,
            glsl_type=glsl_type,
            source_location=node.start_point
        )

        # out/inout params are OpenCL pointers: every READ of one dereferences
        # to *p (incl. as an assignment target -> *p = ..., and a member base
        # -> (*p).x). The one exception is passing a pointer param to another
        # pointer param, which _transform_call_expression unwraps back to bare p.
        if name in self.pointer_params:
            return IR.UnaryOp(
                operator='*',
                operand=ident,
                source_location=node.start_point
            )
        return ident

    def _transform_type_name(self, node: ASTNode) -> str:
        """
        Transform GLSL type name to OpenCL equivalent.

        Args:
            node: Type specifier node

        Returns:
            OpenCL type name string
        """
        glsl_type = node.text

        # Remove precision qualifiers if present
        glsl_type = glsl_type.replace('highp ', '').replace('mediump ', '').replace('lowp ', '')
        glsl_type = glsl_type.strip()

        # Map to OpenCL type
        return self.type_map.get(glsl_type, glsl_type)

    def _get_type_name(self, node: IR.TransformedNode) -> str:
        """
        Get the type name of a transformed IR node.

        Args:
            node: Transformed AST node

        Returns:
            Type name string (e.g., 'mat3', 'float', 'vec2'), or None if type unknown
        """
        # Unwrap ParenthesizedExpression to get to the actual expression
        if isinstance(node, IR.ParenthesizedExpression):
            return self._get_type_name(node.expression)

        # For identifiers, check local type environment first
        if isinstance(node, IR.Identifier):
            if node.name in self.local_types:
                return self.local_types[node.name]

        # Try glsl_type attribute
        if not hasattr(node, 'glsl_type') or not node.glsl_type:
            return None

        # Handle GLSLType objects
        if hasattr(node.glsl_type, 'name'):
            # Only use .name if it's not None
            if node.glsl_type.name is not None:
                return node.glsl_type.name
            # Fall through to str() if .name is None

        # Handle string type names or GLSLType __str__ representation
        # GLSLType.__str__() returns the type name (e.g., 'vec2', 'mat3')
        type_str = str(node.glsl_type)

        # Verify it's a valid type name (not a generic str representation)
        if type_str and not type_str.startswith('<'):
            return type_str

        return None

    def _is_bool_mask(self, node: IR.TransformedNode) -> str:
        """
        True if a node is a boolean-vector mask: a relational expression
        (`a < b`, possibly parenthesized — produced by lowering lessThan/etc.) or
        a bvec-typed value. Used to route mix(a,b,mask) -> select(a,b,mask).
        """
        inner = node.expression if isinstance(node, IR.ParenthesizedExpression) else node
        if isinstance(inner, IR.BinaryOp) and inner.operator in ('<', '<=', '>', '>=', '==', '!='):
            return True
        return self._get_type_name(node) in ('bvec2', 'bvec3', 'bvec4')

    def _transform_vector_conversion_ctor(
        self,
        function_name: str,
        opencl_type: str,
        arg: IR.TransformedNode,
        location: tuple
    ) -> Optional[IR.TransformedNode]:
        """
        Category N: a vector constructor whose single argument is itself a
        vector cannot be emitted as a C-style cast — OpenCL's (T)(...) vector
        literal only broadcasts scalars or assembles component lists, so
        (int2)(float2_expr) is rejected ("invalid conversion between
        ext-vector types") and (float3)(float4_expr) cannot truncate.

        Rewrites:
          element-type change, same width  ivec2(v2)  -> convert_int2(v2)
          truncation, same element base    vec3(v4)   -> v4.xyz
          truncation + element change      ivec2(v3)  -> convert_int2(v3.xy)
          bool mask (relational / bvec)    vec4(a<b)  -> convert_float4((a < b) & 1)
        The mask is `& 1`-normalized because OpenCL vector comparisons yield
        -1 for true where GLSL bvec->number conversion requires 1 (& 1 maps
        both -1- and 1-for-true representations to 1).

        Returns None whenever the existing cast emission is already correct
        (scalar broadcast, component list, identity, widening, unknown
        argument type) so the caller falls through unchanged.
        """
        target = VECTOR_TYPE_INFO.get(function_name)
        if target is None:
            return None  # scalar/matrix/sampler constructor: not this rewrite
        target_base, target_width = target

        arg_type = self._get_type_name(arg)
        arg_info = VECTOR_TYPE_INFO.get(arg_type) if arg_type else None
        if arg_info is None:
            return None  # scalar broadcast or unknown type: keep the cast
        arg_base, arg_width = arg_info
        # Vector comparisons are typed vecN by _infer_binary_op_type but hold
        # an int -1/0 mask at runtime; detect them structurally.
        if self._is_bool_mask(arg):
            arg_base = 'bool'

        if arg_width < target_width:
            return None  # vec4(vec2) is invalid GLSL; leave untouched

        glsl_type = TYPE_NAME_MAP.get(function_name)
        node = arg
        if arg_width > target_width:
            # Truncation is a swizzle, not a cast: vec3(v4) -> v4.xyz
            base = node
            if isinstance(node, (IR.BinaryOp, IR.UnaryOp, IR.TernaryOp)):
                base = IR.ParenthesizedExpression(
                    expression=node, source_location=location)
            node = IR.MemberAccess(
                base=base,
                member='xyzw'[:target_width],
                glsl_type=glsl_type if arg_base == target_base else None,
                source_location=location
            )

        if arg_base == 'bool' and target_base != 'bool':
            mask = node
            if isinstance(mask, (IR.BinaryOp, IR.UnaryOp, IR.TernaryOp)):
                mask = IR.ParenthesizedExpression(
                    expression=mask, source_location=location)
            node = IR.ParenthesizedExpression(
                expression=IR.BinaryOp(
                    operator='&',
                    left=mask,
                    right=IR.IntLiteral(value='1'),
                    source_location=location
                ),
                glsl_type=glsl_type,
                source_location=location
            )
            arg_base = 'int'

        # bvec targets share OpenCL's int-vector representation
        if target_base == 'bool':
            target_base = 'int'

        if arg_base == target_base:
            # Pure truncation or pure mask normalization; None if untouched.
            return node if node is not arg else None
        return IR.CallExpression(
            function=f'convert_{opencl_type}',
            arguments=[node],
            glsl_type=glsl_type,
            source_location=location
        )

    def _vector_width_suffix(self, node: IR.TransformedNode) -> str:
        """
        Width suffix ('', '2', '3', '4') for an as_float/as_uint/as_int call,
        inferred from a scalar/vecN argument's type. Returns '' (scalar) when the
        width is unknown.
        """
        type_name = None
        if isinstance(node, IR.TypeConstructor):
            type_name = node.type_name
        if type_name is None:
            type_name = self._get_type_name(node)
        if type_name and type_name[-1] in '234':
            return type_name[-1]
        return ''

    def _is_matrix_type(self, type_name: str) -> bool:
        """
        Check if a type name is a matrix type.

        Args:
            type_name: Type name string (GLSL or OpenCL: 'mat2'/'matrix2x2', etc.)

        Returns:
            True if type is a matrix
        """
        if not type_name:
            return False
        # Check both GLSL and OpenCL matrix names
        return type_name in ['mat2', 'mat3', 'mat4', 'matrix2x2', 'matrix3x3', 'matrix4x4']

    def _are_all_vector_type(
        self,
        arguments: List[IR.TransformedNode],
        glsl_vec_type: str,
        opencl_vec_type: str
    ) -> bool:
        """
        Check if all arguments are of the expected vector type.

        This is used for detecting matrix column constructors like mat3(vec3, vec3, vec3).

        Args:
            arguments: List of argument nodes
            glsl_vec_type: Expected GLSL vector type ('vec2', 'vec3', 'vec4')
            opencl_vec_type: Expected OpenCL vector type ('float2', 'float3', 'float4')

        Returns:
            True if all arguments are of the expected vector type
        """
        for arg in arguments:
            # Get the type of the argument
            arg_type = self._get_type_name(arg)

            # Check if it's a TypeConstructor with matching type
            if isinstance(arg, IR.TypeConstructor):
                if arg.type_name == opencl_vec_type:
                    continue
                # Also check glsl_type attribute
                if hasattr(arg, 'glsl_type') and arg.glsl_type:
                    if str(arg.glsl_type) == glsl_vec_type:
                        continue

            # Check if it's an identifier with vector type
            if arg_type in [glsl_vec_type, opencl_vec_type]:
                continue

            # If we get here, this argument is not the expected vector type
            return False

        return True

    def _is_vector_type(self, type_name: str) -> bool:
        """
        Check if a type name is a vector type.

        Args:
            type_name: Type name string (e.g., 'float2', 'float3', 'vec2', etc.)

        Returns:
            True if type is a vector
        """
        if not type_name:
            return False
        # Check both GLSL and OpenCL vector names
        vector_types = [
            'vec2', 'vec3', 'vec4',
            'ivec2', 'ivec3', 'ivec4',
            'uvec2', 'uvec3', 'uvec4',
            'bvec2', 'bvec3', 'bvec4',
            'float2', 'float3', 'float4',
            'int2', 'int3', 'int4',
            'uint2', 'uint3', 'uint4'
        ]
        return type_name in vector_types

    def _is_scalar_type(self, type_name: str) -> bool:
        """
        Check if a type name is a scalar type.

        Args:
            type_name: Type name string (e.g., 'float', 'int', 'bool')

        Returns:
            True if type is a scalar
        """
        if not type_name:
            return False
        return type_name in ['float', 'int', 'uint', 'bool']

    def _create_zero_initializer(self, glsl_type: str, opencl_type: str) -> Optional[IR.TransformedNode]:
        """
        Create a zero initializer for undefined variables to match GLSL semantics.

        GLSL implicitly initializes undefined variables to zero, while OpenCL
        leaves them undefined. This method creates appropriate zero initializers
        for scalar, vector, and matrix types to match GLSL behavior.

        Args:
            glsl_type: GLSL type name (e.g., 'float', 'vec3', 'mat2')
            opencl_type: OpenCL type name (e.g., 'float', 'float3', 'matrix2x2')

        Returns:
            IR node representing zero initializer, or None for unsupported types

        Examples:
            float -> IR.FloatLiteral("0.0f")
            int -> IR.IntLiteral("0")
            vec3 -> IR.TypeConstructor("float3", [IR.FloatLiteral("0.0f")])
            mat2 -> IR.CallExpression("GLSL_matrix2x2_diagonal", [IR.FloatLiteral("0.0f")])
        """
        # Scalar float
        if glsl_type == 'float':
            return IR.FloatLiteral(
                value="0.0f",
                glsl_type=TYPE_NAME_MAP['float'],
                source_location=None
            )

        # Scalar int
        if glsl_type == 'int':
            return IR.IntLiteral(
                value="0",
                glsl_type=TYPE_NAME_MAP['int'],
                source_location=None
            )

        # Float vectors (vec2, vec3, vec4)
        if glsl_type in ['vec2', 'vec3', 'vec4']:
            return IR.TypeConstructor(
                type_name=opencl_type,  # float2, float3, float4
                arguments=[IR.FloatLiteral(value="0.0f", glsl_type=TYPE_NAME_MAP['float'], source_location=None)],
                glsl_type=TYPE_NAME_MAP[glsl_type],
                source_location=None
            )

        # Integer vectors (ivec2, ivec3, ivec4)
        if glsl_type in ['ivec2', 'ivec3', 'ivec4']:
            return IR.TypeConstructor(
                type_name=opencl_type,  # int2, int3, int4
                arguments=[IR.IntLiteral(value="0", glsl_type=TYPE_NAME_MAP['int'], source_location=None)],
                glsl_type=TYPE_NAME_MAP[glsl_type],
                source_location=None
            )

        # Matrices (mat2, mat3, mat4) - use diagonal constructor with zero
        if glsl_type == 'mat2':
            return IR.CallExpression(
                function='GLSL_matrix2x2_diagonal',
                arguments=[IR.FloatLiteral(value="0.0f", glsl_type=TYPE_NAME_MAP['float'], source_location=None)],
                glsl_type=TYPE_NAME_MAP['mat2'],
                source_location=None
            )

        if glsl_type == 'mat3':
            return IR.CallExpression(
                function='GLSL_matrix3x3_diagonal',
                arguments=[IR.FloatLiteral(value="0.0f", glsl_type=TYPE_NAME_MAP['float'], source_location=None)],
                glsl_type=TYPE_NAME_MAP['mat3'],
                source_location=None
            )

        if glsl_type == 'mat4':
            return IR.CallExpression(
                function='GLSL_matrix4x4_diagonal',
                arguments=[IR.FloatLiteral(value="0.0f", glsl_type=TYPE_NAME_MAP['float'], source_location=None)],
                glsl_type=TYPE_NAME_MAP['mat4'],
                source_location=None
            )

        # For other types (uint, bool, uvec*, bvec*, structs), return None
        # These types may have different initialization semantics or are less common
        return None

    def _infer_swizzle_type(self, base_type: str, swizzle: str) -> Optional[GLSLType]:
        """
        Infer the result type of a swizzle operation on a vector.

        Swizzles extract components from vectors using patterns like:
        - Coordinate: x, y, z, w (or any combination: xy, xyz, xyzw, etc.)
        - Color: r, g, b, a (or any combination: rg, rgb, rgba, etc.)

        Args:
            base_type: Base vector type (e.g., 'vec3', 'float3', 'ivec2')
            swizzle: Swizzle pattern (e.g., 'xy', 'xyz', 'rg')

        Returns:
            GLSLType of the swizzled result, or None if invalid

        Examples:
            vec3, 'xy' -> vec2
            vec4, 'xyz' -> vec3
            ivec3, 'xy' -> ivec2
            vec2, 'x' -> float
        """
        from ..analyzer.type_checker import TYPE_NAME_MAP

        if not base_type or not swizzle:
            return None

        # Map OpenCL type names to GLSL for easier handling
        opencl_to_glsl = {
            'float2': 'vec2', 'float3': 'vec3', 'float4': 'vec4',
            'int2': 'ivec2', 'int3': 'ivec3', 'int4': 'ivec4',
            'uint2': 'uvec2', 'uint3': 'uvec3', 'uint4': 'uvec4'
        }
        glsl_base = opencl_to_glsl.get(base_type, base_type)

        # Check if base is a vector type
        if not self._is_vector_type(glsl_base):
            return None

        # Validate swizzle pattern
        # GLSL allows xyzw (coordinate) or rgba (color), but not mixed
        coord_chars = set('xyzw')
        color_chars = set('rgba')
        swizzle_chars = set(swizzle)

        is_coord = swizzle_chars.issubset(coord_chars)
        is_color = swizzle_chars.issubset(color_chars)

        if not (is_coord or is_color):
            # Invalid swizzle pattern (mixed or invalid characters)
            return None

        # Determine result type based on swizzle length
        swizzle_len = len(swizzle)

        # Extract base type family (vec, ivec, uvec, bvec)
        if glsl_base.startswith('vec'):
            base_family = 'vec'
            scalar_type = 'float'
        elif glsl_base.startswith('ivec'):
            base_family = 'ivec'
            scalar_type = 'int'
        elif glsl_base.startswith('uvec'):
            base_family = 'uvec'
            scalar_type = 'uint'
        elif glsl_base.startswith('bvec'):
            base_family = 'bvec'
            scalar_type = 'bool'
        else:
            return None

        # Single component -> scalar
        if swizzle_len == 1:
            return TYPE_NAME_MAP.get(scalar_type)

        # Multiple components -> vector of appropriate dimension
        if swizzle_len in [2, 3, 4]:
            result_type = f'{base_family}{swizzle_len}'
            return TYPE_NAME_MAP.get(result_type)

        return None

    def _infer_mul_result_type(self, left_type: str, right_type: str) -> Optional[GLSLType]:
        """
        Infer the result type of matrix multiplication.

        Args:
            left_type: Type of left operand (can be GLSL or OpenCL name)
            right_type: Type of right operand (can be GLSL or OpenCL name)

        Returns:
            Result type of the multiplication
        """
        from ..analyzer.type_checker import TYPE_NAME_MAP

        # Map OpenCL type names back to GLSL for TYPE_NAME_MAP lookup
        opencl_to_glsl = {
            'float2': 'vec2', 'float3': 'vec3', 'float4': 'vec4',
            'int2': 'ivec2', 'int3': 'ivec3', 'int4': 'ivec4',
            'uint2': 'uvec2', 'uint3': 'uvec3', 'uint4': 'uvec4',
            'matrix2x2': 'mat2', 'matrix3x3': 'mat3', 'matrix4x4': 'mat4'
        }

        # Normalize to GLSL names for lookup
        left_glsl = opencl_to_glsl.get(left_type, left_type)
        right_glsl = opencl_to_glsl.get(right_type, right_type)

        # Matrix * Vector -> Vector
        if self._is_matrix_type(left_type) and self._is_vector_type(right_type):
            return TYPE_NAME_MAP.get(right_glsl, TYPE_NAME_MAP.get(right_type))

        # Vector * Matrix -> Vector
        if self._is_vector_type(left_type) and self._is_matrix_type(right_type):
            return TYPE_NAME_MAP.get(left_glsl, TYPE_NAME_MAP.get(left_type))

        # Matrix * Matrix -> Matrix
        if self._is_matrix_type(left_type) and self._is_matrix_type(right_type):
            return TYPE_NAME_MAP.get(left_glsl, TYPE_NAME_MAP.get(left_type))

        return None

    def _get_matrix_mul_function_name(self, left_type: str, right_type: str) -> str:
        """
        Get the correct GLSL_mul_* function name based on operand types.

        Args:
            left_type: Type of left operand (GLSL or OpenCL type name)
            right_type: Type of right operand (GLSL or OpenCL type name)

        Returns:
            Function name like 'GLSL_mul_mat2_vec2', 'GLSL_mul_vec3_mat3', etc.
        """
        # Extract matrix/vector dimensions from type names
        # Handles both GLSL (mat2, vec2) and OpenCL (matrix2x2, float2) names
        def get_dim(type_name):
            # Matrix types
            if type_name in ['mat2', 'matrix2x2']:
                return '2'
            elif type_name in ['mat3', 'matrix3x3']:
                return '3'
            elif type_name in ['mat4', 'matrix4x4']:
                return '4'
            # Vector types
            elif type_name in ['vec2', 'float2']:
                return '2'
            elif type_name in ['vec3', 'float3']:
                return '3'
            elif type_name in ['vec4', 'float4']:
                return '4'
            return None

        left_dim = get_dim(left_type)
        right_dim = get_dim(right_type)

        # Matrix * Vector -> GLSL_mul_matN_vecN
        if self._is_matrix_type(left_type) and self._is_vector_type(right_type):
            return f'GLSL_mul_mat{left_dim}_vec{right_dim}'

        # Vector * Matrix -> GLSL_mul_vecN_matN
        if self._is_vector_type(left_type) and self._is_matrix_type(right_type):
            return f'GLSL_mul_vec{left_dim}_mat{right_dim}'

        # Matrix * Matrix -> GLSL_mul_matN_matN
        if self._is_matrix_type(left_type) and self._is_matrix_type(right_type):
            return f'GLSL_mul_mat{left_dim}_mat{right_dim}'

        # Fallback to generic (shouldn't happen)
        return 'GLSL_mul'

    def _infer_builtin_function_type(
        self,
        function_name: str,
        arguments: List[IR.TransformedNode]
    ) -> Optional[GLSLType]:
        """
        Infer the return type of a built-in GLSL function.

        This is critical for matrix operation detection when function calls
        are used as operands in binary expressions.

        Args:
            function_name: Name of the function (with GLSL_ prefix if applicable)
            arguments: List of transformed argument nodes

        Returns:
            GLSLType of the function's return value, or None if unknown
        """
        from ..analyzer.type_checker import TYPE_NAME_MAP

        # Matrix functions - return same type as input
        if function_name.startswith('GLSL_transpose') or function_name.startswith('GLSL_inverse'):
            if arguments:
                arg_type = self._get_type_name(arguments[0])
                if arg_type:
                    return TYPE_NAME_MAP.get(arg_type)

        # Determinant - always returns float
        elif function_name.startswith('GLSL_determinant'):
            return TYPE_NAME_MAP.get('float')

        # Vector functions that return the same type as their first argument
        # normalize, abs, sign, floor, ceil, trunc, fract, sqrt, inversesqrt, etc.
        vector_passthrough_functions = [
            'GLSL_normalize', 'GLSL_abs', 'GLSL_sign', 'GLSL_floor', 'GLSL_ceil',
            'GLSL_trunc', 'GLSL_fract', 'GLSL_sqrt', 'GLSL_inversesqrt',
            'GLSL_exp', 'GLSL_log', 'GLSL_exp2', 'GLSL_log2',
            'GLSL_sin', 'GLSL_cos', 'GLSL_tan', 'GLSL_asin', 'GLSL_acos', 'GLSL_atan',
            'GLSL_sinh', 'GLSL_cosh', 'GLSL_tanh', 'GLSL_asinh', 'GLSL_acosh', 'GLSL_atanh',
            'GLSL_radians', 'GLSL_degrees',
            'GLSL_faceforward', 'GLSL_reflect', 'GLSL_refract',
            'GLSL_dFdx', 'GLSL_dFdy', 'GLSL_fwidth'
        ]
        if function_name in vector_passthrough_functions and arguments:
            arg_type = self._get_type_name(arguments[0])
            if arg_type:
                # Parameters register OpenCL names ('float3'); normalize so
                # the GLSL-keyed TYPE_NAME_MAP lookup doesn't come up empty.
                return TYPE_NAME_MAP.get(
                    OPENCL_TO_GLSL_NAME.get(arg_type, arg_type))

        # Cross product - returns vec3
        elif function_name == 'GLSL_cross':
            return TYPE_NAME_MAP.get('vec3')

        # Functions that return the same type as their arguments (with multiple args)
        # min, max, clamp, mix, step, smoothstep, pow, mod
        elif function_name in ['GLSL_min', 'GLSL_max', 'GLSL_clamp', 'GLSL_mix',
                                'GLSL_step', 'GLSL_smoothstep', 'GLSL_pow', 'GLSL_mod'] and arguments:
            arg_type = self._get_type_name(arguments[0])
            if arg_type:
                return TYPE_NAME_MAP.get(
                    OPENCL_TO_GLSL_NAME.get(arg_type, arg_type))

        # Functions that return float (scalar reduction functions)
        # length, distance, dot
        elif function_name in ['GLSL_length', 'GLSL_distance', 'GLSL_dot']:
            return TYPE_NAME_MAP.get('float')

        # modf - returns same type as first argument
        elif function_name == 'GLSL_modf' and arguments:
            arg_type = self._get_type_name(arguments[0])
            if arg_type:
                return TYPE_NAME_MAP.get(
                    OPENCL_TO_GLSL_NAME.get(arg_type, arg_type))

        # Texture sampling builtins return vec4 (resolved by the
        # textureHelpers.h overloads). Needed so vec3(texture(...)) is
        # recognized as a truncation (category N).
        elif function_name in ('texture', 'texelFetch', 'textureLod',
                               'textureGrad', 'textureProj'):
            return TYPE_NAME_MAP.get('vec4')

        return None

    # ========================================================================
    # Expressions
    # ========================================================================

    def _infer_binary_op_type(
        self,
        operator: str,
        left: IR.TransformedNode,
        right: IR.TransformedNode
    ) -> Optional[GLSLType]:
        """
        Infer the result type of a binary operation.

        Args:
            operator: Binary operator (+, -, *, /, etc.)
            left: Left operand node
            right: Right operand node

        Returns:
            GLSLType of the result, or None if cannot infer
        """
        left_type = self._get_type_name(left)
        right_type = self._get_type_name(right)

        if not left_type or not right_type:
            return None

        # For arithmetic operations (+, -, *, /, %)
        if operator in ['+', '-', '*', '/', '%']:
            # Scalar op scalar = scalar
            if self._is_scalar_type(left_type) and self._is_scalar_type(right_type):
                # Return the "larger" type (float > int > uint)
                if left_type == 'float' or right_type == 'float':
                    return TYPE_NAME_MAP.get('float')
                elif left_type == 'int' or right_type == 'int':
                    return TYPE_NAME_MAP.get('int')
                return TYPE_NAME_MAP.get(left_type)

            # Vector op vector = vector (same type)
            if self._is_vector_type(left_type) and self._is_vector_type(right_type):
                if left_type == right_type:
                    return TYPE_NAME_MAP.get(left_type)
                # Handle GLSL vs OpenCL type names (vec3 vs float3)
                opencl_to_glsl = {
                    'float2': 'vec2', 'float3': 'vec3', 'float4': 'vec4',
                    'int2': 'ivec2', 'int3': 'ivec3', 'int4': 'ivec4',
                    'uint2': 'uvec2', 'uint3': 'uvec3', 'uint4': 'uvec4'
                }
                left_glsl = opencl_to_glsl.get(left_type, left_type)
                right_glsl = opencl_to_glsl.get(right_type, right_type)
                if left_glsl == right_glsl:
                    return TYPE_NAME_MAP.get(left_glsl)

            # Vector op scalar = vector (or scalar op vector = vector)
            if self._is_vector_type(left_type) and self._is_scalar_type(right_type):
                return TYPE_NAME_MAP.get(left_type)
            if self._is_scalar_type(left_type) and self._is_vector_type(right_type):
                return TYPE_NAME_MAP.get(right_type)

            # Matrix op scalar = matrix (or scalar op matrix = matrix)
            if self._is_matrix_type(left_type) and self._is_scalar_type(right_type):
                return TYPE_NAME_MAP.get(left_type)
            if self._is_scalar_type(left_type) and self._is_matrix_type(right_type):
                return TYPE_NAME_MAP.get(right_type)

        # For comparison operations (<, >, <=, >=, ==, !=)
        # These always return bool (or bvec for vector comparisons)
        if operator in ['<', '>', '<=', '>=', '==', '!=']:
            if self._is_vector_type(left_type):
                # Vector comparison returns bvec (but we use int vec in OpenCL)
                dim = left_type[-1] if left_type[-1].isdigit() else '3'
                return TYPE_NAME_MAP.get(f'vec{dim}')
            return TYPE_NAME_MAP.get('bool')

        # For logical operations (&&, ||)
        if operator in ['&&', '||']:
            return TYPE_NAME_MAP.get('bool')

        return None

    def _transform_binary_expression(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform binary expression (a + b, x * y, etc.).

        Special handling for matrix operations:
        - M * v -> GLSL_mul(M, v)
        - v * M -> GLSL_mul(v, M)
        - M1 * M2 -> GLSL_mul(M1, M2)
        """
        left = self._transform_node(node.left)
        right = self._transform_node(node.right)
        operator = node.operator

        # Check for matrix multiplication operations
        if operator == '*':
            left_type = self._get_type_name(left)
            right_type = self._get_type_name(right)

            # Special handling for function calls that return matrices
            # Check if left operand is a function call that returns a matrix type
            if isinstance(left, IR.CallExpression) and not left_type:
                # Try to infer type from glsl_type attribute first (set by _infer_builtin_function_type)
                if hasattr(left, 'glsl_type') and left.glsl_type:
                    left_type = str(left.glsl_type)
                # Otherwise look up user-defined function in symbol table
                elif left.function in self.symbol_table.symbols:
                    func_symbol = self.symbol_table.lookup(left.function)
                    if func_symbol and hasattr(func_symbol, 'glsl_type'):
                        left_type = str(func_symbol.glsl_type)

            # Check if right operand is a function call that returns a matrix type
            if isinstance(right, IR.CallExpression) and not right_type:
                # Try to infer type from glsl_type attribute first (set by _infer_builtin_function_type)
                if hasattr(right, 'glsl_type') and right.glsl_type:
                    right_type = str(right.glsl_type)
                # Otherwise look up user-defined function in symbol table
                elif right.function in self.symbol_table.symbols:
                    func_symbol = self.symbol_table.lookup(right.function)
                    if func_symbol and hasattr(func_symbol, 'glsl_type'):
                        right_type = str(func_symbol.glsl_type)

            # Check if right operand is a BinaryOp that doesn't have type set yet
            # This handles cases like: m2 * (a / b) where (a / b) is a BinaryOp
            if isinstance(right, IR.BinaryOp) and not right_type:
                # Try to infer type from the BinaryOp's glsl_type attribute
                if hasattr(right, 'glsl_type') and right.glsl_type:
                    right_type = str(right.glsl_type)

            # Check if left operand is a BinaryOp that doesn't have type set yet
            if isinstance(left, IR.BinaryOp) and not left_type:
                # Try to infer type from the BinaryOp's glsl_type attribute
                if hasattr(left, 'glsl_type') and left.glsl_type:
                    left_type = str(left.glsl_type)

            # Check if this is a matrix operation
            is_matrix_op = (
                (self._is_matrix_type(left_type) and self._is_vector_type(right_type)) or
                (self._is_vector_type(left_type) and self._is_matrix_type(right_type)) or
                (self._is_matrix_type(left_type) and self._is_matrix_type(right_type))
            )

            if is_matrix_op:
                # Infer result type for proper type propagation
                result_type = self._infer_mul_result_type(left_type, right_type)

                # Get the correctly typed function name based on operand types
                function_name = self._get_matrix_mul_function_name(left_type, right_type)

                return IR.CallExpression(
                    function=function_name,
                    arguments=[left, right],
                    glsl_type=result_type,
                    source_location=node.start_point
                )

            # Matrix * Scalar - use native OpenCL, no transformation needed

        # Infer result type for this binary operation
        result_type = self._infer_binary_op_type(operator, left, right)

        # Default: keep binary operation as-is
        return IR.BinaryOp(
            operator=operator,
            left=left,
            right=right,
            glsl_type=result_type,
            source_location=node.start_point
        )

    def _transform_unary_expression(self, node: ASTNode) -> IR.UnaryOp:
        """Transform unary expression (-x, !flag, etc.)."""
        # Find operator and operand
        operator = None
        operand_node = None

        for child in node.children:
            if child.type in ['-', '+', '!', '~', '++', '--']:
                operator = child.type
            elif child.type not in ['(', ')']:
                operand_node = child

        if operator is None or operand_node is None:
            raise TransformationError(
                "Invalid unary expression structure",
                node.start_point
            )

        operand = self._transform_node(operand_node)

        return IR.UnaryOp(
            operator=operator,
            operand=operand,
            source_location=node.start_point
        )

    def _transform_call_expression(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform function call.

        Handles:
        - Type constructors: vec2(1.0, 2.0) -> (float2)(1.0f, 2.0f)
        - Built-in functions: sin(x) -> GLSL_sin(x)
        - User functions: unchanged
        - Output parameters: foo(x, y) -> foo(x, &y) if y is out/inout param
        """
        function_node = node.function
        function_name = function_node.text if function_node else ""

        # Transform arguments
        arguments = []
        for arg in node.arguments:
            transformed_arg = self._transform_node(arg)
            if transformed_arg:
                arguments.append(transformed_arg)

        location = node.start_point

        # Category K: GLSL array constructor T[N](...) — parses as a call
        # whose "function" is a subscript expression over the element type
        # name (float[3], vec2[2], MyStruct[9]). The size is discarded: the
        # emitter produces a brace list in initializer position and an
        # unsized compound literal in expression position.
        if function_node is not None and function_node.type == 'subscript_expression':
            base_node = None
            for sub_child in function_node.named_children:
                if sub_child.type in ('identifier', 'type_identifier', 'primitive_type'):
                    base_node = sub_child
                    break
            if base_node is not None and base_node.text in self.type_map:
                return IR.ArrayConstructor(
                    element_type=self.type_map[base_node.text],
                    arguments=arguments,
                    glsl_type=None,
                    source_location=location
                )

        # Check if it's a struct constructor
        if function_name in self.struct_types:
            # This is a struct constructor: Geo(...) -> compound literal { ... }
            # We create a TypeConstructor with the struct name
            # The emitter will handle this specially to emit { arg1, arg2, ... }
            return IR.TypeConstructor(
                type_name=function_name,
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(function_name),
                source_location=location
            )

        # Check if it's a type constructor
        if function_name in self.type_map:
            opencl_type = self.type_map[function_name]

            # Handle matrix constructors specially
            if function_name in ['mat2', 'mat3', 'mat4']:
                return self._transform_matrix_constructor(
                    function_name, opencl_type, arguments, location
                )

            # A single vector-valued argument needs a real conversion or a
            # swizzle, not a C cast (category N).
            if len(arguments) == 1:
                converted = self._transform_vector_conversion_ctor(
                    function_name, opencl_type, arguments[0], location)
                if converted is not None:
                    return converted

            # Vector constructors: vec2(...) -> (float2)(...)
            return IR.TypeConstructor(
                type_name=opencl_type,
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(function_name),
                source_location=location
            )

        # Bit-cast reinterprets -> OpenCL as_* builtins, size-suffixed by arg
        # width (category X). These are native OpenCL builtins, so no GLSL_ helper
        # / Houdini change is needed. uintBitsToFloat/intBitsToFloat -> as_float*,
        # floatBitsToUint -> as_uint*, floatBitsToInt -> as_int*.
        bitcast_map = {
            'uintBitsToFloat': 'as_float', 'intBitsToFloat': 'as_float',
            'floatBitsToUint': 'as_uint', 'floatBitsToInt': 'as_int',
        }
        if function_name in bitcast_map and arguments:
            return IR.CallExpression(
                function=bitcast_map[function_name] + self._vector_width_suffix(arguments[0]),
                arguments=arguments,
                source_location=location
            )

        # GLSL component-wise comparison functions -> OpenCL relational operators
        # (category X). OpenCL has no lessThan()/etc.; the vector relational
        # operators return per-component int vectors. Guard against a user
        # function shadowing one of these names.
        comparison_ops = {
            'lessThan': '<', 'lessThanEqual': '<=',
            'greaterThan': '>', 'greaterThanEqual': '>=',
            'equal': '==', 'notEqual': '!=',
        }
        if (function_name in comparison_ops and len(arguments) == 2
                and function_name not in self.user_function_return_types):
            # Type the mask so downstream consumers (e.g. the category-N
            # vector-conversion ctor) know its width.
            mask_type = self._infer_binary_op_type(
                comparison_ops[function_name], arguments[0], arguments[1])
            return IR.ParenthesizedExpression(
                expression=IR.BinaryOp(
                    operator=comparison_ops[function_name],
                    left=arguments[0],
                    right=arguments[1],
                    glsl_type=mask_type,
                    source_location=location
                ),
                glsl_type=mask_type,
                source_location=location
            )

        # GLSL mix(a, b, m) with a bool-vector mask m is component-wise SELECT
        # (m[i] ? b[i] : a[i]), not interpolation. OpenCL has no GLSL_mix overload
        # for an int/bool-vector mask, so emit select(a, b, m). OpenCL `select`
        # picks b where the mask's sign bit is set, matching our -1-for-true
        # relational results (and GLSL "pick b where bvec true"). The float-t
        # interpolation path is left untouched.
        if (function_name == 'mix' and len(arguments) == 3
                and function_name not in self.user_function_return_types
                and self._is_bool_mask(arguments[2])):
            return IR.CallExpression(
                function='select',
                arguments=[arguments[0], arguments[1], arguments[2]],
                source_location=location
            )

        # Check if it's a built-in function that needs GLSL_ prefix
        # Comprehensive list of all GLSL built-in functions from glslHelpers.h
        # (Session 3: Complete function transformation)
        glsl_builtins = {
            # Angle conversion
            'radians', 'degrees',
            # Trigonometric
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            # Hyperbolic
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            # Exponential/Power/Root
            'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
            # Common/Math
            'abs', 'sign', 'floor', 'ceil', 'trunc', 'fract', 'mod', 'modf',
            'min', 'max', 'clamp', 'mix', 'step', 'smoothstep',
            # Geometric
            'length', 'distance', 'dot', 'cross', 'normalize',
            'faceforward', 'reflect', 'refract',
            # Derivative placeholders (dummy implementations)
            'dFdx', 'dFdy', 'fwidth',
            # Matrix functions (Session 5)
            'transpose', 'inverse', 'determinant',
        }

        if function_name in glsl_builtins:
            function_name = f'GLSL_{function_name}'

        # Add type suffix for mat3/mat4 matrix functions
        # mat2 uses base name (GLSL_transpose), mat3/mat4 use suffixes
        matrix_functions = ['GLSL_transpose', 'GLSL_inverse', 'GLSL_determinant', 'GLSL_matrixCompMult']
        if function_name in matrix_functions and arguments:
            arg_type = self._get_type_name(arguments[0])
            # Handle both GLSL and OpenCL type names
            if arg_type in ['mat3', 'matrix3x3']:
                function_name = f'{function_name}_mat3'
            elif arg_type in ['mat4', 'matrix4x4']:
                function_name = f'{function_name}_mat4'
            # mat2/matrix2x2 uses base name (no suffix)

        # Infer result type for built-in functions
        # This is critical for matrix operation detection when function calls are operands
        glsl_type = self._infer_builtin_function_type(function_name, arguments)

        # If not a built-in function, look up user-defined function in our registry
        # This handles arbitrary user-defined matrix-returning functions
        if glsl_type is None and function_name in self.user_function_return_types:
            # Get the GLSL type name from our registry
            glsl_type_name = self.user_function_return_types[function_name]
            # Convert to GLSLType object for proper type propagation
            glsl_type = TYPE_NAME_MAP.get(glsl_type_name)

        # Handle output parameters (out/inout): the callee wants a pointer.
        if function_name in self.function_signatures:
            param_info = self.function_signatures[function_name]
            for i, (param_name, is_pointer) in enumerate(param_info):
                if is_pointer and i < len(arguments):
                    arg = arguments[i]
                    # A pointer param read was auto-deref'd to *p; pass the
                    # pointer straight through to the callee's pointer param.
                    if (isinstance(arg, IR.UnaryOp) and arg.operator == '*'
                            and isinstance(arg.operand, IR.Identifier)
                            and arg.operand.name in self.pointer_params):
                        arguments[i] = arg.operand
                    # A local variable or array element: take its address.
                    # (Swizzles/MemberAccess are excluded: &v.xy is invalid.)
                    elif isinstance(arg, (IR.Identifier, IR.ArrayAccess)):
                        arguments[i] = IR.UnaryOp(
                            operator='&',
                            operand=arg,
                            source_location=arg.source_location
                        )

        # Regular function call
        return IR.CallExpression(
            function=function_name,
            arguments=arguments,
            glsl_type=glsl_type,
            source_location=location
        )

    def _transform_matrix_constructor(
        self,
        mat_type: str,
        opencl_type: str,
        arguments: List[IR.TransformedNode],
        location: tuple
    ) -> IR.TransformedNode:
        """
        Transform matrix constructor.

        Handles:
        - Diagonal constructors: mat2(1.0) -> GLSL_matrix2x2_diagonal(1.0f)
        - Column constructors: mat3(vec3, vec3, vec3) -> GLSL_mat3_cols(vec3, vec3, vec3)
        - Full constructors: mat2(1,2,3,4) -> GLSL_mat2(1f,2f,3f,4f)
        - Type casting: mat4(mat3_var) -> GLSL_mat4_from_mat3(mat3_var)

        Args:
            mat_type: GLSL matrix type ('mat2', 'mat3', 'mat4')
            opencl_type: OpenCL matrix type ('matrix2x2', 'matrix3x3', 'matrix4x4')
            arguments: Transformed argument list
            location: Source location

        Returns:
            Appropriate IR node for the matrix constructor
        """
        num_args = len(arguments)

        # Diagonal constructor: single scalar argument
        if num_args == 1:
            arg = arguments[0]

            # Check if argument is a matrix (type casting)
            arg_type_name = self._get_type_name(arg)
            if arg_type_name in ['matrix2x2', 'matrix3x3', 'matrix4x4']:
                # Matrix type casting: mat4(mat3_var) -> GLSL_mat4_from_mat3(mat3_var)
                return self._create_matrix_cast(mat_type, arg_type_name, arguments, location)

            # Diagonal constructor: mat2(scalar) -> GLSL_matrix2x2_diagonal(scalar)
            function_name = f'GLSL_{opencl_type}_diagonal'
            return IR.CallExpression(
                function=function_name,
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(mat_type),
                source_location=location
            )

        # Column constructor: mat2(vec2, vec2), mat3(vec3, vec3, vec3), mat4(vec4, vec4, vec4, vec4)
        column_patterns = {
            'mat2': (2, 'vec2', 'float2'),
            'mat3': (3, 'vec3', 'float3'),
            'mat4': (4, 'vec4', 'float4')
        }

        if mat_type in column_patterns:
            expected_cols, vec_type, opencl_vec = column_patterns[mat_type]
            if num_args == expected_cols:
                # Check if all arguments are vector types
                if self._are_all_vector_type(arguments, vec_type, opencl_vec):
                    function_name = f'GLSL_{mat_type}_cols'
                    return IR.CallExpression(
                        function=function_name,
                        arguments=arguments,
                        glsl_type=TYPE_NAME_MAP.get(mat_type),
                        source_location=location
                    )

        # Full matrix constructor
        expected_elements = {'mat2': 4, 'mat3': 9, 'mat4': 16}
        if num_args == expected_elements.get(mat_type, 0):
            # All matrix types use GLSL_mat* functions
            function_name = f'GLSL_{mat_type}'
            return IR.CallExpression(
                function=function_name,
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(mat_type),
                source_location=location
            )

        # Unsupported number of arguments
        raise TransformationError(
            f"Invalid number of arguments for {mat_type} constructor: {num_args}",
            location
        )

    def _create_matrix_cast(
        self,
        target_type: str,
        source_type: str,
        arguments: List[IR.TransformedNode],
        location: tuple
    ) -> IR.CallExpression:
        """
        Create matrix type casting call.

        Examples:
            mat4(mat3_var) -> GLSL_mat4_from_mat3(mat3_var)
            mat3(mat4_var) -> GLSL_mat3_from_mat4(mat4_var, &result) [needs special handling]
        """
        function_name = f'GLSL_{target_type}_from_{source_type}'
        return IR.CallExpression(
            function=function_name,
            arguments=arguments,
            source_location=location
        )

    def _transform_field_expression(self, node: ASTNode) -> IR.MemberAccess:
        """Transform member access (swizzling, struct field)."""
        # field_expression: base.field
        # Use field access so an interleaved comment (a named child) can't shift
        # positional indices and drop the real operand.
        base_node = node.child_by_field_name('argument')
        field_node = node.child_by_field_name('field')
        if base_node is None:
            operands = [c for c in node.named_children if c.type != "comment"]
            base_node = operands[0] if operands else None
            field_node = operands[1] if len(operands) > 1 else None

        base = self._transform_node(base_node)
        field = field_node.text if field_node else ""

        # Try to infer type for matrix operation detection
        glsl_type = None

        # Check if we've tracked this field assignment (e.g., "t.matrix" -> "mat2")
        if isinstance(base, IR.Identifier) and field_node:
            field_key = f"{base.name}.{field}"
            field_type = self.local_types.get(field_key)
            if field_type:
                glsl_type = TYPE_NAME_MAP.get(field_type)
            else:
                # Fallback: try to look up struct type in symbol table
                base_type = self.local_types.get(base.name)
                if base_type:
                    # Check if it's a struct type with fields
                    symbol = self.symbol_table.lookup(base_type)
                    if symbol and hasattr(symbol, 'metadata'):
                        fields = symbol.metadata.get('fields', {})
                        if field in fields:
                            field_type = fields[field]
                            glsl_type = TYPE_NAME_MAP.get(field_type)

        # If type not inferred yet, check for vector swizzle operations
        # This enables matrix operations on swizzled vector components
        # Examples: foo.xy * M2, V3.xyz * M3, V4.xy *= M2
        if glsl_type is None:
            base_type = self._get_type_name(base)
            if base_type and self._is_vector_type(base_type):
                # Try to infer swizzle type
                glsl_type = self._infer_swizzle_type(base_type, field)

        return IR.MemberAccess(
            base=base,
            member=field,
            glsl_type=glsl_type,
            source_location=node.start_point
        )

    def _transform_subscript_expression(self, node: ASTNode) -> IR.ArrayAccess:
        """Transform array subscript (arr[i])."""
        # subscript_expression: base[index]
        # Field access avoids comment nodes shifting positional indices.
        base_node = node.child_by_field_name('argument')
        index_node = node.child_by_field_name('index')
        if base_node is None:
            operands = [c for c in node.named_children if c.type != "comment"]
            base_node = operands[0] if operands else None
            index_node = operands[1] if len(operands) > 1 else None

        base = self._transform_node(base_node)
        index = self._transform_node(index_node) if index_node else None

        # Try to infer type for matrix operation detection
        glsl_type = None

        # Simple inference: if base is an identifier with an array type,
        # the element type is the same as the stored type
        if isinstance(base, IR.Identifier):
            element_type = self.local_types.get(base.name)
            if element_type:
                # For arrays, local_types stores the element type (e.g., "mat2" for mat2[])
                glsl_type = TYPE_NAME_MAP.get(element_type)

        return IR.ArrayAccess(
            base=base,
            index=index,
            glsl_type=glsl_type,
            source_location=node.start_point
        )

    def _transform_conditional_expression(self, node: ASTNode) -> IR.TernaryOp:
        """Transform ternary operator (cond ? a : b)."""
        # Ternary fields: condition ? consequence : alternative.
        # Use field access so interleaved comments don't break the structure.
        cond_node = node.child_by_field_name('condition')
        true_node = node.child_by_field_name('consequence')
        false_node = node.child_by_field_name('alternative')

        if cond_node is None or true_node is None or false_node is None:
            children = [c for c in node.named_children if c.type != "comment"]
            if len(children) != 3:
                raise TransformationError(
                    "Invalid ternary expression structure",
                    node.start_point
                )
            cond_node, true_node, false_node = children

        condition = self._transform_node(cond_node)
        true_expr = self._transform_node(true_node)
        false_expr = self._transform_node(false_node)

        return IR.TernaryOp(
            condition=condition,
            true_expr=true_expr,
            false_expr=false_expr,
            source_location=node.start_point
        )

    def _transform_assignment_expression(self, node: ASTNode) -> IR.AssignmentOp:
        """
        Transform assignment (x = 5, v += w, etc.).

        Special handling for:
        - Matrix compound assignments: v *= M -> v = GLSL_mul(v, M)
        - Pointer parameter assignments: param = value -> *param = value
        """
        # assignment_expression: target = value or target += value
        # Field access avoids comment nodes shifting positional indices.
        target_node = node.child_by_field_name('left')
        value_node = node.child_by_field_name('right')
        if target_node is None:
            operands = [c for c in node.named_children if c.type != "comment"]
            target_node = operands[0] if operands else None
            value_node = operands[1] if len(operands) > 1 else None

        # Find operator
        operator = '='
        for child in node.children:
            if child.type in ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']:
                operator = child.type
                break

        target = self._transform_node(target_node)
        value = self._transform_node(value_node) if value_node else None

        # Note: a pointer-param target (p = ..., p.x = ...) is already
        # dereferenced by _transform_identifier (-> *p / (*p).x), so no extra
        # wrap is needed here.

        # Track field assignments for type inference (e.g., t.matrix = mat2(...))
        if operator == '=' and isinstance(target, IR.MemberAccess) and isinstance(target.base, IR.Identifier):
            value_type = self._get_type_name(value)
            if value_type:
                # Store compound key: "base.field" -> "type"
                field_key = f"{target.base.name}.{target.member}"
                self.local_types[field_key] = value_type

        # Handle compound assignment with matrix multiplication
        if operator == '*=' and value:
            target_type = self._get_type_name(target)
            value_type = self._get_type_name(value)

            # Vector *= Matrix or Matrix *= Matrix
            if (self._is_vector_type(target_type) and self._is_matrix_type(value_type)) or \
               (self._is_matrix_type(target_type) and self._is_matrix_type(value_type)) or \
               (self._is_matrix_type(target_type) and self._is_vector_type(value_type)):
                # Transform to: target = GLSL_mul_*(target, value)
                # Get correctly typed function name
                function_name = self._get_matrix_mul_function_name(target_type, value_type)

                mul_call = IR.CallExpression(
                    function=function_name,
                    arguments=[target, value],
                    source_location=node.start_point
                )
                return IR.AssignmentOp(
                    operator='=',
                    target=target,
                    value=mul_call,
                    source_location=node.start_point
                )

        return IR.AssignmentOp(
            operator=operator,
            target=target,
            value=value,
            source_location=node.start_point
        )

    def _transform_update_expression(self, node: ASTNode) -> IR.TransformedNode:
        """Transform update expression (++i, i--, etc.)."""
        # update_expression: ++var, var++, --var, var--
        # For simplicity, convert to assignment: i++ -> i = i + 1

        # Find operator and operand
        operator = None
        operand_node = None
        is_prefix = False

        for i, child in enumerate(node.children):
            if child.type in ['++', '--']:
                operator = child.type
                is_prefix = (i == 0)  # ++ before operand = prefix
            elif child.type not in ['(', ')']:
                operand_node = child

        if operator is None or operand_node is None:
            raise TransformationError(
                "Invalid update expression structure",
                node.start_point
            )

        operand = self._transform_node(operand_node)

        # OpenCL forbids ++/-- on vector types ("cannot increment value of type
        # 'float4'"). Rewrite a vector ++/-- to a compound assignment that
        # broadcasts (v++ -> v += 1, v-- -> v -= 1). Scalars keep ++/--.
        # The emitter already renders all ++/-- as prefix, so this introduces no
        # new pre/post-fix semantic change for the common statement form.
        if self._is_vector_type(self._get_type_name(operand)):
            return IR.AssignmentOp(
                operator='+=' if operator == '++' else '-=',
                target=operand,
                value=IR.IntLiteral(value='1', source_location=node.start_point),
                source_location=node.start_point
            )

        # The code emitter will handle prefix vs postfix
        return IR.UnaryOp(
            operator=operator,
            operand=operand,
            source_location=node.start_point
        )

    def _transform_parenthesized_expression(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform parenthesized expression - preserve parentheses.

        This is critical for maintaining order of operations when the
        programmer explicitly used parentheses. Without this, expressions like:
            1.0*(2.0/iResolution.y)*(1.0/fov)
        would become:
            1.0f * 2.0f / iResolution.y * 1.0f / fov  (WRONG!)
        instead of:
            1.0f * (2.0f / iResolution.y) * (1.0f / fov)  (CORRECT!)
        """
        # Transform the inner expression
        if node.named_children:
            inner = self._transform_node(node.named_children[0])
            # Wrap in ParenthesizedExpression to preserve parentheses
            return IR.ParenthesizedExpression(
                expression=inner,
                source_location=node.start_point
            )
        return None

    # ========================================================================
    # Statements
    # ========================================================================

    def _transform_expression_statement(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform expression statement (expr;).

        Special handling for GLSL 'discard' statement which becomes 'return;'.
        """
        expr_node = node.named_children[0] if node.named_children else None
        if expr_node is None:
            # Empty statement (a bare/stray `;`, e.g. a trailing `;;`). A no-op in
            # GLSL/C — skip it. The compound-statement and top-level declaration
            # loops filter None, so the empty statement is simply dropped instead
            # of aborting the whole transform.
            return None

        # Check for GLSL 'discard' statement -> transform to 'return;'
        # In GLSL fragment shaders, 'discard' terminates fragment processing
        # In OpenCL, we use 'return;' to exit the kernel function early
        if expr_node.type == 'identifier' and expr_node.text == 'discard':
            return IR.ReturnStatement(
                value=None,
                source_location=node.start_point
            )

        # Transform the expression
        expr = self._transform_node(expr_node)

        return IR.ExpressionStatement(
            expression=expr,
            source_location=node.start_point
        )

    def _is_ct_constant(self, node) -> bool:
        """
        True if `node` is a compile-time-constant initializer that OpenCL
        accepts at program (file) scope.

        Verified on the NVIDIA CUDA target: accepted forms are bare literals and
        pure vector/matrix-literal constructors of constants (recursively),
        optionally wrapped in unary +/-/!/~ or parentheses. ANY binary
        arithmetic (`*`, `/`, `+`, ...) or function call makes the initializer
        "not a compile-time constant" — even under __constant — so those return
        False and get hoisted into the kernel body.

        Returning True here is conservative (leaves the global in place); we only
        return True for forms known to compile, so we never under-hoist a form
        that would fail.
        """
        if node is None:
            return True
        if isinstance(node, (IR.FloatLiteral, IR.IntLiteral, IR.BoolLiteral)):
            return True
        if isinstance(node, IR.ParenthesizedExpression):
            return self._is_ct_constant(node.expression)
        if isinstance(node, IR.UnaryOp):
            return self._is_ct_constant(node.operand)
        if isinstance(node, (IR.TypeConstructor, IR.ArrayConstructor)):
            return all(self._is_ct_constant(arg) for arg in (node.arguments or []))
        # BinaryOp, CallExpression, Identifier, MemberAccess, ArrayAccess,
        # TernaryOp, ArrayInitializer, ... -> not a program-scope constant.
        return False

    def _transform_declaration(self, node: ASTNode):
        """
        Transform variable declaration.

        Handles both single and comma-separated declarations:
        - Single: float x = 1.0;
        - Comma-separated: float x, y, z;
        - Comma with init: int a = 10, b = 20;
        - Const qualifier: const float foo = 0.5;

        Returns IR.Declaration for single declarations,
        IR.DeclarationList for comma-separated declarations.
        """
        type_node = node.child_by_field_name('type')

        if not type_node:
            raise TransformationError(
                "Invalid declaration structure: missing type",
                node.start_point
            )

        # Extract qualifiers (const, etc.) from declaration
        qualifiers = []
        for child in node.children:
            if child.type == 'type_qualifier':
                # type_qualifier node contains the actual qualifier keyword
                for qualifier_child in child.children:
                    if qualifier_child.type == 'const':
                        qualifiers.append('const')

        # Get type name
        glsl_type = type_node.text.strip()
        # Remove precision qualifiers for GLSL type tracking
        glsl_type = glsl_type.replace('highp ', '').replace('mediump ', '').replace('lowp ', '').strip()
        opencl_type = self._transform_type_name(type_node)

        # Collect all declarators (identifiers and init_declarators)
        # Skip type node and punctuation (,;)
        declarators = []
        for child in node.named_children:
            if child.type in ('identifier', 'init_declarator', 'array_declarator'):
                declarators.append(child)

        if not declarators:
            raise TransformationError(
                "Invalid declaration structure: no declarators found",
                node.start_point
            )

        # Transform each declarator into a Declaration node
        declarations = []
        hoisted_any = False  # category A: any declarator hoisted out of file scope
        for declarator in declarators:
            var_name = None
            base_name = None
            initializer_node = None

            # Handle different declarator types
            if declarator.type == 'identifier':
                var_name = declarator.text
                base_name = var_name
            elif declarator.type == 'array_declarator':
                # Array declaration: type name[size]
                var_name = declarator.text
                base_declarator = declarator.child_by_field_name('declarator')
                if base_declarator:
                    base_name = base_declarator.text
            elif declarator.type == 'init_declarator':
                # init_declarator has name and value children
                name_node = declarator.child_by_field_name('declarator')
                if name_node:
                    if name_node.type == 'identifier':
                        var_name = name_node.text
                        base_name = var_name
                    elif name_node.type == 'array_declarator':
                        var_name = name_node.text
                        base_declarator = name_node.child_by_field_name('declarator')
                        if base_declarator:
                            base_name = base_declarator.text
                initializer_node = declarator.child_by_field_name('value')

            if not var_name:
                raise TransformationError(
                    "Could not extract variable name from declarator",
                    declarator.start_point
                )

            # Record variable type in local environment
            if base_name:
                self.local_types[base_name] = glsl_type

            # Transform initializer if present, or create zero initializer for undefined variables
            initializer = None
            if initializer_node:
                initializer = self._transform_node(initializer_node)

                # Category A — hoist a non-constant program-scope initializer.
                # OpenCL rejects any arithmetic/call in a file-scope initializer
                # ("initializer element is not a compile-time constant"), even
                # with __constant. Emit the global bare (mutable, default-zero)
                # and record the real initializer; the host (tests/transpile.py /
                # Houdini @KERNEL) assigns it at the top of the kernel body, the
                # same pattern main_header.cl uses for `static float iTime`.
                # Skips: array globals (whole-array assignment is illegal) and
                # scalar int/uint globals (array-size/loop-bound candidates whose
                # integer constant expressions OpenCL already folds).
                if (self._global_scope
                        and '[' not in var_name
                        and opencl_type not in ('int', 'uint')
                        and not self._is_ct_constant(initializer)):
                    self.hoisted_global_inits.append((var_name, initializer))
                    initializer = None  # bare declaration; assigned in the kernel
                    hoisted_any = True
            else:
                # No explicit initializer - create zero initializer to match GLSL semantics
                # GLSL implicitly initializes undefined variables to zero, while OpenCL
                # leaves them undefined. This creates appropriate zero initializers.
                initializer = self._create_zero_initializer(glsl_type, opencl_type)

                # For arrays, wrap the initializer in ArrayInitializer with curly braces
                # OpenCL requires array initializers to be in the form: type name[size] = {...}
                if initializer and '[' in var_name:
                    initializer = IR.ArrayInitializer(
                        elements=[initializer],
                        glsl_type=None,
                        source_location=None
                    )

            # Create Declaration node (without type_name for DeclarationList)
            declarations.append(IR.Declaration(
                type_name=None,  # Will be set at DeclarationList level
                name=var_name,
                initializer=initializer,
                qualifiers=[],  # Qualifiers will be set at DeclarationList level
                source_location=declarator.start_point
            ))

        # A hoisted global is assigned in the kernel body, so it must not be
        # `const` (assignment to a const is an error). Drop const for the whole
        # statement when any declarator was hoisted.
        if hoisted_any:
            qualifiers = [q for q in qualifiers if q != 'const']

        # Return single Declaration or DeclarationList
        if len(declarations) == 1:
            # Single declaration - set type_name on the Declaration
            return IR.Declaration(
                type_name=opencl_type,
                name=declarations[0].name,
                initializer=declarations[0].initializer,
                qualifiers=qualifiers,
                source_location=node.start_point
            )
        else:
            # Comma-separated declarations
            return IR.DeclarationList(
                type_name=opencl_type,
                declarators=declarations,
                qualifiers=qualifiers,
                source_location=node.start_point
            )

    def _transform_return_statement(self, node: ASTNode) -> IR.ReturnStatement:
        """Transform return statement."""
        # return_statement may have a value or be empty (return;)
        value_node = node.named_children[0] if node.named_children else None

        if not value_node:
            # Empty return statement
            return IR.ReturnStatement(
                value=None,
                source_location=node.start_point
            )

        # Transform the return value expression
        value = self._transform_node(value_node)

        return IR.ReturnStatement(
            value=value,
            source_location=node.start_point
        )

    def _unwrap_syntax_parens(self, node: IR.TransformedNode) -> IR.TransformedNode:
        """
        Unwrap one level of ParenthesizedExpression if present.

        This is used in contexts where parentheses are already enforced by syntax
        (if/while/for conditions) to avoid double parentheses.

        Args:
            node: Transformed node, possibly a ParenthesizedExpression

        Returns:
            The inner expression if node is ParenthesizedExpression, otherwise node
        """
        if isinstance(node, IR.ParenthesizedExpression):
            return node.expression
        return node

    def _transform_if_statement(self, node: ASTNode) -> IR.IfStatement:
        """Transform if statement."""
        condition_node = node.child_by_field_name('condition')
        consequence_node = node.child_by_field_name('consequence')
        alternative_node = node.child_by_field_name('alternative')

        if not condition_node or not consequence_node:
            raise TransformationError(
                "Invalid if statement structure",
                node.start_point
            )

        condition = self._transform_node(condition_node)
        # Unwrap one level of parentheses since if syntax already requires them
        condition = self._unwrap_syntax_parens(condition)
        then_block = self._transform_node(consequence_node)
        else_block = self._transform_node(alternative_node) if alternative_node else None

        return IR.IfStatement(
            condition=condition,
            then_block=then_block,
            else_block=else_block,
            source_location=node.start_point
        )

    def _transform_else_clause(self, node: ASTNode) -> Optional[IR.TransformedNode]:
        """
        Transform else clause.

        The else_clause node wraps either:
        - An if_statement (for else-if chains)
        - A compound_statement (for final else block)

        We need to extract and transform the actual content, skipping the 'else' keyword.

        Args:
            node: else_clause AST node

        Returns:
            Transformed if statement or compound statement
        """
        # The else_clause has 'else' keyword as first child,
        # and the actual content (if_statement or compound_statement) as named child
        for child in node.named_children:
            # Transform the first named child (which is the actual else content)
            return self._transform_node(child)

        # Empty else clause (shouldn't happen in valid GLSL)
        return None

    def _transform_for_statement(self, node: ASTNode) -> IR.ForStatement:
        """Transform for loop."""
        init_node = node.child_by_field_name('initializer')
        condition_node = node.child_by_field_name('condition')
        update_node = node.child_by_field_name('update')
        body_node = node.child_by_field_name('body')

        init = self._transform_node(init_node) if init_node else None
        condition = self._transform_node(condition_node) if condition_node else None
        # Unwrap one level of parentheses since for syntax already requires them
        if condition:
            condition = self._unwrap_syntax_parens(condition)
        update = self._transform_node(update_node) if update_node else None
        body = self._transform_node(body_node) if body_node else None

        return IR.ForStatement(
            init=init,
            condition=condition,
            update=update,
            body=body,
            source_location=node.start_point
        )

    def _transform_while_statement(self, node: ASTNode) -> IR.WhileStatement:
        """Transform while loop."""
        condition_node = node.child_by_field_name('condition')
        body_node = node.child_by_field_name('body')

        if not condition_node or not body_node:
            raise TransformationError(
                "Invalid while statement structure",
                node.start_point
            )

        condition = self._transform_node(condition_node)
        # Unwrap one level of parentheses since while syntax already requires them
        condition = self._unwrap_syntax_parens(condition)
        body = self._transform_node(body_node)

        return IR.WhileStatement(
            condition=condition,
            body=body,
            source_location=node.start_point
        )

    def _transform_do_statement(self, node: ASTNode) -> IR.DoWhileStatement:
        """
        Transform do-while loop.

        Syntax: do { body } while (condition);
        """
        body_node = node.child_by_field_name('body')
        condition_node = node.child_by_field_name('condition')

        if not body_node or not condition_node:
            raise TransformationError(
                "Invalid do-while statement structure",
                node.start_point
            )

        body = self._transform_node(body_node)
        condition = self._transform_node(condition_node)
        # Unwrap one level of parentheses since while syntax already requires them
        condition = self._unwrap_syntax_parens(condition)

        return IR.DoWhileStatement(
            body=body,
            condition=condition,
            source_location=node.start_point
        )

    def _transform_break_statement(self, node: ASTNode) -> IR.BreakStatement:
        """
        Transform break statement.

        Break exits the innermost enclosing loop (for/while/do-while).
        """
        return IR.BreakStatement(
            source_location=node.start_point
        )

    def _transform_continue_statement(self, node: ASTNode) -> IR.ContinueStatement:
        """
        Transform continue statement.

        Continue skips to the next iteration of the innermost enclosing loop.
        """
        return IR.ContinueStatement(
            source_location=node.start_point
        )

    def _transform_compound_statement(self, node: ASTNode) -> IR.CompoundStatement:
        """Transform block statement ({ ... })."""
        statements = []
        for stmt in node.named_children:
            transformed = self._transform_node(stmt)
            if transformed is not None:
                statements.append(transformed)

        return IR.CompoundStatement(
            statements=statements,
            source_location=node.start_point
        )

    # ========================================================================
    # Functions
    # ========================================================================

    def _transform_function_definition(self, node: ASTNode) -> IR.FunctionDefinition:
        """
        Transform function definition.

        Also tracks pointer parameters for proper dereference/address-of handling.
        """
        # function_definition: return_type declarator body
        return_type_node = node.return_type
        declarator = node.declarator
        body_node = node.body

        if not return_type_node or not declarator or not body_node:
            raise TransformationError(
                "Invalid function definition structure",
                node.start_point
            )

        # Transform return type
        return_type = self._transform_type_name(return_type_node)

        # Extract function name
        func_name = node.name

        # Register function return type for later lookup (for matrix operation detection)
        # Store the GLSL type name (before transformation) so we can detect matrix types
        glsl_return_type = return_type_node.text.strip()
        self.user_function_return_types[func_name] = glsl_return_type

        # Transform parameters
        parameters = []
        param_info = []  # For function signature registry
        self.pointer_params.clear()  # Reset for this function

        for param_node in node.parameters:
            param = self._transform_parameter(param_node)
            if param:
                parameters.append(param)
                # Add parameter to local type environment for matrix operation detection
                self.local_types[param.name] = param.type_name

                # Track pointer parameters (for dereference handling in function
                # body). The renderpass ENTRY function's out-param (fragColor) is
                # a host-provided @KERNEL local, not a deref'able pointer, so it
                # is excluded here (a helper that merely shares the name is not).
                if param.is_pointer and func_name != self.entry_function:
                    self.pointer_params.add(param.name)

                # Store parameter info for function signature registry
                param_info.append((param.name, param.is_pointer))

        # Register function signature for call site handling
        self.function_signatures[func_name] = param_info

        # Transform body. Declarations inside a function body are local scope,
        # not program scope, so global-init hoisting must not apply to them.
        prev_global_scope = self._global_scope
        self._global_scope = False
        body = self._transform_node(body_node)
        self._global_scope = prev_global_scope

        # Clear pointer params after transformation
        self.pointer_params.clear()

        # OpenCL C has no overloading: mark user functions overloadable so
        # same-named GLSL functions of different signatures coexist. Shadertoy
        # renderpass ENTRY points must stay unmarked: a host strips their
        # signature and replaces it with a kernel wrapper (tests/transpile.py
        # re-wraps mainImage; Houdini replaces the signature with @KERNEL), which
        # would leave `__attribute__((overloadable))` dangling before the kernel.
        entry_points = {'mainImage', 'mainCubemap', 'mainSound', 'mainVR'}
        overloadable = func_name not in entry_points

        return IR.FunctionDefinition(
            return_type=return_type,
            name=func_name,
            parameters=parameters,
            body=body,
            overloadable=overloadable,
            source_location=node.start_point
        )

    def _transform_parameter(self, node: ASTNode) -> Optional[IR.Parameter]:
        """
        Transform function parameter with GLSL qualifier handling.

        GLSL qualifiers:
        - in: Default, parameter is read-only (remove qualifier)
        - out: Parameter is write-only output (use pointer, except mat3)
        - inout: Parameter is read-write (use pointer, except mat3)
        - const: Keep as-is

        OpenCL transformation:
        - in -> (remove, it's the default)
        - out -> __private TYPE* (pointer for scalars/vectors, no pointer for mat3)
        - inout -> __private TYPE* (same as out)
        - const -> const (unchanged)
        """
        # parameter_declaration: [qualifiers] type declarator

        # Extract type
        type_node = node.child_by_field_name('type')
        if not type_node:
            return None

        param_type = self._transform_type_name(type_node)

        # Extract name
        declarator = node.child_by_field_name('declarator')
        if not declarator:
            return None

        param_name = declarator.text if declarator.type == 'identifier' else ""

        # Array parameter (category K): vec3 pts[4] — keep the name and the
        # bracket suffix. Arrays already have reference semantics in C, so
        # out/inout array params skip the pointer machinery entirely: the
        # body indexes the name directly and call sites pass the array name.
        array_suffix = None
        if declarator.type == 'array_declarator':
            base_declarator = declarator.child_by_field_name('declarator')
            if base_declarator is not None:
                param_name = base_declarator.text
                array_suffix = declarator.text[len(param_name):].strip()

        # Extract GLSL qualifiers (in, out, inout, const)
        glsl_qualifiers = []
        for child in node.children:
            if child.type in ['in', 'out', 'inout', 'const']:
                glsl_qualifiers.append(child.type)

        # Transform qualifiers for OpenCL
        opencl_qualifiers = []
        is_pointer = False

        # Check if this is an output parameter (out or inout)
        is_output_param = 'out' in glsl_qualifiers or 'inout' in glsl_qualifiers

        if is_output_param and array_suffix is None:
            # Output parameters use __private address space
            opencl_qualifiers.append('__private')

            # All output parameters need explicit pointers
            is_pointer = True

        # Keep const qualifier if present (unless the mapped type already
        # carries one, e.g. `const sampler2D` -> `const IMX_Layer*`).
        if ('const' in glsl_qualifiers and not is_output_param
                and not param_type.startswith('const ')):
            opencl_qualifiers.append('const')

        # Note: 'in' qualifier is removed (it's the default in C/OpenCL)

        return IR.Parameter(
            type_name=param_type,
            name=param_name,
            qualifiers=opencl_qualifiers,
            is_pointer=is_pointer,
            array_suffix=array_suffix,
            source_location=node.start_point
        )

    # ========================================================================
    # Structs
    # ========================================================================

    def _transform_struct_specifier(self, node: ASTNode) -> IR.StructDefinition:
        """
        Transform struct definition.

        GLSL struct syntax:
            struct Name {
                type field1;
                type field2, field3;
            };

        OpenCL typedef struct syntax:
            typedef struct {
                type field1;
                type field2;
                type field3;
            } Name;

        Args:
            node: struct_specifier AST node

        Returns:
            StructDefinition IR node
        """
        # Extract struct name (type_identifier)
        struct_name = None
        field_list_node = None

        for child in node.named_children:
            if child.type == 'type_identifier':
                struct_name = child.text
            elif child.type == 'field_declaration_list':
                field_list_node = child

        if not struct_name:
            raise TransformationError(
                "Struct definition missing name",
                node.start_point
            )

        if not field_list_node:
            raise TransformationError(
                f"Struct '{struct_name}' has no field declaration list",
                node.start_point
            )

        # Transform field declarations
        fields = []
        field_info = {}  # For struct type registry

        for field_decl in field_list_node.named_children:
            if field_decl.type != 'field_declaration':
                continue

            # Extract field type and names
            field_type_node = None
            field_names = []       # as emitted, may carry an array suffix
            field_base_names = []  # registry keys (suffix stripped)

            for child in field_decl.named_children:
                # Field type can be either 'primitive_type' (float, int) or 'type_identifier' (vec3, custom types)
                if child.type in ('type_identifier', 'primitive_type'):
                    field_type_node = child
                elif child.type == 'field_identifier':
                    field_names.append(child.text)
                    field_base_names.append(child.text)
                elif child.type == 'array_declarator':
                    # Array field (category K): vec4 data[4]; — emit the name
                    # with its bracket suffix, register the base name with the
                    # element type (mirrors how local array declarations are
                    # tracked in local_types).
                    base = child.child_by_field_name('declarator')
                    if base is not None:
                        field_names.append(child.text)
                        field_base_names.append(base.text)

            if not field_type_node:
                raise TransformationError(
                    f"Field declaration in struct '{struct_name}' missing type",
                    field_decl.start_point
                )

            if not field_names:
                raise TransformationError(
                    f"Field declaration in struct '{struct_name}' missing name(s)",
                    field_decl.start_point
                )

            # Transform GLSL type to OpenCL type
            glsl_type = field_type_node.text
            opencl_type = self.type_map.get(glsl_type, glsl_type)

            # Create StructField node
            fields.append(IR.StructField(
                type_name=opencl_type,
                names=field_names,
                source_location=field_decl.start_point
            ))

            # Register field types for member access inference
            for field_name in field_base_names:
                field_info[field_name] = glsl_type

        # Register struct type in our registry
        self.struct_types[struct_name] = field_info

        # Also add to local_types for declaration type tracking
        # (not strictly necessary, but keeps consistency with other types)
        self.type_map[struct_name] = struct_name  # Struct types don't change name

        return IR.StructDefinition(
            name=struct_name,
            fields=fields,
            source_location=node.start_point
        )

    # ========================================================================
    # Preprocessor Directives (Session 9)
    # ========================================================================

    def _transform_preprocessor(self, node: ASTNode) -> IR.PreprocessorDirective:
        """
        Transform preprocessor directive.

        Preprocessor directives are already transformed by PreprocessorTransformer
        before AST parsing, so we just pass them through as-is.

        Args:
            node: Preprocessor directive AST node

        Returns:
            PreprocessorDirective IR node with raw text
        """
        # Get the raw text of the preprocessor directive
        text = node.text.strip()

        return IR.PreprocessorDirective(
            text=text,
            source_location=node.start_point
        )
