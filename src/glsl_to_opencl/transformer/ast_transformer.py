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
from collections import namedtuple
from typing import Dict, Optional, List
from ..parser.ast_nodes import ASTNode
from ..analyzer.type_checker import TypeChecker, GLSLType, TYPE_NAME_MAP
from ..analyzer.symbol_table import SymbolTable
from . import transformed_ast as IR

# Category A2 — a hoisted program-scope ARRAY global. A whole-array assignment
# (`arr = {...};`) is illegal in C, so an array cannot use the scalar hoist
# channel. Instead the host renders a temp-local-array + copy-loop block (a
# non-constant aggregate initializer is legal on a LOCAL). This record rides the
# SAME ordered `hoisted_global_inits` channel as the scalar `(name, ir)` tuples
# so inter-global declaration order is preserved; the host discriminates by
# type. `size_text` is the array length as it appears in the bare decl (a
# literal like `5`, a symbolic const like `N`, or an element count derived for
# an unsized `[]`), reused verbatim as the copy-loop bound (a kept const global
# is still in scope in the kernel body).
HoistedArrayInit = namedtuple(
    "HoistedArrayInit", "base_name elem_type size_text init_ir"
)

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

# Matrix type name (either family — declarations record 'mat3', parameters
# record 'matrix3x3') -> canonical GLSL name, for matrix cast helper naming
# (GLSL_mat3_from_mat4) and identity detection in matrix constructors.
MATRIX_NAME_TO_GLSL = {
    'mat2': 'mat2', 'mat3': 'mat3', 'mat4': 'mat4',
    'matrix2x2': 'mat2', 'matrix3x3': 'mat3', 'matrix4x4': 'mat4',
}

# Function names defined by the Houdini OpenCL headers that `main_header.cl`
# `#include`s (<interpolate.h>, <matrix.h>, <random.h>, <imx.h>, <imx_filter.h>
# and their transitive includes <typedefines.h>/<util.h>/<imx_internal.h>/
# <imx_filter_internal.h>). A user function of the same name collides: Session 1
# marks user definitions `__attribute__((overloadable))`, but these Houdini
# builtins are UNMARKED, so clang rejects the pair with "redeclaration of X must
# not have the 'overloadable' attribute" + "redefinition of X" (category D2 —
# e.g. shaders defining their own `rotate2D`, `lerp`, `fit`). Such a user
# function is renamed to `sh_<name>` at its definition, forward-declaration
# prototype, and every call site (see `_collect_function_renames`). Names that
# GLSL already remaps elsewhere (the `glsl_builtins` → `GLSL_*` set and the type
# constructors) are intentionally excluded so the rename never desyncs a call
# that was rewritten to `GLSL_*`. Extracted from the headers of Houdini
# 21.0.440; regenerate if the bundled Houdini version changes.
HOUDINI_RESERVED_FUNCTIONS = {
    'CREATE_RAND', 'CREATE_RANDOM', 'SYSfastRandom', 'SYSfloorIL',
    'SYSwang_inthash', 'applySTXform', 'applySTXformInverse',
    'applySTXformInverseVec', 'applySTXformVec', 'asfpreal2', 'asfpreal3',
    'asfpreal4', 'bilinear_interp', 'bilinear_interp_val', 'bilinear_interp_vol',
    'bufferSampleRectClipF4', 'bufferSampleRectF4', 'bufferToImage',
    'bufferToImageVec', 'bufferToPixel', 'bufferToPixelVec', 'bufferToTexture',
    'bufferToTextureVec', 'build_float2', 'build_float3', 'build_float4',
    'centerFromFace', 'computeSubdCurveCoeffsAndIndices',
    'constImageSampleRectClip', 'cornerFromCenter', 'cornerFromCenter2d',
    'dCdxF4', 'dCdxF4aligned', 'dCdyF4', 'dCdyF4aligned', 'det3', 'diag3',
    'dudxAligned', 'dudxAlignedFace', 'dudxCenterAtCorner',
    'dudxCenterAtCorner2d', 'dudxCenterAtFace', 'dudxFaceAtCenter',
    'evaluateCubicCoeffs_3', 'evaluateCubicCoeffs_vload3', 'faceFromCenter',
    'fit', 'fit01', 'fitTo01', 'image3ToWorld', 'image3ToWorldVec',
    'imageToBuffer', 'imageToBufferVec', 'imageToWorld', 'imageToWorldVec',
    'lerp', 'lerpConstant', 'lerpConstant3', 'linesegInterpolationWeights',
    'mat2det', 'mat2fromcols', 'mat2inv', 'mat2mul', 'mat2vecmul',
    'mat3Tvec2mul', 'mat3Tvecmul', 'mat3add', 'mat3copy', 'mat3diag',
    'mat3fromcols', 'mat3fromcolsd', 'mat3identity', 'mat3inv', 'mat3invtol',
    'mat3isPD', 'mat3issym', 'mat3lcombine', 'mat3lincomb2', 'mat3load',
    'mat3makesym', 'mat3mul', 'mat3mulT', 'mat3scale', 'mat3solve_LDLT',
    'mat3store', 'mat3sub', 'mat3vec2mul', 'mat3vecmul', 'mat3zero',
    'mat43vec3mul', 'mat4identity', 'mat4invert', 'mat4solvecol', 'mat4vec2mul',
    'mat4vec3mul', 'mat4vecmul', 'mitchell', 'outerprod3', 'pixelToBuffer',
    'pixelToBufferVec', 'quadInterpolationWeights', 'rotate2D', 'sampleLookup',
    'squaredNorm2', 'squaredNorm3', 'tetInterpolationWeights', 'textureToBuffer',
    'textureToBufferVec', 'trace3', 'transpose2', 'transpose3', 'transpose3d',
    'triInterpolationWeights', 'trilinear_interp', 'trilinear_interp_val',
    'trilinear_interp_vol', 'vec3prod', 'vec3sum', 'vload2f', 'vload3f',
    'vload4f', 'vstore3f', 'vstore4f', 'wh_from_dP', 'worldToImage',
    'worldToImage3', 'worldToImage3Vec', 'worldToImageVec',
}

# GLSL swizzle set 'stpq' has no OpenCL equivalent; remap to 'xyzw' once the
# base is proven to be a vector (struct fields named s/t/p/q must survive).
STPQ_TO_XYZW = str.maketrans('stpq', 'xyzw')


def _strip_type_qualifiers(type_name: str) -> str:
    """Return the bare type name, dropping any leading address-space / const
    qualifiers. Parameter types can arrive as e.g. `const matrix3x3` or
    `__global matrix4x4` (the type-name string carries the qualifier), which
    would defeat a bare-name lookup. Taking the last whitespace-separated
    token yields the underlying type."""
    if not type_name:
        return type_name
    parts = type_name.split()
    return parts[-1] if parts else type_name


class _PreprocRouteAbort(Exception):
    """Category E (Session 53) — a #if-block child is not safely routable
    through the AST; the block falls back to the raw-text passthrough."""


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

        # Names declared as arrays (e.g. `mat3 arr[4]`). local_types stores the
        # element type for arrays, so an array-of-matrix and a bare matrix are
        # indistinguishable by type alone; this set keeps the matrix column
        # subscript rewrite (M[i] -> M.cols[i]) from firing on array indexing.
        self.array_vars = set()

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

        # Set per-shader in transform(): True when the source supplies its own
        # gl_FragCoord (#define or declaration), so category-Q injection is off.
        self._gl_fragcoord_user_provided = False

        # Track function signatures for handling call sites with out parameters.
        # Maps function name -> {arity: list of parameter info (name, is_pointer)}.
        # Bucketing by arity disambiguates overloaded functions (e.g. a by-value
        # `intersect(a,b)` vs an out-param `intersect(a,b,c,d)`): a call selects
        # the overload whose parameter count matches its argument count.
        self.function_signatures = {}

        # Register GLSL built-in functions with out parameters
        # modf(x, out i) - returns fractional part, stores integer part in i
        self.function_signatures['GLSL_modf'] = {
            2: [
                ('x', False),  # input value
                ('i', True)    # output pointer (out parameter)
            ]
        }

        # Copy-in/copy-out state for vector-swizzle out-args. A swizzle (`p.xz`,
        # even `p.z`) is not an addressable lvalue in OpenCL, so a swizzle passed
        # to an out/inout param is lowered to a temp: `{ T _cico = p.xz;
        # f(&_cico); p.xz = _cico; }`. `_cico_active` is set only while
        # transforming an expression-statement (the drain point); the prelude
        # (temp decls) and writeback (copy-out assignments) buffers are drained
        # there into a wrapping CompoundStatement.
        self._cico_active = False
        self._cico_counter = 0
        self._cico_prelude = []
        self._cico_writeback = []

        # Track user-defined function return types for matrix operation detection
        # Maps function name -> GLSL type name (e.g., 'foo' -> 'mat2')
        # This is populated during transformation when function definitions are encountered
        self.user_function_return_types = {}

        # Category AI — user functions with TYPE-overloads (>=2 definitions
        # whose return types differ, e.g. `float f(float)` + `vec4 f(vec4)`).
        # user_function_return_types collapses these to ONE type (last wins),
        # so a call's inferred width is UNTRUSTWORTHY — the vector-conversion
        # ctor (category N) must not truncate on it (would emit `.xyz` on a
        # value that is actually a scalar). Populated by the pre-scan.
        self.overloaded_return_type_fns = set()

        # Category D2 — user functions whose name collides with a Houdini header
        # builtin (HOUDINI_RESERVED_FUNCTIONS). Maps original GLSL name ->
        # `sh_<name>` emitted name. Populated by `_collect_function_renames` in a
        # pre-scan (so call sites rename regardless of definition order), applied
        # at the definition, prototype, and every call site. The tracking dicts
        # above stay keyed by the ORIGINAL name — only the emitted identifier
        # changes.
        self.function_renames = {}

        # Category AE — every user-defined function name (definitions +
        # prototypes), collected by `_collect_function_renames` in the same
        # pre-scan. Used to detect a LOCAL variable that shadows a function.
        self.user_function_names = set()

        # Category AE — active local-variable renames in the CURRENT function
        # body. A local declaration whose name shadows a user function
        # (`float ao = ao(p);`) is renamed (`ao` -> `ao_v`) so the same-name
        # call keeps resolving to the function instead of the local value.
        # Maps original name -> emitted name; reset per function body.
        self.local_renames = {}

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
        # Category A1: names of program-scope int/uint globals left in place as
        # foldable integer constant expressions (possible array sizes). Used to
        # decide whether a later int/uint init is itself foldable by OpenCL.
        self._const_int_globals = set()

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

        # Category Q — gl_FragCoord builtin. Skip our body-local injection when
        # the shader already supplies gl_FragCoord itself: a `#define
        # gl_FragCoord ...` (would rewrite our injected decl into a redefinition)
        # or an own `vec4 gl_FragCoord` declaration. See
        # _transform_function_definition.
        src = ast.text
        self._gl_fragcoord_user_provided = bool(
            re.search(r'#\s*define\s+gl_FragCoord\b', src)
            or re.search(r'\b(?:vec4|float4)\s+gl_FragCoord\b', src)
        )

        # Category Q (Design B) — gl_FragCoord referenced in HELPER functions.
        # Helpers can't see the entry's kernel-local gl_FragCoord, so they read
        # it through the runtime accessor GLSL_glFragCoord() (glslHelpers.h),
        # which reconstructs the pixel coord from get_global_id() + a uniform
        # offset static. The offset static is seeded by the HDA setter
        # `shadertoy_bind_inputs()` at the top of EVERY kernel body (host header),
        # so the transpiler emits no seed of its own (retired Session 58). Only
        # the frag-token regex is set up here, driving the per-function helper
        # injection in _transform_function_definition:
        #  * frag-token regex: `gl_FragCoord` plus every object-macro alias of it
        #    (`#define F gl_FragCoord`). The alias #define survives into the
        #    emitted OpenCL and the device preprocessor expands `F` back to
        #    `gl_FragCoord`, so a helper written with `F.xy` must still receive
        #    the injected local. Resolving aliases here lets one regex cover both.
        self._gl_fragcoord_token_re = None
        if not self._gl_fragcoord_user_provided:
            aliases = set(re.findall(r'#\s*define\s+(\w+)\s+gl_FragCoord\b', src))
            tokens = {'gl_FragCoord'} | aliases
            self._gl_fragcoord_token_re = re.compile(
                r'\b(?:' + '|'.join(re.escape(t) for t in tokens) + r')\b')

        # Category D2 — pre-scan for user functions colliding with a Houdini
        # header builtin; build the rename map before any call site is walked so
        # renaming is order-independent (recursion, call-before-definition).
        self._collect_function_renames(ast)

        # Transform all top-level declarations. Top-level declarations are at
        # program (file) scope; hoisted_global_inits is (re)populated here.
        self.hoisted_global_inits = []
        self._const_int_globals = set()
        self._global_scope = True
        declarations = []
        for decl in ast.named_children:
            # Preserve file-scope comments verbatim (license/attribution
            # blocks — Shadertoy code is CC-licensed). Comments elsewhere
            # (statement/expression positions) are still dropped.
            if decl.type == 'comment':
                declarations.append(IR.Comment(
                    text=decl.text,
                    source_location=decl.start_point
                ))
                continue
            transformed = self._transform_node(decl)
            # Category AH — a struct-definition-with-variable declaration returns
            # [StructDefinition, Declaration]; splice both in as siblings.
            if isinstance(transformed, list):
                declarations.extend(transformed)
            elif transformed is not None:
                declarations.append(transformed)
        self._global_scope = False

        return IR.TranslationUnit(
            declarations=declarations,
            source_location=ast.start_point
        )

    def _collect_function_renames(self, ast: ASTNode) -> None:
        """Category D2 — pre-scan top-level function definitions and prototypes
        for names that collide with a Houdini header builtin
        (HOUDINI_RESERVED_FUNCTIONS) and populate `self.function_renames` with
        `name -> sh_<name>`.

        A user function of a reserved name emits an `overloadable` definition
        that clang refuses to place beside the unmarked Houdini builtin of the
        same name, so a shader defining such a name ALWAYS fails to compile
        today — the rename can only fix, never regress. Only user-defined names
        are collected here, so calls to the Houdini builtin itself (which the
        user never defines) are never rewritten.
        """
        self.function_renames = {}
        # Category AE — collected in the same walk (see __init__).
        self.user_function_names = set()
        # Category AI — collected in the same walk (see __init__). Detect
        # type-overloads (same name, differing return type) order-independently.
        self.overloaded_return_type_fns = set()
        seen_return_types = {}  # func name -> set of distinct GLSL return types

        def _reserve(func_name: str) -> None:
            if func_name and func_name in HOUDINI_RESERVED_FUNCTIONS:
                self.function_renames[func_name] = f"sh_{func_name}"

        for decl in ast.named_children:
            if decl.type == 'function_definition':
                if decl.name:
                    self.user_function_names.add(decl.name)
                    ret_node = decl.return_type
                    if ret_node is not None:
                        rts = seen_return_types.setdefault(decl.name, set())
                        rts.add(ret_node.text.strip())
                        if len(rts) > 1:
                            self.overloaded_return_type_fns.add(decl.name)
                _reserve(decl.name)
            elif decl.type == 'declaration':
                # A body-less prototype: `float rotate2D(vec2, float);`
                for child in decl.named_children:
                    if child.type == 'function_declarator':
                        for gc in child.children:
                            if gc.type == 'identifier':
                                self.user_function_names.add(gc.text)
                                _reserve(gc.text)
                                break
                        break

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
            'comma_expression': self._transform_comma_expression,
            # Genuine OpenCL casts appear only in Stage-0-preprocessed #if
            # block lines ((float2)(...) from vec2(...)) — routed since S53.
            'cast_expression': self._transform_cast_expression,
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
            # Normalize an uppercase 'F' suffix to lowercase 'f' — GLSL accepts
            # both (e.g. 0.95100F), but OpenCL / our FloatLiteral require the
            # lowercase form. Otherwise add 'f' when the suffix is absent.
            if text.endswith('F'):
                text = text[:-1] + 'f'
            elif not text.endswith('f'):
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

    def _transform_cast_expression(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform a genuine C-style cast (Session 53, category E).

        GLSL has no casts, but Stage-0 (PreprocessorTransformer) rewrites
        vector/scalar constructors on #if-block lines to OpenCL cast syntax
        BEFORE parsing (`vec2(x, y)` -> `(float2)(x, y)`), so an AST-routed
        block contains real cast_expression nodes. Re-emit them as a
        TypeConstructor — same `(T)(args)` spelling — with the type attached
        so vector/matrix detection sees through the cast. (Spurious grouping
        mis-parses never reach here: `_disambiguate_casts` dissolved them at
        parse time.)
        """
        type_node = node.child_by_field_name('type')
        value_node = node.child_by_field_name('value')
        if type_node is None or value_node is None:
            raise TransformationError("Malformed cast expression",
                                      node.start_point)
        type_name = type_node.text.strip()

        value = self._transform_node(value_node)
        arguments = [value]
        if isinstance(value, IR.ParenthesizedExpression):
            inner = value.expression
            arguments = (self._flatten_comma_expression(inner)
                         if isinstance(inner, IR.CommaExpression) else [inner])

        glsl_name = OPENCL_TO_GLSL_NAME.get(type_name, type_name)
        return IR.TypeConstructor(
            type_name=type_name,
            arguments=arguments,
            glsl_type=TYPE_NAME_MAP.get(glsl_name),
            source_location=node.start_point,
        )

    def _flatten_comma_expression(self, node) -> list:
        """Flatten a right-nested CommaExpression into an argument list."""
        items = [node.left]
        rest = node.right
        while isinstance(rest, IR.CommaExpression):
            items.append(rest.left)
            rest = rest.right
        items.append(rest)
        return items

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

        # Category AE — a read of a local that shadows a user function is
        # emitted under its renamed name (see _transform_declaration). Call
        # callees are not routed through here, so the function call keeps the
        # original name.
        emit_name = self.local_renames.get(name, name)

        ident = IR.Identifier(
            name=emit_name,
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

        # Sign/complement/inc/dec preserve the operand's type; without this a
        # negated column (-f) defeats matrix ctor / vector detection
        # (categories C/E). '!' excluded: its result is bool, not operand type.
        # '*' (pointer-param deref, produced by _transform_identifier for
        # out/inout reads) also passes through: local_types registers the
        # POINTEE type under the param name, so *p has p's registered type.
        if isinstance(node, IR.UnaryOp) and node.operator in ('-', '+', '~', '++', '--', '*'):
            return self._get_type_name(node.operand)

        # An assignment expression (`x = ...`, `o /= s`) evaluates to its
        # target; without this `ivec2(o /= .7)` can't see o's vector type and
        # falls back to the invalid (int2)(float2) cast (category N).
        if isinstance(node, IR.AssignmentOp):
            return self._get_type_name(node.target)

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

    def _glsl_type_from_name(self, type_name: Optional[str]):
        """
        GLSLType for a type name from EITHER naming family ('vec3'/'float3',
        'mat3'/'matrix3x3'), tolerating qualifier prefixes ('const matrix3x3').
        A struct type name is returned as the bare name string so nested member
        access (a.b.c) can keep resolving through struct_types. None if unknown.
        """
        if not type_name:
            return None
        bare = _strip_type_qualifiers(type_name)
        glsl = OPENCL_TO_GLSL_NAME.get(bare, MATRIX_NAME_TO_GLSL.get(bare, bare))
        resolved = TYPE_NAME_MAP.get(glsl)
        if resolved is not None:
            return resolved
        if bare in self.struct_types:
            return bare
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

    def _vector_ctor_arg_type(self, arg: IR.TransformedNode) -> Optional[str]:
        """
        Vector type name of a single vector-constructor argument, for the
        category-N conversion decision only.

        Falls back for an arithmetic/bitwise BinaryOp whose overall type
        `_get_type_name` can't resolve because ONE operand is statically
        untypeable — the common case being an object-like macro constant
        (`#define scale 20.`, which survives as a bare identifier since object
        macros aren't inlined). If the other operand is a proven vector, the
        result has that vector's width and base (GLSL broadcast /
        componentwise). `ivec2(uv / scale)` then converts (`convert_int2`)
        instead of falling back to the invalid `(int2)(float2)` C cast.

        Safe for the CONVERSION decision specifically: the fallback only fires
        when one operand is a GENUINE vector, so `vec op x` is a vector of that
        vector's shape regardless of the other operand. Deliberately NOT wired
        into `_infer_binary_op_type` (whose result feeds the multi-arg
        truncation budgeter — that path would over-count when a flat
        `local_types` entry is a stale vector, dropping a real ctor arg).
        """
        arg_type = self._get_type_name(arg)
        if arg_type is not None:
            return arg_type
        if isinstance(arg, IR.BinaryOp) and arg.operator in (
                '+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>'):
            left_type = self._get_type_name(arg.left)
            right_type = self._get_type_name(arg.right)
            if self._is_vector_type(left_type) and not right_type:
                return left_type
            if not left_type and self._is_vector_type(right_type):
                return right_type
        return arg_type

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
        # Category AI: if the arg's width traces to a TYPE-overloaded user
        # function, the inferred type is untrustworthy (return types collapse
        # to one — see overloaded_return_type_fns). Truncating on it would emit
        # `.xyz` on what is actually a scalar (`vec3(f(scalar))`). Keep the
        # plain cast: `(float3)(scalar)` broadcasts correctly, and is an
        # identity for a same-type vector.
        if self._expr_type_uses_overloaded_fn(arg):
            return None

        # Scalar-from-vector: GLSL float(vecN)/int(vecN)/uint(vecN)/bool(vecN)
        # extracts component .x (then converts the scalar). OpenCL rejects the
        # C cast (float)(float3_expr) just as it does the vector cast; emit
        # arg.x, wrapping in a scalar cast when the element base also changes
        # (int(vec3) -> (int)(vec3.x)). A scalar-argument ctor keeps the plain
        # cast (its arg_info is None below).
        if function_name in ('float', 'int', 'uint', 'bool'):
            arg_type = self._get_type_name(arg)
            arg_info = VECTOR_TYPE_INFO.get(arg_type) if arg_type else None
            if arg_info is None:
                return None  # scalar / unknown argument: keep the plain cast
            arg_base, _ = arg_info
            base = arg
            if isinstance(arg, (IR.BinaryOp, IR.UnaryOp, IR.TernaryOp)):
                base = IR.ParenthesizedExpression(
                    expression=arg, source_location=location)
            dot_x = IR.MemberAccess(
                base=base,
                member='x',
                glsl_type=TYPE_NAME_MAP.get(arg_base),
                source_location=location
            )
            if arg_base == function_name:
                return dot_x  # base already matches: .x is the scalar
            return IR.TypeConstructor(
                type_name=opencl_type,
                arguments=[dot_x],
                glsl_type=TYPE_NAME_MAP.get(function_name),
                source_location=location
            )

        target = VECTOR_TYPE_INFO.get(function_name)
        if target is None:
            return None  # scalar/matrix/sampler constructor: not this rewrite
        target_base, target_width = target

        arg_type = self._vector_ctor_arg_type(arg)
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
        # Check both GLSL and OpenCL matrix names, seeing through a leading
        # qualifier prefix (const / __global) that parameter types carry.
        bare = _strip_type_qualifiers(type_name)
        return bare in ['mat2', 'mat3', 'mat4', 'matrix2x2', 'matrix3x3', 'matrix4x4']

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
        # GLSL allows xyzw (coordinate), rgba (color) or stpq (texcoord),
        # but not mixed
        coord_chars = set('xyzw')
        color_chars = set('rgba')
        swizzle_chars = set(swizzle)

        texcoord_chars = set('stpq')

        is_coord = swizzle_chars.issubset(coord_chars)
        is_color = swizzle_chars.issubset(color_chars)
        is_texcoord = swizzle_chars.issubset(texcoord_chars)

        if not (is_coord or is_color or is_texcoord):
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

        # Normalize to GLSL names for lookup (stripping qualifier prefixes
        # like 'const matrix3x3' that parameter types can carry)
        left_type = _strip_type_qualifiers(left_type)
        right_type = _strip_type_qualifiers(right_type)
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
        # Qualifier prefixes ('const matrix3x3') pass _is_matrix_type but would
        # defeat the exact-name dim lookup below.
        left_type = _strip_type_qualifiers(left_type)
        right_type = _strip_type_qualifiers(right_type)

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

    def _widest_vector_arg_type(
        self,
        arguments: List[IR.TransformedNode]
    ) -> Optional[str]:
        """
        Type name of the widest-vector argument (genType result of a
        broadcasting builtin like min/max/clamp/step/smoothstep/mix/pow/mod).
        Scalars broadcast to the vector, so the result width follows the widest
        arg. Falls back to the first typed argument when none is a vector.
        """
        best = None
        best_width = -1
        for arg in arguments:
            type_name = self._get_type_name(arg)
            if not type_name:
                continue
            info = VECTOR_TYPE_INFO.get(_strip_type_qualifiers(type_name))
            width = info[1] if info else 0
            if width > best_width:
                best_width = width
                best = type_name
        return best

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

        # Matrix constructor dispatchers (matrix_ops.h). Stage-0 pre-renames
        # `matN(` -> `GLSL_matN(` on #if-block / #define lines BEFORE parsing,
        # so an AST-routed block (Session 53) sees the renamed spelling — type
        # it like the GLSL ctor or `GLSL_mat2(...) * v` misses matmul lowering.
        if function_name in ('GLSL_mat2', 'GLSL_mat3', 'GLSL_mat4'):
            return TYPE_NAME_MAP.get(function_name[5:])

        # Matrix functions - return same type as input. Normalize the argument
        # type (parameters register OpenCL names like 'matrix3x3', possibly
        # qualified) or inverse(param) * v silently misses matmul detection.
        if (function_name.startswith('GLSL_transpose')
                or function_name.startswith('GLSL_inverse')
                or function_name.startswith('GLSL_matrixCompMult')):
            if arguments:
                arg_type = self._get_type_name(arguments[0])
                resolved = self._glsl_type_from_name(arg_type)
                if isinstance(resolved, GLSLType):
                    return resolved

        # outerProduct(vecN, vecN) -> matN
        elif function_name == 'GLSL_outerProduct' and arguments:
            arg_type = _strip_type_qualifiers(self._get_type_name(arguments[0]))
            vec_info = VECTOR_TYPE_INFO.get(arg_type) if arg_type else None
            if vec_info:
                return TYPE_NAME_MAP.get(f'mat{vec_info[1]}')

        # Determinant - always returns float
        elif function_name.startswith('GLSL_determinant'):
            return TYPE_NAME_MAP.get('float')

        # Vector functions that return the same type as their first argument
        # normalize, abs, sign, floor, ceil, trunc, fract, sqrt, inversesqrt, etc.
        # `round`/`roundEven` map to the NATIVE OpenCL round (no GLSL_ prefix),
        # so they must be listed by their bare name — ivec2(round(uv)) (Xs3fRB).
        vector_passthrough_functions = [
            'round', 'roundEven',
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
        # min, max, clamp, mix, step, smoothstep, pow, mod. GLSL broadcasts
        # scalars, so the genType result follows the WIDEST argument, not arg[0]
        # — `step(0.25, p3)` returns vec3 (4ljyRc), not float.
        elif function_name in ['GLSL_min', 'GLSL_max', 'GLSL_clamp', 'GLSL_mix',
                                'GLSL_step', 'GLSL_smoothstep', 'GLSL_pow', 'GLSL_mod'] and arguments:
            arg_type = self._widest_vector_arg_type(arguments)
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

        # For arithmetic operations (+, -, *, /, %) and bitwise/shift ops
        # (&, |, ^, <<, >>). GLSL bitwise ops apply only to int/uint scalars
        # and vectors and follow the same promotion shape as arithmetic
        # (vec op scalar -> vec, vec op vec -> vec, shift keeps the left type
        # since the left operand dominates). Without this, `iuv & 7` infers no
        # type and vec2(iuv & 7) falls back to the invalid (float2)(int2) cast.
        if operator in ['+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>']:
            # Scalar op scalar = scalar
            if self._is_scalar_type(left_type) and self._is_scalar_type(right_type):
                # Return the "larger" type (float > int > uint)
                if left_type == 'float' or right_type == 'float':
                    return TYPE_NAME_MAP.get('float')
                elif left_type == 'int' or right_type == 'int':
                    return TYPE_NAME_MAP.get('int')
                return TYPE_NAME_MAP.get(left_type)

            # Vector op vector = vector (same type). Normalize BEFORE the
            # TYPE_NAME_MAP lookup: parameters register OpenCL names
            # ('float2'), which TYPE_NAME_MAP does not know — the equal-name
            # early return used to silently yield None for float2+float2.
            if self._is_vector_type(left_type) and self._is_vector_type(right_type):
                left_glsl = OPENCL_TO_GLSL_NAME.get(left_type, left_type)
                right_glsl = OPENCL_TO_GLSL_NAME.get(right_type, right_type)
                if left_glsl == right_glsl:
                    return TYPE_NAME_MAP.get(left_glsl)

            # Vector op scalar = vector (or scalar op vector = vector)
            if self._is_vector_type(left_type) and self._is_scalar_type(right_type):
                return TYPE_NAME_MAP.get(OPENCL_TO_GLSL_NAME.get(left_type, left_type))
            if self._is_scalar_type(left_type) and self._is_vector_type(right_type):
                return TYPE_NAME_MAP.get(OPENCL_TO_GLSL_NAME.get(right_type, right_type))

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

    def _resolve_binary_operand_type(
        self,
        operand: IR.TransformedNode,
        type_name: Optional[str]
    ) -> Optional[str]:
        """
        Resolve a binary operand's type name, falling back to the glsl_type
        attribute of a CallExpression / BinaryOp (or a user function's return
        type in the symbol table) when _get_type_name comes up empty. Needed so
        matrix arithmetic detection works even when a side is an un-annotated
        call or nested binary op (e.g. `m2 * (a / b)`).
        """
        if type_name:
            return type_name
        if isinstance(operand, IR.CallExpression):
            if getattr(operand, 'glsl_type', None):
                return str(operand.glsl_type)
            if operand.function in self.symbol_table.symbols:
                func_symbol = self.symbol_table.lookup(operand.function)
                if func_symbol and hasattr(func_symbol, 'glsl_type'):
                    return str(func_symbol.glsl_type)
        if isinstance(operand, IR.BinaryOp):
            if getattr(operand, 'glsl_type', None):
                return str(operand.glsl_type)
        return type_name

    def _transform_matrix_componentwise(
        self,
        operator: str,
        left: IR.TransformedNode,
        right: IR.TransformedNode,
        left_type: Optional[str],
        right_type: Optional[str],
        node: ASTNode
    ) -> Optional[IR.TransformedNode]:
        """
        Category H: componentwise matrix arithmetic that OpenCL's struct matrix
        types can't express with native operators.

        GLSL semantics (all componentwise; `M + s` adds `s` to EVERY element,
        not just the diagonal):
          M * s, s * M -> GLSL_matN_muls(M, s)
          M / s        -> GLSL_matN_divs(M, s)
          s / M        -> GLSL_matN_rdiv(s, M)
          M + s, s + M -> GLSL_matN_adds(M, s)
          M - s        -> GLSL_matN_subs(M, s)
          s - M        -> GLSL_matN_rsub(s, M)
          M + M        -> GLSL_matN_add(A, B)
          M - M        -> GLSL_matN_sub(A, B)
          M / M        -> GLSL_matN_div(A, B)

        Matrix * matrix for `*` is linear-algebra multiplication and is handled
        by the caller before this runs, so it never reaches here. Returns None
        for any non-matrix shape (leaving native emission intact).
        """
        left_mat = self._is_matrix_type(left_type)
        right_mat = self._is_matrix_type(right_type)

        # A "scalar" operand broadcasts componentwise. For +, -, / an operand
        # that is neither matrix nor vector is necessarily scalar even when its
        # type failed to infer (None), because vector±matrix / matrix/vector are
        # illegal GLSL — so a matrix's partner under +,-,/ can only be a scalar.
        # `*` stays strict (an untyped partner could be a vector -> matmul, which
        # is handled by the caller; never steal it as a scalar scale).
        def _is_scalar_operand(tn):
            if self._is_scalar_type(tn):
                return True
            if operator == '*':
                return False
            return not self._is_matrix_type(tn) and not self._is_vector_type(tn)

        left_scalar = (not left_mat) and _is_scalar_operand(left_type)
        right_scalar = (not right_mat) and _is_scalar_operand(right_type)

        # Size (2/3/4) comes from whichever operand is a matrix.
        mat_type = left_type if left_mat else right_type
        glsl_name = MATRIX_NAME_TO_GLSL.get(_strip_type_qualifiers(mat_type or ''))
        if glsl_name is None:
            return None
        n = glsl_name[-1]  # '2' / '3' / '4'
        result_type = TYPE_NAME_MAP.get(glsl_name)

        function_name = None
        arguments = None

        if left_mat and right_scalar:
            suffix = {'*': 'muls', '/': 'divs', '+': 'adds', '-': 'subs'}.get(operator)
            if suffix:
                function_name = f'GLSL_mat{n}_{suffix}'
                arguments = [left, right]
        elif left_scalar and right_mat:
            if operator == '*':
                function_name, arguments = f'GLSL_mat{n}_muls', [right, left]
            elif operator == '+':
                function_name, arguments = f'GLSL_mat{n}_adds', [right, left]
            elif operator == '-':
                function_name, arguments = f'GLSL_mat{n}_rsub', [left, right]
            elif operator == '/':
                function_name, arguments = f'GLSL_mat{n}_rdiv', [left, right]
        elif left_mat and right_mat:
            suffix = {'+': 'add', '-': 'sub', '/': 'div'}.get(operator)
            if suffix:
                function_name = f'GLSL_mat{n}_{suffix}'
                arguments = [left, right]

        if function_name is None:
            return None

        return IR.CallExpression(
            function=function_name,
            arguments=arguments,
            glsl_type=result_type,
            source_location=node.start_point
        )

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

        # Matrix arithmetic (categories: matrix mul, plus category H's
        # componentwise scalar/matrix ops). Resolve operand type names with the
        # call/binaryop fallbacks first so detection works for all of *, /, +, -.
        if operator in ('*', '/', '+', '-'):
            left_type = self._resolve_binary_operand_type(left, self._get_type_name(left))
            right_type = self._resolve_binary_operand_type(right, self._get_type_name(right))

            # M * v / v * M / M * M — matrix multiplication (linear algebra).
            if operator == '*':
                is_matrix_mul = (
                    (self._is_matrix_type(left_type) and self._is_vector_type(right_type)) or
                    (self._is_vector_type(left_type) and self._is_matrix_type(right_type)) or
                    (self._is_matrix_type(left_type) and self._is_matrix_type(right_type))
                )
                if is_matrix_mul:
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

                # Category E fallback: one side is a proven matrix but the
                # partner is statically untypeable (a #define'd identifier, an
                # unresolvable call...). GLSL guarantees the partner is a
                # scalar, matching vector, or matrix — dispatch through the
                # overloadable GLSL_mul (matrix_ops.h) and let clang overload
                # resolution pick. A typed-but-non-matrix pair falls through
                # to native emission unchanged.
                if ((left_type is None) != (right_type is None)) and \
                        self._is_matrix_type(left_type or right_type):
                    return IR.CallExpression(
                        function='GLSL_mul',
                        arguments=[left, right],
                        source_location=node.start_point
                    )

            # Category H: componentwise matrix arithmetic (M*s, s*M, M/s,
            # M+/-s, s+/-M, M+/-M, M/M). OpenCL matrix structs reject native
            # operators; rewrite to a GLSL_matN_* helper.
            componentwise = self._transform_matrix_componentwise(
                operator, left, right, left_type, right_type, node)
            if componentwise is not None:
                return componentwise

        # Infer result type for this binary operation
        result_type = self._infer_binary_op_type(operator, left, right)

        # GLSL aggregate vector comparison (category O): `v1 == v2` yields a
        # SCALAR bool ("all components equal"), `v1 != v2` "any component
        # differs". The OpenCL operators yield an int-vector mask instead,
        # which is invalid wherever a scalar is required (if/ternary/&&/
        # return/bool init). Wrap at the producer: all(l == r) / any(l != r).
        # OpenCL's relational -1-for-true sets the MSB that all()/any() test.
        # Only the OPERATOR spelling is aggregate in GLSL — the lessThan/
        # equal/... builtins are bvec producers and are lowered elsewhere
        # (_transform_call_expression), so their masks stay raw.
        if operator in ('==', '!='):
            left_type = self._get_type_name(left)
            right_type = self._get_type_name(right)

            # GLSL matrix == / != is aggregate equality on a struct type in
            # OpenCL — the operator is rejected outright. Lower to the
            # overloadable GLSL_mat_eq helper (matrix_ops.h).
            if self._is_matrix_type(left_type) or self._is_matrix_type(right_type):
                eq_call = IR.CallExpression(
                    function='GLSL_mat_eq',
                    arguments=[left, right],
                    glsl_type=TYPE_NAME_MAP.get('bool'),
                    source_location=node.start_point
                )
                if operator == '==':
                    return eq_call
                return IR.UnaryOp(
                    operator='!',
                    operand=eq_call,
                    source_location=node.start_point
                )

            if self._is_vector_type(left_type) or self._is_vector_type(right_type):
                return IR.CallExpression(
                    function='all' if operator == '==' else 'any',
                    arguments=[IR.BinaryOp(
                        operator=operator,
                        left=left,
                        right=right,
                        glsl_type=result_type,
                        source_location=node.start_point
                    )],
                    glsl_type=TYPE_NAME_MAP.get('bool'),
                    source_location=node.start_point
                )

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

        # Unary minus on a matrix: OpenCL matrix types are structs, so a raw
        # -M is rejected ("invalid argument type to unary expression"). GLSL
        # -M negates every component == componentwise scale by -1.
        if operator == '-':
            operand_type = self._resolve_binary_operand_type(
                operand, self._get_type_name(operand))
            if self._is_matrix_type(operand_type):
                glsl_name = MATRIX_NAME_TO_GLSL.get(
                    _strip_type_qualifiers(operand_type))
                return IR.CallExpression(
                    function=f'GLSL_mat{glsl_name[-1]}_muls',
                    arguments=[operand, IR.FloatLiteral(value='-1.0f')],
                    glsl_type=TYPE_NAME_MAP.get(glsl_name),
                    source_location=node.start_point
                )

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
        # Category D2 — the callee's original (pre-remap) name, for the Houdini
        # collision rename applied at the user-call return below.
        original_function_name = function_name

        # Transform arguments
        arguments = []
        for arg in node.arguments:
            transformed_arg = self._transform_node(arg)
            if transformed_arg:
                arguments.append(transformed_arg)

        location = node.start_point

        # GLSL array `.length()` method (UNKNOWN sub-cluster): a compile-time
        # element count with no OpenCL equivalent. The post-process builtin
        # regex would otherwise turn `arr.length()` into `arr.GLSL_length()`
        # ("member reference base type '... [N]' is not a structure or union").
        # Rewrite to the standard C count idiom `(sizeof(arr)/sizeof(arr[0]))`,
        # a compile-time constant needing no size tracking. Guarded on the exact
        # shape GLSL's array method takes: a zero-arg call whose callee is a
        # field access named `length` (the free builtin length(v) has an
        # identifier callee and is untouched).
        if (function_node is not None
                and function_node.type == 'field_expression'
                and len(arguments) == 0):
            field_node = function_node.child_by_field_name('field')
            if field_node is not None and field_node.text == 'length':
                base_node = function_node.child_by_field_name('argument')
                if base_node is not None:
                    base_ir = self._transform_node(base_node)
                    elem_ir = IR.ArrayAccess(
                        base=base_ir,
                        index=IR.IntLiteral(value='0', source_location=location),
                        source_location=location,
                    )
                    return IR.ParenthesizedExpression(
                        expression=IR.BinaryOp(
                            operator='/',
                            left=IR.CallExpression(
                                function='sizeof', arguments=[base_ir],
                                source_location=location),
                            right=IR.CallExpression(
                                function='sizeof', arguments=[elem_ir],
                                source_location=location),
                            source_location=location,
                        ),
                        source_location=location,
                    )

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

        # HLSL-style type aliases (category J): shaders that `#define float2
        # vec2` call the constructor by its OpenCL spelling `float2(x, y)`.
        # tree-sitter parses that as an ordinary identifier (float2 is not a
        # GLSL type), so it never matched type_map. Normalize the callee to
        # its GLSL name so the constructor logic below — including the
        # category-N single-arg conversions — emits `(float2)(x, y)`.
        if function_name in OPENCL_TO_GLSL_NAME:
            function_name = OPENCL_TO_GLSL_NAME[function_name]

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

            # Multi-arg vector ctors: GLSL truncates excess components
            # (vec3(vec2, vec4) uses the first 3 of the 2+4), but OpenCL's
            # (float3)(...) literal flattens ALL 6 and clang rejects it
            # (category AF). Swizzle the boundary-crossing arg down to just
            # the components still needed. Only fires on genuine overflow.
            if len(arguments) >= 2 and opencl_type in VECTOR_TYPE_INFO:
                arguments = self._truncate_overflow_ctor_args(
                    opencl_type, arguments, location)

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
            # Matrix functions (Session 5; compMult/outerProduct Session 19)
            'transpose', 'inverse', 'determinant',
            'matrixCompMult', 'outerProduct',
        }

        if function_name in glsl_builtins:
            function_name = f'GLSL_{function_name}'

        # Add type suffix for mat3/mat4 matrix functions
        # mat2 uses base name (GLSL_transpose), mat3/mat4 use suffixes; an
        # UNRESOLVED argument type keeps the bare name, which matrix_ops.h
        # defines as an overloadable dispatcher across all sizes.
        # (GLSL_outerProduct is overloadable-only — never suffixed.)
        matrix_functions = ['GLSL_transpose', 'GLSL_inverse', 'GLSL_determinant', 'GLSL_matrixCompMult']
        if function_name in matrix_functions and arguments:
            arg_type = _strip_type_qualifiers(self._get_type_name(arguments[0]))
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
        # Select the overload whose arity matches this call so a by-value call
        # is not pointerised by a same-named out-param overload (and vice versa).
        if function_name in self.function_signatures:
            param_info = self.function_signatures[function_name].get(len(arguments))
        else:
            param_info = None
        if param_info is not None:
            for i, (param_name, is_pointer) in enumerate(param_info):
                if is_pointer and i < len(arguments):
                    arg = arguments[i]
                    # A pointer param read was auto-deref'd to *p; pass the
                    # pointer straight through to the callee's pointer param.
                    if (isinstance(arg, IR.UnaryOp) and arg.operator == '*'
                            and isinstance(arg.operand, IR.Identifier)
                            and arg.operand.name in self.pointer_params):
                        arguments[i] = arg.operand
                    # A local variable, array element, or struct field: take
                    # its address. A vector swizzle member (v.xy) is excluded —
                    # &v.xy is not a valid address — but a struct field (cam.ray)
                    # is a real lvalue, so &cam.ray is valid and required.
                    elif (isinstance(arg, (IR.Identifier, IR.ArrayAccess))
                          or self._is_struct_field_access(arg)):
                        arguments[i] = IR.UnaryOp(
                            operator='&',
                            operand=arg,
                            source_location=arg.source_location
                        )
                    # A vector swizzle out-arg (`pR(p.xz, ...)` — the hg_sdf
                    # rotate/mirror idiom): not addressable, so lower to
                    # copy-in/copy-out via a temp. Only inside an
                    # expression-statement (where the temp decl + writeback can
                    # be drained into a wrapping block); elsewhere fall through
                    # unchanged.
                    elif self._cico_active and self._is_vector_swizzle(arg):
                        cico = self._make_swizzle_copy_in_out(arg)
                        if cico is not None:
                            arguments[i] = cico

        # Category D2 — a call to a user function whose name collides with a
        # Houdini builtin is emitted under its `sh_<name>` rename. Keyed by the
        # ORIGINAL callee name (type inference / signature lookup above used it).
        if original_function_name in self.function_renames:
            function_name = self.function_renames[original_function_name]

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
        cols = {'mat2': 2, 'mat3': 3, 'mat4': 4}[mat_type]
        total = cols * cols  # GLSL resolves matrix ctors by TOTAL components

        if num_args == 1:
            arg = arguments[0]
            arg_type_name = self._get_type_name(arg)

            # Matrix argument: identity (mat3(m3) is m3) or size cast.
            # MATRIX_NAME_TO_GLSL accepts both name families — declarations
            # record 'mat3' in local_types, parameters record 'matrix3x3'.
            src_glsl = MATRIX_NAME_TO_GLSL.get(arg_type_name)
            if src_glsl is not None:
                if src_glsl == mat_type:
                    return arg
                return self._create_matrix_cast(mat_type, src_glsl, arguments, location)

            # mat2(vec4): the four components fill the matrix column-major
            # (the animated-rotation idiom mat2(cos(t+vec4(...)))). Emitting
            # the diagonal helper here passed a float4 to a float parameter.
            if arg_type_name in VECTOR_TYPE_INFO:
                if mat_type == 'mat2' and VECTOR_TYPE_INFO[arg_type_name][1] == 4:
                    return IR.CallExpression(
                        function='GLSL_mat2_from_vec4',
                        arguments=arguments,
                        glsl_type=TYPE_NAME_MAP.get(mat_type),
                        source_location=location
                    )

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
            'mat2': ('vec2', 'float2'),
            'mat3': ('vec3', 'float3'),
            'mat4': ('vec4', 'float4')
        }
        vec_type, opencl_vec = column_patterns[mat_type]
        if num_args == cols and self._are_all_vector_type(arguments, vec_type, opencl_vec):
            return IR.CallExpression(
                function=f'GLSL_{mat_type}_cols',
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(mat_type),
                source_location=location
            )

        # Full matrix constructor: one scalar per element
        if num_args == total:
            return IR.CallExpression(
                function=f'GLSL_{mat_type}',
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(mat_type),
                source_location=location
            )

        # Mixed scalar/vector runs — mat2(a, -a.y, a.x), mat3(v2, s, v4, v2)…
        # GLSL consumes components in order (column-major), so flatten every
        # vector argument into its components and emit the flat GLSL_matN ctor.
        widths = [self._ctor_component_count(a) for a in arguments]
        if None not in widths and sum(widths) == total:
            return IR.CallExpression(
                function=f'GLSL_{mat_type}',
                arguments=self._flatten_matrix_ctor_args(arguments, widths, location),
                glsl_type=TYPE_NAME_MAP.get(mat_type),
                source_location=location
            )

        # Untypeable arguments with column arity: assume columns. The shader
        # compiled on Shadertoy, so the ctor was valid GLSL, and N args for
        # matN is overwhelmingly the column form — this keeps type-inference
        # gaps (calls, globals) from killing the whole shader at transpile.
        if num_args == cols:
            return IR.CallExpression(
                function=f'GLSL_{mat_type}_cols',
                arguments=arguments,
                glsl_type=TYPE_NAME_MAP.get(mat_type),
                source_location=location
            )

        # Unsupported number of arguments
        raise TransformationError(
            f"Invalid number of arguments for {mat_type} constructor: {num_args}",
            location
        )

    def _ctor_component_count(self, node: IR.TransformedNode) -> Optional[int]:
        """
        Total scalar components a matrix-constructor argument contributes:
        1 for scalars/literals, N for vecN, None when unknown (matrices are
        None too — they are only legal as a sole argument, handled earlier).
        """
        if isinstance(node, (IR.FloatLiteral, IR.IntLiteral, IR.BoolLiteral)):
            return 1
        type_name = self._get_type_name(node)
        if type_name is None:
            return None
        if type_name in VECTOR_TYPE_INFO:
            return VECTOR_TYPE_INFO[type_name][1]
        if self._is_scalar_type(type_name):
            return 1
        return None

    def _flatten_matrix_ctor_args(
        self,
        arguments: List[IR.TransformedNode],
        widths: List[int],
        location: tuple
    ) -> List[IR.TransformedNode]:
        """
        Flatten mixed scalar/vector ctor arguments into a flat component list:
        [a(vec2), s, t] -> [a.x, a.y, s, t]. Non-postfix expressions get
        parenthesized so the swizzle binds: (a + b).x. Vector expressions are
        duplicated per component — fine for the (pure) shader code this serves.
        """
        postfix_safe = (IR.Identifier, IR.MemberAccess, IR.ArrayAccess,
                        IR.CallExpression, IR.ParenthesizedExpression)
        components = []
        for arg, width in zip(arguments, widths):
            if width == 1:
                components.append(arg)
                continue
            base = arg
            if not isinstance(arg, postfix_safe):
                base = IR.ParenthesizedExpression(
                    expression=arg, source_location=location)
            elem_base = VECTOR_TYPE_INFO[self._get_type_name(arg)][0]
            elem_type = TYPE_NAME_MAP.get('float' if elem_base == 'bool' else elem_base)
            for c in 'xyzw'[:width]:
                components.append(IR.MemberAccess(
                    base=base,
                    member=c,
                    glsl_type=elem_type,
                    source_location=location
                ))
        return components

    def _expr_type_uses_user_fn(self, node: IR.TransformedNode) -> bool:
        """
        True if the (vector) type — hence component width — of `node` could be
        derived from a user-function call, whose return type may be an
        overload-mismatched over-count (see _truncate_overflow_ctor_args).
        Walks only the type-DETERMINING sub-expressions: operands of unary /
        binary / assignment / parenthesized nodes, and the callee/args of a
        call. A swizzle's width comes from its member string, not its base, so
        member-access does not need its base inspected here.
        """
        if isinstance(node, IR.CallExpression):
            if node.function in self.user_function_names:
                return True
            return any(self._expr_type_uses_user_fn(a) for a in node.arguments)
        if isinstance(node, IR.ParenthesizedExpression):
            return self._expr_type_uses_user_fn(node.expression)
        if isinstance(node, IR.UnaryOp):
            return self._expr_type_uses_user_fn(node.operand)
        if isinstance(node, IR.BinaryOp):
            return (self._expr_type_uses_user_fn(node.left)
                    or self._expr_type_uses_user_fn(node.right))
        if isinstance(node, IR.AssignmentOp):
            return self._expr_type_uses_user_fn(node.target)
        return False

    def _expr_type_uses_overloaded_fn(self, node: IR.TransformedNode) -> bool:
        """
        Like _expr_type_uses_user_fn, but True only when the type-DETERMINING
        sub-expression is a call to a *type-overloaded* user function (category
        AI) — whose collapsed single return type mis-infers the width. Walks the
        same type-determining spine (call callee/args, unary/binary/assignment/
        parenthesized operands); a swizzle's width comes from its member string,
        not its base, so member-access bases are not inspected.
        """
        if isinstance(node, IR.CallExpression):
            if node.function in self.overloaded_return_type_fns:
                return True
            return any(self._expr_type_uses_overloaded_fn(a)
                       for a in node.arguments)
        if isinstance(node, IR.ParenthesizedExpression):
            return self._expr_type_uses_overloaded_fn(node.expression)
        if isinstance(node, IR.UnaryOp):
            return self._expr_type_uses_overloaded_fn(node.operand)
        if isinstance(node, IR.BinaryOp):
            return (self._expr_type_uses_overloaded_fn(node.left)
                    or self._expr_type_uses_overloaded_fn(node.right))
        if isinstance(node, IR.AssignmentOp):
            return self._expr_type_uses_overloaded_fn(node.target)
        return False

    def _truncate_overflow_ctor_args(
        self,
        opencl_type: str,
        arguments: List[IR.TransformedNode],
        location: tuple
    ) -> List[IR.TransformedNode]:
        """
        GLSL vector constructors truncate excess components; OpenCL's literal
        syntax does not (category AF). When the summed component width of the
        args EXCEEDS the target vector size, budget the target across the args
        and swizzle the boundary-crossing arg down to just the components still
        needed (`vec3(vec2, vec4)` -> `(float3)(v2, v4.x)`), dropping any
        fully-excess trailing args entirely.

        Only truncates on genuine overflow — an exactly-filled or under-filled
        ctor is returned unchanged (never pads; under-fill is a legal/different
        case). If any arg's width can't be inferred, we cannot safely budget,
        so the whole arg list is left untouched (no guess).
        """
        target = VECTOR_TYPE_INFO[opencl_type][1]
        # A width that traces to a user-function return type is UNTRUSTWORTHY:
        # user_function_return_types keeps ONE type per name, so an overloaded
        # fn (`vec2 logc(vec2)` + `vec4 logc(vec4)`) mis-infers width and would
        # over-count -> we would falsely truncate a legal exactly-filled ctor.
        # Builtins (texture, etc.) have fixed signatures, so they stay safe.
        if any(self._expr_type_uses_user_fn(a) for a in arguments):
            return arguments
        widths = [self._ctor_component_count(a) for a in arguments]
        if any(w is None for w in widths):
            return arguments
        if sum(widths) <= target:
            return arguments

        result = []
        remaining = target
        postfix_safe = (IR.Identifier, IR.MemberAccess, IR.ArrayAccess,
                        IR.CallExpression, IR.ParenthesizedExpression)
        for arg, width in zip(arguments, widths):
            if remaining <= 0:
                break  # fully-excess trailing arg: drop it
            if width <= remaining:
                result.append(arg)
                remaining -= width
                continue
            # This arg crosses the boundary: swizzle it to `remaining` comps.
            arg_type = self._get_type_name(arg)
            base = arg
            if not isinstance(arg, postfix_safe):
                base = IR.ParenthesizedExpression(
                    expression=arg, source_location=location)
            elem_base = VECTOR_TYPE_INFO[arg_type][0]
            ocl_base = 'int' if elem_base == 'bool' else elem_base
            swz_type = ocl_base if remaining == 1 else f'{ocl_base}{remaining}'
            result.append(IR.MemberAccess(
                base=base,
                member='xyzw'[:remaining],
                glsl_type=swz_type,
                source_location=location
            ))
            remaining = 0
        return result

    def _create_matrix_cast(
        self,
        target_type: str,
        source_type: str,
        arguments: List[IR.TransformedNode],
        location: tuple
    ) -> IR.CallExpression:
        """
        Create matrix type casting call.

        source_type may arrive in either name family ('mat4' from a
        declaration, 'matrix4x4' from a parameter) — the helper names in
        matrix_ops.h use the GLSL form: GLSL_mat3_from_mat4(mat4_var).
        """
        source_glsl = MATRIX_NAME_TO_GLSL.get(source_type, source_type)
        function_name = f'GLSL_{target_type}_from_{source_glsl}'
        return IR.CallExpression(
            function=function_name,
            arguments=arguments,
            glsl_type=TYPE_NAME_MAP.get(target_type),
            source_location=location
        )

    def _is_struct_field_access(self, node: IR.TransformedNode) -> bool:
        """True if `node` is a struct-field member access (an addressable
        lvalue, e.g. `cam.ray` or `hit.pos`), as opposed to a vector swizzle
        (`v.xy` — not addressable). Used to decide whether an out/inout arg can
        legally have its address taken (`&cam.ray`). A field is addressable iff
        the base resolves to a user struct type registered in struct_types."""
        if not isinstance(node, IR.MemberAccess):
            return False
        base_type = self._get_type_name(node.base)
        if not base_type:
            return False
        return _strip_type_qualifiers(base_type) in self.struct_types

    def _is_vector_swizzle(self, node: IR.TransformedNode) -> bool:
        """True if `node` is a vector swizzle member access (`p.xz`, `v.x`) —
        NOT an addressable lvalue in OpenCL. The complement of
        `_is_struct_field_access` over MemberAccess nodes: the base resolves to
        a vector type rather than a user struct."""
        if not isinstance(node, IR.MemberAccess):
            return False
        base_type = self._get_type_name(node.base)
        if not base_type:
            return False
        base_type = _strip_type_qualifiers(base_type)
        if base_type in self.struct_types:
            return False
        return self._is_vector_type(base_type)

    def _capture_cico(self, transform):
        """Run `transform()` with the swizzle-out-arg copy-in/copy-out buffers
        armed; returns (result, prelude, writeback). Buffers are saved/restored
        so nested statements don't cross-contaminate."""
        saved = (self._cico_active, self._cico_prelude, self._cico_writeback)
        self._cico_active = True
        self._cico_prelude = []
        self._cico_writeback = []
        try:
            result = transform()
            return result, self._cico_prelude, self._cico_writeback
        finally:
            self._cico_active, self._cico_prelude, self._cico_writeback = saved

    def _make_swizzle_copy_in_out(self, arg: IR.MemberAccess):
        """Lower a vector-swizzle out-arg to copy-in/copy-out. Records a temp
        declaration (`T _cicoN = p.xz;`) in `_cico_prelude` and a writeback
        (`p.xz = _cicoN;`) in `_cico_writeback`, and returns `&_cicoN` to
        replace the argument. Returns None (caller leaves the arg unchanged) if
        the swizzle type can't be resolved for the temp declaration."""
        swz_type = self._get_type_name(arg)
        if not swz_type:
            return None
        ocl_type = self.type_map.get(_strip_type_qualifiers(swz_type), swz_type)
        temp_name = f'_cico{self._cico_counter}'
        self._cico_counter += 1
        loc = arg.source_location
        # copy-in: T _cicoN = <swizzle>;
        self._cico_prelude.append(IR.Declaration(
            type_name=ocl_type,
            name=temp_name,
            initializer=arg,
            source_location=loc,
        ))
        # copy-out: <swizzle> = _cicoN;  (reusing the same lvalue node)
        self._cico_writeback.append(IR.ExpressionStatement(
            expression=IR.AssignmentOp(
                operator='=',
                target=arg,
                value=IR.Identifier(name=temp_name, source_location=loc),
                source_location=loc,
            ),
            source_location=loc,
        ))
        return IR.UnaryOp(
            operator='&',
            operand=IR.Identifier(name=temp_name, source_location=loc),
            source_location=loc,
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
                glsl_type = self._glsl_type_from_name(field_type)

        # Struct field type (category E): struct definitions register their
        # field types in struct_types. Resolve the base type via _get_type_name
        # so nested members (a.b.c), deref'd pointer params ((*p).f) and
        # subscripted struct arrays (hits[i].f) all resolve, not just plain
        # identifiers.
        if glsl_type is None and field:
            base_type = self._get_type_name(base)
            if base_type:
                struct_fields = self.struct_types.get(
                    _strip_type_qualifiers(base_type))
                if struct_fields and field in struct_fields:
                    glsl_type = self._glsl_type_from_name(struct_fields[field])

        # If type not inferred yet, check for vector swizzle operations
        # This enables matrix operations on swizzled vector components
        # Examples: foo.xy * M2, V3.xyz * M3, V4.xy *= M2
        if glsl_type is None:
            base_type = self._get_type_name(base)
            if base_type and self._is_vector_type(base_type):
                # Try to infer swizzle type
                glsl_type = self._infer_swizzle_type(base_type, field)
                # OpenCL has no stpq swizzle set: remap p.st -> p.xy. Only
                # here, where the base is a proven vector AND the pattern
                # validated as a swizzle — a struct field named s/t/p/q must
                # pass through untouched.
                if glsl_type is not None and set(field) <= set('stpq'):
                    field = field.translate(STPQ_TO_XYZW)

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

        # Category F — matrix column subscript. GLSL M[i] returns the i-th
        # column vector, but the OpenCL matrix types (matrix2x2 etc.) are structs
        # whose columns live in a `cols[]` array, so M[i] must become M.cols[i]
        # (its type is the column vector vec2/vec3/vec4). Only applies to a bare
        # matrix value: an array-of-matrix (arr[i], tracked in array_vars) or a
        # vector/array subscript falls through unchanged.
        base_is_array = isinstance(base, IR.Identifier) and base.name in self.array_vars
        base_type = self._get_type_name(base) if not base_is_array else None
        if base_type and self._is_matrix_type(base_type):
            glsl_name = MATRIX_NAME_TO_GLSL.get(base_type, base_type)
            column_type = {'mat2': 'vec2', 'mat3': 'vec3', 'mat4': 'vec4'}.get(glsl_name)
            cols = IR.MemberAccess(
                base=base,
                member='cols',
                source_location=node.start_point
            )
            return IR.ArrayAccess(
                base=cols,
                index=index,
                glsl_type=TYPE_NAME_MAP.get(column_type) if column_type else None,
                source_location=node.start_point
            )

        # Try to infer type for matrix operation detection
        glsl_type = None

        # Element-type inference (category E): for arrays local_types stores
        # the ELEMENT type under the base name (e.g. 'float3' for vec3[4]), so
        # arr[i] has exactly the stored type; for a vector, v[i] is a scalar
        # COMPONENT (typing it as the whole vector would mis-route
        # v[0] * M into matmul instead of componentwise scale).
        if isinstance(base, IR.Identifier):
            stored_type = self.local_types.get(base.name)
            if stored_type:
                if base.name in self.array_vars:
                    glsl_type = self._glsl_type_from_name(stored_type)
                else:
                    vec_info = VECTOR_TYPE_INFO.get(
                        _strip_type_qualifiers(stored_type))
                    if vec_info:
                        glsl_type = TYPE_NAME_MAP.get(vec_info[0])
                    else:
                        glsl_type = self._glsl_type_from_name(stored_type)

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

        # GLSL requires both branches to share a type; propagate whichever
        # resolves so a ternary operand ((k ? A : B) * v) keeps matrix/vector
        # detection alive (category E).
        branch_type = (self._get_type_name(true_expr)
                       or self._get_type_name(false_expr))

        return IR.TernaryOp(
            condition=condition,
            true_expr=true_expr,
            false_expr=false_expr,
            glsl_type=self._glsl_type_from_name(branch_type),
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

        # Matrix compound assignment: A op= B -> A = <helper>(A, B). OpenCL's
        # struct matrix types reject native compound operators.
        if operator in ('*=', '+=', '-=', '/=') and value is not None:
            target_type = self._resolve_binary_operand_type(target, self._get_type_name(target))
            value_type = self._resolve_binary_operand_type(value, self._get_type_name(value))

            # *=: matrix/vector multiplication (linear algebra) takes precedence.
            if operator == '*=' and (
               (self._is_vector_type(target_type) and self._is_matrix_type(value_type)) or
               (self._is_matrix_type(target_type) and self._is_matrix_type(value_type)) or
               (self._is_matrix_type(target_type) and self._is_vector_type(value_type))):
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

            # Componentwise matrix arithmetic: M += M, M -= M, M += s, M /= s...
            # (category H). Reuse the binary-expression rewrite with the base op.
            componentwise = self._transform_matrix_componentwise(
                operator[0], target, value, target_type, value_type, node)
            if componentwise is not None:
                return IR.AssignmentOp(
                    operator='=',
                    target=target,
                    value=componentwise,
                    source_location=node.start_point
                )

            # Category E fallback for *=: one side is a proven matrix but the
            # other is statically untypeable — A = GLSL_mul(A, B) and let the
            # overloadable dispatcher (matrix_ops.h) resolve scalar/vector/
            # matrix at compile time.
            if operator == '*=' and \
                    ((target_type is None) != (value_type is None)) and \
                    self._is_matrix_type(target_type or value_type):
                return IR.AssignmentOp(
                    operator='=',
                    target=target,
                    value=IR.CallExpression(
                        function='GLSL_mul',
                        arguments=[target, value],
                        source_location=node.start_point
                    ),
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
        operand_type = self._get_type_name(operand)
        if self._is_vector_type(operand_type):
            return IR.AssignmentOp(
                operator='+=' if operator == '++' else '-=',
                target=operand,
                value=IR.IntLiteral(value='1', source_location=node.start_point),
                source_location=node.start_point
            )

        # Matrix ++/-- adds/subtracts 1 from EVERY element (GLSL semantics).
        # Struct matrix types reject both ++ and the += the vector rewrite
        # uses, so go straight to the componentwise helper:
        # M++ -> M = GLSL_matN_adds(M, 1).
        if self._is_matrix_type(operand_type):
            glsl_name = MATRIX_NAME_TO_GLSL.get(
                _strip_type_qualifiers(operand_type))
            suffix = 'adds' if operator == '++' else 'subs'
            return IR.AssignmentOp(
                operator='=',
                target=operand,
                value=IR.CallExpression(
                    function=f'GLSL_mat{glsl_name[-1]}_{suffix}',
                    arguments=[operand, IR.IntLiteral(value='1')],
                    glsl_type=TYPE_NAME_MAP.get(glsl_name),
                    source_location=node.start_point
                ),
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
        # Transform the inner expression. A comment may sit inside the parens
        # (e.g. `if ( //note\n cond )`); tree-sitter keeps it as a named child,
        # so pick the first NON-comment child — otherwise the comment (which
        # emits nothing) collapses the whole expression to `()` (category AD).
        inner_nodes = [c for c in node.named_children if c.type != 'comment']
        if inner_nodes:
            inner = self._transform_node(inner_nodes[0])
            # Wrap in ParenthesizedExpression to preserve parentheses
            return IR.ParenthesizedExpression(
                expression=inner,
                source_location=node.start_point
            )
        return None

    def _transform_comma_expression(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform a comma (sequence) expression `a, b`.

        GLSL/C evaluate each operand and yield the last. tree-sitter nests it
        right-associatively for 3+ operands (`a, b, c` is
        comma_expression(a, comma_expression(b, c))), so transforming the two
        named children recursively reconstructs the whole chain. Without this
        handler the node fell through to the unknown-type branch and emitted
        nothing, collapsing an enclosing paren to `()` (category AD).
        """
        children = [c for c in node.named_children if c.type != 'comment']
        if len(children) < 2:
            # Degenerate/recovered node — fall back to whatever is there.
            return self._transform_node(children[0]) if children else None
        return IR.CommaExpression(
            left=self._transform_node(children[0]),
            right=self._transform_node(children[-1]),
            source_location=node.start_point,
        )

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
        # In OpenCL, we use 'return;' to exit the kernel function early.
        # Inside a value-returning helper (some Shadertoy code puts `discard` in
        # a non-void function), a bare `return;` is a compile error, so return a
        # zero-valued default of the function's return type instead.
        if expr_node.type == 'identifier' and expr_node.text == 'discard':
            ret_type = self.current_function_return_type
            value = None
            if ret_type and ret_type != 'void':
                value = IR.TypeConstructor(
                    type_name=ret_type,
                    arguments=[IR.IntLiteral(value="0")],
                    source_location=node.start_point,
                )
            return IR.ReturnStatement(
                value=value,
                source_location=node.start_point
            )

        # Transform the expression, capturing any vector-swizzle out-arg
        # copy-in/copy-out temps generated during the call transform.
        expr, prelude, writeback = self._capture_cico(
            lambda: self._transform_node(expr_node))

        stmt = IR.ExpressionStatement(
            expression=expr,
            source_location=node.start_point,
        )
        if prelude or writeback:
            # Wrap in a block: { temp decls; call(&temp,…); writebacks; }
            return IR.CompoundStatement(
                statements=[*prelude, stmt, *writeback],
                source_location=node.start_point,
            )
        return stmt

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
        # An array/brace initializer (A2) is a program-scope constant only if
        # every element is: a zero-vector wrap `{(float3)(0.0f)}` stays at file
        # scope, but an arithmetic element or a synthesized matrix diagonal
        # `{GLSL_matrix3x3_diagonal(0.0f)}` is non-constant and must be hoisted.
        if isinstance(node, IR.ArrayInitializer):
            return all(self._is_ct_constant(e) for e in (node.elements or []))
        # BinaryOp, CallExpression, Identifier, MemberAccess, ArrayAccess,
        # TernaryOp, ... -> not a program-scope constant.
        return False

    def _is_int_foldable(self, node) -> bool:
        """
        Category A1 — True if `node` is an *integer* constant expression OpenCL
        folds at program scope (so a non-`_is_ct_constant` int/uint global with
        this initializer must be LEFT in place — it may be an array size or case
        label). False means OpenCL cannot fold it, so it must be hoisted.

        Foldable: integer/bool literals; unary/parenthesized/binary combinations
        of foldable operands; references to other int/uint globals we kept as
        constants (`self._const_int_globals`); and an int/uint cast whose operand
        is itself foldable (`int(N)`).

        NOT foldable (⇒ hoist): a float literal or float-typed operand, a call,
        a vector/matrix member access (`N.x` — not folded even on a const
        vector), a subscript, or a reference to a hoisted (non-constant) global.
        """
        if node is None:
            return False
        if isinstance(node, (IR.IntLiteral, IR.BoolLiteral)):
            return True
        if isinstance(node, IR.FloatLiteral):
            return False
        if isinstance(node, IR.ParenthesizedExpression):
            return self._is_int_foldable(node.expression)
        if isinstance(node, IR.UnaryOp):
            return self._is_int_foldable(node.operand)
        if isinstance(node, IR.BinaryOp):
            return (self._is_int_foldable(node.left)
                    and self._is_int_foldable(node.right))
        if isinstance(node, IR.Identifier):
            return node.name in self._const_int_globals
        # An int/uint cast is a TypeConstructor over a single argument.
        if isinstance(node, IR.TypeConstructor):
            if node.type_name in ('int', 'uint') and node.arguments:
                return all(self._is_int_foldable(a) for a in node.arguments)
            return False
        # CallExpression, MemberAccess, ArrayAccess, TernaryOp, ... -> not folded.
        return False

    @staticmethod
    def _array_init_len(init_ir) -> Optional[int]:
        """Element count of an array initializer IR, or None if not countable."""
        if isinstance(init_ir, IR.ArrayConstructor):
            return len(init_ir.arguments or [])
        if isinstance(init_ir, IR.ArrayInitializer):
            return len(init_ir.elements or [])
        return None

    def _hoist_array_global(self, base_name, var_name, opencl_type, init_ir):
        """Category A2 — record a non-constant program-scope ARRAY global for
        temp-local + copy-loop hoisting, and return the bare-decl name (the
        array declarator, sized).

        `var_name` is the raw declarator text (`mats[2]`, `positions[]`,
        `grid[N]`). The size text between the brackets is reused as the
        copy-loop bound; an unsized `[]` is filled from the initializer's
        element count so the bare global is correctly dimensioned (a later
        `sizeof(arr)/sizeof(arr[0])` loop bound then folds).
        """
        open_b = var_name.index('[')
        close_b = var_name.rindex(']')
        size_text = var_name[open_b + 1:close_b].strip()
        if not size_text:
            count = self._array_init_len(init_ir)
            size_text = str(count) if count is not None else ''
        self.hoisted_global_inits.append(
            HoistedArrayInit(base_name, opencl_type, size_text, init_ir)
        )
        # Bare decl carries the (now always explicit) size.
        return f"{base_name}[{size_text}]"

    def _unique_shadow_name(self, base_name: str) -> str:
        """Category AE — pick a rename for a local that shadows a user function.

        `base_name` -> `base_name_v`, disambiguated with a counter if that name
        is itself a user function, an existing local, or an in-flight rename.
        """
        candidate = f"{base_name}_v"
        i = 2
        while (candidate in self.user_function_names
               or candidate in self.local_types
               or candidate in self.local_renames.values()):
            candidate = f"{base_name}_v{i}"
            i += 1
        return candidate

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

        # Category AH — an inline struct DEFINITION carrying a trailing variable:
        # `struct Name { ... } var;`. Tree-sitter parses this as a declaration
        # whose `type` is a named struct_specifier with a field list (as opposed
        # to a bare `struct Name var;` reference). The old path passed the whole
        # `struct Name {...}` text through as the variable's type_name, emitting a
        # bare `struct` tag with no typedef — so later bare-name uses (`Name p`)
        # are invalid C ("must use 'struct' tag to refer to type 'Name'"), and the
        # struct is never registered for member-access inference. Route the type
        # through _transform_struct_specifier (emits `typedef struct {...} Name;`
        # and registers struct_types/type_map), then declare the variable(s) with
        # the bare struct name. The StructDefinition is returned alongside the
        # variable declaration and flattened by the caller.
        struct_def_ir = None
        if type_node.type == 'struct_specifier':
            has_fields = any(c.type == 'field_declaration_list'
                             for c in type_node.named_children)
            name_child = next((c for c in type_node.named_children
                               if c.type == 'type_identifier'), None)
            if has_fields and name_child is not None:
                struct_def_ir = self._transform_struct_specifier(type_node)
                glsl_type = struct_def_ir.name
                opencl_type = struct_def_ir.name

        # Collect all declarators (identifiers and init_declarators)
        # Skip type node and punctuation (,;)
        declarators = []
        for child in node.named_children:
            if child.type in ('identifier', 'init_declarator', 'array_declarator'):
                declarators.append(child)

        if not declarators:
            # Function prototype (category S): `float Fn (vec3 p);` parses as a
            # declaration whose declarator is a function_declarator, not an
            # identifier. Transform it like a body-less function definition.
            for child in node.named_children:
                if child.type == 'function_declarator':
                    return self._transform_function_prototype(node, child)
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
            is_array = False
            if declarator.type == 'identifier':
                var_name = declarator.text
                base_name = var_name
            elif declarator.type == 'array_declarator':
                # Array declaration: type name[size]
                is_array = True
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
                        is_array = True
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
                # Remember array-ness so a matrix-array subscript (arr[i], an
                # element access) is not mistaken for a matrix column access.
                if is_array:
                    self.array_vars.add(base_name)

            # Transform initializer if present, or create zero initializer for undefined variables
            initializer = None
            if initializer_node:
                initializer = self._transform_node(initializer_node)

                # Category A — hoist a non-constant program-scope initializer.
                # (Shadow discard happens after this block — see below — so a
                # self-referencing init `float r = r;` still reads the param.)
                # OpenCL rejects any arithmetic/call in a file-scope initializer
                # ("initializer element is not a compile-time constant"), even
                # with __constant. Emit the global bare (mutable, default-zero)
                # and record the real initializer; the host (tests/transpile.py /
                # Houdini @KERNEL) assigns it at the top of the kernel body, the
                # same pattern main_header.cl uses for `static float iTime`.
                # Skips: scalar int/uint globals (array-size/loop-bound
                # candidates whose integer constant expressions OpenCL already
                # folds). Array globals take the A2 temp-local + copy-loop path
                # below (whole-array assignment `arr = {...};` is illegal in C).
                if (self._global_scope
                        and not is_array
                        and not self._is_ct_constant(initializer)):
                    # A1: an int/uint global stays in place only if OpenCL can
                    # fold its initializer (it may be an array size / case
                    # label). A non-foldable int/uint init — a cast of a runtime
                    # value, a vector member access (`N.x*N.y`), or a reference
                    # to a hoisted global — must be hoisted like any other type.
                    if opencl_type in ('int', 'uint'):
                        do_hoist = not self._is_int_foldable(initializer)
                    else:
                        do_hoist = True
                    if do_hoist:
                        self.hoisted_global_inits.append((var_name, initializer))
                        initializer = None  # bare decl; assigned in the kernel
                        hoisted_any = True
                # A2: an array/aggregate global with a non-constant element
                # initializer. Emit it bare and sized, and hoist a temp-local
                # array (a non-constant aggregate init is legal on a LOCAL)
                # copied element-by-element into the bare global in the kernel.
                elif (self._global_scope
                        and is_array
                        and not self._is_ct_constant(initializer)):
                    var_name = self._hoist_array_global(
                        base_name, var_name, opencl_type, initializer)
                    initializer = None  # bare, sized decl; filled in the kernel
                    hoisted_any = True
            else:
                # No explicit initializer - create zero initializer to match GLSL semantics
                # GLSL implicitly initializes undefined variables to zero, while OpenCL
                # leaves them undefined. This creates appropriate zero initializers.
                initializer = self._create_zero_initializer(glsl_type, opencl_type)

                # Category A3 — a synthesized zero-initializer whose value is a
                # CALL (matrices -> GLSL_matrixNxN_diagonal(0.0f)) is not a
                # compile-time constant at program scope. Hoist it like an
                # explicit non-const initializer. Scalar/vector zero-inits are
                # literals (ct-constant) and stay in place; arrays keep their
                # ArrayInitializer below (whole-array assignment is illegal).
                if (self._global_scope
                        and '[' not in var_name
                        and initializer is not None
                        and not self._is_ct_constant(initializer)):
                    self.hoisted_global_inits.append((var_name, initializer))
                    initializer = None  # bare decl; assigned in the kernel body
                    hoisted_any = True

                # For arrays, wrap the initializer in ArrayInitializer with curly braces
                # OpenCL requires array initializers to be in the form: type name[size] = {...}
                if initializer and '[' in var_name:
                    initializer = IR.ArrayInitializer(
                        elements=[initializer],
                        glsl_type=None,
                        source_location=None
                    )
                    # A2 — a synthesized array zero-init whose element is a CALL
                    # (matrix diagonal: `{GLSL_matrix3x3_diagonal(0.0f)}`) is not
                    # a compile-time constant at file scope. Hoist it via the
                    # temp-local + copy loop; C's tail zero-fill makes the
                    # single-element wrap correct for the whole array. Constant
                    # scalar/vector zero wraps (`{0.0f}`, `{(float3)(0.0f)}`)
                    # stay in place.
                    if (self._global_scope
                            and not self._is_ct_constant(initializer)):
                        var_name = self._hoist_array_global(
                            base_name, var_name, opencl_type, initializer)
                        initializer = None  # bare, sized decl; filled in kernel
                        hoisted_any = True

            # A1: remember an int/uint global we KEPT in place (a compile-time
            # constant or OpenCL-foldable integer expression) so a later int/uint
            # initializer referencing it is recognized as foldable too.
            if (self._global_scope
                    and not is_array
                    and opencl_type in ('int', 'uint')
                    and initializer is not None
                    and base_name):
                self._const_int_globals.add(base_name)

            # Category AE — a local whose name shadows a user function it CALLS
            # in its own initializer: `float ao = ao(p);`. In OpenCL the bare
            # name binds to this value, so the `ao(...)` call fails ("called
            # object type 'float' is not a function"); GLSL resolves the call to
            # the function. Rename the LOCAL (and its later reads in this scope)
            # so the function stays callable; the call keeps the original name
            # (call callees are not routed through _transform_identifier).
            #
            # GATED on the initializer actually calling the shadowed name. That
            # construct is ALWAYS a compile error today (the declarator's scope
            # begins before its own initializer), so this can only touch
            # already-failing shaders — never regress a passing one. A local
            # that merely shares a function's name without calling it (a legal,
            # passing shadow, possibly with reads inside a textual #ifdef/#define
            # block our AST rename can't reach) is left untouched.
            emit_name = var_name
            if (not self._global_scope
                    and not is_array
                    and base_name in self.user_function_names
                    and initializer_node is not None
                    and re.search(r'\b' + re.escape(base_name) + r'\s*\(',
                                  initializer_node.text)):
                emit_name = self._unique_shadow_name(base_name)
                self.local_renames[base_name] = emit_name
                self.local_types[emit_name] = glsl_type

            # Category B residual — a local that shadows an out/inout pointer
            # param (`float r` inside a fn taking `inout vec3 r`). From here on
            # in this block the bare name binds to the local, so its reads must
            # NOT be dereferenced. Drop it from pointer_params AFTER the
            # initializer was transformed above (a self-referencing init
            # `float r = r;` still reads the param) — _transform_compound_statement
            # restores the set on block exit, so later param reads deref again.
            if (not self._global_scope
                    and base_name in self.pointer_params):
                self.pointer_params.discard(base_name)

            # Create Declaration node (without type_name for DeclarationList)
            declarations.append(IR.Declaration(
                type_name=None,  # Will be set at DeclarationList level
                name=emit_name,
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
            decl_node = IR.Declaration(
                type_name=opencl_type,
                name=declarations[0].name,
                initializer=declarations[0].initializer,
                qualifiers=qualifiers,
                source_location=node.start_point
            )
        else:
            # Comma-separated declarations
            decl_node = IR.DeclarationList(
                type_name=opencl_type,
                declarators=declarations,
                qualifiers=qualifiers,
                source_location=node.start_point
            )

        # Category AH — emit the typedef'd struct definition ahead of the
        # variable declaration(s). The caller flattens the list into siblings.
        if struct_def_ir is not None:
            return [struct_def_ir, decl_node]
        return decl_node

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
        # Category B residual — block-scope the pointer-param deref set so a
        # nested local that shadows an out/inout param (`float r` inside a
        # function taking `inout vec3 r`) suppresses the deref only for this
        # block. _transform_declaration discards the shadowed name; restoring
        # the snapshot on exit brings the param's deref back for later reads.
        saved_pointer_params = set(self.pointer_params)
        statements = []
        for stmt in node.named_children:
            if stmt.type == 'declaration':
                # A swizzle out-arg inside a declaration INITIALIZER
                # (`float c = pMod1(p.z, s);`) needs the copy-in/copy-out
                # lowering too, but the declaration cannot be block-wrapped
                # (the binding must stay in scope) - splice the temp decl
                # before it and the writeback after it as siblings.
                transformed, prelude, writeback = self._capture_cico(
                    lambda s=stmt: self._transform_node(s))
                statements.extend(prelude)
                # Category AH — a local struct-definition-with-variable returns
                # [StructDefinition, Declaration]; splice both in as siblings.
                if isinstance(transformed, list):
                    statements.extend(transformed)
                elif transformed is not None:
                    statements.append(transformed)
                statements.extend(writeback)
            else:
                transformed = self._transform_node(stmt)
                if isinstance(transformed, list):
                    statements.extend(transformed)
                elif transformed is not None:
                    statements.append(transformed)

        self.pointer_params = saved_pointer_params
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
                # Array params (vec3 pts[4]) index to an element, not a column.
                if getattr(param, 'array_suffix', None):
                    self.array_vars.add(param.name)

                # Entry-function params (custom-named golf signatures like
                # `out vec4 O, vec2 U`) are registered under their GLSL type
                # names: the pre-single-TU pipeline declared them as GLSL
                # alias locals (`vec2 U = fragCoord;`), and several inference
                # paths only recognize GLSL names — OpenCL names here would
                # silently disable category-N conversions / matrix lowering
                # inside the entry body.
                if func_name == self.entry_function:
                    self.local_types[param.name] = OPENCL_TO_GLSL_NAME.get(
                        param.type_name, param.type_name)

                # Track pointer parameters (for dereference handling in function
                # body). The renderpass ENTRY function's out-param (fragColor) is
                # a host-provided @KERNEL local, not a deref'able pointer, so it
                # is excluded here (a helper that merely shares the name is not).
                if param.is_pointer and func_name != self.entry_function:
                    self.pointer_params.add(param.name)

                # Store parameter info for function signature registry
                param_info.append((param.name, param.is_pointer))

        # Register function signature for call site handling, keyed by arity so
        # overloads (same name, different parameter count) coexist.
        self.function_signatures.setdefault(func_name, {})[len(param_info)] = param_info

        # Transform body. Declarations inside a function body are local scope,
        # not program scope, so global-init hoisting must not apply to them.
        # Track the OpenCL return type so a `discard` inside a value-returning
        # helper can lower to a zero-valued return instead of a bare `return;`
        # (which is a compile error in a non-void function).
        prev_global_scope = self._global_scope
        prev_return_type = self.current_function_return_type
        prev_local_renames = self.local_renames  # Category AE — per-body scope
        self.local_renames = {}
        self._global_scope = False
        self.current_function_return_type = return_type
        body = self._transform_node(body_node)
        self.local_renames = prev_local_renames
        self.current_function_return_type = prev_return_type
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

        # Category Q — gl_FragCoord builtin. gl_FragCoord.xy is the pixel-center
        # fragment coordinate; OpenCL has no such builtin, so a reference fails
        # with "undeclared identifier 'gl_FragCoord'". `_gl_fragcoord_token_re`
        # (set in transform()) matches gl_FragCoord and every object-macro alias
        # of it; a match against `body_node.text` also catches uses inside
        # inactive `#ifdef` blocks (the raw source text), which is harmless.
        token_re = self._gl_fragcoord_token_re
        body_uses_fragcoord = bool(
            token_re and isinstance(body, IR.CompoundStatement)
            and token_re.search(body_node.text or ''))

        if func_name == self.entry_function and not self._gl_fragcoord_user_provided \
                and body_uses_fragcoord:
            # ENTRY body — inject the proven fragCoord-based local for the entry's
            # OWN gl_FragCoord reads (it uses the true `fragCoord` the entry
            # receives). The gid->pixel offset static that helpers read via
            # GLSL_glFragCoord() is seeded by the HDA setter shadertoy_bind_inputs()
            # at the top of every kernel body, so no transpiler seed is emitted.
            frag_local = IR.Declaration(
                type_name="float4",
                name="gl_FragCoord",
                initializer=IR.TypeConstructor(
                    type_name="float4",
                    arguments=[
                        IR.Identifier(name="fragCoord"),
                        IR.FloatLiteral(value="0.0f"),
                        IR.FloatLiteral(value="1.0f"),
                    ],
                ),
            )
            body = IR.CompoundStatement(
                statements=[frag_local] + list(body.statements or []),
                source_location=body.source_location,
            )
        elif func_name != self.entry_function and body_uses_fragcoord \
                and not self._gl_fragcoord_user_provided:
            # HELPER body — resolve gl_FragCoord to the runtime accessor. The
            # accessor rebuilds the raw pixel coordinate from get_global_id() +
            # the entry-seeded offset (glslHelpers.h). Injecting a local named
            # gl_FragCoord means zero read-site rewriting (mirrors the entry
            # injection) and CPP-expanded aliases (`F` -> `gl_FragCoord`) bind to
            # it too.
            helper_decl = IR.Declaration(
                type_name="float4",
                name="gl_FragCoord",
                initializer=IR.CallExpression(
                    function="GLSL_glFragCoord",
                    arguments=[],
                ),
            )
            body = IR.CompoundStatement(
                statements=[helper_decl] + list(body.statements or []),
                source_location=body.source_location,
            )

        return IR.FunctionDefinition(
            return_type=return_type,
            # Category D2 — emit `sh_<name>` when the name collides with a
            # Houdini builtin (tracking dicts stay keyed by the original name).
            name=self.function_renames.get(func_name, func_name),
            parameters=parameters,
            body=body,
            overloadable=overloadable,
            source_location=node.start_point
        )

    def _transform_function_prototype(self, node: ASTNode,
                                      declarator: ASTNode) -> IR.FunctionDefinition:
        """
        Transform a function prototype / forward declaration (category S).

        GLSL: `float PrSphDf (vec3 p, float r);` parses as a `declaration`
        whose declarator is a `function_declarator`. Emit it as a body-less
        OpenCL prototype with the SAME parameter transformation and
        `__attribute__((overloadable))` marking as the eventual definition —
        OpenCL rejects an overloadable definition whose earlier declaration
        is not marked.

        The prototype also pre-registers the function's return type and
        out-param signature so call sites that appear before the definition
        (the reason prototypes exist) get correct type inference and
        `&` insertion.
        """
        type_node = node.child_by_field_name('type')
        return_type = self._transform_type_name(type_node)

        # Function name: identifier child of the function_declarator
        func_name = ""
        for child in declarator.children:
            if child.type == 'identifier':
                func_name = child.text
                break
        if not func_name:
            raise TransformationError(
                "Invalid function prototype: missing name",
                node.start_point
            )

        # Register return type (GLSL name) for call-before-definition inference
        self.user_function_return_types[func_name] = type_node.text.strip()

        # Transform parameters exactly like a definition, but without touching
        # local_types/pointer_params (there is no body to scope them to).
        parameters = []
        param_info = []
        for child in declarator.children:
            if child.type != 'parameter_list':
                continue
            for param_node in child.named_children:
                if param_node.type != 'parameter_declaration':
                    continue
                param = self._transform_parameter(param_node)
                if param is None:
                    # Unnamed prototype param (`float Fn(vec3);`) — keep the
                    # arity: emit the type alone.
                    param_type_node = param_node.child_by_field_name('type')
                    if param_type_node is None:
                        continue
                    param = IR.Parameter(
                        type_name=self._transform_type_name(param_type_node),
                        name="",
                        qualifiers=[],
                        is_pointer=False,
                        source_location=param_node.start_point
                    )
                parameters.append(param)
                param_info.append((param.name, param.is_pointer))

        # Pre-register the signature for `&` insertion at call sites that
        # precede the definition (the definition re-registers identically).
        # Keyed by arity so overloads coexist.
        self.function_signatures.setdefault(func_name, {})[len(param_info)] = param_info

        # Same entry-point exclusion as _transform_function_definition: hosts
        # strip/replace the entry signature, so it must stay unmarked.
        entry_points = {'mainImage', 'mainCubemap', 'mainSound', 'mainVR'}
        overloadable = func_name not in entry_points

        return IR.FunctionDefinition(
            return_type=return_type,
            # Category D2 — mirror the definition's `sh_<name>` rename so the
            # prototype and definition keep identical signatures.
            name=self.function_renames.get(func_name, func_name),
            parameters=parameters,
            body=None,
            overloadable=overloadable,
            is_prototype=True,
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
            # Output parameters become pointers, emitted WITHOUT an explicit
            # address-space qualifier (a bare `float4* p`). An explicit
            # `__private` rejects a `__global` argument — and category-A leaves
            # compile-time-constant-init globals at program scope, where OpenCL
            # places them in `__global`, so `save(&gState)` passes a
            # `__global float4*` to the param. A bare pointer acts as the
            # generic address space in the campaign/Houdini build mode (no
            # -cl-std), accepting both `__global` (hoisted globals) and
            # `__private` (locals) arguments; probe-verified on the CUDA target.
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

    def _transform_preprocessor(self, node: ASTNode) -> IR.TransformedNode:
        """
        Transform preprocessor directive.

        Single directives (#define, #undef, ...) pass through as raw text —
        PreprocessorTransformer already transformed them before parsing.

        Category E (Session 53): a statement-level `#if`/`#ifdef` BLOCK inside
        a function body is routed through the normal AST transform instead
        (tree-sitter parses its contents as structured statements, but the
        historical raw-text passthrough left them untyped, so e.g. a matrix
        `*` inside the block never lowered to GLSL_mul). Any parse error or
        unrecognized child falls back to the raw-text passthrough — the worst
        case per block is the status quo. Program-scope blocks (which wrap
        whole functions/declarations) keep the raw path.
        """
        if (node.type in ('preproc_if', 'preproc_ifdef', 'preproc_ifndef')
                and not self._global_scope):
            block = self._try_transform_preproc_block(node)
            if block is not None:
                return block

        # Raw-text passthrough (single directives + fallback for blocks).
        text = node.text.strip()

        return IR.PreprocessorDirective(
            text=text,
            source_location=node.start_point
        )

    # Statement types safe to route through _transform_node from inside a
    # preprocessor conditional block. Anything else (ERROR nodes, expression
    # fragments from a block that splits a statement, ...) aborts the routing
    # and falls back to the raw-text passthrough.
    _PREPROC_ROUTABLE_STMTS = frozenset((
        'declaration', 'expression_statement', 'return_statement',
        'if_statement', 'for_statement', 'while_statement', 'do_statement',
        'break_statement', 'continue_statement', 'compound_statement',
        'preproc_if', 'preproc_ifdef', 'preproc_ifndef',
    ))
    # Single directives legal inside a routed block — kept as raw text lines.
    _PREPROC_RAW_CHILDREN = frozenset((
        'preproc_def', 'preproc_function_def', 'preproc_call',
    ))

    def _try_transform_preproc_block(self, node: ASTNode):
        """Route a clean statement-level conditional block through the AST.

        Returns an IR.PreprocessorBlock, or None if the block is not safely
        routable (parse errors, unknown children, or a transform failure).
        """
        if node.has_error:
            return None
        try:
            segments = []
            self._collect_preproc_segments(node, segments)
        except Exception:
            # Fail-safe: ANY problem routing the block (unknown child shape,
            # a transform bug on unusual content, ...) falls back to the
            # raw-text passthrough — never worse than the pre-S53 behavior.
            return None
        return IR.PreprocessorBlock(
            segments=segments,
            source_location=node.start_point,
        )

    def _collect_preproc_segments(self, node: ASTNode, segments) -> None:
        """Append (directive_line, [transformed statements]) segments for a
        preproc_if/ifdef/elif/else node, recursing through the alternative
        chain. Raises _PreprocRouteAbort on any unroutable child."""
        directive_line = node.text.split('\n', 1)[0].rstrip()
        stmts = []
        segments.append((directive_line, stmts))

        # The condition/name child is part of the directive line — skip it by
        # position (its text is embedded in directive_line already).
        cond = (node.child_by_field_name('name')
                or node.child_by_field_name('condition'))
        cond_point = cond.start_point if cond is not None else None

        for child in node.named_children:
            ctype = child.type
            if cond_point is not None and child.start_point == cond_point:
                continue
            if ctype == 'comment':
                continue
            if ctype in ('preproc_else', 'preproc_elif'):
                # Alternative chain: #elif recurses (it may nest another
                # alternative); #else is a plain terminal segment.
                self._collect_preproc_segments(child, segments)
                continue
            if ctype in self._PREPROC_RAW_CHILDREN:
                stmts.append(IR.PreprocessorDirective(
                    text=child.text.strip(),
                    source_location=child.start_point,
                ))
                continue
            if ctype not in self._PREPROC_ROUTABLE_STMTS:
                raise _PreprocRouteAbort(ctype)
            transformed = self._transform_node(child)
            if transformed is not None:
                stmts.append(transformed)
