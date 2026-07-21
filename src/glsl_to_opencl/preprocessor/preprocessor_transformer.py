"""
Preprocessor Directive Transformer - Session 9.

Transforms GLSL preprocessor directives to OpenCL equivalents.

This module handles:
1. #define macros with float literals: #define PI 3.14159265 -> #define PI 3.14159265f
2. #define macros with function calls: #define random(x) fract(sin(x)) -> #define random(x) GLSL_fract(GLSL_sin(x))
3. Conditional compilation: #if, #ifdef, #ifndef, #else, #elif, #endif (pass through unchanged)

Design:
- String-based processing (preprocessor directives are not part of AST)
- Line-by-line parsing
- Regex-based transformation for macro bodies
- Preserves comments and whitespace

Usage:
    transformer = PreprocessorTransformer()
    transformed_source = transformer.transform(glsl_source)
"""

import re
from typing import List, Set

from .conditional_eval import maybe_preprocess_directives
from .macro_expander import maybe_expand_function_macros


class PreprocessorTransformer:
    """
    Transforms GLSL preprocessor directives to OpenCL equivalents.

    Handles #define macros by transforming:
    - Float literals: adds 'f' suffix
    - Function calls: adds GLSL_ prefix to built-in functions
    - Vector constructors: adds cast-style parentheses (vec2(...) -> (float2)(...))

    Also transforms code inside conditional directives (#ifdef, #else, etc.).
    """

    def __init__(self):
        """Initialize the preprocessor transformer."""
        # List of GLSL built-in functions that need GLSL_ prefix
        # Must match the list in ast_transformer.py
        self.glsl_builtins = {
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
            # Derivative placeholders
            'dFdx', 'dFdy', 'fwidth',
            # Matrix functions
            'transpose', 'inverse', 'determinant',
        }

        # GLSL vector type constructors that need cast-style parentheses
        # Maps GLSL type name to OpenCL type name
        self.vector_types = {
            'vec2': 'float2',
            'vec3': 'float3',
            'vec4': 'float4',
            'ivec2': 'int2',
            'ivec3': 'int3',
            'ivec4': 'int4',
            'uvec2': 'uint2',
            'uvec3': 'uint3',
            'uvec4': 'uint4',
            'bvec2': 'int2',
            'bvec3': 'int3',
            'bvec4': 'int4',
            # HLSL-style aliases (`#define float2 vec2`) used as constructors
            # inside macro / #if bodies: float2(x,y) -> (float2)(x,y). AST call
            # sites are aliased in ast_transformer; these textual regions are
            # not, so they must be cast here. Map name -> itself.
            'float2': 'float2', 'float3': 'float3', 'float4': 'float4',
            'int2': 'int2', 'int3': 'int3', 'int4': 'int4',
            'uint2': 'uint2', 'uint3': 'uint3', 'uint4': 'uint4',
        }

        # GLSL scalar type constructors that become C-style casts
        # float(x) -> (float)(x)  (category V; OpenCL has no float(x) function)
        # Maps GLSL type name to OpenCL type name
        self.scalar_types = {
            'float': 'float',
            'int': 'int',
            'uint': 'uint',
            'bool': 'bool',
        }

        # GLSL matrix types (category J). Constructors map to the overloadable
        # GLSL_matN dispatcher (matrix_ops.h resolves any GLSL arg shape);
        # bare type spellings in declarations map to the OpenCL matrix struct
        # type (Houdini typedefs the GLSL spelling `mat2` as float4, so an
        # untransformed `mat2 R = ...` inside a #if block miscompiles).
        self.matrix_types = {
            'mat2': 'matrix2x2',
            'mat3': 'matrix3x3',
            'mat4': 'matrix4x4',
        }

        # Category N (Session 54): dispatcher name for a SINGLE-argument
        # vector constructor whose argument may be a vector (overloadable
        # GLSL_<type> family in glslHelpers.h — same precedent as GLSL_matN).
        # The C cast `(int2)(U)` miscompiles when U expands to a float2; the
        # dispatcher lets OpenCL overload resolution supply the type. HLSL
        # aliases map to the GLSL-named dispatcher (float2 -> GLSL_vec2).
        self.vector_ctor_dispatchers = {
            'vec2': 'GLSL_vec2', 'vec3': 'GLSL_vec3', 'vec4': 'GLSL_vec4',
            'ivec2': 'GLSL_ivec2', 'ivec3': 'GLSL_ivec3', 'ivec4': 'GLSL_ivec4',
            'uvec2': 'GLSL_uvec2', 'uvec3': 'GLSL_uvec3', 'uvec4': 'GLSL_uvec4',
            'bvec2': 'GLSL_bvec2', 'bvec3': 'GLSL_bvec3', 'bvec4': 'GLSL_bvec4',
            'float2': 'GLSL_vec2', 'float3': 'GLSL_vec3', 'float4': 'GLSL_vec4',
            'int2': 'GLSL_ivec2', 'int3': 'GLSL_ivec3', 'int4': 'GLSL_ivec4',
            'uint2': 'GLSL_uvec2', 'uint3': 'GLSL_uvec3', 'uint4': 'GLSL_uvec4',
        }

        # A ctor argument that is a bare numeric literal is provably a scalar
        # broadcast — the existing cast is already correct, so it is NOT
        # routed to the dispatcher (keeps the changed-output blast radius to
        # constructors whose argument could actually be a vector).
        self._numeric_literal_re = re.compile(
            r'^[+-]?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[fFuU]?$'
        )

        # Object-like macros whose body is a top-level comma list
        # (`#define COLOR_1 0.50, 0.90, 0.95`). `vec3(COLOR_1)` textually
        # scans as ONE argument but expands to THREE — routing it to the
        # dispatcher would produce a 3-arg call with no overload, while the
        # cast expands into a legal component list. Names recorded here keep
        # the cast. (Macro PARAMETERS can never receive an unparenthesized
        # comma list — that is an expansion arity error — so bare parameter
        # arguments stay routable.)
        self.comma_object_macros = set()

        # Track if we're inside a preprocessor conditional block
        # This helps us know whether to transform code lines
        self.inside_conditional = False

        # Macros whose body is a matrix constructor expression, e.g.
        # `#define rot(a) mat2(cos(a),-sin(a),sin(a),cos(a))`. These are
        # matrix-returning "functions" that the AST cannot type (the #define
        # is opaque to it), so `p *= rot(a)` at a use site would emit a raw
        # `float2 *= matrix2x2`. transpile() seeds these into the AST's
        # user_function_return_types so the matmul dispatcher fires. Maps
        # macro name -> GLSL matrix type ('mat2' | 'mat3' | 'mat4').
        self.matrix_macros = {}

    def transform(self, source: str) -> str:
        """
        Transform GLSL source with preprocessor directives.

        Args:
            source: GLSL source code string

        Returns:
            Transformed source code with OpenCL preprocessor directives
        """
        # Stage 0 (category G, Session 38): evaluate/strip constant conditionals
        # and, if the source still fails to parse, expand object-like macro
        # uses (the bounded preprocessing cascade in conditional_eval.py).
        # Gated on parse-failure exactly like the function-macro expander
        # below: a shader that already parses is returned byte-identical.
        source = maybe_preprocess_directives(source)

        # Stage 0a (category P cluster 5): expand function-like macro USES on raw
        # GLSL so tree-sitter can parse their (otherwise invalid) call sites. Runs
        # first, before the body/#if-region transforms below, so expanded tokens
        # are seen as ordinary code by the normal AST path (no double-prefixing).
        # Gated on parse-failure: a shader that already parses is left untouched
        # (OpenCL expands its macros), so a passing shader can never regress.
        source = maybe_expand_function_macros(source)

        lines = source.split('\n')
        transformed_lines = []

        for line in lines:
            transformed_line = self._transform_line(line)
            transformed_lines.append(transformed_line)

        return '\n'.join(transformed_lines)

    def _transform_line(self, line: str) -> str:
        """
        Transform a single line of source code.

        Args:
            line: Source line

        Returns:
            Transformed line
        """
        # Check if this is a preprocessor directive
        stripped = line.lstrip()

        if stripped.startswith('#'):
            # Track conditional block state
            if stripped.startswith('#ifdef') or stripped.startswith('#ifndef') or stripped.startswith('#if'):
                self.inside_conditional = True
            elif stripped.startswith('#endif'):
                self.inside_conditional = False
            # #else and #elif keep us inside the conditional

            # Check if it's a #define directive
            if stripped.startswith('#define'):
                return self._transform_define(line)

            # All other preprocessor directives pass through unchanged
            # (#if, #ifdef, #ifndef, #else, #elif, #endif, #include, etc.)
            return line
        else:
            # Not a preprocessor directive
            # If we're inside a conditional block, transform vector constructors and types
            if self.inside_conditional:
                return self._transform_code_line(line)
            return line

    def _transform_define(self, line: str) -> str:
        """
        Transform a #define directive.

        Handles:
        - Object-like macros: #define PI 3.14159265
        - Function-like macros: #define random(x) fract(sin(x))

        Args:
            line: #define directive line

        Returns:
            Transformed #define directive
        """
        # Match #define with optional whitespace
        # Pattern: #define NAME [BODY]
        # or: #define NAME(PARAMS) [BODY]

        # Extract the line before any comment
        # Handle both // and /* */ style comments
        line_without_comment, comment = self._extract_comment(line)

        # Match the #define pattern
        # Group 1: whitespace before #define
        # Group 2: macro name
        # Group 3: optional (parameters)
        # Group 4: macro body (everything after name or params)
        pattern = r'^(\s*)#define\s+([a-zA-Z_][a-zA-Z0-9_]*)(\([^)]*\))?\s*(.*)$'
        match = re.match(pattern, line_without_comment)

        if not match:
            # Malformed #define, return unchanged
            return line

        indent = match.group(1)
        macro_name = match.group(2)
        params = match.group(3) or ''  # Empty string if no params
        body = match.group(4)

        # Record comma-list object macros BEFORE transforming any later body
        # that might use them as a single ctor argument (category N guard).
        if not params and self._has_top_level_comma(body):
            self.comma_object_macros.add(macro_name)

        # Transform the macro body
        transformed_body = self._transform_macro_body(body)

        # Record matrix-returning macros (body is a bare matrix ctor
        # expression). The outermost token decides the return type, so a
        # statement macro (`v *= mat2(...)`) or a float-valued wrapper
        # (`length(fract(p*=mat2(...)))`) is correctly excluded.
        mm = re.match(r'\s*GLSL_(mat[234])\s*\(', transformed_body)
        if mm:
            self.matrix_macros[macro_name] = mm.group(1)

        # Reconstruct the line
        result = f"{indent}#define {macro_name}{params}"
        if transformed_body:
            result += f" {transformed_body}"

        # Add back comment if present
        if comment:
            result += comment

        return result

    def _extract_comment(self, line: str) -> tuple:
        """
        Extract inline comment from a line.

        Args:
            line: Source line

        Returns:
            Tuple of (line_without_comment, comment_part)
        """
        # Look for // style comments
        comment_pos = line.find('//')
        if comment_pos != -1:
            return line[:comment_pos].rstrip(), ' ' + line[comment_pos:]

        # For now, we don't handle /* */ style comments within a line
        # (they're less common in preprocessor directives)
        return line, ''

    # Category E (Session 53) — literal matrix-ctor multiplication wrap.
    _MAT_CTOR_RE = re.compile(r'\bGLSL_mat[234]\s*\(')

    @staticmethod
    def _balanced_paren_end(s: str, open_idx: int) -> int:
        """Index just past the ')' matching the '(' at open_idx, or -1."""
        depth = 0
        for i in range(open_idx, len(s)):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
                if depth == 0:
                    return i + 1
        return -1

    @staticmethod
    def _scan_operand_left(s: str):
        """Start index of the multiplication operand ENDING at s's end —
        an identifier / call / paren-group / cast / subscript chain like
        `p.xy`, `(p)`, `rot2(a)`, `(float2)(t, t)`, `arr[i].xy` — or None.
        """
        i = len(s)
        while True:
            if i > 0 and s[i - 1] == ')':
                # One or more juxtaposed paren groups (a cast `(T)(args)` is
                # two): consume balanced groups right-to-left.
                while i > 0 and s[i - 1] == ')':
                    depth = 0
                    j = i - 1
                    while j >= 0:
                        if s[j] == ')':
                            depth += 1
                        elif s[j] == '(':
                            depth -= 1
                            if depth == 0:
                                break
                        j -= 1
                    if j < 0 or depth != 0:
                        return None
                    i = j
                # Optional callee/type identifier hugging the '('.
                k = i
                while k > 0 and (s[k - 1].isalnum() or s[k - 1] == '_'):
                    k -= 1
                i = k
            elif i > 0 and s[i - 1] == ']':
                depth = 0
                j = i - 1
                while j >= 0:
                    if s[j] == ']':
                        depth += 1
                    elif s[j] == '[':
                        depth -= 1
                        if depth == 0:
                            break
                    j -= 1
                if j < 0 or depth != 0:
                    return None
                i = j
                continue  # the array name precedes the subscript
            elif i > 0 and (s[i - 1].isalnum() or s[i - 1] == '_'):
                k = i
                while k > 0 and (s[k - 1].isalnum() or s[k - 1] == '_'):
                    k -= 1
                i = k
            else:
                return None
            # A '.' chains another unit to the left (member/swizzle base).
            if i > 0 and s[i - 1] == '.':
                i -= 1
                continue
            return i

    @staticmethod
    def _scan_operand_right(s: str):
        """End index (exclusive) of the multiplication operand STARTING at
        index 0 of s — mirror of _scan_operand_left — or None."""
        i = 0
        n = len(s)
        while True:
            if i < n and s[i] == '(':
                # One or more juxtaposed paren groups (cast syntax is two).
                while i < n and s[i] == '(':
                    end = PreprocessorTransformer._balanced_paren_end(s, i)
                    if end < 0:
                        return None
                    i = end
            elif i < n and (s[i].isalpha() or s[i] == '_'):
                while i < n and (s[i].isalnum() or s[i] == '_'):
                    i += 1
                if i < n and s[i] == '(':
                    end = PreprocessorTransformer._balanced_paren_end(s, i)
                    if end < 0:
                        return None
                    i = end
            else:
                return None
            if i < n and s[i] == '[':
                depth = 0
                j = i
                while j < n:
                    if s[j] == '[':
                        depth += 1
                    elif s[j] == ']':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j >= n:
                    return None
                i = j + 1
            if i < n and s[i] == '.':
                i += 1
                continue
            return i

    def _wrap_matrix_ctor_muls(self, body: str) -> str:
        """Wrap `X *= GLSL_matN(...)`, `X * GLSL_matN(...)` and
        `GLSL_matN(...) * X` in GLSL_mul when the partner operand is a simple
        expression unit. The literal constructor is positive matrix evidence;
        anything ambiguous (operator chains `a/b*M`, unusual operands) is left
        untouched — this pass runs on EVERY #define body and #if-block line,
        so it must never rewrite something it cannot prove."""
        for _ in range(16):  # bounded; each edit consumes one ctor adjacency
            edited = self._wrap_one_matrix_ctor_mul(body)
            if edited is None:
                return body
            body = edited
        return body

    def _wrap_one_matrix_ctor_mul(self, body: str):
        """Apply the first provable wrap; return the new body or None."""
        for m in self._MAT_CTOR_RE.finditer(body):
            c_start = m.start()
            c_end = self._balanced_paren_end(body, m.end() - 1)
            if c_end < 0:
                continue
            ctor = body[c_start:c_end]
            left = body[:c_start].rstrip()

            # ---- X *= GLSL_matN(...) --------------------------------------
            if left.endswith('*='):
                tail = body[c_end:].lstrip()
                # The ctor must BE the whole RHS: a statement/expression
                # boundary may follow (`;`, `}`, `,`, `)`, end), but any
                # operator continuation (`*= M * 2.`) is not provable.
                if tail and tail[0] not in ';,)}':
                    continue
                lhs_region = left[:-2].rstrip()
                lv_start = self._scan_operand_left(lhs_region)
                if lv_start is None:
                    continue
                before = lhs_region[:lv_start].rstrip()
                if before.endswith(('*', '/', '%', '.')):
                    continue
                lv = lhs_region[lv_start:]
                return (body[:lv_start] + f"{lv} = GLSL_mul({lv}, {ctor})"
                        + body[c_end:])

            # ---- X * GLSL_matN(...) ---------------------------------------
            if left.endswith('*') and not left.endswith(('**', '*=')):
                lop_region = left[:-1].rstrip()
                lop_start = self._scan_operand_left(lop_region)
                if lop_start is None:
                    continue
                before = lop_region[:lop_start].rstrip()
                if before.endswith(('*', '/', '%', '.')):
                    continue
                lop = lop_region[lop_start:]
                return (body[:lop_start] + f"GLSL_mul({lop}, {ctor})"
                        + body[c_end:])

            # ---- GLSL_matN(...) * X ---------------------------------------
            # Guard: a `*`/`/` clinging to the ctor's LEFT would change
            # associativity (`a / M * v` is `(a/M)*v`, not `a/(M*v)`).
            if not left.endswith(('*', '/', '%', '.')):
                rest = body[c_end:]
                stripped = rest.lstrip()
                if stripped.startswith('*') and not stripped.startswith(('**', '*=')):
                    rop_region = stripped[1:].lstrip()
                    rop_end = self._scan_operand_right(rop_region)
                    if rop_end is None:
                        continue
                    rop = rop_region[:rop_end]
                    consumed = len(rest) - len(rop_region) + rop_end
                    return (body[:c_start] + f"GLSL_mul({ctor}, {rop})"
                            + body[c_end + consumed:])
        return None

    def _transform_code_line(self, line: str) -> str:
        """
        Transform a code line (non-preprocessor line inside conditional blocks).

        Applies vector constructor transformations to match AST transformer behavior.

        Args:
            line: Code line

        Returns:
            Transformed code line
        """
        # Apply the same transformations as _transform_macro_body
        # This ensures code inside #ifdef blocks gets properly transformed
        return self._transform_macro_body(line)

    def _scan_ctor_args(self, text: str, open_idx: int):
        """
        Balanced-paren scan of a constructor argument list (category N).

        Args:
            text: The full body text.
            open_idx: Index of the constructor's opening '('.

        Returns:
            (arity, arg_text) where arity is the number of TOP-LEVEL
            arguments and arg_text the raw text between the outer parens.
            Returns (None, None) when the list is unscannable — unbalanced
            within the body (e.g. a macro splits the parens) — so the caller
            keeps today's cast behavior for it.
        """
        depth = 0
        arity = 1
        has_content = False
        for i in range(open_idx, len(text)):
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    if not has_content:
                        arity = 0
                    return arity, text[open_idx + 1:i]
            elif depth == 1:
                if c == ',':
                    arity += 1
                elif not c.isspace():
                    has_content = True
        return None, None

    @staticmethod
    def _has_top_level_comma(text: str) -> bool:
        """True if text contains a comma outside any parens/brackets."""
        depth = 0
        for c in text:
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c == ',' and depth <= 0:
                return True
        return False

    def _transform_macro_body(self, body: str) -> str:
        """
        Transform a macro body by applying GLSL transformations.

        Applies:
        1. Vector constructor cast syntax: vec2(...) -> (float2)(...)
        2. Scalar constructor cast syntax: float(...) -> (float)(...)
        3. Float literal suffix: 3.14159 -> 3.14159f
        4. Function call prefix: sin(x) -> GLSL_sin(x)

        Args:
            body: Macro body string

        Returns:
            Transformed macro body
        """
        if not body or not body.strip():
            return body

        # Step 1: Transform vector constructors.
        # Component lists / scalar-literal broadcasts keep the legal cast
        # syntax vec2(a, b) -> (float2)(a, b); a SINGLE argument that could be
        # a vector routes to the overloadable dispatcher vec2(U) ->
        # GLSL_vec2(U) (category N — the cast `(int2)(U)` is invalid when U
        # expands to a float2, and only the OpenCL compiler knows U's type).
        # This must be done BEFORE function call transformation to avoid conflicts
        for glsl_type, opencl_type in sorted(self.vector_types.items(), key=lambda x: len(x[0]), reverse=True):
            # Pattern: vec2 followed by (
            # Use word boundary to avoid partial matches (e.g., vec2d)
            # Negative lookbehind: not preceded by GLSL_ (avoid double-transforming)
            pattern = re.compile(r'(?<!GLSL_)\b' + re.escape(glsl_type) + r'\s*\(')
            dispatcher = self.vector_ctor_dispatchers[glsl_type]

            pieces = []
            pos = 0
            while True:
                match = pattern.search(body, pos)
                if match is None:
                    break
                open_idx = match.end() - 1  # index of the '('
                arity, arg_text = self._scan_ctor_args(body, open_idx)
                if (arity == 1
                        and not self._numeric_literal_re.match(arg_text.strip())
                        and arg_text.strip() not in self.comma_object_macros):
                    replacement = f'{dispatcher}('
                else:
                    # 2+ args (legal OpenCL vector literal), a provably-scalar
                    # literal broadcast, an empty list, or an unscannable
                    # (unbalanced) argument list: keep today's cast behavior.
                    replacement = f'({opencl_type})('
                pieces.append(body[pos:match.start()])
                pieces.append(replacement)
                pos = match.end()  # continue after the '(' — nested ctors still seen
            pieces.append(body[pos:])
            body = ''.join(pieces)

        # Step 1a: Transform matrix constructors to the overloadable GLSL_matN
        # dispatcher — mat2(c,s,-s,c) -> GLSL_mat2(c,s,-s,c),
        # mat2(cos(a+vec4(...))) -> GLSL_mat2((float4)(...)), mat4(1.0) ->
        # GLSL_mat4(1.0) (diagonal). GLSL_mat2/3/4 are overloadable in
        # matrix_ops.h, so every GLSL matrix-ctor arg shape resolves without
        # counting components textually. Must run BEFORE the bare-type map
        # (Step 1c) so `mat2(` is consumed here, not rewritten to `matrix2x2(`.
        for glsl_type in self.matrix_types:
            pattern = r'(?<!GLSL_)\b' + re.escape(glsl_type) + r'\s*\('
            body = re.sub(pattern, f'GLSL_{glsl_type}(', body)

        # Step 1a-2 (category E, Session 53): wrap a `*`/`*=` whose operand is
        # a literal GLSL_matN(...) constructor in the overloadable GLSL_mul.
        # These textual regions never reach the AST matmul lowering, so
        # `v *= mat2(...)` / `(p)*mat2(...)` inside a #define body (or a #if
        # block line that later falls back to raw text) stayed a raw struct
        # multiply. The literal ctor is positive matrix evidence — GLSL_mul
        # has NO vec·vec/scalar·scalar overload, so wrapping without it is
        # forbidden. Runs right after Step 1a (ctors are GLSL_matN( now).
        body = self._wrap_matrix_ctor_muls(body)

        # Step 1b: Transform scalar type constructors to cast syntax
        # float(x) -> (float)(x), int(x) -> (int)(x), etc. (category V)
        # Word boundary + required '(' means declarations (`float x`),
        # existing casts (`(float)(x)`) and identifiers embedding a type name
        # (`intersect(`, `convert_float(`) never match.
        for glsl_type, opencl_type in self.scalar_types.items():
            pattern = r'\b' + re.escape(glsl_type) + r'\s*\('
            body = re.sub(pattern, f'({opencl_type})(', body)

        # Step 1c: Map bare matrix type spellings in declarations / casts
        # (mat2 R = ...) to the OpenCL matrix struct type. Constructors were
        # already rewritten to GLSL_matN in Step 1a, so a remaining `matN`
        # token is a type name. `\b` on both sides leaves GLSL_matN and
        # composite identifiers (GLSL_mul_vec2_mat2) untouched.
        for glsl_type, opencl_type in self.matrix_types.items():
            pattern = r'(?<!GLSL_)\b' + re.escape(glsl_type) + r'\b'
            body = re.sub(pattern, opencl_type, body)

        # Step 2: Transform float literals
        # Pattern: number with decimal point or exponent, not followed by 'f' or 'F'
        # Examples: 3.14159, 1.0, 0.5, 1e4, 1.5e-3
        # Negative lookahead: not followed by 'f', 'F', or digit or letter (to avoid partial matches)

        def add_float_suffix(match):
            """Add 'f' suffix to float literal if not present."""
            number = match.group(0)
            # Check if already has 'f' suffix
            if number.endswith('f') or number.endswith('F'):
                return number
            return number + 'f'

        # Match float literals with decimal point or exponent:
        # - Optional minus sign (not captured in pattern, handled by word boundary)
        # - Digits, optional decimal point with more digits, optional exponent
        # - Must have either a decimal point OR an exponent to be a float
        # - Not followed by 'f', 'F', digit, or identifier character (negative lookahead)

        # Pattern 1: Numbers with decimal point (3.14159, 1.0, 0.5)
        # Must not be followed by 'f', 'F', digit, or identifier char
        float_pattern = r'(?<!\w)(\d+\.\d*(?:[eE][+-]?\d+)?)(?![fF\d])'
        body = re.sub(float_pattern, lambda m: m.group(1) + 'f', body)

        # Pattern 2: Numbers with exponent but no decimal (1e4)
        # Must not be followed by 'f', 'F', digit
        exp_pattern = r'(?<!\w)(\d+[eE][+-]?\d+)(?![fF\d])'
        body = re.sub(exp_pattern, lambda m: m.group(1) + 'f', body)

        # Pattern 3: Decimal point at start (.5, .123)
        # Must not be followed by 'f', 'F', digit
        decimal_pattern = r'(?<!\w)(\.\d+(?:[eE][+-]?\d+)?)(?![fF\d])'
        body = re.sub(decimal_pattern, lambda m: m.group(1) + 'f', body)

        # Step 3: Transform GLSL function calls
        # For each built-in function, add GLSL_ prefix
        # Pattern: function_name followed by '('
        # Use word boundaries to avoid partial matches

        for func_name in sorted(self.glsl_builtins, key=len, reverse=True):
            # Sort by length (descending) to handle longer names first
            # e.g., 'inversesqrt' before 'sqrt'
            pattern = r'\b' + re.escape(func_name) + r'\s*\('
            replacement = f'GLSL_{func_name}('
            body = re.sub(pattern, replacement, body)

        return body
