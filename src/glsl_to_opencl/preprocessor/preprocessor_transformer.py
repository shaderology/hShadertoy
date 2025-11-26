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
        }

        # Track if we're inside a preprocessor conditional block
        # This helps us know whether to transform code lines
        self.inside_conditional = False

    def transform(self, source: str) -> str:
        """
        Transform GLSL source with preprocessor directives.

        Args:
            source: GLSL source code string

        Returns:
            Transformed source code with OpenCL preprocessor directives
        """
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

        # Transform the macro body
        transformed_body = self._transform_macro_body(body)

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

    def _transform_macro_body(self, body: str) -> str:
        """
        Transform a macro body by applying GLSL transformations.

        Applies:
        1. Vector constructor cast syntax: vec2(...) -> (float2)(...)
        2. Float literal suffix: 3.14159 -> 3.14159f
        3. Function call prefix: sin(x) -> GLSL_sin(x)

        Args:
            body: Macro body string

        Returns:
            Transformed macro body
        """
        if not body or not body.strip():
            return body

        # Step 1: Transform vector constructors to cast syntax
        # vec2(...) -> (float2)(...)
        # This must be done BEFORE function call transformation to avoid conflicts
        for glsl_type, opencl_type in sorted(self.vector_types.items(), key=lambda x: len(x[0]), reverse=True):
            # Pattern: vec2 followed by (
            # Use word boundary to avoid partial matches (e.g., vec2d)
            # Negative lookbehind: not preceded by GLSL_ (avoid double-transforming)
            pattern = r'(?<!GLSL_)\b' + re.escape(glsl_type) + r'\s*\('

            def replace_constructor(match):
                """Replace vector constructor with cast syntax."""
                # Get the matched text
                matched = match.group(0)
                # Extract any whitespace between type and (
                ws_match = re.search(r'\s+', matched)
                ws = ws_match.group(0) if ws_match else ''
                # Return cast-style syntax: (float2)(
                return f'({opencl_type})('

            body = re.sub(pattern, replace_constructor, body)

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
