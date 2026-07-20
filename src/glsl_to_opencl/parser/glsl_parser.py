"""
GLSL Parser using tree-sitter-glsl.
"""

import re

from tree_sitter import Language, Parser
import tree_sitter_glsl as tsglsl

# Category UNKNOWN (Session 32): tree-sitter-glsl inherits C's `cast_expression`
# grammar, so `(ident) <expr>` is ambiguous between a parenthesised grouping and
# a C-style type cast. GLSL has NO C casts, yet the GLR parser resolves toward a
# cast in certain operator contexts (e.g. a `/` adjacent to `(x)+term`), which
# both drops operands at emit time AND mis-associates the surrounding operators.
# Bounded re-parse loop below double-parenthesises the offending type span
# (`(rot)` -> `((rot))`, semantically identical for every GLSL type) to force
# the grouping interpretation back.
_MAX_CAST_DISAMBIGUATION_PASSES = 6

from .ast_nodes import TranslationUnit
from .errors import ParseError


# Category K: tree-sitter-glsl rejects two legal GLSL array spellings, so they
# are normalized to the equivalent C-style form before parsing. Both rewrites
# stay on one line (line numbers in ParseErrors are preserved).
#
# 1. Type-first declaration `float[4] pattern` -> `float pattern[4]`.
#    The pattern `ident [ digits? ] ident` followed by = ; , or ) cannot occur
#    in expression contexts of valid GLSL (an identifier never directly
#    follows a subscript), so matching any identifier as the type is safe.
#    The size group is `\w*` (Category P cluster 4): it accepts a *named*
#    compile-time size (`vec2[N] poly`, `ball[BALLCOUNT] balls`) as well as a
#    digit literal. An identifier size never appears in a genuine subscript
#    followed by `ident [=;,)]`, so widening the group stays safe.
# The size may be a constant *expression*, not just a single token — `vec3[SZ*3]
# vertices` (tdjfWc). The size group therefore accepts anything but a bracket,
# and the whole match is pinned to a single line (horizontal whitespace only):
# a multi-line match would let a bracketed range in a trailing comment
# (`// remap to [0,1]`) fuse with the next line's identifier and delete it.
_TYPE_FIRST_ARRAY_DECL = re.compile(
    r'\b([A-Za-z_]\w*)[ \t]*\[[ \t]*([^\]\[\n]*?)[ \t]*\][ \t]+([A-Za-z_]\w*)[ \t]*(?=[=;,)])'
)
# 2. Unsized array constructor `int[](...)` -> `int[1](...)`. The size is a
#    parse placeholder only: the transformer discards it (see
#    IR.ArrayConstructor) and the emitter never writes one.
_UNSIZED_ARRAY_CTOR = re.compile(r'\b([A-Za-z_]\w*)\s*\[\s*\]\s*\(')

# Category T: tree-sitter-glsl accepts a single parameter qualifier but rejects
# the legal GLSL *combination* `const in` / `const out` / `const inout`, so the
# redundant token is collapsed before parsing (single-line rewrite → line
# numbers preserved). `in` is the read-only default, so `const in` keeps the
# `const` (a valid OpenCL value-param qualifier); for `out`/`inout` the pointer
# semantics dominate and `const` (meaningless/illegal on an output param) is
# dropped, leaving the qualifier the transformer's `_transform_parameter`
# already turns into a `__private TYPE*`. The trailing `\b` keeps `const int`
# and `const invariant` untouched (the token after `in`/`out` is a word char).
_CONST_PARAM_QUALIFIER = re.compile(r'\bconst\s+(inout|in|out)\b')

# Category P (cluster 1): tree-sitter-glsl reads a *grouping* paren immediately
# followed by a scalar primitive constructor — `(float(...)` — as a C-style cast
# `(float)(...)` and errors, even though GLSL has no C casts and this is a legal
# constructor call inside a parenthesised expression (`(float(i)/float(N))` is a
# ubiquitous loop idiom). Inserting a parse-neutral unary `+` after the opening
# paren disambiguates it back to an expression. Unary `+` is identity on every
# numeric type and emits fine as OpenCL; `bool` is excluded (unary `+` is illegal
# on bool). Over-matching the call-arglist case (`vec4(float(a))`) is harmless —
# it only adds a no-op `+` inside the argument.
_PAREN_PRIMITIVE_CTOR = re.compile(r'\(\s*(float|int|uint|double)\s*\(')

# Category P (4sjcz1): the same C-cast misread for the `bool` constructor —
# `(bool(x) ? a : b)` is read as the cast `(bool)(x)`. `bool` is excluded from
# the `+` rewrite above (unary `+` is illegal on bool), so it is disambiguated
# with a double logical-not `!!`, which is identity on bool (`!!b == b`), forces
# expression context (a cast operand cannot start with `!`), and emits fine as
# OpenCL. Over-matching the already-legal call/`if` case (`if(bool(x))`) is
# harmless — `!!` leaves the truth value unchanged.
_PAREN_BOOL_CTOR = re.compile(r'\(\s*bool\s*\(')

# Category P (cluster 2): the GLSL ES precision qualifiers `highp`/`mediump`/
# `lowp` are rejected by tree-sitter-glsl wherever they prefix a type, and so is
# the default-precision statement `precision <qual> <type>;`. Both are
# meaningless in OpenCL. The whole default-precision statement is removed first
# (a bare `precision <type>;` would still not parse), then the inline qualifier
# token is stripped off return types, parameters, and declarations. Only
# horizontal whitespace is consumed, so no source line is ever merged and
# ParseError line numbers stay accurate.
_PRECISION_STMT = re.compile(r'\bprecision[ \t]+(?:highp|mediump|lowp)[ \t]+\w+[ \t]*;')
_PRECISION_QUALIFIER = re.compile(r'\b(?:highp|mediump|lowp)[ \t]+')

# Category X: GLSL accepts the redundant square spelling `matNxN` as an exact
# synonym of `matN` (`mat3x3` == `mat3`), in both type-name and constructor
# position. tree-sitter-glsl and OpenCL-C know only `matN`, so the spelling
# leaked through verbatim and clang errored `unknown type name 'mat3x3'`. The
# backreference `\b([234])x\1\b` matches only the square cases and rewrites
# them to `matN`; the non-square spellings (`mat2x4`, `mat3x2`, `mat4x2`) are
# genuinely distinct types and are deliberately left untouched (they need real
# struct types — deferred). Same-line, so ParseError line numbers stay accurate.
_SQUARE_MATRIX_SPELLING = re.compile(r'\bmat([234])x\1\b')

# Category P (cluster 3): tree-sitter-glsl does not recognise GLSL's logical-XOR
# operator `^^`. `a ^^ b` on bools equals `a != b`, but `^^` binds looser than
# almost everything (only `||` is looser) whereas `!=` binds at equality level,
# so a bare token swap mis-groups mixed operands (`b ^^ f()==1`). See
# `_rewrite_logical_xor` for the operand-wrapping scan.
_LOGICAL_XOR = '^^'
# operand-boundary characters at bracket-depth 0 (single-char)
_XOR_LEFT_STOP = frozenset(',;?:([{')
_XOR_RIGHT_STOP = frozenset(',;?:)]}')

# Category U: a user identifier (variable, parameter, or function name) whose
# spelling is an OpenCL / C reserved word. None of these words are GLSL keywords
# — GLSL has no `char`/`kernel`/address-space qualifiers — so in GLSL source they
# can ONLY be user identifiers, yet the emitted kernel treats them as keywords.
# The symptom is either a parse failure (tree-sitter reads `(char >> 4)` as a
# C-cast `(char)`, or rejects `uint char` in some positions) or a clang error
# (`uint char = ...;` -> "cannot combine with previous declaration specifier").
# Every occurrence of a reserved word as an identifier is suffixed with `_`
# (`char` -> `char_`) on the whole source before parsing, so the declaration and
# every use stay consistent across the AST path, the `#ifdef` textual passthrough
# and `#define` bodies alike. Because a shader that uses one of these words as an
# identifier is already failing downstream, the rename can only fix, never
# regress. GLSL's own memory qualifiers `volatile`/`restrict`/`coherent`/
# `readonly`/`writeonly` and the shared keyword `const` are deliberately excluded
# (they are valid in both languages); only the OpenCL-spelled `read_only`/
# `write_only` variants are listed.
OPENCL_RESERVED_WORDS = frozenset({
    # C integer type keywords
    'char', 'short', 'long', 'signed', 'unsigned',
    # C storage-class / misc keywords not present in GLSL
    'static', 'extern', 'register', 'auto', 'typedef', 'union', 'enum',
    'goto', 'sizeof',
    # OpenCL kernel / address-space / access qualifiers
    'kernel', 'global', 'local', 'private', 'constant', 'generic',
    'read_only', 'write_only', 'read_write', 'pipe',
    # OpenCL-only scalar types
    'half',
})

# Category AC: names that OpenCL C *predefines as macros* (`cl_kernel.h` pulls in
# the `<math.h>`/`<float.h>` constants) but GLSL/Shadertoy WebGL do NOT. A GLSL
# shader that wants `M_PI`/`FLT_MAX`/... must declare it itself (`const float
# M_PI = 3.14159;`); at compile the predefined macro expands the *declared name*
# -> `const float 3.14159f = 3.14159f;` -> "expected identifier or '('". Because
# none of these are predefined in GLSL, any occurrence in GLSL source is a user
# identifier, so — exactly like OPENCL_RESERVED_WORDS — a blanket `_` suffix
# rename can only touch already-failing shaders and never regresses a passing one
# (proven per-corpus with the hash blast-radius rig). The integer-limit macros
# (`INT_MAX`, `CHAR_BIT`, ...) are also predefined but are omitted here: no
# corpus shader needs them and they carry a marginally higher risk of being
# used-undefined-yet-compiling, so add them only when a real shader demands it.
OPENCL_PREDEFINED_MACROS = frozenset({
    # math constants (double), plus the single-`_F` and half-`_H` variants OpenCL
    # also predefines
    'M_E', 'M_LOG2E', 'M_LOG10E', 'M_LN2', 'M_LN10', 'M_PI', 'M_PI_2',
    'M_PI_4', 'M_1_PI', 'M_2_PI', 'M_2_SQRTPI', 'M_SQRT2', 'M_SQRT1_2',
    'M_E_F', 'M_LOG2E_F', 'M_LOG10E_F', 'M_LN2_F', 'M_LN10_F', 'M_PI_F',
    'M_PI_2_F', 'M_PI_4_F', 'M_1_PI_F', 'M_2_PI_F', 'M_2_SQRTPI_F',
    'M_SQRT2_F', 'M_SQRT1_2_F',
    'M_E_H', 'M_LOG2E_H', 'M_LOG10E_H', 'M_LN2_H', 'M_LN10_H', 'M_PI_H',
    'M_PI_2_H', 'M_PI_4_H', 'M_1_PI_H', 'M_2_PI_H', 'M_2_SQRTPI_H',
    'M_SQRT2_H', 'M_SQRT1_2_H',
    # <float.h> limits
    'FLT_DIG', 'FLT_MANT_DIG', 'FLT_MAX_10_EXP', 'FLT_MAX_EXP',
    'FLT_MIN_10_EXP', 'FLT_MIN_EXP', 'FLT_RADIX', 'FLT_MAX', 'FLT_MIN',
    'FLT_EPSILON',
    # special float values OpenCL predefines
    'MAXFLOAT', 'HUGE_VALF', 'HUGE_VAL', 'INFINITY', 'NAN',
})

_RESERVED_IDENTIFIER = re.compile(
    r'\b(?:' + '|'.join(sorted(OPENCL_RESERVED_WORDS | OPENCL_PREDEFINED_MACROS))
    + r')\b'
)


def _collapse_const_param_qualifier(match: 're.Match') -> str:
    qualifier = match.group(1)
    return 'const' if qualifier == 'in' else qualifier


def _mask_comments(source: str) -> str:
    """Return a same-length copy with comment bodies blanked to spaces.

    Newlines are preserved so indices and line numbers are unchanged. Used to
    keep `^^` (and its operand scan) from matching inside comments, where the
    XOR operator commonly appears as an ASCII arrow (`^^^`).
    """
    out = []
    i, n = 0, len(source)
    while i < n:
        two = source[i:i + 2]
        if two == '//':
            j = i
            while j < n and source[j] != '\n':
                j += 1
            out.append(' ' * (j - i))
            i = j
        elif two == '/*':
            j = i + 2
            while j < n and source[j - 1:j + 1] != '*/':
                j += 1
            j = min(j + 1, n)
            out.append(''.join('\n' if c == '\n' else ' ' for c in source[i:j]))
            i = j
        else:
            out.append(source[i])
            i += 1
    return ''.join(out)


def _rewrite_logical_xor(source: str) -> str:
    """Rewrite `A ^^ B` -> `(A) != (B)`, wrapping operands to preserve grouping.

    Runs on comment-masked text so `^^` inside comments is ignored; edits are
    spliced into the original source right-to-left so indices stay valid.
    """
    masked = _mask_comments(source)
    positions = []
    start = 0
    while True:
        k = masked.find(_LOGICAL_XOR, start)
        if k == -1:
            break
        positions.append(k)
        start = k + 2

    for k in reversed(positions):
        # scan left for the start of the left operand
        i, depth = k - 1, 0
        while i >= 0:
            ch = masked[i]
            if ch in ')]}':
                depth += 1
            elif ch in '([{':
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if ch in _XOR_LEFT_STOP:
                    break
                if ch == '|' and i > 0 and masked[i - 1] == '|':
                    break
                if ch == '=':
                    prev = masked[i - 1] if i > 0 else ''
                    nxt = masked[i + 1] if i + 1 < len(masked) else ''
                    # `==` `!=` `<=` `>=` are not boundaries; a lone/compound `=` is
                    if prev not in '=!<>' and nxt != '=':
                        break
            i -= 1
        lhs_start = i + 1

        # scan right for the end of the right operand
        j, depth = k + 2, 0
        while j < len(masked):
            ch = masked[j]
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if ch in _XOR_RIGHT_STOP:
                    break
                if ch == '|' and j + 1 < len(masked) and masked[j + 1] == '|':
                    break
                if ch == '^' and j + 1 < len(masked) and masked[j + 1] == '^':
                    break
            j += 1
        rhs_end = j

        # keep whitespace that sits outside the operands out of the parens
        while lhs_start < k and masked[lhs_start] in ' \t\r\n':
            lhs_start += 1
        while rhs_end > k + 2 and masked[rhs_end - 1] in ' \t\r\n':
            rhs_end -= 1

        lhs = source[lhs_start:k].strip()
        rhs = source[k + 2:rhs_end].strip()
        replacement = '(' + lhs + ') != (' + rhs + ')'
        source = source[:lhs_start] + replacement + source[rhs_end:]

    return source


def _rename_reserved_identifiers(source: str) -> str:
    """Suffix `_` onto every OpenCL reserved word or predefined macro used as an
    identifier.

    Runs on a comment-masked copy so a match inside a comment is ignored, then
    splices a single `_` into the original source right-to-left (indices stay
    valid, line count is unchanged). See the `OPENCL_RESERVED_WORDS` (category U)
    and `OPENCL_PREDEFINED_MACROS` (category AC) notes for why a blanket token
    rename is safe here: neither the keywords nor the macros are predefined in
    GLSL, so any match is a user identifier whose shader is already failing.
    """
    masked = _mask_comments(source)
    ends = [m.end() for m in _RESERVED_IDENTIFIER.finditer(masked)]
    for pos in reversed(ends):
        source = source[:pos] + '_' + source[pos:]
    return source


def _normalize_array_syntax(source: str) -> str:
    """Rewrite GLSL spellings that tree-sitter-glsl cannot parse."""
    source = _rename_reserved_identifiers(source)
    source = _SQUARE_MATRIX_SPELLING.sub(r'mat\1', source)
    source = _TYPE_FIRST_ARRAY_DECL.sub(r'\1 \3[\2]', source)
    source = _UNSIZED_ARRAY_CTOR.sub(r'\1[1](', source)
    source = _CONST_PARAM_QUALIFIER.sub(_collapse_const_param_qualifier, source)
    source = _PRECISION_STMT.sub('', source)
    source = _PRECISION_QUALIFIER.sub('', source)
    source = _rewrite_logical_xor(source)
    # after XOR: the wrapping parens can create a fresh `(float(` etc., so the
    # parenthesised-primitive-ctor rewrites must run last.
    source = _PAREN_PRIMITIVE_CTOR.sub(r'(+\1(', source)
    source = _PAREN_BOOL_CTOR.sub(r'(!!bool(', source)
    return source


class GLSLParser:
    """
    Main GLSL parser using tree-sitter-glsl.

    Usage:
        parser = GLSLParser()
        ast = parser.parse(glsl_source)
    """

    def __init__(self):
        """Initialize tree-sitter-glsl parser."""
        self._language = Language(tsglsl.language())
        self._parser = Parser(self._language)

    def parse(self, source: str) -> TranslationUnit:
        """
        Parse GLSL source code into AST.

        Args:
            source: GLSL source code as string

        Returns:
            TranslationUnit (root AST node)

        Raises:
            ParseError: If parsing fails
        """
        # Normalize array spellings tree-sitter-glsl rejects (category K)
        source = _normalize_array_syntax(source)

        # Disambiguate spurious C-style casts (GLSL has none) — see the
        # module-level note. Must run on the final source so the wrapped
        # parens survive into the tree the transformer consumes.
        source = self._disambiguate_casts(source)

        # Convert source to bytes (tree-sitter requirement)
        source_bytes = bytes(source, "utf8")

        # Parse with tree-sitter
        tree = self._parser.parse(source_bytes)
        root_node = tree.root_node

        # Check for parse errors
        if root_node.has_error:
            # Find first error node
            error_node = self._find_error_node(root_node)
            if error_node:
                line, col = error_node.start_point
                raise ParseError(
                    f"Syntax error at line {line + 1}, column {col + 1}",
                    line=line,
                    column=col
                )
            else:
                raise ParseError("Syntax error in GLSL source")

        # Wrap in typed AST
        return TranslationUnit(root_node, source)

    def _disambiguate_casts(self, source: str) -> str:
        """Neutralise spurious C-style `cast_expression` mis-parses.

        GLSL has no C-style casts, so any `cast_expression` tree-sitter-glsl
        emits is really a parenthesised grouping `(x)` that the GLR parser
        mis-resolved as a type cast (this happens for `(ident)+term` when a
        division/multiplication sits adjacent). Left alone, the mis-parse both
        re-associates the surrounding operators and drops the "cast value" at
        emit time (the transformer has no cast handler).

        Fix by double-parenthesising the offending type span in the source
        (`(rot)` -> `((rot))`). Extra parens are semantically identical for
        every GLSL type and force the grouping interpretation, restoring the
        correct precedence tree on re-parse. Byte-offset edits are applied
        right-to-left so earlier offsets stay valid.

        Only the true mis-parse is targeted: it is distinguished by the cast
        VALUE being a `unary_expression` (`(rot) + PI` reads the `+PI` as the
        casted unary). A GENUINE OpenCL cast — which the transformer itself
        emits when lowering a scalar constructor, e.g. `float(i)` ->
        `(float)(i)`, and which this parser re-sees when the pipeline re-parses
        emitted code for `#ifdef` blocks — has a `parenthesized_expression` (or
        bare identifier) value and is left untouched. Wrapping such a cast would
        never remove it (`((float))(i)` is still a cast), so a naive pass would
        diverge. A convergence guard (stop once the count stops shrinking) plus
        the bounded pass count make the loop safe on any input.
        """
        prev_count = None
        for _ in range(_MAX_CAST_DISAMBIGUATION_PASSES):
            data = bytes(source, "utf8")
            tree = self._parser.parse(data)

            spans = []

            def collect(node):
                if node.type == "cast_expression":
                    value_node = node.child_by_field_name("value")
                    type_node = node.child_by_field_name("type")
                    # Only the `(x) <unary>` mis-parse; genuine casts (value is
                    # a parenthesised expr or plain identifier) are left alone.
                    if (type_node is not None and value_node is not None
                            and value_node.type == "unary_expression"):
                        spans.append((type_node.start_byte, type_node.end_byte))
                for child in node.children:
                    collect(child)

            collect(tree.root_node)
            if not spans:
                return source
            # Guard against non-convergence (a cast we cannot dissolve).
            if prev_count is not None and len(spans) >= prev_count:
                return source
            prev_count = len(spans)

            for start, end in sorted(set(spans), reverse=True):
                data = data[:start] + b"(" + data[start:end] + b")" + data[end:]
            source = data.decode("utf8")

        return source

    def _find_error_node(self, node):
        """Find first ERROR node in tree (DFS)."""
        if node.type == "ERROR":
            return node
        for child in node.children:
            error = self._find_error_node(child)
            if error:
                return error
        return None
