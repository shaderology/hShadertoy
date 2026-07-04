"""
GLSL Parser using tree-sitter-glsl.
"""

import re

from tree_sitter import Language, Parser
import tree_sitter_glsl as tsglsl

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
_TYPE_FIRST_ARRAY_DECL = re.compile(
    r'\b([A-Za-z_]\w*)\s*\[\s*(\d*)\s*\]\s+([A-Za-z_]\w*)\s*(?=[=;,)])'
)
# 2. Unsized array constructor `int[](...)` -> `int[1](...)`. The size is a
#    parse placeholder only: the transformer discards it (see
#    IR.ArrayConstructor) and the emitter never writes one.
_UNSIZED_ARRAY_CTOR = re.compile(r'\b([A-Za-z_]\w*)\s*\[\s*\]\s*\(')


def _normalize_array_syntax(source: str) -> str:
    """Rewrite GLSL array spellings that tree-sitter-glsl cannot parse."""
    source = _TYPE_FIRST_ARRAY_DECL.sub(r'\1 \3[\2]', source)
    source = _UNSIZED_ARRAY_CTOR.sub(r'\1[1](', source)
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

    def _find_error_node(self, node):
        """Find first ERROR node in tree (DFS)."""
        if node.type == "ERROR":
            return node
        for child in node.children:
            error = self._find_error_node(child)
            if error:
                return error
        return None
