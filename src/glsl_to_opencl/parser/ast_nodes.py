"""
AST Node classes for GLSL parse tree.

This module provides typed wrappers around tree-sitter nodes.
"""

from typing import List, Optional, Tuple


class ASTNode:
    """
    Base class for all AST nodes.

    Wraps a tree-sitter node with a clean Python API.

    Note: Minimal implementation for TDD - will be completed during implementation phase.
    """

    def __init__(self, ts_node, source: str):
        """
        Initialize AST node.

        Args:
            ts_node: tree-sitter node
            source: Original GLSL source code
        """
        self._node = ts_node
        self._source = source

    @property
    def type(self) -> str:
        """Node type (e.g., 'function_definition')."""
        return self._node.type

    @property
    def text(self) -> str:
        """Source text for this node."""
        return self._node.text.decode('utf8')

    @property
    def start_point(self) -> Tuple[int, int]:
        """Start position (line, column)."""
        return self._node.start_point

    @property
    def end_point(self) -> Tuple[int, int]:
        """End position (line, column)."""
        return self._node.end_point

    @property
    def has_error(self) -> bool:
        """True if this node's subtree contains a parse error (tree-sitter
        error recovery). Used to gate AST routing of #if-block contents."""
        return self._node.has_error

    @property
    def children(self) -> List['ASTNode']:
        """Child nodes (wrapped)."""
        return [wrap_node(child, self._source) for child in self._node.children]

    @property
    def named_children(self) -> List['ASTNode']:
        """Named child nodes only (excludes punctuation)."""
        return [wrap_node(child, self._source) for child in self._node.named_children]

    def child_by_field_name(self, field_name: str) -> Optional['ASTNode']:
        """Get child by field name."""
        child = self._node.child_by_field_name(field_name)
        return wrap_node(child, self._source) if child else None

    def walk(self):
        """
        Depth-first traversal of AST.

        Yields:
            ASTNode: Each node in DFS order
        """
        yield self
        for child in self.children:
            yield from child.walk()

    def __repr__(self):
        text = self.text[:40] if len(self.text) > 40 else self.text
        return f"{self.__class__.__name__}(type={self.type}, text={text!r})"


class TranslationUnit(ASTNode):
    """
    Root AST node representing entire shader.
    """

    @property
    def declarations(self) -> List[ASTNode]:
        """All top-level declarations."""
        return self.named_children

    def get_functions(self) -> List['FunctionDefinition']:
        """Get all function definitions."""
        return [node for node in self.declarations
                if isinstance(node, FunctionDefinition)]

    def get_main_image(self) -> Optional['FunctionDefinition']:
        """
        Find mainImage() function (Shadertoy entry point).

        Returns:
            FunctionDefinition if found, None otherwise
        """
        for func in self.get_functions():
            if func.name == "mainImage":
                return func
        return None


class FunctionDefinition(ASTNode):
    """
    Function definition node.

    Example:
        void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }
    """

    @property
    def return_type(self) -> ASTNode:
        """Return type node (e.g., 'void', 'vec3')."""
        return self.named_children[0] if self.named_children else None

    @property
    def declarator(self) -> ASTNode:
        """Function declarator (name + parameters)."""
        return self.named_children[1] if len(self.named_children) > 1 else None

    @property
    def name(self) -> str:
        """Function name."""
        if not self.declarator:
            return ""

        # Find identifier in declarator
        for child in self.declarator.children:
            if child.type == "identifier":
                return child.text
        return ""

    @property
    def parameters(self) -> List[ASTNode]:
        """Function parameters."""
        if not self.declarator:
            return []

        # Find parameter_list in declarator
        for child in self.declarator.children:
            if child.type == "parameter_list":
                return [p for p in child.named_children
                        if p.type == "parameter_declaration"]
        return []

    @property
    def body(self) -> Optional[ASTNode]:
        """Function body (compound statement)."""
        if len(self.named_children) < 3:
            return None
        body = self.named_children[-1]
        return body if body.type == "compound_statement" else None


class Declaration(ASTNode):
    """Variable declaration."""
    pass


class CallExpression(ASTNode):
    """Function call expression."""

    @property
    def function(self) -> ASTNode:
        """Function being called."""
        return self.named_children[0] if self.named_children else None

    @property
    def arguments(self) -> List[ASTNode]:
        """Function arguments."""
        # Find argument_list
        for child in self.children:
            if child.type == "argument_list":
                # Filter out commas
                return [arg for arg in child.named_children]
        return []


class BinaryExpression(ASTNode):
    """Binary expression (a + b, a * b, etc.)."""

    @property
    def left(self) -> ASTNode:
        """Left operand."""
        # Use tree-sitter field access so interleaved comment nodes (which are
        # *named* children) don't shift positional indices. Fall back to the
        # first non-comment named child if the grammar lacks the field.
        field = self.child_by_field_name("left")
        if field is not None:
            return field
        operands = [c for c in self.named_children if c.type != "comment"]
        return operands[0] if operands else None

    @property
    def operator(self) -> str:
        """Operator (+, -, *, /, etc.)."""
        # Find operator node (non-named child between operands)
        operators = ["+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=",
                     "&&", "||", "&", "|", "^", "<<", ">>"]
        for child in self.children:
            if child.type in operators:
                return child.type
        return ""

    @property
    def right(self) -> ASTNode:
        """Right operand."""
        # See `left`: a comment between operator and right operand appears as a
        # named child, so named_children[1] would wrongly return the comment and
        # the real operand would be dropped. Prefer the grammar's 'right' field.
        field = self.child_by_field_name("right")
        if field is not None:
            return field
        operands = [c for c in self.named_children if c.type != "comment"]
        return operands[1] if len(operands) > 1 else None


def wrap_node(ts_node, source: str) -> ASTNode:
    """
    Wrap tree-sitter node in appropriate AST class.

    Args:
        ts_node: tree-sitter node
        source: Original source code

    Returns:
        Appropriate ASTNode subclass
    """
    node_type = ts_node.type

    # Map tree-sitter types to AST classes
    if node_type == "translation_unit":
        return TranslationUnit(ts_node, source)
    elif node_type == "function_definition":
        return FunctionDefinition(ts_node, source)
    elif node_type == "declaration":
        return Declaration(ts_node, source)
    elif node_type == "call_expression":
        return CallExpression(ts_node, source)
    elif node_type == "binary_expression":
        return BinaryExpression(ts_node, source)

    # Default: Generic ASTNode
    return ASTNode(ts_node, source)
