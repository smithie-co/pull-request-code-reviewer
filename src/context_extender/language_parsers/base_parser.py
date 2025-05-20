"""Base class for language-specific context extraction using Tree-sitter."""
from abc import ABC, abstractmethod
from typing import Optional, Set

from tree_sitter import Node, Parser, Tree

class BaseLanguageParser(ABC):
    """
    Abstract base class for language-specific parsers that use Tree-sitter
    to extract meaningful context around changed code segments.
    """

    def __init__(self, language_name: str, parser: Parser):
        """
        Initializes the base language parser.

        Args:
            language_name: The name of the language (e.g., 'python', 'hcl').
            parser: A Tree-sitter Parser instance pre-configured for the language.
        """
        self.language_name = language_name
        self.parser = parser

    def parse_code(self, code: str) -> Tree:
        """
        Parses the given code string into a Tree-sitter syntax tree.

        Args:
            code: The source code string.

        Returns:
            A Tree-sitter Tree object.
        """
        return self.parser.parse(bytes(code, "utf8"))

    @abstractmethod
    def get_enclosing_block(
        self, 
        file_content: str, 
        tree: Tree, 
        changed_line_numbers: Set[int]
    ) -> Optional[str]:
        """
        Identifies and extracts the most relevant enclosing structural block(s) 
        of code that contain the given changed line numbers.

        The definition of an "enclosing block" is language-specific.
        For example, it could be a function, a class, a method, a resource block, etc.

        Args:
            file_content: The full content of the file as a string.
            tree: The Tree-sitter syntax tree for the entire file_content.
            changed_line_numbers: A set of 1-indexed line numbers that have been modified.

        Returns:
            A string containing the source code of the enclosing block(s),
            or None if no relevant block can be determined or if changes are trivial
            (e.g., only comments, whitespace).
        """
        pass

    def _get_node_text(self, node: Node, file_content_bytes: bytes) -> str:
        """Helper to extract text for a given node."""
        return file_content_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _find_smallest_node_enclosing_lines(
        self, 
        root_node: Node, 
        target_lines_0_indexed: Set[int]
    ) -> Optional[Node]:
        """
        Finds the smallest syntax tree node that completely encloses all target lines.
        This is a generic helper and might need to be adapted or used in conjunction
        with language-specific logic in subclasses.

        Args:
            root_node: The root node of the syntax tree (or a relevant subtree).
            target_lines_0_indexed: A set of 0-indexed line numbers to be enclosed.

        Returns:
            The smallest Node that encloses all lines, or None.
        """
        if not target_lines_0_indexed:
            return None

        min_target_line = min(target_lines_0_indexed)
        max_target_line = max(target_lines_0_indexed)

        best_node: Optional[Node] = None

        def descend(node: Node) -> None:
            nonlocal best_node
            # Node lines are 0-indexed
            node_start_line = node.start_point[0]
            node_end_line = node.end_point[0]

            # Check if this node encloses all target lines
            if node_start_line <= min_target_line and node_end_line >= max_target_line:
                # This node is a candidate. Is it smaller than the current best_node?
                if best_node is None or (node.end_byte - node.start_byte) < (best_node.end_byte - best_node.start_byte):
                    # Check if it's not just a generic 'source_file' or 'program' if better options exist
                    # This heuristic might need refinement.
                    if node.type not in ["source_file", "program", "module"]: # TODO: Add more generic top-level types
                         best_node = node
                    elif best_node is None : # If it's the only option so far
                        best_node = node


                # Continue searching in children for a potentially smaller enclosing node
                for child in node.children:
                    descend(child)
            # If the node is entirely before or after the target range, no need to check its children further
            # unless some children could span across (less likely for well-formed blocks).
            # This simple check prunes some branches.
            elif node_end_line < min_target_line or node_start_line > max_target_line:
                return # No overlap
            else: # Partial overlap, children might still contain the full block
                 for child in node.children:
                    descend(child)


        descend(root_node)
        
        # If the best node is still the root (e.g. source_file) and it has only one significant child,
        # prefer that child if it also covers the lines.
        if best_node and best_node.type in ["source_file", "program", "module"] and len(best_node.children) == 1:
            child = best_node.children[0]
            if child.start_point[0] <= min_target_line and child.end_point[0] >= max_target_line:
                best_node = child
        
        return best_node

    def _get_lines_from_node(self, node: Node) -> Set[int]:
        """Returns a set of 1-indexed line numbers covered by the node."""
        return set(range(node.start_point[0] + 1, node.end_point[0] + 2))

    def _get_text_from_lines(self, file_content: str, start_line_1_indexed: int, end_line_1_indexed: int) -> str:
        lines = file_content.splitlines()
        # Adjust for 0-indexing and exclusive end for slicing
        start_idx = max(0, start_line_1_indexed - 1)
        end_idx = min(len(lines), end_line_1_indexed)
        return "\n".join(lines[start_idx:end_idx]) 