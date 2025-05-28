"""Python specific parser for Tree-sitter context extraction."""
from typing import Optional, Set, List

from tree_sitter import Node, Tree

from .base_parser import BaseLanguageParser

class PythonParser(BaseLanguageParser):
    """
    Python-specific language parser.
    Identifies enclosing functions, classes, or top-level blocks for changed lines.
    """

    RELEVANT_NODE_TYPES = {
        "function_definition",
        "class_definition",
        # "decorated_definition", # Often wraps function or class definitions
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        # Top-level statements (e.g. assignments, expressions) can be harder to delimit
        # without including too much. The generic _find_smallest_node_enclosing_lines
        # might pick these up if they are small enough.
    }

    def get_enclosing_block(
        self, 
        file_content: str, 
        tree: Tree, 
        changed_line_numbers: Set[int]
    ) -> Optional[str]:
        """
        Identifies the enclosing function, class, or significant block for Python code.
        """
        if not changed_line_numbers:
            return None

        file_content_bytes = bytes(file_content, "utf8")
        root_node = tree.root_node
        
        # Convert 1-indexed to 0-indexed for Tree-sitter
        target_lines_0_indexed = {line - 1 for line in changed_line_numbers}
        min_target_line_0_indexed = min(target_lines_0_indexed)
        max_target_line_0_indexed = max(target_lines_0_indexed)

        candidate_nodes: List[Node] = []

        def find_relevant_nodes(node: Node):
            # Check if the current node itself is one of our target types and spans the changed lines
            node_start_line = node.start_point[0]
            node_end_line = node.end_point[0]

            if node.type in self.RELEVANT_NODE_TYPES:                
                if node_start_line <= min_target_line_0_indexed and node_end_line >= max_target_line_0_indexed:
                    candidate_nodes.append(node)
                    # If we found a relevant node, we might not need to go deeper within this branch
                    # for THIS specific type of collection, but other relevant types might be nested.
                    # return # Stop descending if this node is a good candidate that covers all changes.
            
            # If the node (or its children) could possibly span the changed lines, continue search
            if not (node_end_line < min_target_line_0_indexed or node_start_line > max_target_line_0_indexed):
                 for child in node.children:
                    find_relevant_nodes(child)
            
        find_relevant_nodes(root_node)

        if not candidate_nodes:
            # Fallback to the generic smallest enclosing block if no specific type found
            # This might pick up module-level code or smaller expressions.
            fallback_node = self._find_smallest_node_enclosing_lines(root_node, target_lines_0_indexed)
            if fallback_node:
                # Avoid returning the entire file if it's too large or just a generic module node
                if fallback_node.type in ["module", "source_file"] and (fallback_node.end_point[0] - fallback_node.start_point[0]) > 50:
                    # If it's the whole file and it's big, try to get a snippet around the first changed line
                    start_line = max(1, min(changed_line_numbers) - 10)
                    end_line = min(file_content.count('\n') + 1, max(changed_line_numbers) + 10)
                    return self._get_text_from_lines(file_content, start_line, end_line)
                return self._get_node_text(fallback_node, file_content_bytes)
            return None

        # Select the smallest candidate node that covers all changes
        # (find_relevant_nodes might add parent nodes too, so we filter)
        best_candidate: Optional[Node] = None
        for node in candidate_nodes:
            node_start_line = node.start_point[0]
            node_end_line = node.end_point[0]
            if node_start_line <= min_target_line_0_indexed and node_end_line >= max_target_line_0_indexed:
                if best_candidate is None or (node.end_byte - node.start_byte) < (best_candidate.end_byte - best_candidate.start_byte):
                    best_candidate = node
        
        if best_candidate:
            # Special handling for decorated definitions to include the decorator
            if best_candidate.parent and best_candidate.parent.type == "decorated_definition":
                 # Check if the decorated_definition still covers all target lines
                parent_start_line = best_candidate.parent.start_point[0]
                parent_end_line = best_candidate.parent.end_point[0]
                if parent_start_line <= min_target_line_0_indexed and parent_end_line >= max_target_line_0_indexed:
                    return self._get_node_text(best_candidate.parent, file_content_bytes)
            return self._get_node_text(best_candidate, file_content_bytes)

        return None 