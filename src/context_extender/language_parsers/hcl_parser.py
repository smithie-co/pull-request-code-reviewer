"""HCL (Terraform) specific parser for Tree-sitter context extraction."""
from typing import Optional, Set, List

from tree_sitter import Node, Tree

from .base_parser import BaseLanguageParser

class HCLParser(BaseLanguageParser):
    """
    HCL-specific language parser (primarily for Terraform).
    Identifies enclosing resource, data, variable, output, etc., blocks.
    """

    # Common top-level block types in Terraform/HCL
    # You can find these types by inspecting HCL syntax trees from tree-sitter-hcl
    RELEVANT_NODE_TYPES = {
        "resource_block",
        "data_block",
        "provider_block",
        "variable_block",
        "output_block",
        "locals_block",
        "module_block",
        "terraform_block",
        # Individual attributes might be too granular, but blocks are good.
    }

    def get_enclosing_block(
        self, 
        file_content: str, 
        tree: Tree, 
        changed_line_numbers: Set[int]
    ) -> Optional[str]:
        """
        Identifies the enclosing HCL block for Terraform code.
        """
        if not changed_line_numbers:
            return None

        file_content_bytes = bytes(file_content, "utf8")
        root_node = tree.root_node
        target_lines_0_indexed = {line - 1 for line in changed_line_numbers}
        min_target_line_0_indexed = min(target_lines_0_indexed)
        max_target_line_0_indexed = max(target_lines_0_indexed)

        candidate_nodes: List[Node] = []

        def find_relevant_nodes(node: Node):
            node_start_line = node.start_point[0]
            node_end_line = node.end_point[0]

            if node.type in self.RELEVANT_NODE_TYPES:
                if node_start_line <= min_target_line_0_indexed and node_end_line >= max_target_line_0_indexed:
                    candidate_nodes.append(node)
                    # Typically, HCL blocks don't nest other major relevant block types in a way
                    # that we'd prefer a child over a parent that already spans the change.
                    # So, we can often stop descending this branch if a relevant block is found.
                    # However, keeping it might find a tighter block if an inner attribute caused a large block match.
                    # For HCL, usually the block itself is the right granularity.
                    # return 
            
            if not (node_end_line < min_target_line_0_indexed or node_start_line > max_target_line_0_indexed):
                for child in node.children:
                    find_relevant_nodes(child)

        find_relevant_nodes(root_node)

        if not candidate_nodes:
            # Fallback for HCL might be less useful than Python if it just grabs a single attribute line.
            # The _find_smallest_node_enclosing_lines might be okay for simple expressions.
            fallback_node = self._find_smallest_node_enclosing_lines(root_node, target_lines_0_indexed)
            if fallback_node:
                # Avoid returning the entire file if it's too large
                if fallback_node.type == "configuration_file" and (fallback_node.end_point[0] - fallback_node.start_point[0]) > 50:
                    start_line = max(1, min(changed_line_numbers) - 5) # Smaller context for HCL fallback
                    end_line = min(file_content.count('\n') + 1, max(changed_line_numbers) + 5)
                    return self._get_text_from_lines(file_content, start_line, end_line)
                return self._get_node_text(fallback_node, file_content_bytes)
            return None

        best_candidate: Optional[Node] = None
        for node in candidate_nodes:
            node_start_line = node.start_point[0]
            node_end_line = node.end_point[0]
            if node_start_line <= min_target_line_0_indexed and node_end_line >= max_target_line_0_indexed:
                if best_candidate is None or (node.end_byte - node.start_byte) < (best_candidate.end_byte - best_candidate.start_byte):
                    best_candidate = node
        
        if best_candidate:
            return self._get_node_text(best_candidate, file_content_bytes)

        return None 