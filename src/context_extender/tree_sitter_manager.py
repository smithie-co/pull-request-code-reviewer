"""Manages Tree-sitter parsers and language grammars."""
from pathlib import Path
from typing import Optional, Dict

from tree_sitter import Language, Parser

# Path to the directory where compiled grammar libraries are stored.
# Users will need to ensure that a 'languages.so' (or .dll on Windows)
# file, containing all required grammars, is placed here.
# Alternatively, individual .so/.dll files per language could be managed.
GRAMMARS_DIR = Path(__file__).parent / "grammars"
COMPILED_GRAMMAR_FILE = GRAMMARS_DIR / "languages.so" # TODO: Make this configurable or discoverable

# Placeholder for language name to grammar mapping if we load them individually
# e.g., "python": "tree_sitter_python" (if using named .so files)
LANGUAGE_GRAMMAR_MAP: Dict[str, str] = {
    "python": "python",
    "hcl": "hcl", 
    "csharp": "c_sharp",  # Common symbol for tree-sitter-c-sharp
    "javascript": "javascript",
    "typescript": "typescript",
    "sql": "sql", # For generic SQL, from tree-sitter-sql
    "partiql": "partiql", # For AWS PartiQL, from tree-sitter-partiql
    "xml": "xml",
    # Ensure the actual symbol name used during compilation matches these values.
    # For example, tree-sitter-typescript provides 'typescript' and 'tsx'.
}

class TreeSitterManager:
    """
    Manages the loading of Tree-sitter language grammars and provides parsers.
    """
    def __init__(self):
        self.languages: Dict[str, Language] = {}
        self._load_grammars()

    def _load_grammars(self) -> None:
        """
        Loads all specified language grammars from the compiled library.
        Assumes a single shared library file (e.g., languages.so) contains all grammars.
        """
        if not COMPILED_GRAMMAR_FILE.exists():
            # In a real scenario, you might trigger a build process here or log a critical error.
            # For now, we'll raise an error.
            # Consider automatically building if source grammar dirs are provided.
            print(f"Grammar library not found: {COMPILED_GRAMMAR_FILE}")
            print("Please ensure grammars are compiled and the library is in the grammars/ directory.")
            # raise FileNotFoundError(f"Grammar library not found: {COMPILED_GRAMMAR_FILE}") # Or handle gracefully
            return # Allow to proceed without grammars for now for initial setup

        for lang_name, grammar_symbol_name in LANGUAGE_GRAMMAR_MAP.items():
            try:
                self.languages[lang_name] = Language(str(COMPILED_GRAMMAR_FILE), grammar_symbol_name)
                print(f"Successfully loaded grammar for: {lang_name}")
            except Exception as e:
                print(f"Error loading grammar for {lang_name} ({grammar_symbol_name}) from {COMPILED_GRAMMAR_FILE}: {e}")
                # Potentially re-raise or log this as a critical failure for the language.

    def get_parser(self, language_name: str) -> Optional[Parser]:
        """
        Returns a Tree-sitter Parser configured for the specified language.

        Args:
            language_name: The name of the language (e.g., "python", "hcl").

        Returns:
            A Parser instance if the language is supported, None otherwise.
        """
        language = self.languages.get(language_name.lower())
        if language:
            parser = Parser()
            parser.set_language(language)
            return parser
        print(f"Language '{language_name}' not supported or grammar not loaded.")
        return None

    @staticmethod
    def build_grammars(output_path: str = str(COMPILED_GRAMMAR_FILE), grammar_dirs: Optional[Dict[str, str]] = None) -> None:
        """
        Compiles tree-sitter grammars from specified directories into a shared library.

        Args:
            output_path: The path to save the compiled shared library (e.g., "grammars/languages.so").
            grammar_dirs: A dictionary mapping language names to their source directory paths.
                          Example: {"python": "vendor/tree-sitter-python", "hcl": "vendor/tree-sitter-hcl"}
                          If None, tries to find common ones. (This part is complex and might need refinement)
        
        Note: Requires a C compiler to be available in the environment.
        """
        if grammar_dirs is None:
            # This is a simplistic attempt to locate grammars.
            # In a real project, these paths would be explicitly configured or managed as submodules.
            print("grammar_dirs not provided. Auto-detection is not yet robustly implemented.")
            print("Please provide paths to grammar source directories (e.g., cloned from GitHub).")
            # Example for manual setup:
            # grammar_dirs = {
            #     "python": "path/to/tree-sitter-python",
            #     "hcl": "path/to/tree-sitter-hcl"
            # }
            return

        # Ensure the output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        source_paths = []
        for lang, path_str in grammar_dirs.items():
            path = Path(path_str)
            if not path.is_dir():
                print(f"Grammar directory for {lang} not found: {path}")
                continue
            
            # Common locations for grammar definition
            parser_c = path / "src" / "parser.c"
            scanner_c = path / "src" / "scanner.c" # Or .cc for C++ scanners
            scanner_cc = path / "src" / "scanner.cc"

            if parser_c.exists():
                source_paths.append(str(path)) # tree-sitter build_library expects dir paths
            else:
                print(f"parser.c not found in {path / 'src'} for language {lang}")
        
        if not source_paths:
            print("No valid grammar source directories found to compile.")
            return

        try:
            print(f"Attempting to compile grammars from: {source_paths} into {output_path}")
            Language.build_library(
                output_path,
                source_paths # List of paths to grammar directories
            )
            print(f"Successfully compiled grammars to: {output_path}")
        except Exception as e:
            print(f"Error compiling grammars: {e}")
            print("Ensure a C compiler is installed and configured in your PATH.")
            print("And that the provided paths are correct tree-sitter grammar directories.")

# Example usage (for testing purposes, if run directly)
if __name__ == '__main__':
    # This is where you would specify the paths to your cloned grammar repositories
    # For example, if you cloned them into a 'vendor/' directory:
    grammar_sources = {
        "python": "vendor/tree-sitter-python",
        "hcl": "vendor/tree-sitter-hcl"
        # Add other languages and their paths here
    }
    
    # Create the grammars directory if it doesn't exist
    GRAMMARS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the grammars (this needs to be run once, or when grammars are updated)
    # TreeSitterManager.build_grammars(grammar_dirs=grammar_sources)

    # Now, instantiate the manager, which will try to load the compiled library
    manager = TreeSitterManager()
    
    if manager.languages:
        print("\nAvailable parsers:")
        for lang_name in manager.languages.keys():
            print(f"- {lang_name}")

        # Test getting a parser
        py_parser = manager.get_parser("python")
        if py_parser:
            print("\nPython parser obtained successfully.")
            code_snippet = "def hello():\n  print('Hello, world!')"
            tree = py_parser.parse(bytes(code_snippet, "utf8"))
            print(f"Parsed Python snippet. Root node: {tree.root_node.type}")
            print(tree.root_node.sexp())

        hcl_parser = manager.get_parser("hcl")
        if hcl_parser:
            print("\nTerraform (HCL) parser obtained successfully.")
            # Basic HCL, ensure your hcl grammar supports this
            hcl_snippet = 'resource "aws_instance" "example" { ami = "ami-0c55b31ad2535f600" }'
            tree = hcl_parser.parse(bytes(hcl_snippet, "utf8"))
            print(f"Parsed HCL snippet. Root node: {tree.root_node.type}")
            print(tree.root_node.sexp())

    else:
        print("\nNo grammars were loaded. Ensure 'languages.so' (or .dll) exists in the grammars/ directory.")
        print(f"Expected at: {COMPILED_GRAMMAR_FILE.resolve()}")
        print("You might need to run the build_grammars() function or compile them manually.")
        print("Example to build (ensure vendor/tree-sitter-python etc. exist):")
        print("# TreeSitterManager.build_grammars(grammar_dirs={'python': 'vendor/tree-sitter-python', 'hcl': 'vendor/tree-sitter-hcl'})") 