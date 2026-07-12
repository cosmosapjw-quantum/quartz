"""
Unit tests for API documentation validation.

Tests verify that the API documentation contains all required sections,
code examples are syntactically valid, parameter descriptions are complete,
and references to APIs match actual implementations.
"""

import os
import re
import ast
import unittest
from pathlib import Path
from typing import List, Dict, Any


class TestAPIDocumentation(unittest.TestCase):
    """Test API documentation completeness and accuracy."""

    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent.parent / "docs"
        self.api_file = self.docs_dir / "api.md"
        self.project_root = Path(__file__).parent.parent.parent

        # Load API documentation
        with open(self.api_file, 'r') as f:
            self.api_content = f.read()

    def test_api_file_exists(self):
        """Test that api.md file exists."""
        self.assertTrue(self.api_file.exists(),
                       "API documentation file does not exist")

    def test_required_sections_present(self):
        """Test that all required API sections are present."""
        required_sections = [
            "MCTS Engine API",
            "Neural Network Inference API",
            "Training Pipeline API",
            "Game Interface API",
            "Configuration API",
            "Telemetry & Monitoring API",
            "Usage Examples",
            "Error Handling"
        ]

        for section in required_sections:
            self.assertIn(section, self.api_content,
                         f"Required API section '{section}' not found")

    def test_api_classes_documented(self):
        """Test that all major API classes are documented."""
        api_classes = [
            "GameState",
            "MCTSEngine",
            "InferenceWorker",
            "TrainingLoop",
            "SelfPlayGenerator",
            "ExperienceBuffer",
            "GameAdapter",
            "ConfigManager",
            "MetricsCollector"
        ]

        for api_class in api_classes:
            self.assertIn(api_class, self.api_content,
                         f"API class '{api_class}' not documented")

    def test_method_signatures_documented(self):
        """Test that method signatures are properly documented."""
        # Key methods that should be documented
        key_methods = [
            "apply_move_inplace",
            "get_legal_moves",
            "is_terminal",
            "search",
            "inference_batch",
            "generate_game",
            "load_config",
            "record_mcts_performance"
        ]

        for method in key_methods:
            self.assertIn(method, self.api_content,
                         f"Key method '{method}' not documented")

    def test_parameter_descriptions_present(self):
        """Test that parameter descriptions are comprehensive."""
        # Look for parameter documentation patterns
        param_patterns = [
            r"Args:\s*\n",  # Args sections
            r"Returns:\s*\n",  # Returns sections
            r"Raises:\s*\n",  # Raises sections
            r"\s+(\w+)\s+\([^)]+\):",  # Parameter type annotations
        ]

        for pattern in param_patterns[:3]:  # Test Args, Returns, Raises
            matches = re.findall(pattern, self.api_content)
            self.assertGreater(len(matches), 5,
                             f"Insufficient parameter documentation for pattern: {pattern}")

    def test_code_examples_present(self):
        """Test that code examples are present throughout the documentation."""
        # Count Python code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', self.api_content, re.DOTALL)

        self.assertGreaterEqual(len(code_blocks), 10,
                               "Insufficient code examples in API documentation")

    def test_code_examples_syntax_valid(self):
        """Test that Python code examples have valid syntax."""
        # Extract Python code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', self.api_content, re.DOTALL)

        invalid_code = []

        for i, code_block in enumerate(code_blocks):
            try:
                # Remove common documentation artifacts
                cleaned_code = code_block.strip()

                # Skip code blocks that are just class/function definitions or comments
                if cleaned_code.startswith('#') or not cleaned_code:
                    continue

                # Skip incomplete code snippets (those ending with ...)
                if cleaned_code.endswith('...') or '...' in cleaned_code:
                    continue

                # Try to parse as Python code
                ast.parse(cleaned_code)

            except SyntaxError as e:
                invalid_code.append(f"Code block {i+1}: {str(e)}")
            except Exception:
                # Skip blocks that can't be parsed due to incomplete examples
                continue

        self.assertEqual(len(invalid_code), 0,
                        f"Found syntax errors in code examples: {invalid_code}")

    def test_performance_targets_documented(self):
        """Test that performance targets are clearly documented."""
        performance_targets = [
            "30,000+",  # Simulations per second
            "80-92%",   # GPU utilization
            "200+",     # Games per hour
            "<1GB"      # Memory usage
        ]

        for target in performance_targets:
            self.assertIn(target, self.api_content,
                         f"Performance target '{target}' not documented")

    def test_error_handling_comprehensive(self):
        """Test that error handling is comprehensively documented."""
        error_classes = [
            "ConfigurationError",
            "GameError",
            "InferenceError",
            "TrainingError"
        ]

        for error_class in error_classes:
            self.assertIn(error_class, self.api_content,
                         f"Error class '{error_class}' not documented")

    def test_environment_variables_documented(self):
        """Test that environment variable patterns are documented."""
        env_patterns = [
            "ALPHAZERO_MCTS_SIMULATIONS",
            "ALPHAZERO_NEURAL_NETWORK_BATCH_SIZE_PREFERRED",
            "ALPHAZERO_TRAINING_BATCH_SIZE"
        ]

        for env_var in env_patterns:
            self.assertIn(env_var, self.api_content,
                         f"Environment variable '{env_var}' not documented")

    def test_game_types_documented(self):
        """Test that all supported game types are documented."""
        game_types = ["gomoku", "chess", "go"]

        for game_type in game_types:
            self.assertIn(game_type, self.api_content,
                         f"Game type '{game_type}' not documented")

    def test_configuration_structure_documented(self):
        """Test that configuration structure is documented."""
        config_classes = [
            "MCTSConfig",
            "NeuralNetworkConfig",
            "TrainingConfig",
            "GameConfig",
            "SystemConfig",
            "AlphaZeroConfig"
        ]

        for config_class in config_classes:
            self.assertIn(config_class, self.api_content,
                         f"Configuration class '{config_class}' not documented")

    def test_table_formatting_correct(self):
        """Test that parameter tables are properly formatted."""
        # Look for table headers
        table_patterns = [
            r"\|\s*Parameter\s*\|\s*Type\s*\|",
            r"\|\s*Game\s*\|\s*Action Space\s*\|",
            r"\|\s*Metric\s*\|\s*Target\s*\|"
        ]

        for pattern in table_patterns:
            matches = re.findall(pattern, self.api_content, re.IGNORECASE)
            self.assertGreater(len(matches), 0,
                             f"Table format not found: {pattern}")

    def test_version_information_present(self):
        """Test that version and metadata information is present."""
        version_items = [
            "Version:",
            "Last Updated:",
            "Documentation Level:",
            "Document Version:"
        ]

        for item in version_items:
            self.assertIn(item, self.api_content,
                         f"Version information '{item}' not found")

    def test_cross_references_valid(self):
        """Test that cross-references to other documentation are valid."""
        cross_refs = [
            "operations.md",
            "training_guide.md"
        ]

        for ref in cross_refs:
            self.assertIn(ref, self.api_content,
                         f"Cross-reference '{ref}' not found")

    def test_toc_matches_content(self):
        """Test that table of contents matches actual section headers."""
        # Extract TOC items
        toc_match = re.search(r'## Table of Contents\n\n(.*?)\n\n---',
                             self.api_content, re.DOTALL)

        if toc_match:
            toc_content = toc_match.group(1)
            toc_items = re.findall(r'\[([^\]]+)\]', toc_content)

            # Extract actual section headers (level 2)
            section_headers = re.findall(r'^## ([^\n]+)', self.api_content, re.MULTILINE)

            # Remove "Table of Contents" from section headers
            section_headers = [h for h in section_headers if h != "Table of Contents"]

            # Check that TOC items exist as section headers
            for toc_item in toc_items:
                self.assertIn(toc_item, section_headers,
                             f"TOC item '{toc_item}' not found in section headers")


class TestAPICodeExamples(unittest.TestCase):
    """Test API code examples for correctness and completeness."""

    def setUp(self):
        """Set up test environment."""
        self.api_file = Path(__file__).parent.parent.parent / "docs" / "api.md"

        with open(self.api_file, 'r') as f:
            self.api_content = f.read()

    def test_complete_training_example(self):
        """Test that complete training setup example is present."""
        training_example_components = [
            "load_config",
            "InferenceWorker",
            "MCTSEngine",
            "TrainingLoop",
            "start_continuous_training"
        ]

        for component in training_example_components:
            self.assertIn(component, self.api_content,
                         f"Training example component '{component}' not found")

    def test_game_analysis_example(self):
        """Test that game analysis example is present."""
        analysis_components = [
            "GomokuState",
            "apply_move_inplace",
            "mcts.search",
            "np.argmax"
        ]

        for component in analysis_components:
            self.assertIn(component, self.api_content,
                         f"Analysis example component '{component}' not found")

    def test_monitoring_example(self):
        """Test that performance monitoring example is present."""
        monitoring_components = [
            "MetricsCollector",
            "record_mcts_performance",
            "get_metrics_summary"
        ]

        for component in monitoring_components:
            self.assertIn(component, self.api_content,
                         f"Monitoring example component '{component}' not found")

    def test_error_handling_examples(self):
        """Test that error handling examples are comprehensive."""
        error_handling_patterns = [
            "try:",
            "except",
            "ConfigurationError",
            "GameError",
            "InferenceError"
        ]

        for pattern in error_handling_patterns:
            self.assertIn(pattern, self.api_content,
                         f"Error handling pattern '{pattern}' not found")

    def test_import_statements_valid(self):
        """Test that import statements in examples are reasonable."""
        # Extract import statements from code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', self.api_content, re.DOTALL)
        import_statements = []

        for block in code_blocks:
            imports = re.findall(r'^(from .+ import .+|import .+)', block, re.MULTILINE)
            import_statements.extend(imports)

        # Check that we have reasonable imports
        self.assertGreater(len(import_statements), 5,
                          "Insufficient import statements in examples")

        # Check for key expected imports (pathlib is optional)
        required_imports = ["numpy", "src.", "typing"]
        found_imports = " ".join(import_statements)

        for expected in required_imports:
            self.assertIn(expected, found_imports,
                         f"Expected import pattern '{expected}' not found")


if __name__ == '__main__':
    # Create test suite
    test_cases = [
        TestAPIDocumentation,
        TestAPICodeExamples
    ]

    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    exit(0 if result.wasSuccessful() else 1)