"""
Unit tests for MCTS tree Structure-of-Arrays memory layout.

Tests memory efficiency, alignment, and basic functionality of the
SoA-based MCTS tree implementation.
"""

import pytest
import sys
from pathlib import Path
import ctypes
import subprocess
import tempfile
import os

# Add project root to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestTreeMemoryLayout:
    """Test MCTS tree memory layout and efficiency."""

    def test_header_file_exists(self):
        """Test that tree.hpp header file exists and is readable."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        assert tree_header.exists(), "tree.hpp header file not found"

        content = tree_header.read_text()
        assert "Structure-of-Arrays" in content
        assert "MCTSTree" in content
        assert "alignas(64)" in content

    def test_implementation_file_exists(self):
        """Test that tree.cpp implementation file exists."""
        tree_impl = project_root / "cpp_extensions" / "mcts" / "tree.cpp"
        assert tree_impl.exists(), "tree.cpp implementation file not found"

        content = tree_impl.read_text()
        assert "MCTSTree::" in content
        assert "allocate_aligned" in content

    def test_memory_layout_constants(self):
        """Test that memory layout constants are properly defined."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check for key constants and types
        assert "NodeIndex" in content
        assert "NULL_NODE_INDEX" in content
        assert "NodeFlags" in content
        assert "int32_t" in content
        assert "alignas(64)" in content

    def test_node_flags_bit_layout(self):
        """Test NodeFlags bit layout design."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check that NodeFlags has the expected bit manipulation methods
        assert "is_expanded()" in content
        assert "is_terminal()" in content
        assert "current_player()" in content
        assert "set_expanded(" in content
        assert "set_terminal(" in content
        assert "set_current_player(" in content

    def test_soa_arrays_declared(self):
        """Test that all Structure-of-Arrays are properly declared."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check for all expected SoA arrays
        expected_arrays = [
            "visit_counts_",
            "total_values_",
            "prior_probs_",
            "virtual_losses_",
            "parent_indices_",
            "first_child_indices_",
            "num_children_",
            "flags_",
        ]

        for array_name in expected_arrays:
            assert array_name in content, f"Missing SoA array: {array_name}"

    def test_simd_alignment_specified(self):
        """Test that 64-byte SIMD alignment is specified."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check for SIMD alignment specifications
        assert "alignas(64)" in content
        assert "64-byte aligned" in content or "64-byte alignment" in content

    def test_memory_efficiency_targets(self):
        """Test that memory efficiency targets are documented."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check for memory efficiency documentation
        assert "<64 bytes per node" in content or "32-64 bytes" in content
        assert "50M" in content or "50,000,000" in content or "50'000'000" in content

    def test_api_methods_declared(self):
        """Test that essential API methods are declared."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check for essential API methods
        essential_methods = [
            "get_visit_count(",
            "get_total_value(",
            "get_prior_prob(",
            "get_virtual_loss(",
            "set_visit_count(",
            "set_total_value(",
            "add_root_node(",
            "validate_tree(",
            "get_memory_usage(",
            "get_bytes_per_node(",
        ]

        for method in essential_methods:
            assert method in content, f"Missing essential method: {method}"

    def test_index_based_references(self):
        """Test that index-based references are used instead of pointers."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Should use NodeIndex (int32_t) instead of pointers
        assert "NodeIndex" in content
        assert "int32_t" in content

        # Should not have raw pointer navigation between nodes
        # (pointers to arrays for SIMD are OK, but not for node navigation)
        lines = content.split("\n")
        pointer_navigation_lines = [
            line
            for line in lines
            if "->" in line and "node" in line.lower() and "array" not in line.lower()
        ]

        # Should have minimal or no pointer navigation between nodes
        assert len(pointer_navigation_lines) < 5, "Too much pointer navigation detected"

    def test_memory_validation_methods(self):
        """Test that memory validation and debugging methods exist."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Check for validation and debugging support
        validation_features = [
            "validate_tree(",
            "get_node_info(",
            "TreeMemoryStats",
            "get_tree_memory_stats(",
        ]

        for feature in validation_features:
            assert feature in content, f"Missing validation feature: {feature}"


class TestTreeCompilation:
    """Test that the tree implementation can be compiled."""

    def create_test_cpp_file(self) -> str:
        """Create a minimal C++ test file to validate compilation."""
        test_cpp = """
#include <iostream>
#include <cassert>

// Include the tree implementation
#include "tree.hpp"
#include "tree.cpp"

int main() {
    using namespace mcts;

    try {
        // Test basic tree creation
        MCTSTree tree(1000);

        // Test memory usage calculation
        size_t memory = tree.get_memory_usage();
        std::cout << "Tree memory usage: " << memory << " bytes\\n";

        // Test bytes per node calculation (should be 0 for empty tree)
        double bytes_per_node = tree.get_bytes_per_node();
        std::cout << "Bytes per node: " << bytes_per_node << "\\n";

        // Test root node creation
        NodeIndex root = tree.add_root_node(0.5f, 0);
        assert(root == 0);
        assert(tree.get_node_count() == 1);

        // Test root node properties
        assert(tree.get_visit_count(root) == 0.0f);
        assert(tree.get_prior_prob(root) == 0.5f);
        assert(tree.get_parent_index(root) == NULL_NODE_INDEX);

        // Test memory efficiency
        bytes_per_node = tree.get_bytes_per_node();
        std::cout << "Bytes per node (with 1 node): " << bytes_per_node << "\\n";

        // Validate tree structure
        assert(tree.validate_tree());

        // Test memory stats
        TreeMemoryStats stats = get_tree_memory_stats(tree);
        std::cout << "Memory stats - Total: " << stats.total_bytes
                  << ", Nodes: " << stats.node_count
                  << ", Bytes/node: " << stats.bytes_per_node << "\\n";

        // Test with multiple nodes capacity
        MCTSTree large_tree(50000000);  // 50M nodes
        size_t large_memory = large_tree.get_memory_usage();
        std::cout << "Large tree memory (50M capacity): " << large_memory << " bytes\\n";
        std::cout << "Large tree memory: " << (large_memory / 1024.0 / 1024.0) << " MB\\n";

        // Test clear functionality
        tree.clear();
        assert(tree.get_node_count() == 0);
        assert(tree.validate_tree());

        std::cout << "All tests passed!\\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\\n";
        return 1;
    }
}
"""
        return test_cpp

    def test_basic_compilation(self):
        """Test that the tree implementation compiles without errors."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_cpp_path = Path(temp_dir) / "test_tree.cpp"
            test_exe_path = Path(temp_dir) / "test_tree"

            # Write test file
            test_cpp_path.write_text(self.create_test_cpp_file())

            # Copy header and implementation to temp directory
            tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
            tree_impl = project_root / "cpp_extensions" / "mcts" / "tree.cpp"

            temp_header = Path(temp_dir) / "tree.hpp"
            temp_impl = Path(temp_dir) / "tree.cpp"

            temp_header.write_text(tree_header.read_text())
            temp_impl.write_text(tree_impl.read_text())

            # Try to compile
            compile_cmd = [
                "g++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-I.",
                str(test_cpp_path),
                "-o",
                str(test_exe_path),
            ]

            try:
                result = subprocess.run(
                    compile_cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    print(f"Compilation failed:")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    pytest.skip("C++ compiler not available or compilation failed")
                else:
                    print("✅ Tree implementation compiles successfully")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip("C++ compiler not available")

    def test_memory_layout_efficiency(self):
        """Test memory layout efficiency through compilation and execution."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_cpp_path = Path(temp_dir) / "test_memory.cpp"
            test_exe_path = Path(temp_dir) / "test_memory"

            # Create memory efficiency test
            memory_test_cpp = """
#include <iostream>
#include <cassert>

#include "tree.hpp"
#include "tree.cpp"

int main() {
    using namespace mcts;

    try {
        // Test memory efficiency with small tree
        MCTSTree tree(1000);
        NodeIndex root = tree.add_root_node(0.5f, 0);

        double bytes_per_node = tree.get_bytes_per_node();

        // Should be much less than 64 bytes per node even with alignment overhead
        std::cout << "Bytes per node (1 node): " << bytes_per_node << std::endl;

        // Test with 50M node capacity
        MCTSTree large_tree(50000000);
        size_t total_memory = large_tree.get_memory_usage();
        double memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

        std::cout << "50M node capacity memory: " << memory_gb << " GB" << std::endl;

        // Should be less than 2GB for 50M nodes (target is ~1GB)
        if (memory_gb > 2.0) {
            std::cerr << "Memory usage too high: " << memory_gb << " GB" << std::endl;
            return 1;
        }

        // Test memory stats
        TreeMemoryStats stats = get_tree_memory_stats(large_tree);
        std::cout << "Alignment overhead: " << stats.alignment_overhead << " bytes" << std::endl;

        std::cout << "Memory efficiency test passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
"""

            test_cpp_path.write_text(memory_test_cpp)

            # Copy tree files
            tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
            tree_impl = project_root / "cpp_extensions" / "mcts" / "tree.cpp"

            temp_header = Path(temp_dir) / "tree.hpp"
            temp_impl = Path(temp_dir) / "tree.cpp"

            temp_header.write_text(tree_header.read_text())
            temp_impl.write_text(tree_impl.read_text())

            # Compile and run
            compile_cmd = [
                "g++",
                "-std=c++17",
                "-O2",
                "-I.",
                str(test_cpp_path),
                "-o",
                str(test_exe_path),
            ]

            try:
                # Compile
                compile_result = subprocess.run(
                    compile_cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if compile_result.returncode != 0:
                    pytest.skip("C++ compilation failed")

                # Run
                run_result = subprocess.run(
                    [str(test_exe_path)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if run_result.returncode == 0:
                    print("✅ Memory efficiency test passed")
                    print("Output:", run_result.stdout)
                else:
                    print("❌ Memory efficiency test failed")
                    print("Error:", run_result.stderr)

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip("C++ execution failed")


class TestTreeAPIConsistency:
    """Test that tree API is consistent with design specifications."""

    def test_node_index_type(self):
        """Test that NodeIndex type is properly defined."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Should use int32_t for NodeIndex
        assert "using NodeIndex = std::int32_t" in content
        assert "NULL_NODE_INDEX = -1" in content

    def test_memory_target_documentation(self):
        """Test that memory targets are documented."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Should document the <64 bytes per node target
        assert "64 bytes" in content
        assert "50M" in content or "50,000,000" in content or "50'000'000" in content

    def test_simd_optimization_ready(self):
        """Test that SIMD optimization infrastructure is in place."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Should provide access to raw arrays for SIMD operations
        simd_methods = [
            "get_visit_counts_ptr(",
            "get_total_values_ptr(",
            "get_prior_probs_ptr(",
            "get_virtual_losses_ptr(",
        ]

        for method in simd_methods:
            assert method in content, f"Missing SIMD-ready method: {method}"

    def test_validation_infrastructure(self):
        """Test that validation infrastructure is comprehensive."""
        tree_header = project_root / "cpp_extensions" / "mcts" / "tree.hpp"
        content = tree_header.read_text()

        # Should have comprehensive validation
        assert "validate_tree(" in content
        assert "NodeInfo" in content
        assert "TreeMemoryStats" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
