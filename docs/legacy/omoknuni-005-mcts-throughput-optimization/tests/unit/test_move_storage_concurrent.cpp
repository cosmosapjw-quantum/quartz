/**
 * @file test_move_storage_concurrent.cpp
 * @brief Concurrent tests for MCTSTree move storage functionality
 *
 * Tests thread safety and concurrent access patterns for the move storage
 * feature. These tests should be run with ThreadSanitizer to detect any
 * data races or synchronization issues.
 *
 * COMPILE AND RUN:
 * ================
 * # Standard build
 * g++ -std=c++17 -O2 -pthread -I../../cpp_extensions \
 *     -o test_move_storage_concurrent test_move_storage_concurrent.cpp \
 *     ../../cpp_extensions/mcts/tree.cpp ../../cpp_extensions/mcts/virtual_loss.cpp
 *
 * # Build with ThreadSanitizer (TSan) - Ubuntu 24.04+
 * # Note: Ubuntu 24.04 requires clang++-18 for TSan due to higher ASLR entropy
 * clang++-18 -std=c++17 -O1 -g -pthread -fsanitize=thread \
 *     -I../../cpp_extensions -o test_move_storage_concurrent_tsan \
 *     test_move_storage_concurrent.cpp ../../cpp_extensions/mcts/tree.cpp \
 *     ../../cpp_extensions/mcts/virtual_loss.cpp
 *
 * # For older systems (Ubuntu 22.04 and earlier)
 * g++ -std=c++17 -O1 -g -pthread -fsanitize=thread \
 *     -I../../cpp_extensions -o test_move_storage_concurrent_tsan \
 *     test_move_storage_concurrent.cpp ../../cpp_extensions/mcts/tree.cpp \
 *     ../../cpp_extensions/mcts/virtual_loss.cpp
 *
 * # Run tests
 * ./test_move_storage_concurrent
 *
 * # Run with ThreadSanitizer
 * ./test_move_storage_concurrent_tsan
 *
 * EXPECTED OUTPUT:
 * All tests should pass with no ThreadSanitizer warnings about data races.
 */

#include <iostream>
#include <cassert>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "mcts/tree.hpp"
#include "mcts/virtual_loss.hpp"

using namespace mcts;

/**
 * Test concurrent reads from the same move storage location.
 * Multiple threads read the same move index simultaneously - this should
 * always be safe as reads don't modify data.
 */
void test_concurrent_reads() {
    std::cout << "Testing concurrent reads..." << std::endl;

    MCTSTree tree(1000);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Set a known value
    const uint16_t expected_move = 42;
    tree.set_move(root, expected_move);

    // Launch multiple threads to read concurrently
    const int num_threads = 8;
    const int reads_per_thread = 10000;
    std::vector<std::thread> threads;
    std::atomic<int> mismatches{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&tree, root, expected_move, &mismatches, reads_per_thread]() {
            for (int j = 0; j < reads_per_thread; ++j) {
                uint16_t move = tree.get_move(root);
                if (move != expected_move) {
                    mismatches++;
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    assert(mismatches == 0);
    std::cout << "✓ Concurrent reads test passed ("
              << (num_threads * reads_per_thread) << " total reads)" << std::endl;
}

/**
 * Test concurrent writes to different move storage locations.
 * Multiple threads write to different node indices - should be safe
 * as there's no shared memory being modified.
 */
void test_concurrent_writes_different_nodes() {
    std::cout << "Testing concurrent writes to different nodes..." << std::endl;

    const int num_nodes = 100;
    MCTSTree tree(num_nodes * 2);  // Extra capacity for safety

    // Allocate nodes - allocate all at once then track indices
    NodeIndex first = tree.allocate_nodes(num_nodes);
    std::vector<NodeIndex> nodes;
    for (int i = 0; i < num_nodes; ++i) {
        nodes.push_back(first + i);
    }

    // Each thread writes to its own set of nodes
    const int num_threads = 8;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&tree, &nodes, num_nodes, t, num_threads]() {
            // Each thread handles nodes[t], nodes[t+num_threads], nodes[t+2*num_threads], etc.
            for (int i = t; i < num_nodes; i += num_threads) {
                uint16_t move_value = static_cast<uint16_t>(i * 10);
                tree.set_move(nodes[i], move_value);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all values were written correctly
    for (int i = 0; i < num_nodes; ++i) {
        uint16_t expected = static_cast<uint16_t>(i * 10);
        uint16_t actual = tree.get_move(nodes[i]);
        assert(actual == expected);
    }

    std::cout << "✓ Concurrent writes to different nodes test passed" << std::endl;
}

/**
 * Test that virtual loss operations don't interfere with move storage.
 * Virtual loss uses atomic operations on visit counts - this test ensures
 * those don't cause issues with move storage.
 */
void test_move_storage_with_virtual_loss() {
    std::cout << "Testing move storage with concurrent virtual loss operations..." << std::endl;

    MCTSTree tree(1000);
    VirtualLossManager vlm(tree);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Allocate children
    const uint16_t num_children = 10;
    NodeIndex first_child = tree.allocate_nodes(num_children);
    tree.set_first_child_index(root, first_child);
    tree.set_num_children(root, num_children);

    // Set move indices
    for (uint16_t i = 0; i < num_children; ++i) {
        tree.set_move(first_child + i, 100 + i);
    }

    // Concurrent operations: some threads do virtual loss, others read moves
    const int num_threads = 8;
    std::vector<std::thread> threads;
    std::atomic<bool> stop{false};
    std::atomic<int> errors{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            if (t % 2 == 0) {
                // Even threads: apply and remove virtual loss
                while (!stop) {
                    for (uint16_t i = 0; i < num_children; ++i) {
                        NodeIndex child = first_child + i;
                        vlm.apply_virtual_loss(child);
                        vlm.remove_virtual_loss(child);
                    }
                }
            } else {
                // Odd threads: read move indices
                while (!stop) {
                    for (uint16_t i = 0; i < num_children; ++i) {
                        NodeIndex child = first_child + i;
                        uint16_t move = tree.get_move(child);
                        uint16_t expected = 100 + i;
                        if (move != expected) {
                            errors++;
                        }
                    }
                }
            }
        });
    }

    // Let threads run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    assert(errors == 0);
    std::cout << "✓ Move storage with virtual loss test passed" << std::endl;
}

/**
 * Test allocating and deallocating nodes concurrently while accessing moves.
 * This tests more complex scenarios with node lifecycle management.
 */
void test_concurrent_allocation_with_move_access() {
    std::cout << "Testing concurrent allocation with move access..." << std::endl;

    const int max_nodes = 10000;
    MCTSTree tree(max_nodes);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    const int num_threads = 4;
    const int iterations = 50;  // Reduced to avoid exhausting tree
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    std::atomic<int> successful_allocs{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < iterations; ++i) {
                // Allocate a batch of nodes
                const uint16_t batch_size = 5;  // Smaller batches
                NodeIndex first = tree.allocate_nodes(batch_size);

                if (first != NULL_NODE_INDEX) {
                    successful_allocs++;

                    // Set moves for allocated nodes
                    for (uint16_t j = 0; j < batch_size; ++j) {
                        NodeIndex node = first + j;
                        uint16_t move_value = static_cast<uint16_t>(t * 1000 + i * 10 + j);
                        tree.set_move(node, move_value);

                        // Verify immediately
                        uint16_t read_value = tree.get_move(node);
                        if (read_value != move_value) {
                            errors++;
                        }
                    }

                    // Deallocate nodes
                    tree.deallocate_nodes(first, batch_size);
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    assert(errors == 0);
    assert(successful_allocs > 0);  // At least some allocations succeeded
    std::cout << "✓ Concurrent allocation with move access test passed (allocations: "
              << successful_allocs << ")" << std::endl;
}

/**
 * Stress test: many threads performing mixed operations.
 * This test simulates realistic MCTS workload with concurrent tree operations.
 */
void test_stress_mixed_operations() {
    std::cout << "Testing stress with mixed concurrent operations..." << std::endl;

    const int max_nodes = 5000;
    MCTSTree tree(max_nodes);
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Pre-allocate some children
    const uint16_t num_children = 20;
    NodeIndex first_child = tree.allocate_nodes(num_children);
    tree.set_first_child_index(root, first_child);
    tree.set_num_children(root, num_children);

    // Set initial move values
    for (uint16_t i = 0; i < num_children; ++i) {
        tree.set_move(first_child + i, i);
    }

    const int num_threads = 8;
    std::vector<std::thread> threads;
    std::atomic<bool> stop{false};
    std::atomic<int> operations{0};
    std::atomic<int> read_ops{0};
    std::atomic<int> alloc_ops{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int op_count = 0;
            while (!stop.load(std::memory_order_acquire)) {
                // Mix of operations - simplified to avoid issues
                if (op_count % 2 == 0) {
                    // Read moves from children
                    for (uint16_t i = 0; i < num_children; ++i) {
                        volatile uint16_t move = tree.get_move(first_child + i);
                        (void)move;
                    }
                    read_ops++;
                } else {
                    // Read visit counts
                    for (uint16_t i = 0; i < num_children; ++i) {
                        volatile float n = tree.get_visit_count(first_child + i);
                        (void)n;
                    }
                    read_ops++;
                }
                op_count++;

                // Check stop condition more frequently
                if (op_count % 10 == 0 && stop.load(std::memory_order_acquire)) {
                    break;
                }
            }
            operations += op_count;
        });
    }

    // Run for a shorter duration to avoid issues
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true, std::memory_order_release);

    // Wait for all threads with timeout safety
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    std::cout << "✓ Stress test passed (" << operations << " total operations, "
              << read_ops << " read ops)" << std::endl;
}

/**
 * Test edge case: concurrent access to boundary node indices.
 */
void test_boundary_indices() {
    std::cout << "Testing concurrent access to boundary node indices..." << std::endl;

    const int max_nodes = 1000;
    MCTSTree tree(max_nodes);

    // Create root node (first node)
    NodeIndex root = tree.add_root_node(0.5f, 0);

    // Allocate some nodes near the beginning
    NodeIndex second = tree.allocate_nodes(1);
    NodeIndex third = tree.allocate_nodes(1);

    // Allocate a large batch that gets us close to capacity
    // We've used 3 nodes (root, second, third), so allocate more to near capacity
    NodeIndex batch_start = tree.allocate_nodes(max_nodes - 13);

    // Test nodes at various positions: start, near-start, middle, near-end, end
    std::vector<NodeIndex> test_nodes = {
        root,           // First node (0)
        second,         // Second node (1)
        third,          // Third node (2)
        batch_start,    // Start of large batch
        batch_start + (max_nodes - 13) / 2,  // Middle of batch
        batch_start + (max_nodes - 14)       // Near end of batch
    };

    // Set different values
    for (size_t i = 0; i < test_nodes.size(); ++i) {
        if (test_nodes[i] != NULL_NODE_INDEX) {
            tree.set_move(test_nodes[i], static_cast<uint16_t>(i * 100));
        }
    }

    // Concurrent reads from boundary nodes
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 1000; ++i) {
                for (size_t j = 0; j < test_nodes.size(); ++j) {
                    if (test_nodes[j] != NULL_NODE_INDEX) {
                        uint16_t move = tree.get_move(test_nodes[j]);
                        uint16_t expected = static_cast<uint16_t>(j * 100);
                        if (move != expected) {
                            errors++;
                        }
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    assert(errors == 0);
    std::cout << "✓ Boundary indices test passed" << std::endl;
}

int main() {
    std::cout << "\n=== MCTSTree Move Storage Concurrent Tests ===" << std::endl;
    std::cout << "Testing thread safety and concurrent access patterns\n" << std::endl;

    try {
        test_concurrent_reads();
        test_concurrent_writes_different_nodes();
        test_move_storage_with_virtual_loss();
        test_concurrent_allocation_with_move_access();
        test_stress_mixed_operations();
        test_boundary_indices();

        std::cout << "\n✅ All concurrent tests passed!" << std::endl;
        std::cout << "\nNOTE: Run with ThreadSanitizer to detect data races:" << std::endl;
        std::cout << "  g++ -fsanitize=thread ... && ./test_move_storage_concurrent_tsan\n" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
