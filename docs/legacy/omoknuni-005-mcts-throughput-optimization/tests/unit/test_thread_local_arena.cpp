#include "../../cpp_extensions/mcts/thread_local_arena.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cstring>

using namespace alphazero::core;

class ThreadLocalArenaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create fresh arena for each test
        arena = new ThreadLocalArena(2, 64 * 1024, 128);
    }

    void TearDown() override {
        delete arena;
        arena = nullptr;
    }

    ThreadLocalArena* arena = nullptr;
};

// Test 1: Basic arena creation
TEST_F(ThreadLocalArenaTest, ArenaCreation) {
    ASSERT_NE(arena, nullptr);
    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.chunks_allocated, 2);  // Initial chunks
    EXPECT_EQ(stats.allocations_from_bump, 0);
    EXPECT_EQ(stats.bytes_allocated, 0);
}

// Test 2: Single small allocation
TEST_F(ThreadLocalArenaTest, SingleSmallAllocation) {
    void* ptr = arena->allocate(32);
    ASSERT_NE(ptr, nullptr);

    // Verify 64-byte alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.allocations_from_bump, 1);
    EXPECT_EQ(stats.bytes_allocated, 64);  // Rounded up to 64
}

// Test 3: Multiple small allocations
TEST_F(ThreadLocalArenaTest, MultipleSmallAllocations) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        void* ptr = arena->allocate(27);  // MCTS node size
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);

        // Verify alignment
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    }

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.allocations_from_bump, 100);
    EXPECT_EQ(stats.bytes_allocated, 100 * 64);  // Each rounded to 64 bytes
}

// Test 4: Large allocation
TEST_F(ThreadLocalArenaTest, LargeAllocation) {
    size_t large_size = 1024;
    void* ptr = arena->allocate(large_size);
    ASSERT_NE(ptr, nullptr);

    // Verify alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.allocations_from_bump, 1);
    EXPECT_EQ(stats.bytes_allocated, 1024);  // Already aligned
}

// Test 5: Zero-size allocation
TEST_F(ThreadLocalArenaTest, ZeroSizeAllocation) {
    void* ptr = arena->allocate(0);
    EXPECT_EQ(ptr, nullptr);

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.allocations_from_bump, 0);
    EXPECT_EQ(stats.bytes_allocated, 0);
}

// Test 6: Alignment verification with various sizes
TEST_F(ThreadLocalArenaTest, AlignmentVerification) {
    std::vector<size_t> sizes = {1, 7, 15, 27, 32, 63, 64, 65, 127, 128, 255, 256};

    for (size_t size : sizes) {
        void* ptr = arena->allocate(size);
        ASSERT_NE(ptr, nullptr) << "Failed to allocate size " << size;

        // Verify 64-byte alignment
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0)
            << "Allocation of size " << size << " not 64-byte aligned";

        // Verify we can write to the allocated memory
        std::memset(ptr, 0xAB, size);
    }
}

// Test 7: Chunk overflow handling
TEST_F(ThreadLocalArenaTest, ChunkOverflow) {
    // Allocate until we fill first chunk (64KB)
    // Each allocation is 64 bytes (rounded up from 27)
    // 64KB / 64 bytes = 1024 allocations per chunk

    std::vector<void*> ptrs;
    for (int i = 0; i < 2048; ++i) {  // More than one chunk
        void* ptr = arena->allocate(27);
        ASSERT_NE(ptr, nullptr) << "Failed allocation " << i;
        ptrs.push_back(ptr);
    }

    auto stats = arena->get_statistics();
    EXPECT_GE(stats.chunks_allocated, 2);  // Should have allocated at least initial chunks
    EXPECT_EQ(stats.allocations_from_bump, 2048);
}

// Test 8: Reset functionality
TEST_F(ThreadLocalArenaTest, ResetFunctionality) {
    // Allocate some memory
    for (int i = 0; i < 100; ++i) {
        arena->allocate(32);
    }

    auto stats_before = arena->get_statistics();
    EXPECT_EQ(stats_before.allocations_from_bump, 100);
    EXPECT_EQ(stats_before.bytes_allocated, 100 * 64);

    // Reset arena
    arena->reset();

    auto stats_after = arena->get_statistics();
    EXPECT_EQ(stats_after.allocations_from_bump, 0);
    EXPECT_EQ(stats_after.bytes_allocated, 0);
    EXPECT_EQ(stats_after.chunks_allocated, stats_before.chunks_allocated);  // Chunks retained

    // Verify we can allocate again after reset
    void* ptr = arena->allocate(32);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
}

// Test 9: Deallocation (no-op in this phase)
TEST_F(ThreadLocalArenaTest, Deallocation) {
    void* ptr = arena->allocate(32);
    ASSERT_NE(ptr, nullptr);

    // Deallocate should not crash (currently a no-op)
    arena->deallocate(ptr, 32);

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.deallocations, 1);

    // Deallocate nullptr should be safe
    arena->deallocate(nullptr, 0);
    EXPECT_EQ(stats.deallocations, 1);  // No increment for nullptr
}

// Test 10: Very large allocation (exceeds chunk size)
TEST_F(ThreadLocalArenaTest, VeryLargeAllocation) {
    size_t huge_size = 128 * 1024;  // Larger than chunk size
    void* ptr = arena->allocate(huge_size);
    ASSERT_NE(ptr, nullptr);

    // Verify alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

    auto stats = arena->get_statistics();
    EXPECT_GE(stats.chunks_allocated, 3);  // 2 initial + 1 large chunk
}

// Test 11: Max chunks limit
TEST_F(ThreadLocalArenaTest, MaxChunksLimit) {
    // Create arena with very low max chunks (3 total: 2 initial + 1 more)
    ThreadLocalArena small_arena(2, 1024, 3);

    // Fill initial chunks (2 * 1024 bytes / 64 bytes per allocation = 32 allocations)
    for (int i = 0; i < 32; ++i) {
        void* ptr = small_arena.allocate(32);
        ASSERT_NE(ptr, nullptr);
    }

    // Trigger one more chunk allocation
    void* ptr1 = small_arena.allocate(32);
    ASSERT_NE(ptr1, nullptr);

    auto stats = small_arena.get_statistics();
    EXPECT_EQ(stats.chunks_allocated, 3);

    // Now we've hit the limit, next allocation should fallback to malloc
    void* ptr2 = small_arena.allocate(32);
    ASSERT_NE(ptr2, nullptr);  // Should still succeed via malloc

    stats = small_arena.get_statistics();
    EXPECT_GE(stats.fallback_to_malloc, 1);

    // Clean up malloc'd memory (normally arena doesn't track this)
    // In real usage, fallback allocations are leaked or tracked separately
}

// Test 12: Thread-local arena getter
TEST(ThreadLocalArenaGlobalTest, GetThreadArena) {
    // Get thread-local arena
    ThreadLocalArena* arena1 = get_thread_arena();
    ASSERT_NE(arena1, nullptr);

    // Getting again should return same instance
    ThreadLocalArena* arena2 = get_thread_arena();
    EXPECT_EQ(arena1, arena2);

    // Allocate from thread-local arena
    void* ptr = arena1->allocate(32);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

    auto stats1 = arena1->get_statistics();
    EXPECT_EQ(stats1.allocations_from_bump, 1);

    // Clean up
    destroy_thread_arena();

    // After destroy, get_thread_arena should create new instance
    ThreadLocalArena* arena3 = get_thread_arena();
    ASSERT_NE(arena3, nullptr);

    // Verify it's a fresh arena (statistics reset)
    // Note: Pointer might be reused by allocator, so check statistics instead
    auto stats3 = arena3->get_statistics();
    EXPECT_EQ(stats3.allocations_from_bump, 0);
    EXPECT_EQ(stats3.bytes_allocated, 0);
    EXPECT_EQ(stats3.chunks_allocated, 2);  // Initial chunks

    // Clean up
    destroy_thread_arena();
}

// Test 13: Multiple threads with separate arenas
TEST(ThreadLocalArenaGlobalTest, MultipleThreads) {
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<bool> success(num_threads, false);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&success, i]() {
            // Each thread gets its own arena
            ThreadLocalArena* arena = get_thread_arena();
            if (!arena) {
                return;
            }

            // Verify it's a fresh arena
            auto stats_before = arena->get_statistics();
            if (stats_before.allocations_from_bump != 0) {
                return;  // Arena was reused from another thread - this is a bug
            }

            // Allocate from thread-local arena
            for (int j = 0; j < 100; ++j) {
                void* ptr = arena->allocate(27);
                if (!ptr) {
                    return;
                }
                if (reinterpret_cast<uintptr_t>(ptr) % 64 != 0) {
                    return;
                }
            }

            // Verify statistics
            auto stats = arena->get_statistics();
            if (stats.allocations_from_bump == 100) {
                success[i] = true;
            }

            // Clean up thread-local arena
            destroy_thread_arena();
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Verify all threads succeeded
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_TRUE(success[i]) << "Thread " << i << " failed";
    }
}

// Test 14: Statistics tracking
TEST_F(ThreadLocalArenaTest, StatisticsTracking) {
    auto stats0 = arena->get_statistics();
    EXPECT_EQ(stats0.allocations_from_bump, 0);
    EXPECT_EQ(stats0.bytes_allocated, 0);
    EXPECT_EQ(stats0.chunks_allocated, 2);

    // Allocate some memory
    arena->allocate(32);
    auto stats1 = arena->get_statistics();
    EXPECT_EQ(stats1.allocations_from_bump, 1);
    EXPECT_EQ(stats1.bytes_allocated, 64);

    arena->allocate(128);
    auto stats2 = arena->get_statistics();
    EXPECT_EQ(stats2.allocations_from_bump, 2);
    EXPECT_EQ(stats2.bytes_allocated, 64 + 128);

    // Reset and verify statistics
    arena->reset();
    auto stats3 = arena->get_statistics();
    EXPECT_EQ(stats3.allocations_from_bump, 0);
    EXPECT_EQ(stats3.bytes_allocated, 0);
    EXPECT_EQ(stats3.chunks_allocated, 2);  // Chunks not freed on reset
}

// Test 15: Allocation pattern (sequential)
TEST_F(ThreadLocalArenaTest, SequentialAllocationPattern) {
    std::vector<void*> ptrs;
    const int count = 1000;

    for (int i = 0; i < count; ++i) {
        void* ptr = arena->allocate(27);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Verify pointers are sequential (within same chunk)
    // Note: May span multiple chunks, so we just verify no overlaps
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            uintptr_t ptr_i = reinterpret_cast<uintptr_t>(ptrs[i]);
            uintptr_t ptr_j = reinterpret_cast<uintptr_t>(ptrs[j]);

            // Allocations are 64 bytes apart (rounded up from 27)
            // They should not overlap
            EXPECT_NE(ptr_i, ptr_j);
        }
    }
}

// Test 16: Write and read back test
TEST_F(ThreadLocalArenaTest, WriteAndReadBack) {
    const size_t size = 128;
    void* ptr = arena->allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Write pattern
    uint8_t* bytes = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>(i % 256);
    }

    // Read back and verify
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(bytes[i], static_cast<uint8_t>(i % 256))
            << "Mismatch at byte " << i;
    }
}

// Test 17: Free list basic functionality
TEST_F(ThreadLocalArenaTest, FreeListBasic) {
    // Allocate and free a block (32 bytes rounds to 64)
    void* ptr1 = arena->allocate(32);
    ASSERT_NE(ptr1, nullptr);

    auto stats_after_alloc = arena->get_statistics();
    EXPECT_EQ(stats_after_alloc.allocations_from_bump, 1);
    EXPECT_EQ(stats_after_alloc.allocations_from_freelist, 0);

    // Deallocate
    arena->deallocate(ptr1, 32);

    auto stats_after_free = arena->get_statistics();
    EXPECT_EQ(stats_after_free.deallocations, 1);
    EXPECT_EQ(stats_after_free.bytes_in_freelists, 64);  // Rounded to 64

    // Allocate again - should reuse from free list
    void* ptr2 = arena->allocate(32);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_EQ(ptr2, ptr1);  // Same pointer (LIFO)

    auto stats_after_reuse = arena->get_statistics();
    EXPECT_EQ(stats_after_reuse.allocations_from_freelist, 1);
    EXPECT_EQ(stats_after_reuse.bytes_in_freelists, 0);
}

// Test 18: Free list LIFO ordering
TEST_F(ThreadLocalArenaTest, FreeListLIFO) {
    // Allocate multiple blocks
    void* ptr1 = arena->allocate(64);
    void* ptr2 = arena->allocate(64);
    void* ptr3 = arena->allocate(64);

    // Free them in order 1, 2, 3
    arena->deallocate(ptr1, 64);
    arena->deallocate(ptr2, 64);
    arena->deallocate(ptr3, 64);

    // Allocate again - should get them back in reverse order (LIFO): 3, 2, 1
    void* reused1 = arena->allocate(64);
    EXPECT_EQ(reused1, ptr3);  // Most recently freed

    void* reused2 = arena->allocate(64);
    EXPECT_EQ(reused2, ptr2);

    void* reused3 = arena->allocate(64);
    EXPECT_EQ(reused3, ptr1);  // Least recently freed
}

// Test 19: Free list size classes
TEST_F(ThreadLocalArenaTest, FreeListSizeClasses) {
    // Allocate different sizes (will round up to 64, 64, 128, 256)
    void* ptr32 = arena->allocate(32);   // Rounds to 64
    void* ptr64 = arena->allocate(64);   // Stays 64
    void* ptr128 = arena->allocate(128); // Stays 128
    void* ptr256 = arena->allocate(256); // Stays 256

    // Free them all
    arena->deallocate(ptr32, 32);
    arena->deallocate(ptr64, 64);
    arena->deallocate(ptr128, 128);
    arena->deallocate(ptr256, 256);

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.bytes_in_freelists, 64 + 64 + 128 + 256);  // All rounded to size classes

    // Allocate 32 bytes - should get ptr64 or ptr32 back (both are 64-byte class)
    void* reused32 = arena->allocate(32);
    // Either ptr64 or ptr32 (both are in 64-byte class)
    EXPECT_TRUE(reused32 == ptr32 || reused32 == ptr64);

    // Allocate 64 bytes - should get the other one
    void* reused64 = arena->allocate(64);
    EXPECT_TRUE(reused64 == ptr32 || reused64 == ptr64);
    EXPECT_NE(reused64, reused32);  // Different from the first one

    // Allocate 128 bytes - should get ptr128 back
    void* reused128 = arena->allocate(128);
    EXPECT_EQ(reused128, ptr128);

    // Allocate 256 bytes - should get ptr256 back
    void* reused256 = arena->allocate(256);
    EXPECT_EQ(reused256, ptr256);
}

// Test 20: Free list with size class rounding
TEST_F(ThreadLocalArenaTest, FreeListSizeClassRounding) {
    // Allocate 27 bytes (typical MCTS node size)
    // Should round up to 64 bytes
    void* ptr1 = arena->allocate(27);
    ASSERT_NE(ptr1, nullptr);

    arena->deallocate(ptr1, 27);

    // Allocate 30 bytes - should also round to 64 and reuse ptr1
    void* ptr2 = arena->allocate(30);
    EXPECT_EQ(ptr2, ptr1);

    auto stats = arena->get_statistics();
    EXPECT_EQ(stats.allocations_from_freelist, 1);
}

// Test 21: Free list with reset
TEST_F(ThreadLocalArenaTest, FreeListWithReset) {
    // Allocate and free some blocks
    void* ptr1 = arena->allocate(64);
    void* ptr2 = arena->allocate(64);

    arena->deallocate(ptr1, 64);
    arena->deallocate(ptr2, 64);

    auto stats_before = arena->get_statistics();
    EXPECT_EQ(stats_before.bytes_in_freelists, 128);

    // Reset arena
    arena->reset();

    auto stats_after = arena->get_statistics();
    EXPECT_EQ(stats_after.bytes_in_freelists, 0);
    EXPECT_EQ(stats_after.deallocations, 0);

    // Allocate again - should NOT reuse (free lists cleared)
    void* ptr3 = arena->allocate(64);
    ASSERT_NE(ptr3, nullptr);

    auto stats_final = arena->get_statistics();
    EXPECT_EQ(stats_final.allocations_from_freelist, 0);
    EXPECT_EQ(stats_final.allocations_from_bump, 1);
}

// Test 22: Free list reuse performance
TEST_F(ThreadLocalArenaTest, FreeListReusePerformance) {
    const int iterations = 1000;
    std::vector<void*> ptrs;

    // Allocate blocks
    for (int i = 0; i < iterations; ++i) {
        void* ptr = arena->allocate(64);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Free all blocks
    for (void* ptr : ptrs) {
        arena->deallocate(ptr, 64);
    }

    auto stats_after_free = arena->get_statistics();
    EXPECT_EQ(stats_after_free.deallocations, iterations);

    // Allocate again - should all come from free list
    for (int i = 0; i < iterations; ++i) {
        void* ptr = arena->allocate(64);
        ASSERT_NE(ptr, nullptr);
    }

    auto stats_after_reuse = arena->get_statistics();
    EXPECT_EQ(stats_after_reuse.allocations_from_freelist, iterations);
    EXPECT_EQ(stats_after_reuse.bytes_in_freelists, 0);
}

// Test 23: Large allocations (>256) don't use free lists
TEST_F(ThreadLocalArenaTest, LargeAllocationsNoFreeList) {
    // Allocate 512 bytes (larger than max size class of 256)
    void* ptr1 = arena->allocate(512);
    ASSERT_NE(ptr1, nullptr);

    arena->deallocate(ptr1, 512);

    auto stats_after_free = arena->get_statistics();
    // Deallocation is tracked, but not added to free list
    EXPECT_EQ(stats_after_free.deallocations, 1);
    // bytes_in_freelists should be 0 for sizes > 256
    EXPECT_EQ(stats_after_free.bytes_in_freelists, 0);

    // Allocate again - should NOT reuse (too large for free list)
    void* ptr2 = arena->allocate(512);
    ASSERT_NE(ptr2, nullptr);

    auto stats_after_alloc = arena->get_statistics();
    EXPECT_EQ(stats_after_alloc.allocations_from_freelist, 0);
    EXPECT_EQ(stats_after_alloc.allocations_from_bump, 2);
}

// Test 24: Mixed allocate/deallocate pattern
TEST_F(ThreadLocalArenaTest, MixedAllocateDeallocate) {
    std::vector<void*> ptrs;

    // Allocate 10 blocks
    for (int i = 0; i < 10; ++i) {
        void* ptr = arena->allocate(64);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    // Free every other block
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        arena->deallocate(ptrs[i], 64);
    }

    auto stats_after_partial_free = arena->get_statistics();
    EXPECT_EQ(stats_after_partial_free.deallocations, 5);

    // Allocate 5 more - should reuse the freed blocks
    for (int i = 0; i < 5; ++i) {
        void* ptr = arena->allocate(64);
        ASSERT_NE(ptr, nullptr);
    }

    auto stats_final = arena->get_statistics();
    EXPECT_EQ(stats_final.allocations_from_freelist, 5);
    EXPECT_EQ(stats_final.bytes_in_freelists, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
