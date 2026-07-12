# Unit tests for MCTS components

# Node pool tests
add_executable(test_node_pool
    test_node_pool.cpp
)

target_link_libraries(test_node_pool
    PRIVATE
    mcts_core
    gtest_main
    gtest
)

# Add compiler optimizations for realistic performance testing
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_node_pool PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_node_pool)

# PUCT selection tests
add_executable(test_selection
    test_selection.cpp
)

target_link_libraries(test_selection
    PRIVATE
    mcts_core
    gtest_main
    gtest
)

# Add compiler optimizations for realistic performance testing
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_selection PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_selection)

# Instrumentation tests
add_executable(test_instrumentation_metrics
    test_instrumentation_metrics.cpp
)

target_link_libraries(test_instrumentation_metrics
    PRIVATE
    mcts_core
    gtest_main
    gtest
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_instrumentation_metrics PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_instrumentation_metrics)

# Root pre-expansion tests
add_executable(test_root_pre_expansion
    test_root_pre_expansion.cpp
)

target_link_libraries(test_root_pre_expansion
    PRIVATE
    mcts_core
    games_core
    utils_core
    gtest_main
    gtest
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_root_pre_expansion PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_root_pre_expansion)

# Thread affinity tests
add_executable(test_thread_affinity
    test_thread_affinity.cpp
)

target_link_libraries(test_thread_affinity
    PRIVATE
    mcts_core
    gtest_main
    gtest
    pthread
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_thread_affinity PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_thread_affinity)

# Lock-free queue tests
add_executable(test_lock_free_queue
    test_lock_free_queue.cpp
)

target_link_libraries(test_lock_free_queue
    PRIVATE
    mcts_core
    gtest_main
    gtest
    pthread
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_lock_free_queue PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_lock_free_queue)

# Thread-local arena tests
add_executable(test_thread_local_arena
    test_thread_local_arena.cpp
)

target_link_libraries(test_thread_local_arena
    PRIVATE
    mcts_core
    gtest_main
    gtest
    pthread
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_thread_local_arena PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_thread_local_arena)

# Enhanced thread-local allocation tests (T009e)
add_executable(test_enhanced_thread_local_allocation
    test_enhanced_thread_local_allocation.cpp
)

target_link_libraries(test_enhanced_thread_local_allocation
    PRIVATE
    mcts_core
    gtest_main
    gtest
    pthread
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_enhanced_thread_local_allocation PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_enhanced_thread_local_allocation)

# Create a simple test runner for all unit tests
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    DEPENDS test_node_pool test_selection test_instrumentation_metrics test_root_pre_expansion test_thread_affinity test_lock_free_queue test_thread_local_arena test_enhanced_thread_local_allocation
    COMMENT "Running all unit tests"
)
