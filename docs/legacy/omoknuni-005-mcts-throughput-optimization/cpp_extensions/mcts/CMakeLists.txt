# MCTS C++ extension module
cmake_minimum_required(VERSION 3.15)
project(mcts_core)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA (optional, for pinned memory in dlpack_bridge)
find_package(CUDAToolkit QUIET)

# Sanitizer configuration (optional, enabled via cmake args)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Sanitizer flags (as list to avoid quoting issues)
set(SANITIZER_FLAGS)
if(ENABLE_ASAN)
    list(APPEND SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer -g -O1)
    message(STATUS "AddressSanitizer enabled")
endif()
if(ENABLE_TSAN)
    list(APPEND SANITIZER_FLAGS -fsanitize=thread -g -O1)
    message(STATUS "ThreadSanitizer enabled")
endif()
if(ENABLE_UBSAN)
    list(APPEND SANITIZER_FLAGS -fsanitize=undefined -g -O1)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
endif()

# Create static library for MCTS components
add_library(mcts_core STATIC
    tree.cpp
    tree.hpp
    tiny_node.hpp
    tiny_node_tree.cpp
    tiny_node_tree.hpp
    tree_adapter.cpp
    tree_adapter.hpp
    selection.cpp
    selection.hpp
    virtual_loss.cpp
    virtual_loss.hpp
    backup.cpp
    backup.hpp
    instrumentation.cpp
    instrumentation.hpp
    simulation_runner.cpp
    simulation_runner.hpp
    async_inference_queue.cpp
    async_inference_queue.hpp
    continuous_simulation_runner.cpp
    continuous_simulation_runner.hpp
    thread_affinity.cpp
    thread_affinity.hpp
    thread_local_arena.cpp
    thread_local_arena.hpp
    state_pool.cpp
    state_pool.hpp
    batch_inference_callback.hpp
    batch_inference_coordinator.cpp
    batch_inference_coordinator.hpp
    dlpack_bridge.cpp
    dlpack_bridge.hpp
    # Profiling framework (header-only + implementation files)
    profiling/enhanced_profiler.cpp
    profiling/enhanced_profiler.hpp
    profiling/validation.cpp
    profiling/validation.hpp
    profiling/enhanced_metrics.hpp
    profiling/thread_metrics.hpp
    profiling/statistical_analyzer.hpp
    profiling/export.hpp
)

# Set compile features and flags
target_compile_features(mcts_core PUBLIC cxx_std_17)
target_include_directories(mcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with OpenMP
target_link_libraries(mcts_core PUBLIC OpenMP::OpenMP_CXX)

# Link with CUDA runtime if available (for pinned memory)
if(CUDAToolkit_FOUND)
    target_link_libraries(mcts_core PUBLIC CUDA::cudart)
    message(STATUS "CUDA runtime found, enabling pinned memory support")
else()
    message(STATUS "CUDA runtime not found, using malloc fallback")
endif()

# Add compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mcts_core PRIVATE
        -Wall -Wextra
        -march=native  # Use native CPU architecture features
        -mavx2         # Enable AVX2 for vectorized PUCT selection
        -mfma          # Enable FMA instructions
        -fopenmp
        ${SANITIZER_FLAGS}
    )
    if(SANITIZER_FLAGS)
        target_link_options(mcts_core PRIVATE ${SANITIZER_FLAGS})
    endif()
endif()

# Add compile definitions
target_compile_definitions(mcts_core PRIVATE
    _REENTRANT
    ALIGNMENT=64
    PROFILE_LEVEL_VALUE=3  # Enable FULL profiling (all 295 metrics)
)

# Enable position independent code for Python extension
set_target_properties(mcts_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Minimal compilation test - links against mcts_core library
# Create minimal main function for compilation test
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test_main.cpp"
"#include \"tree.hpp\"
#include \"selection.hpp\"
#include \"virtual_loss.hpp\"
#include \"backup.hpp\"
#include <iostream>

int main() {
    std::cout << \"MCTS components compiled successfully!\" << std::endl;

    // Basic instantiation test
    mcts::MCTSTree tree(1000);
    mcts::VirtualLossConfig vl_config;
    mcts::BackupConfig backup_config;

    std::cout << \"Tree nodes: \" << tree.get_node_count() << std::endl;
    std::cout << \"All components instantiated successfully!\" << std::endl;
    return 0;
}
")

add_executable(mcts_compile_test "${CMAKE_CURRENT_BINARY_DIR}/test_main.cpp")
target_include_directories(mcts_compile_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Link against mcts_core library which includes profiling
target_link_libraries(mcts_compile_test PRIVATE mcts_core OpenMP::OpenMP_CXX pthread)

# Add AVX2 support to test executable
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mcts_compile_test PRIVATE
        -Wall -Wextra
        -march=native  # Use native CPU architecture features
        -mavx2         # Enable AVX2 for vectorized PUCT selection
        -mfma          # Enable FMA instructions
        -fopenmp
    )
endif()

# Register compilation test
add_test(NAME mcts_compilation COMMAND mcts_compile_test)

# Python bindings module (optional, requires pybind11)
if(pybind11_FOUND)
    pybind11_add_module(mcts_py python_bindings.cpp dlpack_python.cpp)

    # Link with MCTS core library and games library (for IGameState)
    # CRITICAL: Must link OpenMP explicitly when linking static library to shared object
    target_link_libraries(mcts_py PRIVATE mcts_core games_core utils_core OpenMP::OpenMP_CXX)

    # Include directories
    target_include_directories(mcts_py PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/cpp_extensions/utils
        ${CMAKE_SOURCE_DIR}/cpp_extensions/games
    )

    # Compiler definitions for Python module
    target_compile_definitions(mcts_py PRIVATE
        VERSION_INFO="${PROJECT_VERSION}"
        PYBIND11_DETAILED_ERROR_MESSAGES
        PROFILE_LEVEL_VALUE=3  # Enable FULL profiling (all 295 metrics)
    )

    # Install the module to the Python package
    install(TARGETS mcts_py
        COMPONENT python_modules
        DESTINATION ${SKBUILD_PLATLIB_DIR}
    )

    message(STATUS "MCTS Python bindings configured")
else()
    message(STATUS "pybind11 not found, skipping MCTS Python bindings")
endif()

message(STATUS "MCTS core library configured with all components")
