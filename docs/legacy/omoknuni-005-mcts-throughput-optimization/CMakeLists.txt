cmake_minimum_required(VERSION 3.18...3.25)
project(omoknuni LANGUAGES CXX CUDA)

# Sanitizer options
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if((ENABLE_ASAN AND ENABLE_TSAN) OR (ENABLE_ASAN AND ENABLE_UBSAN) OR (ENABLE_TSAN AND ENABLE_UBSAN))
    message(FATAL_ERROR "Multiple sanitizers enabled. Please enable only one of ENABLE_ASAN, ENABLE_TSAN, ENABLE_UBSAN.")
endif()

set(SANITIZER_COMPILE_FLAGS "")
set(SANITIZER_LINK_FLAGS "")

if(ENABLE_ASAN)
    list(APPEND SANITIZER_COMPILE_FLAGS "-fsanitize=address" "-fno-omit-frame-pointer" "-g" "-O1")
    list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=address")
elseif(ENABLE_TSAN)
    list(APPEND SANITIZER_COMPILE_FLAGS "-fsanitize=thread" "-fno-omit-frame-pointer" "-g" "-O1")
    list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=thread")
elseif(ENABLE_UBSAN)
    list(APPEND SANITIZER_COMPILE_FLAGS "-fsanitize=undefined" "-fno-omit-frame-pointer" "-g" "-O1")
    list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=undefined")
endif()

if(SANITIZER_COMPILE_FLAGS)
    message(STATUS "Sanitizer build enabled with flags: ${SANITIZER_COMPILE_FLAGS}")
    add_compile_options(${SANITIZER_COMPILE_FLAGS})
    add_link_options(${SANITIZER_LINK_FLAGS})
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard if CUDA is available
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Compiler-specific optimizations for Ryzen 5900X
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=znver3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=znver3")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Google Test for unit testing
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Fetch Google Test if not found
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Enable testing
    enable_testing()
    include(GoogleTest)
    message(STATUS "Google Test will be downloaded and built automatically")
else()
    enable_testing()
    include(GoogleTest)
    message(STATUS "Found Google Test")
endif()

# Find PyTorch (optional, for C++ libtorch integration)
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found PyTorch")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# CUDA configuration
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 Ti architecture
endif()

# Include directories
include_directories(cpp_extensions)

# Add subdirectories for C++ extensions
add_subdirectory(cpp_extensions/mcts)
add_subdirectory(cpp_extensions/games)
add_subdirectory(cpp_extensions/utils)

# Add tests subdirectory if testing is enabled
if(BUILD_TESTING OR GTest_FOUND OR TARGET gtest)
    add_subdirectory(tests)
endif()

# Global compiler flags for performance
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    add_compile_definitions(NDEBUG)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-ffast-math -funroll-loops -finline-functions)
    endif()
endif()

# Alignment and SIMD optimizations
add_compile_definitions(ALIGNMENT=64)  # 64-byte alignment for cache lines

# Thread safety flags
add_compile_definitions(_REENTRANT)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
