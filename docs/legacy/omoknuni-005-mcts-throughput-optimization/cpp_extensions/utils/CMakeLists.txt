# Utils C++ extension module

# Collect all utility source files
set(UTILS_SOURCES
    # Core game infrastructure
    igamestate.cpp
    # game_export.cpp  # Moved to games/interface.cpp for T023

    # Utility functions
    zobrist_hash.cpp
    logger.cpp
    hash_specializations.cpp
)

# Create static library for utilities (shared by other modules)
add_library(utils_core STATIC ${UTILS_SOURCES})

# Enable Position Independent Code for static library
set_target_properties(utils_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link libraries
target_link_libraries(utils_core PUBLIC OpenMP::OpenMP_CXX)

# Include directories
target_include_directories(utils_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler definitions
target_compile_definitions(utils_core PUBLIC
    VERSION_INFO="${PROJECT_VERSION}"
)

# Set C++ standard
target_compile_features(utils_core PUBLIC cxx_std_17)