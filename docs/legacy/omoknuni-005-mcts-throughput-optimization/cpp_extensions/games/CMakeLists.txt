# Games C++ extension module

# Collect all game source files (without Python bindings)
set(GAMES_CORE_SOURCES
    # Chess implementation
    chess/chess_state.cpp
    chess/chess_rules.cpp
    chess/chess960.cpp

    # Gomoku implementation
    gomoku/gomoku_state.cpp
    gomoku/gomoku_rules.cpp

    # Go implementation
    go/go_state.cpp
    go/go_rules.cpp

    # Game adapter interface
    interface.cpp
)

# Create static library for game implementations (used by both mcts_py and alphazero_py)
add_library(games_core STATIC ${GAMES_CORE_SOURCES})
target_compile_features(games_core PUBLIC cxx_std_17)
target_include_directories(games_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp_extensions/utils
)
target_link_libraries(games_core PUBLIC
    OpenMP::OpenMP_CXX
    utils_core
)
set_target_properties(games_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Create pybind11 module (only Python bindings, links games_core library)
pybind11_add_module(alphazero_py python_bindings.cpp)

# Link libraries
target_link_libraries(alphazero_py PRIVATE
    games_core
    OpenMP::OpenMP_CXX
    utils_core
)

# Include directories
target_include_directories(alphazero_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp_extensions/utils
)

# Compiler definitions
target_compile_definitions(alphazero_py PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
    PYBIND11_DETAILED_ERROR_MESSAGES
)

# Install the module to the Python package
install(TARGETS alphazero_py
    COMPONENT python_modules
    DESTINATION ${SKBUILD_PLATLIB_DIR}
)