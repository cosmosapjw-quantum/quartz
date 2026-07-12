# Test configuration for omoknuni

# Add subdirectories for different test types
add_subdirectory(unit)

# Optionally add other test types when they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/CMakeLists.txt")
    add_subdirectory(integration)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/performance/CMakeLists.txt")
    add_subdirectory(performance)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/contract/CMakeLists.txt")
    add_subdirectory(contract)
endif()