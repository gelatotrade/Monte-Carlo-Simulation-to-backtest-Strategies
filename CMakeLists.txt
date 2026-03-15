cmake_minimum_required(VERSION 3.14)
project(OptionsBacktester VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release optimisation by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address -fno-omit-frame-pointer")

# Main executable
add_executable(backtest src/main.cpp)
target_include_directories(backtest PRIVATE include)

# Threading support (for potential future parallelisation)
find_package(Threads REQUIRED)
target_link_libraries(backtest PRIVATE Threads::Threads)

# Ensure output directory exists at build time
add_custom_command(TARGET backtest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/output
    COMMENT "Creating output directory"
)

# Install
install(TARGETS backtest DESTINATION bin)
