cmake_minimum_required(VERSION 3.18)
project(BitDB VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put built executables into the project's build/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/build")

# ─────────────────────────────────────────────────────────────────────────────
# Python 3 Discovery
# ─────────────────────────────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "══════════════════════════════════════════════════")
message(STATUS "  BitDB CMake Configuration")
message(STATUS "══════════════════════════════════════════════════")
message(STATUS "  C++ Standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python3 Executable : ${Python3_EXECUTABLE}")
message(STATUS "  Python3 Version    : ${Python3_VERSION}")
message(STATUS "  Python3 Includes   : ${Python3_INCLUDE_DIRS}")
message(STATUS "  Python3 Libraries  : ${Python3_LIBRARIES}")
message(STATUS "  Output Directory   : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "══════════════════════════════════════════════════")

# ─────────────────────────────────────────────────────────────────────────────
# Core Embedding Library (embed.cpp + PathConfig.h)
# ─────────────────────────────────────────────────────────────────────────────
add_library(bitdb_embed STATIC
    src/embed.cpp
    src/embed.h
    src/PathConfig.h
)

target_include_directories(bitdb_embed PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(bitdb_embed PUBLIC
    Python3::Python
)

# ─────────────────────────────────────────────────────────────────────────────
# Target Executables
# ─────────────────────────────────────────────────────────────────────────────

# 1. Build.exe (PDF Ingestion & Segment Index Builder)
add_executable(Build src/Build.cpp)
target_link_libraries(Build PRIVATE bitdb_embed)

# 2. BitDBSearch.exe (Query Search Engine)
add_executable(BitDBSearch src/Search.cpp)
target_link_libraries(BitDBSearch PRIVATE bitdb_embed)

# 3. print_catalog.exe (Catalog inspection utility)
add_executable(print_catalog printers/print_catalog.cpp)
target_include_directories(print_catalog PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# 4. print_segment_dir.exe (Segment index inspection utility)
add_executable(print_segment_dir printers/print_segment_dir.cpp)
target_include_directories(print_segment_dir PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Optimization flags for MSVC and GCC/Clang
if(MSVC)
    target_compile_options(bitdb_embed PRIVATE /O2 /W3)
    target_compile_options(Build PRIVATE /O2 /W3)
    target_compile_options(BitDBSearch PRIVATE /O2 /W3)
else()
    target_compile_options(bitdb_embed PRIVATE -O2 -Wall)
    target_compile_options(Build PRIVATE -O2 -Wall)
    target_compile_options(BitDBSearch PRIVATE -O2 -Wall)
endif()
