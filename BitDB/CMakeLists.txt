cmake_minimum_required(VERSION 3.18)
project(BitDB VERSION 4.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put built executables into the project's build/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/build")

# ─────────────────────────────────────────────────────────────────────────────
# Build Options (AVX2 vs Scalar / Portable Fallback)
# ─────────────────────────────────────────────────────────────────────────────
option(ENABLE_AVX2 "Enable AVX2 SIMD acceleration for primary targets" ON)
option(BUILD_PORTABLE_TARGETS "Build separate portable (scalar fallback) targets alongside AVX2 targets" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Python 3 Discovery
# ─────────────────────────────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
# Compiler Flag Definitions (GCC, Clang, MSVC)
# ─────────────────────────────────────────────────────────────────────────────
if(MSVC)
    set(BITDB_BASE_FLAGS /std:c++17 /O2 /W3 /EHsc)
    set(BITDB_AVX2_FLAGS /arch:AVX2)
else()
    set(BITDB_BASE_FLAGS -std=c++17 -O2 -Wall)
    set(BITDB_AVX2_FLAGS -mavx2 -mpopcnt)
endif()

if(ENABLE_AVX2)
    set(BITDB_PRIMARY_FLAGS ${BITDB_BASE_FLAGS} ${BITDB_AVX2_FLAGS})
    set(AVX2_STATUS_MSG "ENABLED")
else()
    set(BITDB_PRIMARY_FLAGS ${BITDB_BASE_FLAGS})
    set(AVX2_STATUS_MSG "DISABLED (Scalar Fallback)")
endif()

message(STATUS "══════════════════════════════════════════════════")
message(STATUS "  BitDB CMake Configuration")
message(STATUS "══════════════════════════════════════════════════")
message(STATUS "  C++ Standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python3 Executable    : ${Python3_EXECUTABLE}")
message(STATUS "  Python3 Version       : ${Python3_VERSION}")
message(STATUS "  Python3 Includes      : ${Python3_INCLUDE_DIRS}")
message(STATUS "  Python3 Libraries     : ${Python3_LIBRARIES}")
message(STATUS "  Output Directory      : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  Primary AVX2 SIMD     : ${AVX2_STATUS_MSG}")
message(STATUS "  Portable Target Build : ${BUILD_PORTABLE_TARGETS}")
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

target_compile_options(bitdb_embed PRIVATE ${BITDB_PRIMARY_FLAGS})

# ─────────────────────────────────────────────────────────────────────────────
# Primary Target Executables
# ─────────────────────────────────────────────────────────────────────────────

# 1. Build.exe (PDF Ingestion & Segment Index Builder)
add_executable(Build src/Build.cpp)
target_link_libraries(Build PRIVATE bitdb_embed)
target_compile_options(Build PRIVATE ${BITDB_PRIMARY_FLAGS})

# 2. BitDBSearch.exe (Query Search Engine)
add_executable(BitDBSearch src/Search.cpp)
target_link_libraries(BitDBSearch PRIVATE bitdb_embed)
target_compile_options(BitDBSearch PRIVATE ${BITDB_PRIMARY_FLAGS})

# 3. print_catalog.exe (Catalog inspection utility - portable)
add_executable(print_catalog printers/print_catalog.cpp)
target_include_directories(print_catalog PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(print_catalog PRIVATE ${BITDB_BASE_FLAGS})

# 4. print_segment_dir.exe (Segment index inspection utility - portable)
add_executable(print_segment_dir printers/print_segment_dir.cpp)
target_include_directories(print_segment_dir PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(print_segment_dir PRIVATE ${BITDB_BASE_FLAGS})

# 5. Watchdog.exe (Native auto-sync daemon - portable)
add_executable(Watchdog src/Watchdog.cpp)
target_include_directories(Watchdog PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(Watchdog PRIVATE ${BITDB_BASE_FLAGS})

# 6. HaltonProbes.exe (Deterministic 32-probe generator - portable)
add_executable(HaltonProbes src/HaltonProbes.cpp)
target_include_directories(HaltonProbes PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(HaltonProbes PRIVATE ${BITDB_BASE_FLAGS})

# 7. test_suite.exe (Unit & invariant test suite)
add_executable(test_suite src/test_suite.cpp)
target_include_directories(test_suite PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(test_suite PRIVATE ${BITDB_PRIMARY_FLAGS})

# ─────────────────────────────────────────────────────────────────────────────
# Separate Portable Targets (Scalar fallback binaries without AVX2)
# ─────────────────────────────────────────────────────────────────────────────
if(BUILD_PORTABLE_TARGETS)
    add_library(bitdb_embed_portable STATIC
        src/embed.cpp
        src/embed.h
        src/PathConfig.h
    )
    target_include_directories(bitdb_embed_portable PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${Python3_INCLUDE_DIRS}
    )
    target_link_libraries(bitdb_embed_portable PUBLIC
        Python3::Python
    )
    target_compile_options(bitdb_embed_portable PRIVATE ${BITDB_BASE_FLAGS})

    # Build_portable
    add_executable(Build_portable src/Build.cpp)
    target_link_libraries(Build_portable PRIVATE bitdb_embed_portable)
    target_compile_options(Build_portable PRIVATE ${BITDB_BASE_FLAGS})

    # BitDBSearch_portable
    add_executable(BitDBSearch_portable src/Search.cpp)
    target_link_libraries(BitDBSearch_portable PRIVATE bitdb_embed_portable)
    target_compile_options(BitDBSearch_portable PRIVATE ${BITDB_BASE_FLAGS})

    # test_suite_portable
    add_executable(test_suite_portable src/test_suite.cpp)
    target_include_directories(test_suite_portable PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(test_suite_portable PRIVATE ${BITDB_BASE_FLAGS})
endif()
