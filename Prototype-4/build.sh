#!/usr/bin/env bash
set -e

echo ""
echo "=================================================="
echo "  BitDB Prototype-4 - Unified Build System (Unix)"
echo "=================================================="
echo ""

# Ensure required directories exist
mkdir -p build ingestor DataStorage

if command -v cmake >/dev/null 2>&1; then
    echo "[*] CMake detected. Building via CMake..."
    cmake -B build_cmake -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build build_cmake --config Release
    echo ""
    echo "=================================================="
    echo "  BUILD SUCCESSFUL"
    echo "=================================================="
    echo "  Binaries in: build/"
    exit 0
fi

echo "[*] CMake not found. Compiling with g++/clang++..."
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')")
PY_LD=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION'))")

CXX=g++
if ! command -v g++ >/dev/null 2>&1; then
    CXX=clang++
fi

FLAGS="-std=c++17 -O2 -Isrc -I${PY_INC}"
LIBS="-L${PY_LIB} -lpython${PY_LD}"

$CXX $FLAGS src/Build.cpp src/embed.cpp $LIBS -o build/Build
$CXX $FLAGS src/Search.cpp src/embed.cpp $LIBS -o build/BitDBSearch
$CXX -std=c++17 -O2 -Isrc printers/print_catalog.cpp -o build/print_catalog
$CXX -std=c++17 -O2 -Isrc printers/print_segment_dir.cpp -o build/print_segment_dir

echo "=================================================="
echo "  BUILD SUCCESSFUL"
echo "=================================================="
