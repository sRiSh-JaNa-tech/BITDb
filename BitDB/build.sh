#!/usr/bin/env bash
if [ "$1" = "watch" ]; then
    if [ -f "build/Watchdog" ]; then
        echo "[*] Starting BitDB Native C++ Auto-Sync Watchdog on ./ingestor..."
        ./build/Watchdog "${@:2}"
        exit 0
    fi
    echo "[*] Starting BitDB Auto-Sync Watchdog on ./ingestor (Python fallback)..."
    python3 scripts/db_watchdog.py "${@:2}"
    exit 0
fi

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
$CXX -std=c++17 -O2 -Isrc src/Watchdog.cpp -o build/Watchdog
$CXX -std=c++17 -O2 -Isrc src/HaltonProbes.cpp -o build/HaltonProbes
$CXX -std=c++17 -O2 -Isrc printers/print_catalog.cpp -o build/print_catalog
$CXX -std=c++17 -O2 -Isrc printers/print_segment_dir.cpp -o build/print_segment_dir
$CXX -std=c++17 -O2 -mavx2 -mpopcnt -Isrc src/test_suite.cpp -o build/test_suite

echo "=================================================="
echo "  BUILD SUCCESSFUL"
echo "=================================================="
echo ""
echo "  USAGE:"
echo "    1. Ingest:      ./build/Build"
echo "    2. Query:       ./build/BitDBSearch \"your search query\" [N] [probes]"
echo "       List files:  ./build/BitDBSearch --list"
echo "       Interactive: ./build/BitDBSearch --interactive --probes 4"
echo "    3. Watchdog:    ./build.sh watch"
echo "    4. Catalog:     ./build/print_catalog"
echo "    5. Segments:    ./build/print_segment_dir"
echo "=================================================="
