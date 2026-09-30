#!/usr/bin/env bash
set -e

echo ""
echo "=================================================="
echo "  BitDB Automated Test & Invariant Verification"
echo "=================================================="
echo ""

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo "[*] build/ directory missing. Running build.sh first..."
    ./build.sh
fi

TESTS_FAILED=0

echo "──────────────────────────────────────────────────"
echo " [STEP 1/5] Invariant & Unit Test Suite"
echo "──────────────────────────────────────────────────"
if [ -f "build/test_suite" ]; then
    ./build/test_suite || TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo "[!] build/test_suite missing. Compiling..."
    g++ -std=c++17 -O2 -mavx2 -mpopcnt -Isrc src/test_suite.cpp -o build/test_suite
    ./build/test_suite || TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "──────────────────────────────────────────────────"
echo " [STEP 2/5] Native Halton Probes Generator"
echo "──────────────────────────────────────────────────"
if [ -f "build/HaltonProbes" ]; then
    ./build/HaltonProbes || TESTS_FAILED=$((TESTS_FAILED + 1))
else
    g++ -std=c++17 -O2 -Isrc src/HaltonProbes.cpp -o build/HaltonProbes
    ./build/HaltonProbes || TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "──────────────────────────────────────────────────"
echo " [STEP 3/5] Document Catalog Integrity Audit"
echo "──────────────────────────────────────────────────"
if [ -f "build/print_catalog" ]; then
    ./build/print_catalog || TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "──────────────────────────────────────────────────"
echo " [STEP 4/5] 256-Segment Balance & Skew Audit"
echo "──────────────────────────────────────────────────"
if [ -f "build/print_segment_dir" ]; then
    ./build/print_segment_dir || TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "──────────────────────────────────────────────────"
echo " [STEP 5/5] Live End-to-End Query Verification"
echo "──────────────────────────────────────────────────"
if [ -f "build/BitDBSearch" ]; then
    ./build/BitDBSearch "vector index on SSD" 3 4 || TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

echo "=================================================="
if [ $TESTS_FAILED -eq 0 ]; then
    echo "  ALL TESTS PASSED SUCCESSFULLY! (0 Failures)"
else
    echo "  TEST RUN FINISHED WITH $TESTS_FAILED FAILURE(S)."
fi
echo "=================================================="
echo ""

exit $TESTS_FAILED
