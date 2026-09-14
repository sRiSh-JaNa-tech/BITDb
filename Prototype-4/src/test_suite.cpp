// ════════════════════════════════════════════════════════════════════════
// test_suite.cpp — Comprehensive Unit & Invariant Test Suite for Prototype-4
// Validates:
//   1. Physical 128 KB Extent Geometry and Memory Alignment
//   2. Multi-Index Hashing (MIH) Bit Manipulation & 1-Bit Neighbor Invariants
//   3. AVX2 Harley-Seal Popcount vs Scalar Ground Truth
//   4. Asymmetric Distance Computation (ADC) Numeric Correctness
//   5. Cauchy-Schwarz WAND Geometric Upper Bounding
//   6. On-Disk Binary Storage & Extent Block Invariants
// ════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>
#include <filesystem>
#include <iomanip>

#include "probe_vectors.h"
#include "Routing.h"
#include "PathConfig.h"

using namespace std;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────
// Prototype-4 Layout Constants & Geometry
// ─────────────────────────────────────────────────
static constexpr uint32_t DIMS          = 384;
static constexpr uint32_t NUM_SEGMENTS  = BitDB::NUM_SEGMENTS;
static constexpr uint32_t MAGIC         = 0x42444234u;
static constexpr uint32_t VERSION       = 4;

static constexpr uint32_t EXTENT_BYTES     = 131072; // 128 KB
static constexpr uint32_t EXTENT_CAPACITY  = 283;
static constexpr uint32_t HEADER_BYTES     = 512;
static constexpr uint32_t CODES_BYTES      = EXTENT_CAPACITY * BitDB::BINARY_CODE_BYTES; // 13584
static constexpr uint32_t STAGE1_BYTES     = HEADER_BYTES + CODES_BYTES;                 // 14096
static constexpr uint32_t EMBEDDINGS_BYTES = EXTENT_CAPACITY * DIMS;                     // 108672
static constexpr uint32_t METADATA_BYTES   = EXTENT_CAPACITY * 28;                       // 7924
static constexpr uint32_t PADDING_BYTES    = EXTENT_BYTES - (STAGE1_BYTES + EMBEDDINGS_BYTES + METADATA_BYTES); // 380

#pragma pack(push, 1)
struct ExtentHeader {
    uint32_t extent_id;
    uint32_t record_count;
    uint32_t next_extent_idx;
    uint8_t  reserved[500];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ChunkRecordMeta {
    uint64_t text_offset;
    uint32_t text_length;
    uint32_t doc_id;
    uint32_t page_num;
    uint32_t chunk_idx_in_page;
    uint32_t signature;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ExtentBlock {
    ExtentHeader    header;
    uint8_t         codes[EXTENT_CAPACITY][BitDB::BINARY_CODE_BYTES];
    int8_t          embeddings[EXTENT_CAPACITY][DIMS];
    ChunkRecordMeta metadata[EXTENT_CAPACITY];
    uint8_t         padding[PADDING_BYTES];
};
#pragma pack(pop)

#pragma pack(push, 1)
struct SegDirHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_segments;
    uint32_t reserved;
};

struct SegEntry {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t ext_chain_head;
    float    centroid[DIMS];
    float    max_radius;
};
#pragma pack(pop)

// ─────────────────────────────────────────────────
// Test Harness Utilities
// ─────────────────────────────────────────────────
static int g_testsPassed = 0;
static int g_testsFailed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            cerr << "  [FAIL] " << msg << " (line " << __LINE__ << ")\n"; \
            g_testsFailed++; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        cout << "[TEST] Running " << #fn << "...\n"; \
        if (fn()) { \
            cout << "  [PASS] " << #fn << "\n"; \
            g_testsPassed++; \
        } else { \
            cerr << "  [FAILED] " << #fn << "\n"; \
        } \
    } while (0)

// Reference scalar popcount
static uint32_t scalar_hamming(const uint8_t* a, const uint8_t* b, size_t n_bytes) {
    uint32_t dist = 0;
    for (size_t i = 0; i < n_bytes; ++i) {
        dist += (uint32_t)__builtin_popcount(static_cast<unsigned int>(a[i] ^ b[i]));
    }
    return dist;
}

// ─────────────────────────────────────────────────
// Test 1: Layout & Page Alignment
// ─────────────────────────────────────────────────
static bool test_physical_layout() {
    TEST_ASSERT(sizeof(ExtentHeader) == 512, "ExtentHeader must be exactly 512 bytes");
    TEST_ASSERT(sizeof(ChunkRecordMeta) == 28, "ChunkRecordMeta must be exactly 28 bytes");
    TEST_ASSERT(sizeof(ExtentBlock) == EXTENT_BYTES, "ExtentBlock must be exactly 131,072 bytes (128 KB)");
    TEST_ASSERT(EXTENT_CAPACITY == 283, "EXTENT_CAPACITY must be 283 records");
    TEST_ASSERT(STAGE1_BYTES == 14096, "Stage 1 block must be 14,096 bytes");
    TEST_ASSERT(EMBEDDINGS_BYTES == 108672, "Embeddings block must be 108,672 bytes");
    TEST_ASSERT(METADATA_BYTES == 7924, "Metadata block must be 7,924 bytes");
    TEST_ASSERT(PADDING_BYTES == 380, "Padding must be exactly 380 bytes");
    TEST_ASSERT(STAGE1_BYTES + EMBEDDINGS_BYTES + METADATA_BYTES + PADDING_BYTES == EXTENT_BYTES, "Section sum must equal 131072");
    return true;
}

// ─────────────────────────────────────────────────
// Test 2: Multi-Index Hashing (MIH) Substrings
// ─────────────────────────────────────────────────
static bool test_mih_substrings() {
    // 1. Exact split test
    uint32_t sig = 0xA1B2C3D4u;
    std::vector<uint8_t> sub = BitDB::extract_mih_substrings(sig);

    TEST_ASSERT(sub[0] == 0xD4, "Substring 0 must be 0xD4 (low byte)");
    TEST_ASSERT(sub[1] == 0xC3, "Substring 1 must be 0xC3");
    TEST_ASSERT(sub[2] == 0xB2, "Substring 2 must be 0xB2");
    TEST_ASSERT(sub[3] == 0xA1, "Substring 3 must be 0xA1 (high byte)");

    // 2. Edge values
    sub = BitDB::extract_mih_substrings(0x00000000u);
    TEST_ASSERT(sub[0] == 0 && sub[1] == 0 && sub[2] == 0 && sub[3] == 0, "All zeros failed");

    sub = BitDB::extract_mih_substrings(0xFFFFFFFFu);
    TEST_ASSERT(sub[0] == 0xFF && sub[1] == 0xFF && sub[2] == 0xFF && sub[3] == 0xFF, "All ones failed");

    // 3. 1-Bit Hamming Neighbor Invariant
    for (uint32_t val = 0; val < 256; ++val) {
        vector<uint8_t> neighbors;
        for (int b = 0; b < 8; ++b) {
            uint8_t flipped = static_cast<uint8_t>(val ^ (1u << b));
            neighbors.push_back(flipped);
            // Hamming distance between val and flipped must be exactly 1
            TEST_ASSERT(__builtin_popcount(val ^ flipped) == 1, "Hamming distance must be exactly 1");
        }
        // Check uniqueness of 8 neighbors
        for (size_t i = 0; i < neighbors.size(); ++i) {
            for (size_t j = i + 1; j < neighbors.size(); ++j) {
                TEST_ASSERT(neighbors[i] != neighbors[j], "Neighbor values must be distinct");
            }
        }
    }

    return true;
}

// ─────────────────────────────────────────────────
// Test 3: AVX2 Harley-Seal Popcount vs Scalar Reference
// ─────────────────────────────────────────────────
static bool test_avx2_popcount() {
    uint8_t codeZeros[BitDB::BINARY_CODE_BYTES] = {};
    uint8_t codeOnes[BitDB::BINARY_CODE_BYTES];
    memset(codeOnes, 0xFF, sizeof(codeOnes));

    // Zero vs Zero
    TEST_ASSERT(BitDB::hamming_distance_code(codeZeros, codeZeros) == 0, "Zero vs Zero must be 0");
    // Ones vs Ones
    TEST_ASSERT(BitDB::hamming_distance_code(codeOnes, codeOnes) == 0, "Ones vs Ones must be 0");
    // Zeros vs Ones
    TEST_ASSERT(BitDB::hamming_distance_code(codeZeros, codeOnes) == 384, "Zeros vs Ones must be 384");

    // Single bit tests across all 384 bits
    for (int bit = 0; bit < 384; ++bit) {
        uint8_t testCode[BitDB::BINARY_CODE_BYTES] = {};
        testCode[bit / 8] |= (1 << (bit % 8));
        uint32_t avxDist = BitDB::hamming_distance_code(codeZeros, testCode);
        uint32_t scalDist = scalar_hamming(codeZeros, testCode, BitDB::BINARY_CODE_BYTES);
        TEST_ASSERT(avxDist == 1, "Single bit flip must yield distance 1");
        TEST_ASSERT(avxDist == scalDist, "AVX2 must match scalar reference for single bit");
    }

    // 500 Pseudo-random test vectors
    mt19937 rng(1337);
    uniform_int_distribution<int> dist(0, 255);
    for (int iter = 0; iter < 500; ++iter) {
        uint8_t a[BitDB::BINARY_CODE_BYTES];
        uint8_t b[BitDB::BINARY_CODE_BYTES];
        for (size_t i = 0; i < BitDB::BINARY_CODE_BYTES; ++i) {
            a[i] = static_cast<uint8_t>(dist(rng));
            b[i] = static_cast<uint8_t>(dist(rng));
        }

        uint32_t avxRes = BitDB::hamming_distance_code(a, b);
        uint32_t scalRes = scalar_hamming(a, b, BitDB::BINARY_CODE_BYTES);
        TEST_ASSERT(avxRes == scalRes, "AVX2 popcount must match scalar reference exactly");
    }

    return true;
}

// ─────────────────────────────────────────────────
// Test 4: Asymmetric Distance Computation (ADC)
// ─────────────────────────────────────────────────
static bool test_adc_scoring() {
    int8_t query[DIMS];
    for (size_t i = 0; i < DIMS; ++i) {
        query[i] = (i % 2 == 0) ? 5 : -5;
    }

    uint8_t code[BitDB::BINARY_CODE_BYTES];
    // Set all bits to 1
    memset(code, 0xFF, sizeof(code));

    int32_t adcScore = BitDB::adc_score(query, code);

    // Manual sum: every bit is 1, so dot product with +1 for each bit
    int32_t expected = 0;
    for (size_t i = 0; i < DIMS; ++i) {
        expected += query[i];
    }

    TEST_ASSERT(adcScore == expected, "ADC score must equal sum of query entries for all-ones code");

    // Invert code: all bits 0
    memset(code, 0x00, sizeof(code));
    adcScore = BitDB::adc_score(query, code);
    TEST_ASSERT(adcScore == -expected, "ADC score must equal negative sum for all-zeros code");

    return true;
}

// ─────────────────────────────────────────────────
// Test 5: Cauchy-Schwarz WAND Geometric Upper Bound
// ─────────────────────────────────────────────────
static bool test_wand_upper_bounding() {
    mt19937 rng(42);
    normal_distribution<float> norm(0.0f, 1.0f);

    float centroid[DIMS];
    float query[DIMS];
    float qNormSq = 0.0f;
    for (size_t i = 0; i < DIMS; ++i) {
        centroid[i] = norm(rng);
        query[i] = norm(rng);
        qNormSq += query[i] * query[i];
    }
    float qNorm = sqrt(qNormSq);

    float radius = 5.0f;

    // Cauchy-Schwarz bound: MaxScore(q, E) = (q . C) + ||q|| * R
    float dotQC = 0.0f;
    for (size_t i = 0; i < DIMS; ++i) dotQC += query[i] * centroid[i];
    float maxScore = dotQC + qNorm * radius;

    // Generate 500 test points inside the hypersphere of radius R around centroid
    for (int iter = 0; iter < 500; ++iter) {
        float point[DIMS];
        float deltaNormSq = 0.0f;
        for (size_t i = 0; i < DIMS; ++i) {
            point[i] = norm(rng);
            deltaNormSq += point[i] * point[i];
        }
        float dNorm = sqrt(deltaNormSq);
        float scale = (radius * (iter / 500.0f)) / (dNorm > 1e-6f ? dNorm : 1.0f);

        for (size_t i = 0; i < DIMS; ++i) {
            point[i] = centroid[i] + point[i] * scale;
        }

        // Compute true dot product (q . point)
        float dotQP = 0.0f;
        for (size_t i = 0; i < DIMS; ++i) dotQP += query[i] * point[i];

        TEST_ASSERT(dotQP <= maxScore + 1e-4f, "Point dot product must not exceed Cauchy-Schwarz WAND bound");
    }

    return true;
}

// ─────────────────────────────────────────────────
// Test 6: On-Disk Storage & Extent Block Integrity
// ─────────────────────────────────────────────────
static bool test_storage_integrity() {
    fs::path storeFile = PathConfig::getChunkStoreFile();
    fs::path segFile   = PathConfig::getSegmentDirFile();
    fs::path mihFile   = PathConfig::getMihTableFile();
    fs::path catFile   = PathConfig::getDocCatalogFile();

    TEST_ASSERT(fs::exists(storeFile), "chunk_store.bin must exist");
    uintmax_t storeSize = fs::file_size(storeFile);
    TEST_ASSERT(storeSize > 0, "chunk_store.bin must not be empty");
    TEST_ASSERT(storeSize % EXTENT_BYTES == 0, "chunk_store.bin size must be an exact multiple of 128 KB");

    uint32_t numExtents = static_cast<uint32_t>(storeSize / EXTENT_BYTES);
    cout << "    [Info] Verified " << numExtents << " physical 128 KB extents (" << (storeSize / (1024*1024)) << " MB)\n";

    TEST_ASSERT(fs::exists(mihFile), "mih_table.bin must exist");
    TEST_ASSERT(fs::file_size(mihFile) == 32768, "mih_table.bin must be exactly 32,768 bytes (4 * 256 * 32)");

    TEST_ASSERT(fs::exists(segFile), "segment_dir.bin must exist");
    ifstream segIn(segFile, ios::binary);
    TEST_ASSERT(segIn.is_open(), "Could not open segment_dir.bin");
    SegDirHeader hdr;
    segIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    TEST_ASSERT(hdr.magic == MAGIC, "segment_dir.bin magic mismatch");
    TEST_ASSERT(hdr.version == VERSION, "segment_dir.bin version mismatch");
    TEST_ASSERT(hdr.num_segments == NUM_SEGMENTS, "segment_dir.bin num_segments mismatch");

    SegEntry segDir[NUM_SEGMENTS];
    segIn.read(reinterpret_cast<char*>(segDir), sizeof(segDir));
    segIn.close();

    uint32_t totalChunks = 0;
    for (uint32_t s = 0; s < NUM_SEGMENTS; ++s) {
        totalChunks += segDir[s].chunk_count;
        TEST_ASSERT(!isnan(segDir[s].max_radius) && !isinf(segDir[s].max_radius), "max_radius must not be NaN/Inf");
        TEST_ASSERT(segDir[s].max_radius >= 0.0f, "max_radius must be non-negative");
    }
    cout << "    [Info] Total indexed chunks across 256 segments: " << totalChunks << "\n";
    TEST_ASSERT(totalChunks > 0, "Indexed chunks must be positive");

    return true;
}

// ─────────────────────────────────────────────────
// Main Test Runner Entrypoint
// ─────────────────────────────────────────────────
int main() {
    cout << "═══════════════════════════════════════════════════════════\n";
    cout << "  BitDB Prototype-4 Test Suite: Correctness & Invariants\n";
    cout << "═══════════════════════════════════════════════════════════\n";

    RUN_TEST(test_physical_layout);
    RUN_TEST(test_mih_substrings);
    RUN_TEST(test_avx2_popcount);
    RUN_TEST(test_adc_scoring);
    RUN_TEST(test_wand_upper_bounding);
    RUN_TEST(test_storage_integrity);

    cout << "\n───────────────────────────────────────────────────────────\n";
    cout << "  Test Summary: " << g_testsPassed << " passed, " << g_testsFailed << " failed\n";
    cout << "───────────────────────────────────────────────────────────\n";

    return (g_testsFailed == 0) ? 0 : 1;
}
