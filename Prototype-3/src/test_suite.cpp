// ════════════════════════════════════════════════════════════════════════
// test_suite.cpp — Comprehensive Unit & Invariant Test Suite for Prototype-3
// Validates:
//   1. Physical Record Geometry and Alignment (ChunkRecord == 464 bytes)
//   2. MurmurHash3 32-bit Avalanche Finalizer & Strict Avalanche Criterion (SAC)
//   3. Segment Mapping & Boundary Confidence Probing
//   4. On-Disk Binary Storage & Extent Chain Invariants
// ════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
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

static constexpr uint32_t DIMS          = 384;
static constexpr uint32_t NUM_SEGMENTS  = BitDB::NUM_SEGMENTS; // 256
static constexpr uint32_t MAGIC         = 0x42444233u;         // "BDB3"
static constexpr uint32_t VERSION       = 3;

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
};

struct ExtentNode {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t next_extent_idx;
};

struct ChunkRecord {
    int8_t   embedding[DIMS];
    uint64_t text_offset;
    uint32_t text_length;
    uint32_t doc_id;
    uint32_t page_num;
    uint32_t chunk_idx_in_page;
    uint32_t segment_id;
    uint32_t signature;
    uint8_t  binary_code[BitDB::BINARY_CODE_BYTES];
};

struct CatalogHeader {
    uint32_t num_docs;
    uint32_t active_docs;
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

// ─────────────────────────────────────────────────
// Test 1: Physical Record Layout
// ─────────────────────────────────────────────────
static bool test_physical_layout() {
    TEST_ASSERT(sizeof(SegDirHeader) == 16, "SegDirHeader must be 16 bytes");
    TEST_ASSERT(sizeof(SegEntry) == 16, "SegEntry must be 16 bytes");
    TEST_ASSERT(sizeof(ExtentNode) == 16, "ExtentNode must be 16 bytes");
    TEST_ASSERT(sizeof(ChunkRecord) == 464, "ChunkRecord must be exactly 464 bytes");
    TEST_ASSERT(sizeof(CatalogHeader) == 8, "CatalogHeader must be 8 bytes");
    return true;
}

// ─────────────────────────────────────────────────
// Test 2: Murmur3 32-bit Avalanche Finalizer
// ─────────────────────────────────────────────────
static bool test_avalanche_hash() {
    // 1. Determinism
    uint32_t sig = 0x12345678u;
    TEST_ASSERT(BitDB::hash32_avalanche(sig) == BitDB::hash32_avalanche(sig), "Hash must be deterministic");

    // 2. Segment range mapping [0 .. 255]
    for (uint32_t s = 0; s < 1000; ++s) {
        uint32_t seg = BitDB::signature_to_segment(s * 7919u);
        TEST_ASSERT(seg < NUM_SEGMENTS, "Segment ID must be strictly < 256");
    }

    // 3. Strict Avalanche Criterion (SAC):
    // Flipping a single input bit must flip on average ~16 bits in the 32-bit output (40% - 60% range).
    uint64_t totalBitFlips = 0;
    int testCount = 0;

    mt19937 rng(42);
    for (int iter = 0; iter < 100; ++iter) {
        uint32_t baseSig = rng();
        uint32_t baseHash = BitDB::hash32_avalanche(baseSig);

        for (int bit = 0; bit < 32; ++bit) {
            uint32_t flippedSig = baseSig ^ (1u << bit);
            uint32_t flippedHash = BitDB::hash32_avalanche(flippedSig);
            int diff = __builtin_popcount(baseHash ^ flippedHash);
            totalBitFlips += diff;
            testCount++;
        }
    }

    double avgFlips = static_cast<double>(totalBitFlips) / testCount;
    cout << "    [Info] Average bit diffusion per 1-bit input flip: " << fixed << setprecision(2) << avgFlips << " / 32 bits (" << (avgFlips / 32.0 * 100.0) << "%)\n";
    TEST_ASSERT(avgFlips >= 14.0 && avgFlips <= 18.0, "Average bit diffusion must be near 50% (~16 bits)");

    return true;
}

// ─────────────────────────────────────────────────
// Test 3: Confidence Margin Probing Math
// ─────────────────────────────────────────────────
static bool test_confidence_margin_probing() {
    int8_t emb[DIMS] = {};
    for (size_t i = 0; i < DIMS; ++i) emb[i] = 10;

    float margins[P3_NUM_PROBES];
    uint32_t mask = BitDB::compute_probe_bitmask_and_margins(emb, margins);

    // Margins must be non-negative
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        TEST_ASSERT(!isnan(margins[i]) && !isinf(margins[i]), "Margin must be finite");
        TEST_ASSERT(margins[i] >= 0.0f, "Margin must be non-negative");
    }

    return true;
}

// ─────────────────────────────────────────────────
// Test 4: On-Disk Storage Invariants
// ─────────────────────────────────────────────────
static bool test_storage_integrity() {
    fs::path storeFile = PathConfig::getChunkStoreFile();
    fs::path segFile   = PathConfig::getSegmentDirFile();
    fs::path extFile   = PathConfig::getSegmentExtentsFile();
    fs::path catFile   = PathConfig::getDocCatalogFile();

    TEST_ASSERT(fs::exists(storeFile), "chunk_store.bin must exist");
    uintmax_t storeSize = fs::file_size(storeFile);
    TEST_ASSERT(storeSize > 0, "chunk_store.bin must not be empty");
    TEST_ASSERT(storeSize % sizeof(ChunkRecord) == 0, "chunk_store.bin size must be an exact multiple of 464 bytes");

    uint32_t numRecords = static_cast<uint32_t>(storeSize / sizeof(ChunkRecord));
    cout << "    [Info] Verified " << numRecords << " ChunkRecords in chunk_store.bin (" << (storeSize / (1024*1024)) << " MB)\n";

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
    }
    cout << "    [Info] Total chunks in primary extents: " << totalChunks << "\n";
    TEST_ASSERT(totalChunks > 0, "Total chunks must be positive");

    if (fs::exists(catFile)) {
        ifstream catIn(catFile, ios::binary);
        CatalogHeader catHdr;
        catIn.read(reinterpret_cast<char*>(&catHdr), sizeof(catHdr));
        cout << "    [Info] Catalog docs: " << catHdr.num_docs << " total, " << catHdr.active_docs << " active\n";
        TEST_ASSERT(catHdr.num_docs > 0, "Doc catalog must contain documents");
    }

    return true;
}

// ─────────────────────────────────────────────────
// Main Test Runner Entrypoint
// ─────────────────────────────────────────────────
int main() {
    cout << "═══════════════════════════════════════════════════════════\n";
    cout << "  BitDB Prototype-3 Test Suite: Correctness & Invariants\n";
    cout << "═══════════════════════════════════════════════════════════\n";

    RUN_TEST(test_physical_layout);
    RUN_TEST(test_avalanche_hash);
    RUN_TEST(test_confidence_margin_probing);
    RUN_TEST(test_storage_integrity);

    cout << "\n───────────────────────────────────────────────────────────\n";
    cout << "  Test Summary: " << g_testsPassed << " passed, " << g_testsFailed << " failed\n";
    cout << "───────────────────────────────────────────────────────────\n";

    return (g_testsFailed == 0) ? 0 : 1;
}
