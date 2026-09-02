// ════════════════════════════════════════════════════════════════════════
// Search.cpp — Prototype-3 Flat Segment-Chain Search Engine
//
// Query flow:
//   1. Embed query text → 384-dim int8 vector
//   2. Compute 32-bit probe bitmask from query
//   3. Map bitmask to primary segment_id (and optionally ±1 neighbors)
//   4. Load segment_dir.bin (4KB — always in RAM)
//   5. Seek to matching segment(s) in chunk_store.bin, read those records
//   6. Score each chunk by int8 dot product
//   7. For top-K: read passage text from pdf_text.bin
//   8. Return results with PDF filename, page number, passage, score
//
// Usage: BitDBSearch.exe "<query text>" [N_results]
// ════════════════════════════════════════════════════════════════════════

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <set>

#include "embed.h"
#include "probe_vectors.h"
#include "PathConfig.h"

using namespace std;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────
// Constants (must match Build.cpp)
// ─────────────────────────────────────────────────

static constexpr uint32_t DIMS          = 384;
static constexpr uint32_t NUM_SEGMENTS  = 256;
static constexpr uint32_t MAGIC         = 0x42444233u;
static constexpr uint32_t CHUNK_RECORD_SZ = 416;

// ─────────────────────────────────────────────────
// Binary layout structs (must match Build.cpp exactly)
// ─────────────────────────────────────────────────

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
    uint32_t reserved;
};
struct ChunkRecord {
    int8_t   embedding[DIMS];
    uint64_t text_offset;
    uint32_t text_length;
    uint32_t doc_id;
    uint32_t page_num;
    uint32_t chunk_idx_in_page;
    uint32_t segment_id;
    uint32_t reserved;
};
struct CatalogHeader {
    uint32_t num_docs;
    uint32_t reserved;
};
struct DocEntry {
    uint32_t doc_id;
    uint32_t page_count;
    uint64_t chunk_store_start;
    uint32_t chunk_count;
    uint32_t reserved;
    char     filename[232];
};
#pragma pack(pop)

static_assert(sizeof(SegDirHeader) == 16,  "SegDirHeader size mismatch");
static_assert(sizeof(SegEntry)     == 16,  "SegEntry size mismatch");
static_assert(sizeof(ChunkRecord)  == 416, "ChunkRecord size mismatch");
static_assert(sizeof(CatalogHeader)== 8,   "CatalogHeader size mismatch");
static_assert(sizeof(DocEntry)     == 256, "DocEntry size mismatch");

// ─────────────────────────────────────────────────
// Probe-based segment routing (identical to Build.cpp)
// ─────────────────────────────────────────────────

static uint32_t compute_probe_bitmask(const vector<int8_t>& emb) {
    uint32_t mask = 0;
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < DIMS; ++j) {
            dot += static_cast<float>(emb[j]) * PROBE_VECTORS[i][j];
        }
        if (dot > 0.0f) mask |= (1u << i);
    }
    return mask;
}

static uint32_t bitmask_to_segment(uint32_t mask) {
    return mask & (NUM_SEGMENTS - 1);
}

// ─────────────────────────────────────────────────
// Fast int8 dot product similarity
// ─────────────────────────────────────────────────

static int32_t dot_int8(const int8_t* a, const int8_t* b, uint32_t dims) {
    int32_t score = 0;
    for (uint32_t i = 0; i < dims; ++i) {
        score += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return score;
}

// ─────────────────────────────────────────────────
// Result struct
// ─────────────────────────────────────────────────

struct SearchResult {
    int32_t  score;
    uint32_t doc_id;
    uint32_t page_num;
    uint64_t text_offset;
    uint32_t text_length;
    string   passage;
    string   filename;
};

// ─────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " \"<query text>\" [N_results]\n";
        return 1;
    }
    string queryText = argv[1];
    int topN = 5;
    if (argc >= 3) {
        try { topN = stoi(argv[2]); } catch (...) {}
    }

    const string binDir    = PathConfig::getDataStorageDir().string();
    const string segFile   = PathConfig::getSegmentDirFile().string();
    const string chunkFile = PathConfig::getChunkStoreFile().string();
    const string textFile  = PathConfig::getPdfTextFile().string();
    const string catFile   = PathConfig::getDocCatalogFile().string();

    auto t0 = chrono::high_resolution_clock::now();

    // ─── Step 1: Embed query ───
    cout << "[Search] Embedding query: \"" << queryText << "\"\n";
    init_python();
    auto tEmb0 = chrono::high_resolution_clock::now();
    vector<vector<int8_t>> qEmbs = embed_chunks({queryText});
    auto tEmb1 = chrono::high_resolution_clock::now();
    if (qEmbs.empty() || qEmbs[0].size() != DIMS) {
        cerr << "[Search] ERROR: Failed to generate query embedding.\n";
        finalize_python();
        return 1;
    }
    const vector<int8_t>& queryVec = qEmbs[0];
    finalize_python();
    double embedMs = chrono::duration<double, milli>(tEmb1 - tEmb0).count();
    cout << "[Search] Embedding done in " << embedMs << " ms\n";

    // ─── Step 2: Load segment_dir.bin (tiny — always fits in RAM) ───
    cout << "[Search] Loading segment directory...\n";
    SegEntry segDir[NUM_SEGMENTS] = {};
    {
        ifstream segIn(segFile, ios::binary);
        if (!segIn) {
            cerr << "[Search] ERROR: segment_dir.bin not found. Run Build.exe first.\n";
            return 1;
        }
        SegDirHeader hdr;
        segIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.magic != MAGIC) {
            cerr << "[Search] ERROR: segment_dir.bin has wrong magic. Rebuild index.\n";
            return 1;
        }
        segIn.read(reinterpret_cast<char*>(segDir), sizeof(segDir));
    }

    // ─── Step 3: Compute query segment(s) ───
    uint32_t qMask   = compute_probe_bitmask(queryVec);
    uint32_t primSeg = bitmask_to_segment(qMask);

    // Probe up to 3 adjacent segments for better recall
    // (bit-flip neighbors: flip lowest bit, second-lowest bit)
    set<uint32_t> segsToSearch;
    segsToSearch.insert(primSeg);
    segsToSearch.insert(bitmask_to_segment(qMask ^ 1u));         // flip bit 0
    segsToSearch.insert(bitmask_to_segment(qMask ^ 2u));         // flip bit 1
    segsToSearch.insert(bitmask_to_segment(qMask ^ 3u));         // flip bits 0+1

    cout << "[Search] Primary segment: " << primSeg
         << " | Probing " << segsToSearch.size() << " segments\n";

    uint32_t totalCandidates = 0;
    for (uint32_t s : segsToSearch) totalCandidates += segDir[s].chunk_count;
    cout << "[Search] Candidate chunks: " << totalCandidates << "\n";

    // ─── Step 4: Load catalog (small — fits in RAM) ───
    vector<DocEntry> catalog;
    {
        ifstream catIn(catFile, ios::binary);
        if (catIn) {
            CatalogHeader hdr;
            catIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            catalog.resize(hdr.num_docs);
            for (uint32_t i = 0; i < hdr.num_docs; ++i) {
                catIn.read(reinterpret_cast<char*>(&catalog[i]), sizeof(DocEntry));
            }
        }
    }

    // Build doc_id → filename lookup
    vector<string> docFilenames(catalog.size());
    for (const auto& de : catalog) {
        if (de.doc_id < docFilenames.size()) {
            docFilenames[de.doc_id] = string(de.filename);
        }
    }

    // ─── Step 5: Score candidate chunks from matching segments ───
    auto tSearch0 = chrono::high_resolution_clock::now();

    vector<SearchResult> results;
    results.reserve(totalCandidates);

    {
        ifstream chunkIn(chunkFile, ios::binary);
        if (!chunkIn) {
            cerr << "[Search] ERROR: chunk_store.bin not found. Run Build.exe first.\n";
            return 1;
        }

        ChunkRecord rec;
        for (uint32_t segId : segsToSearch) {
            const SegEntry& seg = segDir[segId];
            if (seg.chunk_count == 0) continue;

            // Seek to segment start
            chunkIn.seekg(static_cast<streamoff>(seg.chunk_store_offset), ios::beg);

            for (uint32_t ci = 0; ci < seg.chunk_count; ++ci) {
                chunkIn.read(reinterpret_cast<char*>(&rec), sizeof(ChunkRecord));
                if (!chunkIn) break;

                int32_t score = dot_int8(queryVec.data(), rec.embedding, DIMS);

                SearchResult sr;
                sr.score       = score;
                sr.doc_id      = rec.doc_id;
                sr.page_num    = rec.page_num;
                sr.text_offset = rec.text_offset;
                sr.text_length = rec.text_length;
                if (rec.doc_id < docFilenames.size()) {
                    sr.filename = docFilenames[rec.doc_id];
                } else {
                    sr.filename = "unknown";
                }
                results.push_back(sr);
            }
        }
    }

    // ─── Step 6: Sort by score descending, keep top-K ───
    int numToReturn = min((int)results.size(), topN);
    partial_sort(results.begin(), results.begin() + numToReturn, results.end(),
                 [](const SearchResult& a, const SearchResult& b){ return a.score > b.score; });
    results.resize(numToReturn);

    auto tSearch1 = chrono::high_resolution_clock::now();
    double searchMs = chrono::duration<double, milli>(tSearch1 - tSearch0).count();

    // ─── Step 7: Load passage text for top-K from pdf_text.bin ───
    {
        ifstream textIn(textFile, ios::binary);
        for (auto& sr : results) {
            if (!textIn) break;
            textIn.seekg(static_cast<streamoff>(sr.text_offset), ios::beg);
            sr.passage.resize(sr.text_length, '\0');
            textIn.read(&sr.passage[0], sr.text_length);
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double totalMs = chrono::duration<double, milli>(t1 - t0).count();

    // ─── Step 8: Display results ───
    cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    cout << "  TOP " << numToReturn << " RESULTS for: \"" << queryText << "\"\n";
    cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    for (int i = 0; i < numToReturn; ++i) {
        const auto& r = results[i];
        cout << "  ┌─ Rank " << (i + 1) << " ─────────────────────────────────────────\n";
        cout << "  │  Score   : " << r.score << "\n";
        cout << "  │  File    : " << r.filename << "\n";
        cout << "  │  Page    : " << r.page_num << "\n";
        cout << "  │  Passage : \"" << r.passage.substr(0, 300)
             << (r.passage.size() > 300 ? "..." : "") << "\"\n";
        cout << "  └──────────────────────────────────────────────────────\n\n";
    }

    cout << "  ╔═ TIMING ═════════════════════════════════════════════╗\n";
    cout << "  ║  Embedding   : " << embedMs  << " ms\n";
    cout << "  ║  Disk search : " << searchMs << " ms  (" << totalCandidates << " chunks scored)\n";
    cout << "  ║  Total       : " << totalMs  << " ms\n";
    cout << "  ╚══════════════════════════════════════════════════════╝\n";

    return 0;
}
