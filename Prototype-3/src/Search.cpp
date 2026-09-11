// ════════════════════════════════════════════════════════════════════════
// Search.cpp — Prototype-3 Advanced Segment-Chain Search Engine
//
// Resolves:
//   Shortcoming 1: Full 32-bit signature exploitation via avalanche bit-mixing
//   Shortcoming 3: Adaptive multi-segment probing based on boundary confidence
//   Shortcoming 6: Bounded Top-K min-heap candidate selection (O(K) RAM)
//   Shortcoming 7: Two-stage search with fast Hamming signature pre-filter
//   Shortcoming 8: Single bulk SSD I/O read per extent
//   Shortcoming 9: Persistent interactive daemon mode eliminating Python startup
//
// Usage:
//   BitDBSearch.exe "<query text>" [N_results] [N_probes]
//   BitDBSearch.exe --interactive [--probes N]
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
#include <queue>
#include <unordered_set>
#include <unordered_map>

#include "embed.h"
#include "probe_vectors.h"
#include "Routing.h"
#include "PathConfig.h"

using namespace std;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────
// Constants & Formats (identical to Build.cpp)
// ─────────────────────────────────────────────────

static constexpr uint32_t DIMS          = 384;
static constexpr uint32_t NUM_SEGMENTS  = BitDB::NUM_SEGMENTS;
static constexpr uint32_t MAGIC         = 0x42444233u;
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

struct DocEntry {
    uint32_t doc_id;
    uint32_t page_count;
    uint64_t chunk_store_start;
    uint32_t chunk_count;
    uint32_t is_deleted;
    char     filename[232];
};
#pragma pack(pop)

// ─────────────────────────────────────────────────
// Candidate Hit for Bounded Top-K Min-Heap (Shortcoming 6)
// ─────────────────────────────────────────────────

struct CandidateHit {
    int32_t  score;
    uint32_t doc_id;
    uint32_t page_num;
    uint64_t text_offset;
    uint32_t text_length;

    // Min-heap ordering: lowest score at top of heap so it can be evicted
    bool operator>(const CandidateHit& other) const {
        return score > other.score;
    }
};

struct FinalResult {
    int32_t  score;
    uint32_t doc_id;
    uint32_t page_num;
    string   filename;
    string   passage;
};

// ─────────────────────────────────────────────────
// Fast int8 dot product
// ─────────────────────────────────────────────────

static inline int32_t dot_int8(const int8_t* a, const int8_t* b, uint32_t dims) {
    int32_t score = 0;
    for (uint32_t i = 0; i < dims; ++i) {
        score += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return score;
}

// ─────────────────────────────────────────────────
// Search Engine Session (Handles queries with warm model)
// ─────────────────────────────────────────────────

class SearchEngine {
private:
    SegEntry segDir[NUM_SEGMENTS] = {};
    vector<ExtentNode> extents;
    vector<DocEntry> catalog;
    unordered_map<uint32_t, string> docFilenames;
    unordered_set<uint32_t> tombstonedDocs;

    string chunkFile;
    string textFile;
    bool ready = false;

public:
    bool init() {
        const string segFile = PathConfig::getSegmentDirFile().string();
        const string extFile = PathConfig::getSegmentExtentsFile().string();
        const string catFile = PathConfig::getDocCatalogFile().string();
        chunkFile = PathConfig::getChunkStoreFile().string();
        textFile  = PathConfig::getPdfTextFile().string();

        // 1. Load segment directory (4KB, RAM-resident)
        ifstream segIn(segFile, ios::binary);
        if (!segIn) {
            cerr << "[Search] ERROR: segment_dir.bin not found. Run Build.exe first.\n";
            return false;
        }
        SegDirHeader hdr;
        segIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (hdr.magic != MAGIC || hdr.version != VERSION || hdr.num_segments != NUM_SEGMENTS) {
            cerr << "[Search] ERROR: incompatible segment index. Run Build.exe --rebuild.\n";
            return false;
        }
        segIn.read(reinterpret_cast<char*>(segDir), sizeof(segDir));
        segIn.close();

        // 2. Load extents if present
        if (fs::exists(extFile)) {
            ifstream extIn(extFile, ios::binary);
            if (extIn) {
                ExtentNode node;
                while (extIn.read(reinterpret_cast<char*>(&node), sizeof(node))) {
                    extents.push_back(node);
                }
            }
        }

        // 3. Load catalog
        ifstream catIn(catFile, ios::binary);
        if (catIn) {
            CatalogHeader catHdr;
            catIn.read(reinterpret_cast<char*>(&catHdr), sizeof(catHdr));
            catalog.resize(catHdr.num_docs);
            catIn.read(reinterpret_cast<char*>(catalog.data()), catHdr.num_docs * sizeof(DocEntry));
            for (const auto& de : catalog) {
                if (de.is_deleted) {
                    tombstonedDocs.insert(de.doc_id);
                } else {
                    docFilenames[de.doc_id] = string(de.filename);
                }
            }
        }

        ready = true;
        return true;
    }

    void executeQuery(const string& queryText, int topK = 5, size_t numProbes = 4) {
        if (!ready) return;

        auto tTotal0 = chrono::high_resolution_clock::now();

        // ─── Step 1: Embed query ───
        auto tEmb0 = chrono::high_resolution_clock::now();
        vector<vector<int8_t>> qEmbs = embed_chunks({queryText});
        auto tEmb1 = chrono::high_resolution_clock::now();
        if (qEmbs.empty() || qEmbs[0].size() != DIMS) {
            cerr << "[Search] ERROR: Failed to generate query embedding.\n";
            return;
        }
        const vector<int8_t>& queryVec = qEmbs[0];
        uint8_t queryCode[BitDB::BINARY_CODE_BYTES];
        BitDB::compute_binary_code(queryVec.data(), queryCode);
        double embedMs = chrono::duration<double, milli>(tEmb1 - tEmb0).count();

        // ─── Step 2: Compute 32-bit probe bitmask & margin-adaptive segments ───
        float margins[P3_NUM_PROBES] = {};
        uint32_t qMask = BitDB::compute_probe_bitmask_and_margins(queryVec.data(), margins);
        vector<uint32_t> segsToSearch = BitDB::get_adaptive_probe_segments(qMask, margins, numProbes);

        cout << "[Search] Probing " << segsToSearch.size() << " segment(s) via confidence margins (Primary Seg: "
             << segsToSearch[0] << ")\n";

        // ─── Step 3: Scan Extents using Single Bulk Reads & Bounded Min-Heap ───
        auto tDisk0 = chrono::high_resolution_clock::now();

        ifstream chunkIn(chunkFile, ios::binary);
        if (!chunkIn) {
            cerr << "[Search] ERROR: Cannot open chunk_store.bin\n";
            return;
        }

        // Bounded Top-K Min-Heap (Shortcoming 6: strictly O(K) memory)
        priority_queue<CandidateHit, vector<CandidateHit>, greater<CandidateHit>> minHeap;
        uint32_t totalScored = 0;
        uint32_t totalCandidates = 0;
        uint64_t totalBytesRead = 0;

        for (uint32_t segId : segsToSearch) {
            const SegEntry& seg = segDir[segId];
            if (seg.chunk_count == 0 && seg.ext_chain_head == 0) continue;

            // Collect all extents for this segment
            vector<pair<uint64_t, uint32_t>> segExtents;
            if (seg.chunk_count > 0) {
                segExtents.push_back({seg.chunk_store_offset, seg.chunk_count});
            }
            uint32_t extIdx = seg.ext_chain_head;
            while (extIdx > 0 && extIdx <= extents.size()) {
                const auto& node = extents[extIdx - 1];
                if (node.chunk_count > 0) {
                    segExtents.push_back({node.chunk_store_offset, node.chunk_count});
                }
                extIdx = node.next_extent_idx;
            }

            // Read bounded contiguous blocks. This preserves sequential I/O while
            // preventing a large extent from becoming a large RAM allocation.
            for (const auto& [offset, count] : segExtents) {
                totalCandidates += count;
                constexpr uint64_t BULK_BYTES = 1ull << 20;
                const uint32_t recordsPerBlock = std::max<uint32_t>(
                    1, static_cast<uint32_t>(BULK_BYTES / sizeof(ChunkRecord)));
                uint32_t processed = 0;
                while (processed < count) {
                    const uint32_t blockCount = std::min(recordsPerBlock, count - processed);
                    vector<ChunkRecord> buffer(blockCount);
                    const uint64_t blockOffset = offset + static_cast<uint64_t>(processed) * sizeof(ChunkRecord);
                    const size_t readBytes = static_cast<size_t>(blockCount) * sizeof(ChunkRecord);
                    chunkIn.seekg(static_cast<streamoff>(blockOffset), ios::beg);
                    chunkIn.read(reinterpret_cast<char*>(buffer.data()), static_cast<streamsize>(readBytes));
                    if (chunkIn.gcount() != static_cast<streamsize>(readBytes)) {
                        cerr << "[Search] ERROR: short read in chunk_store.bin.\n";
                        break;
                    }
                    totalBytesRead += readBytes;

                    // Two-stage filtering: compact binary code, then int8 dot product.
                    for (uint32_t i = 0; i < blockCount; ++i) {
                        const ChunkRecord& rec = buffer[i];

                    // Skip tombstoned documents
                    if (tombstonedDocs.count(rec.doc_id)) continue;

                        const uint32_t binaryDistance =
                            BitDB::hamming_distance_code(queryCode, rec.binary_code);
                        if (binaryDistance > 224) continue;

                    int32_t score = dot_int8(queryVec.data(), rec.embedding, DIMS);
                    totalScored++;

                    if (static_cast<int>(minHeap.size()) < topK) {
                        minHeap.push({score, rec.doc_id, rec.page_num, rec.text_offset, rec.text_length});
                    } else if (score > minHeap.top().score) {
                        minHeap.pop();
                        minHeap.push({score, rec.doc_id, rec.page_num, rec.text_offset, rec.text_length});
                    }
                    }
                    processed += blockCount;
                }
            }
        }
        chunkIn.close();
        auto tDisk1 = chrono::high_resolution_clock::now();
        double diskMs = chrono::duration<double, milli>(tDisk1 - tDisk0).count();

        // ─── Step 4: Extract Top-K from Heap in Descending Order ───
        vector<CandidateHit> topHits;
        while (!minHeap.empty()) {
            topHits.push_back(minHeap.top());
            minHeap.pop();
        }
        reverse(topHits.begin(), topHits.end());

        // ─── Step 5: Sparse seek into pdf_text.bin for winning Top-K passages only ───
        auto tText0 = chrono::high_resolution_clock::now();
        vector<FinalResult> results;
        results.reserve(topHits.size());

        ifstream textIn(textFile, ios::binary);
        for (const auto& hit : topHits) {
            FinalResult fr;
            fr.score   = hit.score;
            fr.doc_id  = hit.doc_id;
            fr.page_num = hit.page_num;
            auto it = docFilenames.find(hit.doc_id);
            fr.filename = (it != docFilenames.end()) ? it->second : "unknown.pdf";

            if (textIn && hit.text_length > 0) {
                textIn.seekg(static_cast<streamoff>(hit.text_offset), ios::beg);
                fr.passage.resize(hit.text_length, '\0');
                textIn.read(&fr.passage[0], hit.text_length);
            }
            results.push_back(fr);
        }
        textIn.close();
        auto tText1 = chrono::high_resolution_clock::now();
        double textMs = chrono::duration<double, milli>(tText1 - tText0).count();

        auto tTotal1 = chrono::high_resolution_clock::now();
        double totalMs = chrono::duration<double, milli>(tTotal1 - tTotal0).count();

        // ─── Step 6: Render Results ───
        cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
        cout << "  TOP " << results.size() << " RESULTS for: \"" << queryText << "\"\n";
        cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            cout << "  ┌─ Rank " << (i + 1) << " ─────────────────────────────────────────\n";
            cout << "  │  Score   : " << r.score << "\n";
            cout << "  │  File    : " << r.filename << "\n";
            cout << "  │  Page    : " << r.page_num << "\n";
            string snippet = r.passage.substr(0, 300);
            for (auto& c : snippet) if (c == '\r' || c == '\n') c = ' ';
            cout << "  │  Passage : \"" << snippet << (r.passage.size() > 300 ? "..." : "") << "\"\n";
            cout << "  └──────────────────────────────────────────────────────\n\n";
        }

        cout << "  ╔═ LATENCY & I/O PROFILE ══════════════════════════════╗\n";
        cout << "  ║  Query Embedding : " << fixed << setprecision(2) << embedMs  << " ms\n";
        cout << "  ║  SSD Extent Scan : " << diskMs   << " ms  (" << totalScored << " scored / " << totalCandidates << " candidates)\n";
        cout << "  ║  Bulk I/O Read   : " << (totalBytesRead / 1024.0) << " KB in " << segsToSearch.size() << " segments\n";
        cout << "  ║  Passage Fetch   : " << textMs   << " ms\n";
        cout << "  ║  Total Latency   : " << totalMs  << " ms\n";
        cout << "  ╚══════════════════════════════════════════════════════╝\n\n";
    }
};

// ─────────────────────────────────────────────────
// Main Entrypoint
// ─────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    ios::sync_with_stdio(true);

    bool interactive = false;
    string queryText = "";
    int topK = 5;
    size_t numProbes = 4;
    int positionalNumericArgs = 0;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--interactive" || arg == "--daemon" || arg == "-i") {
            interactive = true;
        } else if (arg == "--probes" && i + 1 < argc) {
            numProbes = static_cast<size_t>(std::clamp(stoi(argv[++i]), 1, P3_NUM_PROBES));
        } else if (queryText.empty() && arg[0] != '-') {
            queryText = arg;
        } else if (isdigit(arg[0])) {
            const int value = stoi(arg);
            if (positionalNumericArgs++ == 0) {
                topK = max(1, value);
            } else {
                numProbes = static_cast<size_t>(std::clamp(value, 1, P3_NUM_PROBES));
            }
        }
    }

    if (argc < 2 && !interactive) {
        interactive = true;
    }

    cout << "══════════════════════════════════════════════════\n";
    cout << "  BitDB Prototype-3 Search Engine\n";
    cout << "══════════════════════════════════════════════════\n";

    // ── Persistent Python & Model Loading (Shortcoming 9) ──
    cout << "[*] Initializing Python interpreter & hardware accelerator...\n";
    init_python();

    // Warm-up inference
    cout << "[*] Warming up model pipeline...\n";
    embed_chunks({"system warm up query"});
    cout << "[*] Model pre-warmed and ready in memory.\n";

    SearchEngine engine;
    if (!engine.init()) {
        finalize_python();
        return 1;
    }

    if (interactive) {
        cout << "\n[Interactive Daemon Mode Active] Type query and press Enter (or 'exit' to quit):\n";
        string line;
        while (true) {
            cout << "bitdb> ";
            if (!getline(cin, line)) break;
            if (line == "exit" || line == "quit" || line == "q") break;
            if (line.empty()) continue;

            engine.executeQuery(line, topK, numProbes);
        }
    } else {
        engine.executeQuery(queryText, topK, numProbes);
    }

    fflush(stdout);
    quick_exit(0);
}
