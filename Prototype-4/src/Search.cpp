// ════════════════════════════════════════════════════════════════════════
// Search.cpp — Prototype-4 High-Performance Columnar Vector Search Engine
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
static constexpr uint32_t MAGIC         = 0x42444234u;
static constexpr uint32_t VERSION       = 4;

#pragma pack(push, 1)
struct SegDirHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_segments;
    uint32_t reserved;
};

#pragma pack(push, 1)
struct SegEntry {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t ext_chain_head;
    float    centroid[DIMS];
    float    max_radius;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ExtentNode {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t next_extent_idx;
    float    centroid[DIMS];
    float    max_radius;
};
#pragma pack(pop)

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
static_assert(sizeof(ExtentHeader) == HEADER_BYTES, "ExtentHeader must be 512 bytes");

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
static_assert(sizeof(ChunkRecordMeta) == 28, "ChunkRecordMeta must be 28 bytes");

#pragma pack(push, 1)
struct ExtentBlock {
    ExtentHeader    header;
    uint8_t         binary_codes[EXTENT_CAPACITY][BitDB::BINARY_CODE_BYTES];
    int8_t          embeddings[EXTENT_CAPACITY][DIMS];
    ChunkRecordMeta metadata[EXTENT_CAPACITY];
    uint8_t         padding[PADDING_BYTES];
};
#pragma pack(pop)
static_assert(sizeof(ExtentBlock) == EXTENT_BYTES, "ExtentBlock must be exactly 128 KB");

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
// Candidate Hit for Bounded Top-K Min-Heap
// ─────────────────────────────────────────────────

struct CandidateHit {
    int32_t  score;
    uint32_t doc_id;
    uint32_t page_num;
    uint64_t text_offset;
    uint32_t text_length;

    CandidateHit(int32_t s, uint32_t d, uint32_t p, uint64_t to, uint32_t tl)
        : score(s), doc_id(d), page_num(p), text_offset(to), text_length(tl) {}

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
    uint8_t mih_tables[4][256][32] = {};
    bool hasMih = false;

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

        // 1. Load segment directory
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

        // 4. Load MIH tables if present
        const string mihFile = PathConfig::getMihTableFile().string();
        if (fs::exists(mihFile)) {
            ifstream mihIn(mihFile, ios::binary);
            if (mihIn) {
                mihIn.read(reinterpret_cast<char*>(mih_tables), sizeof(mih_tables));
                hasMih = true;
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

        // ─── Step 2: Probe Candidate Segments via Multi-Index Hashing (MIH) ───
        uint32_t qMask = BitDB::compute_probe_bitmask(queryVec.data());
        vector<uint32_t> segsToSearch;

        if (hasMih) {
            uint8_t seg_mask[32] = {};
            for (int m = 0; m < 4; ++m) {
                uint8_t sub_val = static_cast<uint8_t>((qMask >> (m * 8)) & 0xFF);
                for (int b = 0; b < 32; ++b) {
                    seg_mask[b] |= mih_tables[m][sub_val][b];
                }
                for (int bit = 0; bit < 8; ++bit) {
                    uint8_t flip_val = sub_val ^ static_cast<uint8_t>(1u << bit);
                    for (int b = 0; b < 32; ++b) {
                        seg_mask[b] |= mih_tables[m][flip_val][b];
                    }
                }
            }
            for (uint32_t seg = 0; seg < NUM_SEGMENTS; ++seg) {
                if (seg_mask[seg / 8] & (1u << (seg % 8))) {
                    segsToSearch.push_back(seg);
                }
            }
        }

        if (segsToSearch.empty()) {
            uint32_t baseSeg = BitDB::signature_to_segment(qMask);
            segsToSearch.push_back(baseSeg);
        }

        cout << "[Search] Probing " << segsToSearch.size() << " segment(s) via MIH routing\n";

        // ─── Step 3: Scan Extents using 2-Stage Asymmetric I/O & Bounded Min-Heap ───
        auto tDisk0 = chrono::high_resolution_clock::now();

        ifstream chunkIn(chunkFile, ios::binary);
        if (!chunkIn) {
            cerr << "[Search] ERROR: Cannot open chunk_store.bin\n";
            return;
        }

        priority_queue<CandidateHit, vector<CandidateHit>, greater<CandidateHit>> minHeap;
        uint32_t totalScored = 0;
        uint32_t totalCandidates = 0;
        uint64_t totalBytesRead = 0;

        float query_len = 0.0f;
        for (int d = 0; d < DIMS; ++d) {
            query_len += static_cast<float>(queryVec[d]) * static_cast<float>(queryVec[d]);
        }
        query_len = sqrt(query_len);

        for (uint32_t segId : segsToSearch) {
            const SegEntry& seg = segDir[segId];
            if (seg.chunk_count == 0 && seg.ext_chain_head == 0) continue;

            vector<ExtentNode> segExtents;
            if (seg.chunk_count > 0) {
                ExtentNode node = {};
                node.chunk_store_offset = seg.chunk_store_offset;
                node.chunk_count = seg.chunk_count;
                memcpy(node.centroid, seg.centroid, sizeof(float)*DIMS);
                node.max_radius = seg.max_radius;
                segExtents.push_back(node);
            }
            uint32_t extIdx = seg.ext_chain_head;
            while (extIdx > 0 && extIdx <= extents.size()) {
                segExtents.push_back(extents[extIdx - 1]);
                extIdx = extents[extIdx - 1].next_extent_idx;
            }

            for (const auto& node : segExtents) {
                // Evaluate WAND Bound: (q · C) + ||q|| * R
                float q_dot_c = 0.0f;
                for (int d = 0; d < DIMS; ++d) {
                    q_dot_c += static_cast<float>(queryVec[d]) * node.centroid[d];
                }
                float max_score_bound = q_dot_c + query_len * node.max_radius;
                
                // Safe Early-Exit Pruning
                if (minHeap.size() == static_cast<size_t>(topK) && max_score_bound < static_cast<float>(minHeap.top().score)) {
                    continue;
                }

                // Stage 1: Read Header & Contiguous Binary Codes (~14 KB)
                ExtentHeader hdr;
                uint8_t extent_codes[EXTENT_CAPACITY][BitDB::BINARY_CODE_BYTES];

                chunkIn.seekg(static_cast<streamoff>(node.chunk_store_offset), ios::beg);
                chunkIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
                chunkIn.read(reinterpret_cast<char*>(extent_codes), hdr.record_count * BitDB::BINARY_CODE_BYTES);
                totalBytesRead += sizeof(hdr) + (hdr.record_count * BitDB::BINARY_CODE_BYTES);
                totalCandidates += hdr.record_count;

                if (hdr.record_count == 0) continue;

                // Stage 1 Filter: Asymmetric Distance Computation (ADC)
                struct CandScore {
                    uint32_t idx;
                    int32_t adc_score;
                };
                vector<CandScore> passed;
                passed.reserve(hdr.record_count);

                for (uint32_t i = 0; i < hdr.record_count; ++i) {
                    int32_t score = BitDB::adc_score(queryVec.data(), extent_codes[i]);
                    passed.push_back({i, score});
                }

                // Dynamic Survival Quotas: rank candidates by ADC score and keep top 15%
                sort(passed.begin(), passed.end(), [](const CandScore& a, const CandScore& b) {
                    return a.adc_score > b.adc_score;
                });
                size_t quota = max<size_t>(8, static_cast<size_t>(ceil(hdr.record_count * 0.15)));
                quota = min(quota, passed.size());

                // Hardware I/O Crossover:
                // If surviving candidate count exceeds 40 records (~15% of extent),
                // sequential bulk read of the remaining extent is faster than scattered seeks.
                constexpr size_t CROSSOVER_THRESHOLD = 40;

                if (quota > CROSSOVER_THRESHOLD) {
                    // Bulk sequential read path
                    uint64_t bulk_offset = node.chunk_store_offset + STAGE1_BYTES;
                    size_t remaining_bytes = EXTENT_BYTES - STAGE1_BYTES;
                    vector<char> bulk_buf(remaining_bytes);
                    chunkIn.seekg(static_cast<streamoff>(bulk_offset), ios::beg);
                    chunkIn.read(bulk_buf.data(), remaining_bytes);
                    totalBytesRead += remaining_bytes;

                    const int8_t (*embeddings)[DIMS] = reinterpret_cast<const int8_t(*)[DIMS]>(bulk_buf.data());
                    const ChunkRecordMeta* meta = reinterpret_cast<const ChunkRecordMeta*>(bulk_buf.data() + EMBEDDINGS_BYTES);

                    for (size_t c = 0; c < quota; ++c) {
                        uint32_t idx = passed[c].idx;
                        if (tombstonedDocs.count(meta[idx].doc_id)) continue;
                        int32_t score = dot_int8(queryVec.data(), embeddings[idx], DIMS);
                        totalScored++;

                        if (minHeap.size() < static_cast<size_t>(topK)) {
                            minHeap.push(CandidateHit(score, meta[idx].doc_id, meta[idx].page_num, meta[idx].text_offset, meta[idx].text_length));
                        } else if (score > minHeap.top().score) {
                            minHeap.pop();
                            minHeap.push(CandidateHit(score, meta[idx].doc_id, meta[idx].page_num, meta[idx].text_offset, meta[idx].text_length));
                        }
                    }
                } else {
                    // Scattered payload read path: read only surviving candidates' int8 embeddings and metadata
                    for (size_t c = 0; c < quota; ++c) {
                        uint32_t idx = passed[c].idx;
                        int8_t emb[DIMS];
                        ChunkRecordMeta meta;

                        uint64_t emb_offset = node.chunk_store_offset + STAGE1_BYTES + (idx * DIMS);
                        chunkIn.seekg(static_cast<streamoff>(emb_offset), ios::beg);
                        chunkIn.read(reinterpret_cast<char*>(emb), DIMS);

                        uint64_t meta_offset = node.chunk_store_offset + STAGE1_BYTES + EMBEDDINGS_BYTES + (idx * sizeof(ChunkRecordMeta));
                        chunkIn.seekg(static_cast<streamoff>(meta_offset), ios::beg);
                        chunkIn.read(reinterpret_cast<char*>(&meta), sizeof(meta));
                        totalBytesRead += DIMS + sizeof(meta);

                        if (tombstonedDocs.count(meta.doc_id)) continue;
                        int32_t score = dot_int8(queryVec.data(), emb, DIMS);
                        totalScored++;

                        if (minHeap.size() < static_cast<size_t>(topK)) {
                            minHeap.push(CandidateHit(score, meta.doc_id, meta.page_num, meta.text_offset, meta.text_length));
                        } else if (score > minHeap.top().score) {
                            minHeap.pop();
                            minHeap.push(CandidateHit(score, meta.doc_id, meta.page_num, meta.text_offset, meta.text_length));
                        }
                    }
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
    cout << "  BitDB Prototype-4 Columnar Search Engine\n";
    cout << "══════════════════════════════════════════════════\n";

    // ── Persistent Python & Model Loading ──
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
