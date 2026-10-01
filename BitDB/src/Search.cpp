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

#ifdef _WIN32
#include <windows.h>
#endif

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
    uint64_t file_size;
    uint64_t last_write_time;
    char     filename[216];
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
    float    cosine_sim;
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
        const string manifestFile = PathConfig::getManifestFile().string();
        chunkFile = PathConfig::getChunkStoreFile().string();
        textFile  = PathConfig::getPdfTextFile().string();

        // 0. Verify manifest if present
        if (fs::exists(manifestFile)) {
            ifstream manIn(manifestFile, ios::binary);
            if (manIn) {
                PathConfig::StorageManifest man;
                if (manIn.read(reinterpret_cast<char*>(&man), sizeof(man)) && manIn.gcount() == sizeof(man)) {
                    if (!man.is_valid()) {
                        cerr << "[Search] WARNING: Storage manifest verification failed (checksum/version mismatch). Rebuilding is recommended.\n";
                    }
                }
            }
        }

        // 1. Load segment directory
        ifstream segIn(segFile, ios::binary);
        if (!segIn) {
            cerr << "[Search] ERROR: segment_dir.bin not found. Run Build.exe first.\n";
            return false;
        }
        SegDirHeader hdr;
        if (!segIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)) || segIn.gcount() != sizeof(hdr)) {
            cerr << "[Search] ERROR: Failed to read SegDirHeader from " << segFile << "\n";
            return false;
        }
        if (hdr.magic != MAGIC || hdr.version != VERSION || hdr.num_segments != NUM_SEGMENTS) {
            cerr << "[Search] ERROR: incompatible segment index. Run Build.exe --rebuild.\n";
            return false;
        }
        if (!segIn.read(reinterpret_cast<char*>(segDir), sizeof(segDir)) || segIn.gcount() != sizeof(segDir)) {
            cerr << "[Search] ERROR: Incomplete read of segment directory data from " << segFile << "\n";
            return false;
        }
        segIn.close();

        // 2. Load extents if present
        if (fs::exists(extFile)) {
            ifstream extIn(extFile, ios::binary);
            if (extIn) {
                ExtentNode node;
                while (extIn.read(reinterpret_cast<char*>(&node), sizeof(node))) {
                    if (extIn.gcount() == sizeof(node)) {
                        extents.push_back(node);
                    }
                }
            }
        }

        // 3. Load catalog
        ifstream catIn(catFile, ios::binary);
        if (catIn) {
            CatalogHeader catHdr;
            if (catIn.read(reinterpret_cast<char*>(&catHdr), sizeof(catHdr)) && catIn.gcount() == sizeof(catHdr)) {
                catalog.resize(catHdr.num_docs);
                size_t expectedBytes = catHdr.num_docs * sizeof(DocEntry);
                if (catIn.read(reinterpret_cast<char*>(catalog.data()), expectedBytes) && static_cast<size_t>(catIn.gcount()) == expectedBytes) {
                    for (const auto& de : catalog) {
                        if (de.is_deleted) {
                            tombstonedDocs.insert(de.doc_id);
                        } else {
                            docFilenames[de.doc_id] = string(de.filename);
                        }
                    }
                } else {
                    cerr << "[Search] ERROR: Incomplete read of doc_catalog entries in " << catFile << "\n";
                    return false;
                }
            }
        }

        // 4. Load MIH tables if present
        const string mihFile = PathConfig::getMihTableFile().string();
        if (fs::exists(mihFile)) {
            ifstream mihIn(mihFile, ios::binary);
            if (mihIn) {
                if (mihIn.read(reinterpret_cast<char*>(mih_tables), sizeof(mih_tables)) && mihIn.gcount() == sizeof(mih_tables)) {
                    hasMih = true;
                }
            }
        }

        ready = true;
        return true;
    }

    bool initCatalogOnly() {
        const string catFile = PathConfig::getDocCatalogFile().string();
        ifstream catIn(catFile, ios::binary);
        if (!catIn) {
            cerr << "[Search] ERROR: doc_catalog.bin not found at: " << catFile << "\n";
            cerr << "         Please run Build.exe to index documents.\n";
            return false;
        }
        CatalogHeader catHdr;
        if (!catIn.read(reinterpret_cast<char*>(&catHdr), sizeof(catHdr)) || catIn.gcount() != sizeof(catHdr)) {
            cerr << "[Search] ERROR: Failed to read CatalogHeader from " << catFile << "\n";
            return false;
        }
        catalog.resize(catHdr.num_docs);
        size_t expectedBytes = catHdr.num_docs * sizeof(DocEntry);
        if (!catIn.read(reinterpret_cast<char*>(catalog.data()), expectedBytes) || static_cast<size_t>(catIn.gcount()) != expectedBytes) {
            cerr << "[Search] ERROR: Incomplete read of doc_catalog entries from " << catFile << "\n";
            return false;
        }
        for (const auto& de : catalog) {
            if (de.is_deleted) {
                tombstonedDocs.insert(de.doc_id);
            } else {
                docFilenames[de.doc_id] = string(de.filename);
            }
        }
        return true;
    }

    void listFiles(bool showDeleted = false) const {
        uint32_t activeCount = 0;
        uint32_t totalChunks = 0;
        uint32_t totalPages  = 0;
        for (const auto& de : catalog) {
            if (!de.is_deleted) {
                activeCount++;
                totalChunks += de.chunk_count;
                totalPages += de.page_count;
            }
        }

        cout << "\n========================================================================================\n";
        cout << "  BITDB CATALOG: " << activeCount << " active document(s) (" << totalPages << " pages, " << totalChunks << " vector chunks)\n";
        cout << "========================================================================================\n";
        cout << "  Doc ID | Pages | Chunks | Status    | Filename\n";
        cout << "---------|-------|--------|-----------|-------------------------------------------------\n";

        if (catalog.empty()) {
            cout << "  (Database is empty. Drop PDFs into ./ingestor and run Build.exe)\n";
        } else {
            bool anyShown = false;
            for (const auto& de : catalog) {
                if (de.is_deleted && !showDeleted) continue;
                anyShown = true;
                printf("    %4u | %5u | %6u | %-9s | %s\n",
                       de.doc_id, de.page_count, de.chunk_count,
                       (de.is_deleted ? "DELETED" : "ACTIVE"),
                       de.filename);
            }
            if (!anyShown) {
                cout << "  (No active documents found. Run Build.exe to re-index)\n";
            }
        }
        cout << "========================================================================================\n\n";
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

        float query_len = 0.0f;
        for (uint32_t d = 0; d < DIMS; ++d) {
            query_len += static_cast<float>(queryVec[d]) * static_cast<float>(queryVec[d]);
        }
        query_len = sqrt(query_len);

        // ─── Step 2: Multi-Probe Primary Segment Ranking (numProbes = number of segment IDs) ───
        // In Prototype-4, each document is assigned to one of 256 primary Voronoi segment partitions
        // determined by the first 8 bits of the 32-bit signature (signature_to_segment = mask & 0xFF).
        // Multi-probe routing ranks the 8 segment-routing hyperplane margins and visits adjacent
        // 1-bit and 2-bit neighbor segment IDs up to numProbes partitions.
        // NOTE: numProbes specifies the count of partition segment IDs probed, not 32 signature dimensions.
        //
        // numProbes=1  → exact primary segment only (fastest, lowest recall)
        // numProbes=8+ → primary segment + top uncertain 1-bit and 2-bit neighbor segment IDs

        float margins[P3_NUM_PROBES];
        uint32_t qMask = BitDB::compute_probe_bitmask_and_margins(queryVec.data(), margins);
        vector<uint32_t> segsToSearch;

        // Always add the primary exact-match segment
        {
            uint32_t baseSeg = BitDB::signature_to_segment(qMask);
            segsToSearch.push_back(baseSeg);
        }

        if (numProbes > 1) {
            // Rank the 8 segment-defining hyperplanes (bits 0..7) by ascending confidence margin
            // (smallest margin = closest to Voronoi decision boundary = most likely neighbor)
            struct ProbeMargin {
                int   bit;
                float margin;
            };
            vector<ProbeMargin> ranked;
            ranked.reserve(8);
            for (int i = 0; i < 8; ++i) {
                ranked.push_back({i, margins[i]});
            }
            std::sort(ranked.begin(), ranked.end(),
                [](const ProbeMargin& a, const ProbeMargin& b) { return a.margin < b.margin; });

            // 1-bit flips in order of lowest margin (1-Hamming distance adjacent Voronoi cells)
            for (size_t i = 0; i < ranked.size() && segsToSearch.size() < numProbes; ++i) {
                uint32_t flippedMask = qMask ^ (1u << ranked[i].bit);
                uint32_t neighborSeg = BitDB::signature_to_segment(flippedMask);
                if (neighborSeg < NUM_SEGMENTS && (segDir[neighborSeg].chunk_count > 0 || segDir[neighborSeg].ext_chain_head > 0)) {
                    if (std::find(segsToSearch.begin(), segsToSearch.end(), neighborSeg) == segsToSearch.end()) {
                        segsToSearch.push_back(neighborSeg);
                    }
                }
            }

            // 2-bit flips on top uncertain bits if more probes requested (diagonal Voronoi cells)
            if (segsToSearch.size() < numProbes) {
                for (size_t i = 0; i < ranked.size() && segsToSearch.size() < numProbes; ++i) {
                    for (size_t j = i + 1; j < ranked.size() && segsToSearch.size() < numProbes; ++j) {
                        uint32_t flippedMask = qMask ^ (1u << ranked[i].bit) ^ (1u << ranked[j].bit);
                        uint32_t neighborSeg = BitDB::signature_to_segment(flippedMask);
                        if (neighborSeg < NUM_SEGMENTS && (segDir[neighborSeg].chunk_count > 0 || segDir[neighborSeg].ext_chain_head > 0)) {
                            if (std::find(segsToSearch.begin(), segsToSearch.end(), neighborSeg) == segsToSearch.end()) {
                                segsToSearch.push_back(neighborSeg);
                            }
                        }
                    }
                }
            }

            // Fallback for remaining probe slots: rank unvisited segments by geometric centroid proximity in R^384
            if (segsToSearch.size() < numProbes) {
                struct SegGeoRank {
                    float score_bound;
                    uint32_t seg_id;
                };
                vector<SegGeoRank> otherSegs;
                for (uint32_t s = 0; s < NUM_SEGMENTS; ++s) {
                    if (segDir[s].chunk_count > 0 || segDir[s].ext_chain_head > 0) {
                        if (std::find(segsToSearch.begin(), segsToSearch.end(), s) == segsToSearch.end()) {
                            float s_q_dot_c = 0.0f;
                            for (uint32_t d = 0; d < DIMS; ++d) {
                                s_q_dot_c += (static_cast<float>(queryVec[d]) / 127.0f) * segDir[s].centroid[d];
                            }
                            float s_bound = s_q_dot_c + (query_len / 127.0f) * segDir[s].max_radius;
                            otherSegs.push_back({s_bound, s});
                        }
                    }
                }
                std::sort(otherSegs.begin(), otherSegs.end(),
                    [](const SegGeoRank& a, const SegGeoRank& b) { return a.score_bound > b.score_bound; });
                for (const auto& item : otherSegs) {
                    if (segsToSearch.size() >= numProbes) break;
                    segsToSearch.push_back(item.seg_id);
                }
            }
        }

        cout << "[Search] Probing " << segsToSearch.size() << " segment partition(s) (numProbes=" << numProbes << " segment IDs): [";
        for (size_t s_i = 0; s_i < segsToSearch.size(); ++s_i) {
            cout << (s_i > 0 ? ", " : "") << segsToSearch[s_i];
        }
        cout << "] via margin-ranked Voronoi routing\n";



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
                // Exact Cauchy-Schwarz WAND Bound: MaxScore(q, E) = (q · C) + ||q|| * R
                // Mathematically rigorous upper bound on maximum possible inner product in extent.
                float q_dot_c = 0.0f;
                for (int d = 0; d < DIMS; ++d) {
                    q_dot_c += static_cast<float>(queryVec[d]) * node.centroid[d];
                }
                
                float max_score_bound = q_dot_c + query_len * node.max_radius;

                // Safe Early-Exit Pruning (Zero I/O if bound cannot exceed current minHeap threshold)
                if (minHeap.size() == static_cast<size_t>(topK) && max_score_bound < static_cast<float>(minHeap.top().score)) {
                    totalCandidates += node.chunk_count; // Tally bypassed candidates for telemetry
                    continue;
                }

                // Stage 1: Read Header & Contiguous Binary Codes (~14 KB)
                ExtentHeader hdr;
                uint8_t extent_codes[EXTENT_CAPACITY][BitDB::BINARY_CODE_BYTES];

                // ── Extent Integrity Validation ──
                chunkIn.seekg(0, ios::end);
                uint64_t fileSize = static_cast<uint64_t>(chunkIn.tellg());
                uint64_t extentEnd = node.chunk_store_offset + EXTENT_BYTES;
                if (extentEnd > fileSize) {
                    cerr << "[Search] WARNING: Corrupt extent at offset " << node.chunk_store_offset
                         << " extends beyond file size (" << fileSize << " bytes). Skipping.\n";
                    totalCandidates += node.chunk_count;
                    continue;
                }

                chunkIn.seekg(static_cast<streamoff>(node.chunk_store_offset), ios::beg);
                if (!chunkIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)) || chunkIn.gcount() != sizeof(hdr)) {
                    cerr << "[Search] WARNING: Failed to read extent header at offset "
                         << node.chunk_store_offset << ". Skipping.\n";
                    totalCandidates += node.chunk_count;
                    continue;
                }

                // Guard against corrupt record_count causing buffer overflow into extent_codes.
                if (hdr.record_count == 0) continue;
                if (hdr.record_count > EXTENT_CAPACITY) {
                    cerr << "[Search] WARNING: Corrupt extent header at offset " << node.chunk_store_offset
                         << ": record_count=" << hdr.record_count
                         << " exceeds EXTENT_CAPACITY (" << EXTENT_CAPACITY << "). Skipping.\n";
                    totalCandidates += node.chunk_count;
                    continue;
                }

                size_t codes_bytes = hdr.record_count * BitDB::BINARY_CODE_BYTES;
                if (!chunkIn.read(reinterpret_cast<char*>(extent_codes), codes_bytes) || static_cast<size_t>(chunkIn.gcount()) != codes_bytes) {
                    cerr << "[Search] WARNING: Failed to read binary codes at offset "
                         << node.chunk_store_offset << ". Skipping.\n";
                    totalCandidates += node.chunk_count;
                    continue;
                }
                totalBytesRead += sizeof(hdr) + codes_bytes;
                totalCandidates += hdr.record_count;

                // ── Stage 1: ADC pre-sort for I/O optimisation ONLY ──
                struct CandScore {
                    uint32_t idx;
                    int32_t  adc_score;
                };
                vector<CandScore> allCands;
                allCands.reserve(hdr.record_count);
                for (uint32_t i = 0; i < hdr.record_count; ++i) {
                    allCands.push_back({i, BitDB::adc_score(queryVec.data(), extent_codes[i])});
                }
                std::sort(allCands.begin(), allCands.end(),
                    [](const CandScore& a, const CandScore& b) { return a.adc_score > b.adc_score; });

                // ── I/O Crossover Heuristic ──
                constexpr size_t CROSSOVER_THRESHOLD = 40;

                if (hdr.record_count > CROSSOVER_THRESHOLD) {
                    // Bulk sequential read of int8 embeddings + metadata
                    uint64_t bulk_offset = node.chunk_store_offset + STAGE1_BYTES;
                    size_t remaining_bytes = EXTENT_BYTES - STAGE1_BYTES;
                    vector<char> bulk_buf(remaining_bytes);
                    chunkIn.seekg(static_cast<streamoff>(bulk_offset), ios::beg);
                    if (!chunkIn.read(bulk_buf.data(), remaining_bytes) || static_cast<size_t>(chunkIn.gcount()) != remaining_bytes) {
                        cerr << "[Search] WARNING: Incomplete read of bulk extent payload at offset "
                             << bulk_offset << ". Skipping extent.\n";
                        continue;
                    }
                    totalBytesRead += remaining_bytes;

                    const int8_t (*embeddings)[DIMS] = reinterpret_cast<const int8_t(*)[DIMS]>(bulk_buf.data());
                    const ChunkRecordMeta* meta = reinterpret_cast<const ChunkRecordMeta*>(bulk_buf.data() + EMBEDDINGS_BYTES);

                    // Score ALL candidates (ADC order ensures heap fills fast)
                    for (const auto& cand : allCands) {
                        uint32_t idx = cand.idx;
                        if (tombstonedDocs.count(meta[idx].doc_id)) continue;
                        int32_t score = dot_int8(queryVec.data(), embeddings[idx], DIMS);
                        totalScored++;

                        if (minHeap.size() < static_cast<size_t>(topK)) {
                            minHeap.push(CandidateHit(score, meta[idx].doc_id, meta[idx].page_num,
                                                      meta[idx].text_offset, meta[idx].text_length));
                        } else if (score > minHeap.top().score) {
                            minHeap.pop();
                            minHeap.push(CandidateHit(score, meta[idx].doc_id, meta[idx].page_num,
                                                      meta[idx].text_offset, meta[idx].text_length));
                        }
                    }
                } else {
                    // Scattered reads for small extents — read each record's embedding + meta individually
                    for (const auto& cand : allCands) {
                        uint32_t idx = cand.idx;
                        int8_t emb[DIMS];
                        ChunkRecordMeta meta;

                        uint64_t emb_offset  = node.chunk_store_offset + STAGE1_BYTES + (uint64_t(idx) * DIMS);
                        uint64_t meta_offset = node.chunk_store_offset + STAGE1_BYTES + EMBEDDINGS_BYTES + (uint64_t(idx) * sizeof(ChunkRecordMeta));

                        chunkIn.seekg(static_cast<streamoff>(emb_offset), ios::beg);
                        if (!chunkIn.read(reinterpret_cast<char*>(emb), DIMS) || chunkIn.gcount() != DIMS) {
                            continue;
                        }
                        chunkIn.seekg(static_cast<streamoff>(meta_offset), ios::beg);
                        if (!chunkIn.read(reinterpret_cast<char*>(&meta), sizeof(meta)) || chunkIn.gcount() != sizeof(meta)) {
                            continue;
                        }
                        totalBytesRead += DIMS + sizeof(ChunkRecordMeta);

                        if (tombstonedDocs.count(meta.doc_id)) continue;
                        int32_t score = dot_int8(queryVec.data(), emb, DIMS);
                        totalScored++;

                        if (minHeap.size() < static_cast<size_t>(topK)) {
                            minHeap.push(CandidateHit(score, meta.doc_id, meta.page_num,
                                                      meta.text_offset, meta.text_length));
                        } else if (score > minHeap.top().score) {
                            minHeap.pop();
                            minHeap.push(CandidateHit(score, meta.doc_id, meta.page_num,
                                                      meta.text_offset, meta.text_length));
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
            fr.score      = hit.score;
            fr.cosine_sim = query_len > 1e-6f ? std::clamp(static_cast<float>(hit.score) / (query_len * 127.0f), -1.0f, 1.0f) : 0.0f;
            fr.doc_id     = hit.doc_id;
            fr.page_num   = hit.page_num;
            auto it = docFilenames.find(hit.doc_id);
            fr.filename = (it != docFilenames.end()) ? it->second : "unknown.pdf";

            if (textIn && hit.text_length > 0) {
                textIn.seekg(static_cast<streamoff>(hit.text_offset), ios::beg);
                fr.passage.resize(hit.text_length, '\0');
                if (textIn.read(&fr.passage[0], hit.text_length)) {
                    fr.passage.resize(static_cast<size_t>(textIn.gcount()));
                } else {
                    fr.passage.clear();
                }
            }
            results.push_back(fr);
        }
        textIn.close();
        auto tText1 = chrono::high_resolution_clock::now();
        double textMs = chrono::duration<double, milli>(tText1 - tText0).count();

        auto tTotal1 = chrono::high_resolution_clock::now();
        double totalMs = chrono::duration<double, milli>(tTotal1 - tTotal0).count();

        // --- Step 6: Render Results ---
        cout << "\n==================================================================\n";
        cout << "  TOP " << results.size() << " RESULTS for: \"" << queryText << "\"\n";
        cout << "==================================================================\n\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            cout << "  +-- Rank " << (i + 1) << " -----------------------------------------\n";
            cout << "  |  Score   : " << r.score << " (Cosine Sim: " << fixed << setprecision(4) << r.cosine_sim << ")\n";
            cout << "  |  File    : " << r.filename << "\n";
            cout << "  |  Page    : " << r.page_num << "\n";
            string snippet = r.passage;
            for (auto& c : snippet) if (c == '\r' || c == '\n') c = ' ';
            // Remove hardcoded 300 char limit to display full multi-sentence chunks
            if (snippet.size() > 1000) {
                snippet = snippet.substr(0, 1000) + "...";
            }
            cout << "  |  Passage : \"" << snippet << "\"\n";
            cout << "  +------------------------------------------------------\n\n";
        }

        cout << "  +-- LATENCY & I/O PROFILE ----------------------------+\n";
        cout << "  |  Query Embedding : " << fixed << setprecision(2) << embedMs << " ms\n";
        cout << "  |  Segments Probed : " << segsToSearch.size() << " (numProbes=" << numProbes << " segment IDs)\n";
        cout << "  |  Records Scored  : " << totalScored << " / " << totalCandidates << " candidates in probed extents\n";
        cout << "  |  SSD Extent Scan : " << diskMs << " ms\n";
        cout << "  |  Bulk I/O Read   : " << (totalBytesRead / 1024.0) << " KB\n";
        cout << "  |  Passage Fetch   : " << textMs << " ms\n";
        cout << "  |  Total Latency   : " << totalMs << " ms\n";
        cout << "  +-----------------------------------------------------+\n\n";
    }
};

// ─────────────────────────────────────────────────
// Main Entrypoint
// ─────────────────────────────────────────────────

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    ios::sync_with_stdio(true);

    bool interactive = false;
    bool listOnly = false;
    bool showDeleted = false;
    string queryText = "";
    int topK = 5;
    size_t numProbes = 4;
    int positionalNumericArgs = 0;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cout << "==================================================\n";
            cout << "  BitDB Prototype-4 Columnar Search Engine\n";
            cout << "==================================================\n";
            cout << "USAGE:\n";
            cout << "  BitDBSearch.exe \"<query>\" [topK] [numProbes]\n";
            cout << "  BitDBSearch.exe --list [--all]\n";
            cout << "  BitDBSearch.exe --interactive [--probes N]\n\n";
            cout << "OPTIONS:\n";
            cout << "  --help, -h          Show this help message and exit\n";
            cout << "  --list, -l, --files List all documents and files stored in the database\n";
            cout << "  --all, -a           Include tombstoned/deleted files when listing\n";
            cout << "  --interactive, -i   Launch in persistent REPL mode\n";
            cout << "  --probes N          Number of partition segment IDs to probe (default 4)\n";
            cout << "                      Specifies count of Voronoi segment IDs (0..255) to search\n";
            cout << "                      via margin ranking of the 8 segment hyperplane bits.\n";
            cout << "                      (Note: specifies number of segment IDs, not 32 signature dimensions)\n\n";
            cout << "REPL COMMANDS (Interactive Mode Only):\n";
            cout << "  list, files, ls     List all documents stored in the database\n";
            cout << "  list --all          List all documents including deleted/tombstoned ones\n";
            cout << "  --K [number]        Append to any query to override the top-K limit for that search\n";
            cout << "                      Example: bitdb> flash memory buffer management --K 10\n";
            cout << "  exit, quit, q       Exit interactive mode\n\n";
            cout << "EXAMPLES:\n";
            cout << "  BitDBSearch.exe --list\n";
            cout << "  BitDBSearch.exe \"flash memory buffer management\" 10 8\n";
            cout << "  BitDBSearch.exe --interactive\n";
            cout << "==================================================\n";
            return 0;
        } else if (arg == "--list" || arg == "-l" || arg == "--files" || arg == "list" || arg == "files") {
            listOnly = true;
        } else if (arg == "--all" || arg == "-a") {
            showDeleted = true;
        } else if (arg == "--interactive" || arg == "--daemon" || arg == "-i") {
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

    // Fast-path: List stored database files without loading heavy Python or neural model weights
    if (listOnly) {
        SearchEngine engine;
        if (!engine.initCatalogOnly()) {
            return 1;
        }
        engine.listFiles(showDeleted);
        return 0;
    }

    if (argc < 2 && !interactive) {
        interactive = true;
    }

    cout << "==================================================\n";
    cout << "  BitDB Prototype-4 Columnar Search Engine\n";
    cout << "==================================================\n";

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
        cout << "\n[Interactive Daemon Mode Active] Type query, 'list' for stored files, or 'exit' to quit:\n";
        string line;
        while (true) {
            cout << "bitdb> ";
            if (!getline(cin, line)) break;
            if (line == "exit" || line == "quit" || line == "q") break;
            if (line.empty()) continue;

            // Interactive list command
            if (line == "list" || line == "files" || line == "--list" || line == "--files" || line == "ls") {
                engine.listFiles(false);
                continue;
            }
            if (line == "list --all" || line == "files --all" || line == "ls -a" || line == "ls --all") {
                engine.listFiles(true);
                continue;
            }

            int currentTopK = topK;
            size_t kPos = line.rfind("--K ");
            if (kPos != string::npos) {
                try {
                    currentTopK = stoi(line.substr(kPos + 4));
                    line = line.substr(0, kPos);
                    while (!line.empty() && line.back() == ' ') line.pop_back();
                } catch (...) {}
            }

            engine.executeQuery(line, currentTopK, numProbes);
        }
    } else {
        engine.executeQuery(queryText, topK, numProbes);
    }

    fflush(stdout);
    quick_exit(0);
}
