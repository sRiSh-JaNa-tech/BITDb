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

    bool initCatalogOnly() {
        const string catFile = PathConfig::getDocCatalogFile().string();
        ifstream catIn(catFile, ios::binary);
        if (!catIn) {
            cerr << "[Search] ERROR: doc_catalog.bin not found at: " << catFile << "\n";
            cerr << "         Please run Build.exe to index documents.\n";
            return false;
        }
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

        // ─── Step 2: Probe Candidate Segments via Multi-Index Hashing (MIH) ───
        //
        // numProbes controls how many additional Hamming-1 neighbours are included
        // beyond the exact-match primary segment. Probe bits are ranked by their
        // normalised margin (|q · probe_i|) — bits close to the decision boundary
        // are most likely to cross it and should be probed first.
        //
        // numProbes=1  → exact segment only (fastest, lowest recall)
        // numProbes=32 → all 32 probe dimensions flipped (slowest, highest recall)

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

            // Fallback for remaining slots if empty segments were skipped
            if (segsToSearch.size() < numProbes) {
                uint32_t baseSeg = segsToSearch[0];
                vector<pair<int, uint32_t>> otherSegs;
                for (uint32_t s = 0; s < NUM_SEGMENTS; ++s) {
                    if (s != baseSeg && (segDir[s].chunk_count > 0 || segDir[s].ext_chain_head > 0)) {
                        if (std::find(segsToSearch.begin(), segsToSearch.end(), s) == segsToSearch.end()) {
                            int dist = BitDB::hamming_distance_32(baseSeg, s);
                            otherSegs.push_back({dist, s});
                        }
                    }
                }
                std::sort(otherSegs.begin(), otherSegs.end());
                for (const auto& p : otherSegs) {
                    if (segsToSearch.size() >= numProbes) break;
                    segsToSearch.push_back(p.second);
                }
            }
        }

        cout << "[Search] Probing " << segsToSearch.size() << " segment(s) (numProbes=" << numProbes << "): [";
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
                // Auto-adjusting WAND bound for production
                // We adapt the aggressiveness multiplier based on the query's cosine similarity to the centroid.
                float centroid_len_sq = 0.0f;
                for (int d = 0; d < DIMS; ++d) centroid_len_sq += node.centroid[d] * node.centroid[d];
                float centroid_len = sqrt(centroid_len_sq);
                
                // Calculate normalized cosine similarity [-1.0, 1.0]
                float cos_sim = (query_len * centroid_len > 0.0f) ? (q_dot_c / (query_len * centroid_len)) : 0.0f;
                
                // If highly similar (dense region), we are conservative (alpha ~ 1.0)
                // If distant (fringe region), we prune aggressively (alpha drops to 0.3)
                float alpha = std::min(1.0f, std::max(0.3f, cos_sim));
                
                float max_score_bound = q_dot_c + query_len * (node.max_radius * alpha);
                // Safe Early-Exit Pruning
                if (minHeap.size() == static_cast<size_t>(topK) && max_score_bound < static_cast<float>(minHeap.top().score)) {
                    totalCandidates += node.chunk_count; // Tally bypassed candidates for telemetry
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

                // ── Stage 1: ADC pre-sort for I/O optimisation ONLY ──
                // We compute an ADC score for every record in this extent to decide
                // whether to use a bulk sequential read or scattered per-record seeks.
                // IMPORTANT: the ADC score does NOT filter out any records — all
                // hdr.record_count records proceed to the int8 exact-scoring stage.
                // Eliminating candidates based on ADC alone would be unsafe because
                // binary quantisation introduces approximation error that can invert
                // the ranking of the true nearest neighbour.
                struct CandScore {
                    uint32_t idx;
                    int32_t  adc_score;
                };
                vector<CandScore> allCands;
                allCands.reserve(hdr.record_count);
                for (uint32_t i = 0; i < hdr.record_count; ++i) {
                    allCands.push_back({i, BitDB::adc_score(queryVec.data(), extent_codes[i])});
                }
                // Sort by ADC score descending — highest ADC first.
                // This orders later I/O reads to access likely-high-scoring embeddings
                // first so the heap threshold rises quickly (better heap pruning).
                std::sort(allCands.begin(), allCands.end(),
                    [](const CandScore& a, const CandScore& b) { return a.adc_score > b.adc_score; });

                // ── I/O Crossover Heuristic ──
                // If the extent has more than CROSSOVER_THRESHOLD records, a single
                // sequential bulk read of the remainder is cheaper than N scattered seeks.
                constexpr size_t CROSSOVER_THRESHOLD = 40;

                if (hdr.record_count > CROSSOVER_THRESHOLD) {
                    // Bulk sequential read of int8 embeddings + metadata
                    uint64_t bulk_offset = node.chunk_store_offset + STAGE1_BYTES;
                    size_t remaining_bytes = EXTENT_BYTES - STAGE1_BYTES;
                    vector<char> bulk_buf(remaining_bytes);
                    chunkIn.seekg(static_cast<streamoff>(bulk_offset), ios::beg);
                    chunkIn.read(bulk_buf.data(), remaining_bytes);
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
                        chunkIn.read(reinterpret_cast<char*>(emb), DIMS);
                        chunkIn.seekg(static_cast<streamoff>(meta_offset), ios::beg);
                        chunkIn.read(reinterpret_cast<char*>(&meta), sizeof(meta));
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

        // --- Step 6: Render Results ---
        cout << "\n==================================================================\n";
        cout << "  TOP " << results.size() << " RESULTS for: \"" << queryText << "\"\n";
        cout << "==================================================================\n\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            cout << "  +-- Rank " << (i + 1) << " -----------------------------------------\n";
            cout << "  |  Score   : " << r.score << "\n";
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
        cout << "  |  Segments Probed : " << segsToSearch.size() << " (numProbes=" << numProbes << ")\n";
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
            cout << "  --probes N          Number of adjacent segments to probe (default 4)\n";
            cout << "                      Higher probes = better accuracy but higher latency.\n\n";
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
