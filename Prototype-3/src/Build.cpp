// ════════════════════════════════════════════════════════════════════════
// Build.cpp — Prototype-3 Advanced Segment-Chain Index Builder
//
// Resolves:
//   Shortcoming 1: Full 32-bit signature exploitation via avalanche bit-mixing
//   Shortcoming 4: Incremental extent-based insertion, tombstones, crash safety
//   Shortcoming 5: Segment imbalance diagnostics and metrics
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
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <map>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "embed.h"
#include "probe_vectors.h"
#include "Routing.h"
#include "PathConfig.h"

using namespace std;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────
// Constants & Formats
// ─────────────────────────────────────────────────

static constexpr uint32_t DIMS          = 384;
static constexpr uint32_t NUM_SEGMENTS  = BitDB::NUM_SEGMENTS; // 256
static constexpr uint32_t MAGIC         = 0x42444233u;         // "BDB3"
static constexpr uint32_t VERSION       = 3;                   // Version 3 adds 384-bit compact codes

// ─────────────────────────────────────────────────
// Binary layout structs
// ─────────────────────────────────────────────────

#pragma pack(push, 1)
struct SegDirHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_segments;
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(SegDirHeader) == 16, "SegDirHeader must be 16 bytes");

#pragma pack(push, 1)
struct SegEntry {
    uint64_t chunk_store_offset;   // byte offset of primary chunk record in chunk_store.bin
    uint32_t chunk_count;          // number of chunk records in primary extent
    uint32_t ext_chain_head;       // 0 if single extent, or 1-based index into segment_extents.bin
};
#pragma pack(pop)
static_assert(sizeof(SegEntry) == 16, "SegEntry must be 16 bytes");

#pragma pack(push, 1)
struct ExtentNode {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t next_extent_idx;      // 0 if end of chain, or 1-based index
};
#pragma pack(pop)
static_assert(sizeof(ExtentNode) == 16, "ExtentNode must be 16 bytes");

#pragma pack(push, 1)
struct ChunkRecord {
    int8_t   embedding[DIMS];      // 384 bytes — int8 quantized embedding
    uint64_t text_offset;          //   8 bytes — byte offset in pdf_text.bin
    uint32_t text_length;          //   4 bytes — byte length of passage text
    uint32_t doc_id;               //   4 bytes — index into doc_catalog.bin
    uint32_t page_num;             //   4 bytes — 0-indexed page number
    uint32_t chunk_idx_in_page;    //   4 bytes — chunk index within this page
    uint32_t segment_id;           //   4 bytes — which segment this belongs to
    uint32_t signature;            //   4 bytes — full 32-bit probe bitmask
    uint8_t  binary_code[BitDB::BINARY_CODE_BYTES];
};
#pragma pack(pop)
static_assert(sizeof(ChunkRecord) == 464, "ChunkRecord must be 464 bytes");

#pragma pack(push, 1)
struct CatalogHeader {
    uint32_t num_docs;
    uint32_t active_docs;
};
#pragma pack(pop)
static_assert(sizeof(CatalogHeader) == 8, "CatalogHeader must be 8 bytes");

#pragma pack(push, 1)
struct DocEntry {
    uint32_t doc_id;               //   4
    uint32_t page_count;           //   4
    uint64_t chunk_store_start;    //   8
    uint32_t chunk_count;          //   4
    uint32_t is_deleted;           //   4 — 0 = active, 1 = tombstoned
    char     filename[232];        // 232 — null-terminated PDF filename
};
#pragma pack(pop)
static_assert(sizeof(DocEntry) == 256, "DocEntry must be 256 bytes");

// ─────────────────────────────────────────────────
// Helper: Atomic file replacement (Crash-Safe Commit)
// ─────────────────────────────────────────────────

static bool atomic_commit_file(const fs::path& tmpPath, const fs::path& dstPath) {
#if defined(_WIN32)
    if (ReplaceFileW(dstPath.wstring().c_str(), tmpPath.wstring().c_str(), NULL, REPLACEFILE_IGNORE_MERGE_ERRORS, NULL, NULL)) {
        return true;
    }
    if (MoveFileExW(tmpPath.wstring().c_str(), dstPath.wstring().c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return true;
    }
    return false;
#else
    std::error_code ec;
    fs::rename(tmpPath, dstPath, ec);
    return !ec;
#endif
}

// ─────────────────────────────────────────────────
// Diagnostics: Print Segment Balance Metrics
// ─────────────────────────────────────────────────

static void report_segment_balance(const SegEntry* segDir, const vector<ExtentNode>& extents) {
    uint32_t counts[NUM_SEGMENTS] = {};
    uint32_t totalChunks = 0;
    uint32_t usedSegs = 0;
    uint32_t minCount = UINT32_MAX;
    uint32_t maxCount = 0;

    for (uint32_t i = 0; i < NUM_SEGMENTS; ++i) {
        uint32_t c = segDir[i].chunk_count;
        uint32_t extIdx = segDir[i].ext_chain_head;
        while (extIdx > 0 && extIdx <= extents.size()) {
            c += extents[extIdx - 1].chunk_count;
            extIdx = extents[extIdx - 1].next_extent_idx;
        }
        counts[i] = c;
        totalChunks += c;
        if (c > 0) {
            usedSegs++;
            minCount = min(minCount, c);
            maxCount = max(maxCount, c);
        }
    }
    if (usedSegs == 0) minCount = 0;

    double mean = totalChunks / static_cast<double>(NUM_SEGMENTS);
    double varSum = 0.0;
    for (uint32_t i = 0; i < NUM_SEGMENTS; ++i) {
        double diff = counts[i] - mean;
        varSum += diff * diff;
    }
    double stddev = sqrt(varSum / NUM_SEGMENTS);
    double maxMeanRatio = mean > 0 ? (maxCount / mean) : 0.0;

    // Gini coefficient
    vector<uint32_t> sortedCounts(counts, counts + NUM_SEGMENTS);
    sort(sortedCounts.begin(), sortedCounts.end());
    double diffSum = 0.0;
    for (size_t i = 0; i < NUM_SEGMENTS; ++i) {
        for (size_t j = 0; j < NUM_SEGMENTS; ++j) {
            diffSum += abs(static_cast<double>(sortedCounts[i]) - static_cast<double>(sortedCounts[j]));
        }
    }
    double gini = (totalChunks > 0) ? (diffSum / (2.0 * NUM_SEGMENTS * totalChunks)) : 0.0;

    cout << "\n  ─── Segment Balance Diagnostics (Shortcoming 5) ───\n";
    cout << "  Used Segments : " << usedSegs << " / " << NUM_SEGMENTS << "\n";
    cout << "  Min Occupancy : " << minCount << "\n";
    cout << "  Max Occupancy : " << maxCount << "\n";
    cout << "  Mean Per Seg  : " << fixed << setprecision(2) << mean << "\n";
    cout << "  Std Deviation : " << stddev << "\n";
    cout << "  Max/Mean Ratio: " << maxMeanRatio << "x\n";
    cout << "  Gini Index    : " << gini << " (0 = perfectly uniform)\n";
}

// ─────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    ios::sync_with_stdio(true);
    PathConfig::ensureDirectories();

    bool forceRebuild = false;
    bool forceCompact = false;
    int deleteDocId = -1;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--rebuild") forceRebuild = true;
        else if (arg == "--compact") forceCompact = true;
        else if (arg == "--delete" && i + 1 < argc) {
            deleteDocId = stoi(argv[++i]);
        }
    }

    const string pdfDir    = PathConfig::getIngestorDir().string();
    const string binDir    = PathConfig::getDataStorageDir().string();
    const string chunkFile = PathConfig::getChunkStoreFile().string();
    const string textFile  = PathConfig::getPdfTextFile().string();
    const string segFile   = PathConfig::getSegmentDirFile().string();
    const string extFile   = PathConfig::getSegmentExtentsFile().string();
    const string catFile   = PathConfig::getDocCatalogFile().string();

    cout << "══════════════════════════════════════════════════\n";
    cout << "  BitDB Prototype-3 — Advanced Segment-Chain Builder\n";
    cout << "══════════════════════════════════════════════════\n";
    cout << "  Ingestor   : " << pdfDir << "\n";
    cout << "  DataStorage: " << binDir << "\n";

    // ── Handle Document Deletion (--delete <docId>) ──
    if (deleteDocId >= 0) {
        cout << "\n[*] Processing deletion for DocId: " << deleteDocId << "\n";
        ifstream catIn(catFile, ios::binary);
        if (!catIn) {
            cerr << "Catalog not found.\n";
            return 1;
        }
        CatalogHeader hdr;
        catIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        vector<DocEntry> catalog(hdr.num_docs);
        catIn.read(reinterpret_cast<char*>(catalog.data()), hdr.num_docs * sizeof(DocEntry));
        catIn.close();

        bool found = false;
        for (auto& de : catalog) {
            if (de.doc_id == static_cast<uint32_t>(deleteDocId)) {
                de.is_deleted = 1;
                found = true;
                break;
            }
        }
        if (!found) {
            cerr << "DocId " << deleteDocId << " not found in catalog.\n";
            return 1;
        }

        string tmpCat = catFile + ".tmp";
        ofstream catOut(tmpCat, ios::binary | ios::trunc);
        uint32_t active = 0;
        for (const auto& de : catalog) if (!de.is_deleted) active++;
        hdr.active_docs = active;
        catOut.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        catOut.write(reinterpret_cast<const char*>(catalog.data()), catalog.size() * sizeof(DocEntry));
        catOut.close();

        atomic_commit_file(tmpCat, catFile);
        cout << "  DocId " << deleteDocId << " tombstoned successfully (Active docs: " << active << ").\n";
        return 0;
    }

    // ── Load Existing State ──
    unordered_set<string> alreadyIndexed;
    vector<DocEntry> catalog;
    uint32_t nextDocId = 0;
    SegEntry segDir[NUM_SEGMENTS] = {};
    vector<ExtentNode> extents;
    bool indexExists = false;

    if (!forceRebuild && fs::exists(segFile) && fs::exists(catFile)) {
        // Load catalog
        ifstream catIn(catFile, ios::binary);
        if (catIn) {
            CatalogHeader hdr;
            catIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            catalog.resize(hdr.num_docs);
            catIn.read(reinterpret_cast<char*>(catalog.data()), hdr.num_docs * sizeof(DocEntry));
            for (const auto& de : catalog) {
                if (!de.is_deleted) {
                    alreadyIndexed.insert(string(de.filename));
                }
                nextDocId = max(nextDocId, de.doc_id + 1);
            }
        }

        // Load segment directory
        ifstream segIn(segFile, ios::binary);
        if (segIn) {
            SegDirHeader hdr;
            segIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            if (hdr.magic == MAGIC && hdr.version == VERSION && hdr.num_segments == NUM_SEGMENTS) {
                segIn.read(reinterpret_cast<char*>(segDir), sizeof(segDir));
                indexExists = true;
            } else if (hdr.magic == MAGIC && !forceRebuild) {
                cerr << "[Build] ERROR: Existing index format is incompatible (version "
                     << hdr.version << "). Run Build.exe --rebuild.\n";
                return 1;
            }
        }

        // Load extents
        if (fs::exists(extFile)) {
            ifstream extIn(extFile, ios::binary);
            if (extIn) {
                ExtentNode node;
                while (extIn.read(reinterpret_cast<char*>(&node), sizeof(node))) {
                    extents.push_back(node);
                }
            }
        }
    }

    // ── Scan ingestor for new PDFs ──
    vector<fs::path> pdfFiles;
    if (fs::exists(pdfDir) && fs::is_directory(pdfDir)) {
        for (const auto& entry : fs::directory_iterator(pdfDir)) {
            if (entry.is_regular_file()) {
                string ext = entry.path().extension().string();
                for (auto& c : ext) c = static_cast<char>(tolower(c));
                if (ext == ".pdf") {
                    pdfFiles.push_back(entry.path());
                }
            }
        }
    }
    sort(pdfFiles.begin(), pdfFiles.end());

    // Filter PDFs to only those not yet indexed
    vector<fs::path> pendingPdfs;
    for (const auto& p : pdfFiles) {
        if (forceRebuild || !alreadyIndexed.count(p.filename().string())) {
            pendingPdfs.push_back(p);
        }
    }

    if (pendingPdfs.empty() && !forceCompact && !forceRebuild) {
        cout << "\n[*] Database is up-to-date. No new PDFs found in ingestor/.\n";
        cout << "  Existing documents: " << catalog.size() << "\n";
        report_segment_balance(segDir, extents);
        return 0;
    }

    // ── Full Rebuild or Compaction Routine ──
    if (forceRebuild) {
        cout << "\n[*] Performing FULL REBUILD from ingestor/...\n";
        catalog.clear();
        extents.clear();
        memset(segDir, 0, sizeof(segDir));
        alreadyIndexed.clear();
        nextDocId = 0;
        indexExists = false;
    }

    // ── Ingestion Phase ──
    init_python();

    uint64_t textWriteOffset = 0;
    {
        ifstream textIn(textFile, ios::binary | ios::ate);
        if (textIn && !forceRebuild) {
            textWriteOffset = static_cast<uint64_t>(textIn.tellg());
        }
    }

    ofstream textOut(textFile, forceRebuild ? (ios::binary | ios::trunc) : (ios::binary | ios::app));
    if (!textOut) {
        cerr << "FATAL: Cannot open " << textFile << "\n";
        finalize_python();
        return 1;
    }

    // Accumulate newly extracted chunks grouped by segment: seg_id -> vector<ChunkRecord>
    map<uint32_t, vector<ChunkRecord>> newSegmentChunks;
    vector<DocEntry> newDocs;

    cout << "\n[*] Processing " << pendingPdfs.size() << " PDF(s)...\n";
    int pdfIdx = 0;
    for (const auto& pdfPath : pendingPdfs) {
        string basename = pdfPath.filename().string();
        pdfIdx++;
        cout << "\n  [" << pdfIdx << "/" << pendingPdfs.size() << "] " << basename << "\n";

        vector<PageText> pages = pdf_to_text_pages(pdfPath.generic_string());
        if (pages.empty()) {
            cout << "    [Warning] No text extracted. Skipping.\n";
            continue;
        }

        uint32_t docId = nextDocId++;
        uint32_t docChunkCount = 0;
        uint64_t firstChunkOffset = 0;
        bool hasFirstOffset = false;

        for (const auto& page : pages) {
            vector<ChunkInfo> chunks = chunk_text_with_offsets(page.text);
            if (chunks.empty()) continue;

            vector<string> sentences;
            sentences.reserve(chunks.size());
            for (const auto& c : chunks) sentences.push_back(c.text);

            vector<vector<int8_t>> embeddings = embed_chunks(sentences);
            if (embeddings.size() != chunks.size()) {
                cerr << "    [Warning] Embedding count mismatch on page " << page.pageNum << "\n";
                continue;
            }

            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                const ChunkInfo& chunk = chunks[ci];
                const vector<int8_t>& emb = embeddings[ci];

                uint64_t txtOffset = textWriteOffset;
                textOut.write(chunk.text.data(), static_cast<streamsize>(chunk.text.size()));
                textWriteOffset += chunk.text.size();

                // Compute 32-bit probe bitmask and map to segment using 32-bit avalanche hash
                uint32_t bitmask = BitDB::compute_probe_bitmask(emb.data());
                uint32_t segId   = BitDB::signature_to_segment(bitmask);

                ChunkRecord rec = {};
                uint32_t toCopy = static_cast<uint32_t>(min(emb.size(), (size_t)DIMS));
                memcpy(rec.embedding, emb.data(), toCopy);
                rec.text_offset       = txtOffset;
                rec.text_length       = static_cast<uint32_t>(chunk.text.size());
                rec.doc_id            = docId;
                rec.page_num          = static_cast<uint32_t>(page.pageNum);
                rec.chunk_idx_in_page = static_cast<uint32_t>(ci);
                rec.segment_id        = segId;
                rec.signature         = bitmask; // Store full 32-bit signature
                BitDB::compute_binary_code(emb.data(), rec.binary_code);

                newSegmentChunks[segId].push_back(rec);
                docChunkCount++;
            }
        }

        DocEntry de = {};
        de.doc_id            = docId;
        de.page_count        = static_cast<uint32_t>(pages.size());
        de.chunk_store_start = 0; // will be updated
        de.chunk_count       = docChunkCount;
        de.is_deleted        = 0;
        strncpy(de.filename, basename.c_str(), sizeof(de.filename) - 1);
        de.filename[sizeof(de.filename) - 1] = '\0';
        newDocs.push_back(de);
        cout << "    -> Extracted " << docChunkCount << " chunks across " << pages.size() << " pages.\n";
    }

    textOut.close();
    finalize_python();

    // ── Incremental Extent-Based Chunk Append (Shortcoming 4) ──
    // If index already exists and not full rebuild, we directly append new segment chunks to chunk_store.bin!
    // This achieves zero rewrite of existing chunks and minimal peak RAM.
    if (indexExists && !forceRebuild && !forceCompact) {
        cout << "\n[*] Incrementally appending new chunks to chunk_store.bin (Zero Rewrite)...\n";
        ofstream chunkAppend(chunkFile, ios::binary | ios::app);
        if (!chunkAppend) {
            cerr << "FATAL: Cannot open " << chunkFile << " for append.\n";
            return 1;
        }

        // Get current end of chunk_store.bin
        chunkAppend.seekp(0, ios::end);
        uint64_t currentOffset = static_cast<uint64_t>(chunkAppend.tellp());

        for (auto& [segId, recs] : newSegmentChunks) {
            if (recs.empty()) continue;
            uint32_t count = static_cast<uint32_t>(recs.size());
            uint64_t extentOffset = currentOffset;

            // Write chunk records directly
            chunkAppend.write(reinterpret_cast<const char*>(recs.data()), count * sizeof(ChunkRecord));
            currentOffset += count * sizeof(ChunkRecord);

            // Update segment directory
            if (segDir[segId].chunk_count == 0) {
                // First extent for this segment
                segDir[segId].chunk_store_offset = extentOffset;
                segDir[segId].chunk_count = count;
                segDir[segId].ext_chain_head = 0;
            } else {
                // Segment already has chunks: append an ExtentNode to chain!
                ExtentNode node;
                node.chunk_store_offset = extentOffset;
                node.chunk_count = count;
                node.next_extent_idx = segDir[segId].ext_chain_head;
                extents.push_back(node);
                segDir[segId].ext_chain_head = static_cast<uint32_t>(extents.size()); // 1-based index
            }
        }
        chunkAppend.close();

        // Append new docs to catalog
        for (const auto& de : newDocs) catalog.push_back(de);
    } else {
        // Full write or compaction: coalesce all chunks into contiguous single extents per segment
        cout << "\n[*] Coalescing and writing contiguous segment layout...\n";
        vector<ChunkRecord> allRecords;

        // If compacting existing database:
        if (indexExists && !forceRebuild) {
            ifstream chunkIn(chunkFile, ios::binary);
            if (chunkIn) {
                ChunkRecord rec;
                while (chunkIn.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
                    // Filter tombstoned documents
                    bool isTombstoned = false;
                    for (const auto& de : catalog) {
                        if (de.doc_id == rec.doc_id && de.is_deleted) {
                            isTombstoned = true;
                            break;
                        }
                    }
                    if (!isTombstoned) {
                        // Re-route segment_id using full 32-bit avalanche hash
                        uint32_t sig = (rec.signature != 0) ? rec.signature : BitDB::compute_probe_bitmask(rec.embedding);
                        rec.signature = sig;
                        rec.segment_id = BitDB::signature_to_segment(sig);
                        allRecords.push_back(rec);
                    }
                }
            }
        }

        // Add new chunks
        for (auto& [segId, recs] : newSegmentChunks) {
            for (auto& r : recs) allRecords.push_back(r);
        }

        // Sort all chunks by segment_id
        stable_sort(allRecords.begin(), allRecords.end(), [](const ChunkRecord& a, const ChunkRecord& b) {
            return a.segment_id < b.segment_id;
        });

        // Write chunk_store.bin.tmp
        string tmpChunk = chunkFile + ".tmp";
        ofstream chunkOut(tmpChunk, ios::binary | ios::trunc);
        if (!chunkOut) {
            cerr << "FATAL: Cannot open " << tmpChunk << "\n";
            return 1;
        }

        memset(segDir, 0, sizeof(segDir));
        extents.clear();

        uint32_t curSeg = UINT32_MAX;
        uint64_t offset = 0;
        for (const auto& rec : allRecords) {
            if (rec.segment_id != curSeg) {
                segDir[rec.segment_id].chunk_store_offset = offset;
                curSeg = rec.segment_id;
            }
            segDir[rec.segment_id].chunk_count++;
            chunkOut.write(reinterpret_cast<const char*>(&rec), sizeof(ChunkRecord));
            offset += sizeof(ChunkRecord);
        }
        chunkOut.close();
        atomic_commit_file(tmpChunk, chunkFile);

        for (const auto& de : newDocs) catalog.push_back(de);
    }

    // ── Atomic Commit of segment_extents.bin ──
    string tmpExt = extFile + ".tmp";
    ofstream extOut(tmpExt, ios::binary | ios::trunc);
    if (extOut) {
        if (!extents.empty()) {
            extOut.write(reinterpret_cast<const char*>(extents.data()), extents.size() * sizeof(ExtentNode));
        }
        extOut.close();
        atomic_commit_file(tmpExt, extFile);
    }

    // ── Atomic Commit of segment_dir.bin ──
    string tmpSeg = segFile + ".tmp";
    ofstream segOut(tmpSeg, ios::binary | ios::trunc);
    if (!segOut) {
        cerr << "FATAL: Cannot open " << tmpSeg << "\n";
        return 1;
    }
    SegDirHeader segHdr = {MAGIC, VERSION, NUM_SEGMENTS, 0};
    segOut.write(reinterpret_cast<const char*>(&segHdr), sizeof(segHdr));
    segOut.write(reinterpret_cast<const char*>(segDir), sizeof(segDir));
    segOut.close();
    atomic_commit_file(tmpSeg, segFile);

    // ── Atomic Commit of doc_catalog.bin ──
    string tmpCat = catFile + ".tmp";
    ofstream catOut(tmpCat, ios::binary | ios::trunc);
    if (!catOut) {
        cerr << "FATAL: Cannot open " << tmpCat << "\n";
        return 1;
    }
    uint32_t activeCount = 0;
    for (const auto& de : catalog) if (!de.is_deleted) activeCount++;
    CatalogHeader catHdr = {static_cast<uint32_t>(catalog.size()), activeCount};
    catOut.write(reinterpret_cast<const char*>(&catHdr), sizeof(catHdr));
    catOut.write(reinterpret_cast<const char*>(catalog.data()), catalog.size() * sizeof(DocEntry));
    catOut.close();
    atomic_commit_file(tmpCat, catFile);

    // ── Summary & Diagnostics ──
    cout << "\n══════════════════════════════════════════════════\n";
    cout << "  BUILD & INGESTION COMPLETE\n";
    cout << "══════════════════════════════════════════════════\n";
    cout << "  Total Documents   : " << catalog.size() << " (" << activeCount << " active)\n";
    cout << "  Extents in Chains : " << extents.size() << "\n";
    report_segment_balance(segDir, extents);
    cout << "══════════════════════════════════════════════════\n";

    fflush(stdout);
    quick_exit(0);
}
