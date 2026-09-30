// ════════════════════════════════════════════════════════════════════════
// Build.cpp — Prototype-4 Columnar Extent Index Builder & MIH Router
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
#include <cstdlib>

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
static constexpr uint32_t MAGIC         = 0x42444234u;         // "BDB4"
static constexpr uint32_t VERSION       = 4;                   // Version 4: Columnar 128KB Extents & MIH

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
    float    centroid[DIMS];       // WAND bound: extent centroid
    float    max_radius;           // WAND bound: max distance to centroid
};
#pragma pack(pop)
static_assert(sizeof(SegEntry) == 1556, "SegEntry must be 1556 bytes");

#pragma pack(push, 1)
struct ExtentNode {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t next_extent_idx;      // 0 if end of chain, or 1-based index
    float    centroid[DIMS];       // WAND bound: extent centroid
    float    max_radius;           // WAND bound: max distance to centroid
};
#pragma pack(pop)
static_assert(sizeof(ExtentNode) == 1556, "ExtentNode must be 1556 bytes");

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

struct RawRecord {
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
// Telemetry Globals
// ─────────────────────────────────────────────────
static uint64_t global_bit_tally[32] = {0};
static std::vector<float> global_extent_radii;

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

static void write_extent_blocks(
    uint32_t segId,
    const vector<RawRecord>& recs,
    ofstream& chunkAppend,
    SegEntry* segDir,
    vector<ExtentNode>& extents,
    uint8_t mih_tables[4][256][32]) 
{
    for (size_t i = 0; i < recs.size(); i += EXTENT_CAPACITY) {
        size_t end = min(recs.size(), i + EXTENT_CAPACITY);
        uint32_t count = static_cast<uint32_t>(end - i);
        
        ExtentBlock block = {};
        block.header.extent_id = segId; 
        block.header.record_count = count;
        block.header.next_extent_idx = 0; 
        
        float centroid[DIMS] = {0};
        
        for (size_t j = 0; j < count; ++j) {
            const auto& r = recs[i + j];
            memcpy(block.binary_codes[j], r.binary_code, BitDB::BINARY_CODE_BYTES);
            memcpy(block.embeddings[j], r.embedding, DIMS);
            block.metadata[j].text_offset = r.text_offset;
            block.metadata[j].text_length = r.text_length;
            block.metadata[j].doc_id = r.doc_id;
            block.metadata[j].page_num = r.page_num;
            block.metadata[j].chunk_idx_in_page = r.chunk_idx_in_page;
            block.metadata[j].signature = r.signature;
            
            for (int d = 0; d < DIMS; ++d) {
                centroid[d] += static_cast<float>(r.embedding[d]);
            }

            // Register in Multi-Index Hashing (MIH) tables
            uint32_t sig = r.signature;
            for (int m = 0; m < 4; ++m) {
                uint8_t sub_val = static_cast<uint8_t>((sig >> (m * 8)) & 0xFF);
                mih_tables[m][sub_val][segId / 8] |= static_cast<uint8_t>(1u << (segId % 8));
            }
        }
        
        for (int d = 0; d < DIMS; ++d) {
            centroid[d] /= count;
        }
        
        float max_radius = 0.0f;
        for (size_t j = 0; j < count; ++j) {
            float dist_sq = 0.0f;
            for (int d = 0; d < DIMS; ++d) {
                float diff = static_cast<float>(block.embeddings[j][d]) - centroid[d];
                dist_sq += diff * diff;
            }
            float dist = sqrt(dist_sq);
            if (dist > max_radius) max_radius = dist;
        }
        
        // Log telemetry
        global_extent_radii.push_back(max_radius);
        
        uint64_t extentOffset = static_cast<uint64_t>(chunkAppend.tellp());
        chunkAppend.write(reinterpret_cast<const char*>(&block), sizeof(ExtentBlock));
        
        if (segDir[segId].chunk_count == 0) {
            segDir[segId].chunk_store_offset = extentOffset;
            segDir[segId].chunk_count = count;
            segDir[segId].ext_chain_head = 0;
            memcpy(segDir[segId].centroid, centroid, sizeof(float)*DIMS);
            segDir[segId].max_radius = max_radius;
        } else {
            ExtentNode node = {};
            node.chunk_store_offset = extentOffset;
            node.chunk_count = count;
            node.next_extent_idx = segDir[segId].ext_chain_head;
            memcpy(node.centroid, centroid, sizeof(float)*DIMS);
            node.max_radius = max_radius;
            extents.push_back(node);
            segDir[segId].ext_chain_head = static_cast<uint32_t>(extents.size());
        }
    }
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

    cout << "\n  --- Segment Balance Diagnostics ---\n";
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
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    ios::sync_with_stdio(true);
    PathConfig::ensureDirectories();

    bool forceRebuild = false;
    bool forceCompact = false;
    bool watchMode = false;
    int deleteDocId = -1;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cout << "==================================================\n";
            cout << "  BitDB Prototype-4 Columnar Extent Builder\n";
            cout << "==================================================\n";
            cout << "USAGE:\n";
            cout << "  Build.exe [OPTIONS]\n\n";
            cout << "OPTIONS:\n";
            cout << "  --help, -h          Show this help message and exit\n";
            cout << "  --watch, -w         Launch background auto-sync watchdog on ./ingestor\n";
            cout << "  --rebuild           Force full re-indexing of all PDFs in ingestor\n";
            cout << "  --compact           Force garbage collection & compaction of storage\n";
            cout << "  --delete <docId>    Tombstone specific document by ID and compact\n";
            cout << "==================================================\n";
            return 0;
        } else if (arg == "--watch" || arg == "-w") {
            watchMode = true;
        } else if (arg == "--rebuild") {
            forceRebuild = true;
        } else if (arg == "--compact") {
            forceCompact = true;
        } else if (arg == "--delete" && i + 1 < argc) {
            deleteDocId = stoi(argv[++i]);
        }
    }

    if (watchMode) {
        cout << "[*] Launching BitDB Background Auto-Sync Watchdog on ./ingestor...\n";
        fs::path nativeWatchdog = PathConfig::getProjectRoot() / "build" / "Watchdog.exe";
        if (fs::exists(nativeWatchdog)) {
            string cmd = "\"" + nativeWatchdog.string() + "\"";
            return system(cmd.c_str());
        }

        fs::path watchdogScript = PathConfig::getScriptsDir() / "db_watchdog.py";
        if (!fs::exists(watchdogScript)) {
            cerr << "[!] Watchdog executable or script not found.\n";
            return 1;
        }
#ifdef _WIN32
        string cmd = "python \"" + watchdogScript.string() + "\"";
#else
        string cmd = "python3 \"" + watchdogScript.string() + "\"";
#endif
        return system(cmd.c_str());
    }

    const string pdfDir    = PathConfig::getIngestorDir().string();
    const string binDir    = PathConfig::getDataStorageDir().string();
    const string chunkFile = PathConfig::getChunkStoreFile().string();
    const string textFile  = PathConfig::getPdfTextFile().string();
    const string segFile   = PathConfig::getSegmentDirFile().string();
    const string extFile   = PathConfig::getSegmentExtentsFile().string();
    const string catFile   = PathConfig::getDocCatalogFile().string();
    const string mihFile   = PathConfig::getMihTableFile().string();

    cout << "==================================================\n";
    cout << "  BitDB Prototype-4 - Advanced Columnar Extent Builder\n";
    cout << "==================================================\n";
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
    uint8_t mih_tables[4][256][32] = {};
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

        // Load MIH tables
        if (fs::exists(mihFile)) {
            ifstream mihIn(mihFile, ios::binary);
            if (mihIn) {
                mihIn.read(reinterpret_cast<char*>(mih_tables), sizeof(mih_tables));
            }
        }
    }

    // ── Scan ingestor recursively for all PDFs (including subdirectories) ──
    vector<fs::path> pdfFiles;
    if (fs::exists(pdfDir) && fs::is_directory(pdfDir)) {
        std::error_code ec;
        for (const auto& entry : fs::recursive_directory_iterator(pdfDir, fs::directory_options::skip_permission_denied, ec)) {
            if (ec) continue;
            std::error_code fileEc;
            if (entry.is_regular_file(fileEc)) {
                string ext = entry.path().extension().string();
                for (auto& c : ext) c = static_cast<char>(tolower(c));
                if (ext == ".pdf") {
                    pdfFiles.push_back(entry.path());
                }
            }
        }
    }
    sort(pdfFiles.begin(), pdfFiles.end());

    // Filter PDFs to only those not yet indexed and track physical files
    vector<fs::path> pendingPdfs;
    unordered_set<string> physicalPdfs;
    
    for (const auto& p : pdfFiles) {
        string relPath = fs::relative(p, pdfDir).generic_string();
        string filename = p.filename().string();
        physicalPdfs.insert(relPath);
        physicalPdfs.insert(filename);
        
        if (forceRebuild || (!alreadyIndexed.count(relPath) && !alreadyIndexed.count(filename))) {
            pendingPdfs.push_back(p);
        }
    }

    // ── Auto-Detect Deleted Documents (Sync Database to Folder) ──
    if (indexExists && !forceRebuild) {
        bool detectedDeletions = false;
        for (auto& de : catalog) {
            if (!de.is_deleted) {
                // If a previously indexed document is no longer on disk, delete it
                if (physicalPdfs.find(string(de.filename)) == physicalPdfs.end()) {
                    cout << "\n[!] Auto-Sync: Detected deleted file '" << de.filename << "'. Tombstoning DocId " << de.doc_id << ".\n";
                    de.is_deleted = 1;
                    detectedDeletions = true;
                }
            }
        }
        // If we detected deletions, force a compaction so they are physically removed from binaries
        if (detectedDeletions) {
            forceCompact = true;
            cout << "[*] Forcing database compaction to physically remove deleted vectors from binary files...\n";
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
        memset(mih_tables, 0, sizeof(mih_tables));
        alreadyIndexed.clear();
        nextDocId = 0;
        indexExists = false;
    }

    // ── Ingestion Phase ──
    init_python();

    // ── Text Store Write ──
    // Write text data to pdf_text.bin.tmp first; after all index files are
    // atomically committed we rename it to pdf_text.bin. This guarantees that
    // a crash mid-ingestion cannot produce a partially-written text store.
    const string tmpTextFile = textFile + ".tmp";

    uint64_t textWriteOffset = 0;
    if (!forceRebuild) {
        // Preserve existing content by copying it into the .tmp file first
        ifstream existingText(textFile, ios::binary);
        if (existingText) {
            ofstream tmpTextOut(tmpTextFile, ios::binary | ios::trunc);
            tmpTextOut << existingText.rdbuf();
            existingText.seekg(0, ios::end);
            textWriteOffset = static_cast<uint64_t>(existingText.tellg());
        }
    }

    ofstream textOut(tmpTextFile,
        forceRebuild ? (ios::binary | ios::trunc) : (ios::binary | ios::app));
    if (!textOut) {
        cerr << "FATAL: Cannot open " << tmpTextFile << "\n";
        finalize_python();
        return 1;
    }

    // Accumulate newly extracted chunks grouped by segment: seg_id -> vector<RawRecord>
    map<uint32_t, vector<RawRecord>> newSegmentChunks;
    vector<DocEntry> newDocs;

    cout << "\n[*] Processing " << pendingPdfs.size() << " PDF(s)...\n";
    int pdfIdx = 0;
    for (const auto& pdfPath : pendingPdfs) {
        string relPath = fs::relative(pdfPath, pdfDir).generic_string();
        string basename = pdfPath.filename().string();
        pdfIdx++;
        cout << "\n  [" << pdfIdx << "/" << pendingPdfs.size() << "] " << relPath << "\n";

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

                // Compute 32-bit probe bitmask and map to segment
                uint32_t bitmask = BitDB::compute_probe_bitmask(emb.data());
                uint32_t segId   = BitDB::signature_to_segment(bitmask);

                for (int b = 0; b < 32; b++) {
                    if ((bitmask >> b) & 1) global_bit_tally[b]++;
                }

                RawRecord rec = {};
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
        string storeName = (relPath.size() < sizeof(de.filename)) ? relPath : basename;
        strncpy(de.filename, storeName.c_str(), sizeof(de.filename) - 1);
        de.filename[sizeof(de.filename) - 1] = '\0';
        newDocs.push_back(de);
        cout << "    -> Extracted " << docChunkCount << " chunks across " << pages.size() << " pages.\n";
    }

    textOut.close();
    finalize_python();

    // ── Incremental Extent-Based Chunk Append ──
    if (indexExists && !forceRebuild && !forceCompact) {
        cout << "\n[*] Incrementally appending new chunks to chunk_store.bin (Zero Rewrite)...\n";
        ofstream chunkAppend(chunkFile, ios::binary | ios::app);
        if (!chunkAppend) {
            cerr << "FATAL: Cannot open " << chunkFile << " for append.\n";
            return 1;
        }

        // Get current end of chunk_store.bin
        chunkAppend.seekp(0, ios::end);

        for (auto& [segId, recs] : newSegmentChunks) {
            if (recs.empty()) continue;
            write_extent_blocks(segId, recs, chunkAppend, segDir, extents, mih_tables);
        }
        chunkAppend.close();

        // Append new docs to catalog
        for (const auto& de : newDocs) catalog.push_back(de);
    } else {
        // Full write or compaction: coalesce all chunks into contiguous single extents per segment
        cout << "\n[*] Coalescing and writing contiguous segment layout...\n";
        vector<RawRecord> allRecords;

        // If compacting existing database:
        if (indexExists && !forceRebuild) {
            ifstream chunkIn(chunkFile, ios::binary);
            if (chunkIn) {
                ExtentBlock block;
                while (chunkIn.read(reinterpret_cast<char*>(&block), sizeof(block))) {
                    for (uint32_t j = 0; j < block.header.record_count; ++j) {
                        RawRecord rec = {};
                        memcpy(rec.binary_code, block.binary_codes[j], BitDB::BINARY_CODE_BYTES);
                        memcpy(rec.embedding, block.embeddings[j], DIMS);
                        rec.text_offset = block.metadata[j].text_offset;
                        rec.text_length = block.metadata[j].text_length;
                        rec.doc_id = block.metadata[j].doc_id;
                        rec.page_num = block.metadata[j].page_num;
                        rec.chunk_idx_in_page = block.metadata[j].chunk_idx_in_page;
                        rec.signature = block.metadata[j].signature;
                        rec.segment_id = block.header.extent_id;

                        // Filter tombstoned documents
                        bool isTombstoned = false;
                        for (const auto& de : catalog) {
                            if (de.doc_id == rec.doc_id && de.is_deleted) {
                                isTombstoned = true;
                                break;
                            }
                        }
                        if (!isTombstoned) {
                            uint32_t sig = BitDB::compute_probe_bitmask(rec.embedding);
                            rec.signature = sig;
                            rec.segment_id = BitDB::signature_to_segment(sig);
                            for (int b = 0; b < 32; b++) {
                                if ((sig >> b) & 1) global_bit_tally[b]++;
                            }
                            allRecords.push_back(rec);
                        }
                    }
                }
            }
        }

        // Add new chunks
        for (auto& [segId, recs] : newSegmentChunks) {
            for (auto& r : recs) allRecords.push_back(r);
        }

        // Sort all chunks by segment_id
        stable_sort(allRecords.begin(), allRecords.end(), [](const RawRecord& a, const RawRecord& b) {
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
        memset(mih_tables, 0, sizeof(mih_tables));
        extents.clear();

        // Group by segment and write blocks
        uint32_t curSeg = UINT32_MAX;
        vector<RawRecord> curSegRecords;
        
        for (const auto& rec : allRecords) {
            if (rec.segment_id != curSeg) {
                if (!curSegRecords.empty()) {
                    std::sort(curSegRecords.begin(), curSegRecords.end(), [](const RawRecord& a, const RawRecord& b) {
                        return a.signature < b.signature;
                    });
                    write_extent_blocks(curSeg, curSegRecords, chunkOut, segDir, extents, mih_tables);
                    curSegRecords.clear();
                }
                curSeg = rec.segment_id;
            }
            curSegRecords.push_back(rec);
        }
        if (!curSegRecords.empty()) {
            std::sort(curSegRecords.begin(), curSegRecords.end(), [](const RawRecord& a, const RawRecord& b) {
                return a.signature < b.signature;
            });
            write_extent_blocks(curSeg, curSegRecords, chunkOut, segDir, extents, mih_tables);
        }

        chunkOut.close();
        atomic_commit_file(tmpChunk, chunkFile);

        for (const auto& de : newDocs) catalog.push_back(de);
    }

    // ── Atomic Commit of mih_table.bin ──
    string tmpMih = mihFile + ".tmp";
    ofstream mihOut(tmpMih, ios::binary | ios::trunc);
    if (mihOut) {
        mihOut.write(reinterpret_cast<const char*>(mih_tables), sizeof(mih_tables));
        mihOut.close();
        atomic_commit_file(tmpMih, mihFile);
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

    // ── Atomic Commit of pdf_text.bin (final step) ──
    // Close is implicit since textOut went out of scope; now rename .tmp -> final.
    atomic_commit_file(tmpTextFile, textFile);

    // ── Summary & Diagnostics ──
    cout << "\n==================================================\n";
    cout << "  BUILD & INGESTION COMPLETE\n";
    cout << "==================================================\n";
    cout << "  Total Documents   : " << catalog.size() << " (" << activeCount << " active)\n";
    cout << "  Extents in Chains : " << extents.size() << "\n";
    report_segment_balance(segDir, extents);
    cout << "==================================================\n";

    // ── Dump Telemetry JSON ──
    std::string telPath = (PathConfig::getProjectRoot() / "eda_output" / "build_telemetry.json").generic_string();
    std::ofstream telOut(telPath, std::ios::trunc);
    if (telOut) {
        telOut << "{\n";
        telOut << "  \"bit_tally\": [";
        for (int i=0; i<32; i++) {
            telOut << global_bit_tally[i] << (i<31 ? ", " : "");
        }
        telOut << "],\n";
        telOut << "  \"extent_radii\": [";
        for (size_t i=0; i<global_extent_radii.size(); i++) {
            telOut << global_extent_radii[i] << (i<global_extent_radii.size()-1 ? ", " : "");
        }
        telOut << "]\n";
        telOut << "}\n";
        telOut.close();
        cout << "[*] Dumped ingestion telemetry to: " << telPath << "\n";
    }

    fflush(stdout);
    quick_exit(0);
}
