// ════════════════════════════════════════════════════════════════════════
// Build.cpp — Prototype-3 PDF Ingestion & Index Builder
//
// Scans ingestor/ for .pdf files, extracts text page-by-page,
// chunks each page into sentences, embeds them, computes a 32-bit
// probe bitmask for each chunk, routes it to a segment bucket, then
// writes everything to SSD-resident binary files:
//
//   DataStorage/segment_dir.bin   — 4KB segment directory (tiny, in RAM)
//   DataStorage/chunk_store.bin   — all chunk embeddings (SSD-resident)
//   DataStorage/pdf_text.bin      — raw passage UTF-8 text (SSD-resident)
//   DataStorage/doc_catalog.bin   — PDF metadata (small, in RAM)
//
// Usage: Build.exe
// Reads:  Prototype-3/ingestor/*.pdf
// Writes: Prototype-3/DataStorage/
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

#include "embed.h"
#include "probe_vectors.h"
#include "PathConfig.h"

using namespace std;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────

static constexpr uint32_t DIMS          = 384;
static constexpr uint32_t NUM_SEGMENTS  = 256;   // must be power of 2
static constexpr uint32_t MAGIC         = 0x42444233u;  // "BDB3"
static constexpr uint32_t VERSION       = 1;

// ─────────────────────────────────────────────────
// Binary layout structs
// ─────────────────────────────────────────────────

// segment_dir.bin header (16 bytes)
#pragma pack(push, 1)
struct SegDirHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_segments;
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(SegDirHeader) == 16, "SegDirHeader must be 16 bytes");

// segment_dir.bin per-segment entry (16 bytes)
#pragma pack(push, 1)
struct SegEntry {
    uint64_t chunk_store_offset;   // byte offset of first chunk record in chunk_store.bin
    uint32_t chunk_count;          // number of chunk records in this segment
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(SegEntry) == 16, "SegEntry must be 16 bytes");

// chunk_store.bin per-chunk record (416 bytes, cache-line friendly)
#pragma pack(push, 1)
struct ChunkRecord {
    int8_t   embedding[DIMS];      // 384 bytes — int8 quantized embedding
    uint64_t text_offset;          //   8 bytes — byte offset in pdf_text.bin
    uint32_t text_length;          //   4 bytes — byte length of passage text
    uint32_t doc_id;               //   4 bytes — index into doc_catalog.bin
    uint32_t page_num;             //   4 bytes — 0-indexed page number
    uint32_t chunk_idx_in_page;    //   4 bytes — chunk index within this page
    uint32_t segment_id;           //   4 bytes — which segment this belongs to
    uint32_t reserved;             //   4 bytes — padding to 416 bytes
};
#pragma pack(pop)
static_assert(sizeof(ChunkRecord) == 416, "ChunkRecord must be 416 bytes");

// doc_catalog.bin header (8 bytes)
#pragma pack(push, 1)
struct CatalogHeader {
    uint32_t num_docs;
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(CatalogHeader) == 8, "CatalogHeader must be 8 bytes");

// doc_catalog.bin per-document entry (256 bytes)
#pragma pack(push, 1)
struct DocEntry {
    uint32_t doc_id;               //   4
    uint32_t page_count;           //   4
    uint64_t chunk_store_start;    //   8 — byte offset of first chunk for this doc in chunk_store.bin
    uint32_t chunk_count;          //   4
    uint32_t reserved;             //   4
    char     filename[232];        // 232 — null-terminated PDF filename (basename)
};
#pragma pack(pop)
static_assert(sizeof(DocEntry) == 256, "DocEntry must be 256 bytes");

// ─────────────────────────────────────────────────
// Probe-based segment routing
// ─────────────────────────────────────────────────

// Compute 32-bit probe bitmask: bit i = (dot(embedding, PROBE_VECTORS[i]) > 0)
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

// Map bitmask to segment_id (mod NUM_SEGMENTS)
static uint32_t bitmask_to_segment(uint32_t mask) {
    return mask & (NUM_SEGMENTS - 1);
}

// ─────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────

int main() {
    PathConfig::ensureDirectories();

    const string pdfDir    = PathConfig::getIngestorDir().string();
    const string binDir    = PathConfig::getDataStorageDir().string();
    const string chunkFile = PathConfig::getChunkStoreFile().string();
    const string textFile  = PathConfig::getPdfTextFile().string();
    const string segFile   = PathConfig::getSegmentDirFile().string();
    const string catFile   = PathConfig::getDocCatalogFile().string();

    cout << "══════════════════════════════════════════════════\n";
    cout << "  BitDB Prototype-3 — PDF Index Builder\n";
    cout << "══════════════════════════════════════════════════\n";
    cout << "  ingestor   : " << pdfDir << "\n";
    cout << "  DataStorage: " << binDir << "\n\n";

    // ── Load existing doc catalog to avoid re-ingesting already-indexed PDFs ──
    unordered_set<string> alreadyIndexed;
    uint32_t nextDocId = 0;

    // Track segment membership during this build run
    // segments[i] = list of (chunk_store_offset_of_record)
    // We'll build a temporary in-memory routing table then write segment_dir.bin
    struct PendingChunk {
        ChunkRecord rec;
        uint32_t    seg_id;
    };
    vector<PendingChunk> allChunks;

    // ── Load existing catalog ──
    vector<DocEntry> catalog;
    {
        ifstream catIn(catFile, ios::binary);
        if (catIn) {
            CatalogHeader hdr;
            catIn.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            for (uint32_t i = 0; i < hdr.num_docs; ++i) {
                DocEntry de;
                catIn.read(reinterpret_cast<char*>(&de), sizeof(de));
                catalog.push_back(de);
                alreadyIndexed.insert(string(de.filename));
                nextDocId = max(nextDocId, de.doc_id + 1);
            }
            cout << "  Loaded " << catalog.size() << " existing docs from catalog.\n";
        }
    }

    // ── Load existing chunk records ──
    vector<PendingChunk> existingChunks;
    {
        ifstream chunkIn(chunkFile, ios::binary);
        if (chunkIn) {
            ChunkRecord rec;
            while (chunkIn.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
                existingChunks.push_back({rec, rec.segment_id});
            }
            cout << "  Loaded " << existingChunks.size() << " existing chunk records.\n";
        }
    }
    allChunks = existingChunks;

    // ── Load existing pdf_text.bin to know current end offset ──
    uint64_t textWriteOffset = 0;
    {
        ifstream textIn(textFile, ios::binary | ios::ate);
        if (textIn) textWriteOffset = static_cast<uint64_t>(textIn.tellg());
    }

    // ── Initialize Python ──
    init_python();

    // ── Scan ingestor for new PDFs ──
    vector<fs::path> pdfFiles;
    if (fs::exists(pdfDir) && fs::is_directory(pdfDir)) {
        for (const auto& entry : fs::directory_iterator(pdfDir)) {
            if (entry.is_regular_file()) {
                string ext = entry.path().extension().string();
                // lowercase extension check
                for (auto& c : ext) c = static_cast<char>(tolower(c));
                if (ext == ".pdf") {
                    pdfFiles.push_back(entry.path());
                }
            }
        }
    }
    sort(pdfFiles.begin(), pdfFiles.end());

    if (pdfFiles.empty()) {
        cout << "  No .pdf files found in ingestor/. Drop PDFs there and re-run.\n";
        finalize_python();
        return 0;
    }

    // ── Open append streams for pdf_text.bin and chunk records ──
    // We'll accumulate new chunks in memory, then write sorted by segment at end
    ofstream textOut(textFile, ios::binary | ios::app);
    if (!textOut) {
        cerr << "  FATAL: Cannot open " << textFile << "\n";
        finalize_python();
        return 1;
    }

    // New chunks added in this run
    vector<PendingChunk> newChunks;
    vector<DocEntry>     newDocs;

    int pdfIdx = 0;
    for (const auto& pdfPath : pdfFiles) {
        string basename = pdfPath.filename().string();

        if (alreadyIndexed.count(basename)) {
            cout << "\n  [SKIP] Already indexed: " << basename << "\n";
            continue;
        }

        pdfIdx++;
        cout << "\n  ══════════════════════════════════════\n";
        cout << "  [" << pdfIdx << "] " << basename << "\n";
        cout << "  ══════════════════════════════════════\n";
        cout << "  Extracting pages...\n";

        vector<PageText> pages = pdf_to_text_pages(pdfPath.generic_string());
        if (pages.empty()) {
            cout << "  Warning: No text extracted from PDF. Skipping.\n";
            continue;
        }
        cout << "  Extracted " << pages.size() << " pages.\n";

        uint32_t docId = nextDocId++;
        uint64_t docChunkStart = 0;
        bool     firstChunkForDoc = true;
        uint32_t docChunkCount = 0;

        for (const auto& page : pages) {
            cout << "    Page " << page.pageNum << ": " << page.text.size() << " chars -> ";

            // Chunk the page text into sentences
            vector<ChunkInfo> chunks = chunk_text_with_offsets(page.text);
            if (chunks.empty()) {
                cout << "0 sentences (skipped)\n";
                continue;
            }
            cout << chunks.size() << " sentences\n";

            // Collect sentences for batch embedding
            vector<string> sentences;
            sentences.reserve(chunks.size());
            for (const auto& c : chunks) sentences.push_back(c.text);

            // Batch embed
            vector<vector<int8_t>> embeddings = embed_chunks(sentences);
            if (embeddings.size() != chunks.size()) {
                cerr << "    Warning: Embedding count mismatch. Skipping page.\n";
                continue;
            }

            // Process each chunk
            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                const ChunkInfo& chunk = chunks[ci];
                const vector<int8_t>& emb = embeddings[ci];

                // Write text to pdf_text.bin
                uint64_t txtOffset = textWriteOffset;
                textOut.write(chunk.text.data(), static_cast<streamsize>(chunk.text.size()));
                textWriteOffset += chunk.text.size();

                // Compute segment id
                uint32_t bitmask = compute_probe_bitmask(emb);
                uint32_t segId   = bitmask_to_segment(bitmask);

                // Build ChunkRecord
                ChunkRecord rec = {};
                uint32_t toCopy = static_cast<uint32_t>(min(emb.size(), (size_t)DIMS));
                memcpy(rec.embedding, emb.data(), toCopy);
                rec.text_offset       = txtOffset;
                rec.text_length       = static_cast<uint32_t>(chunk.text.size());
                rec.doc_id            = docId;
                rec.page_num          = static_cast<uint32_t>(page.pageNum);
                rec.chunk_idx_in_page = static_cast<uint32_t>(ci);
                rec.segment_id        = segId;
                rec.reserved          = 0;

                if (firstChunkForDoc) {
                    // Will be set after sorting — placeholder
                    docChunkStart = 0;
                    firstChunkForDoc = false;
                }
                docChunkCount++;
                newChunks.push_back({rec, segId});
            }
        }

        // Build DocEntry
        DocEntry de = {};
        de.doc_id          = docId;
        de.page_count      = static_cast<uint32_t>(pages.size());
        de.chunk_store_start = 0;  // will fix after sorting
        de.chunk_count     = docChunkCount;
        de.reserved        = 0;
        // Copy filename (truncated if needed)
        strncpy(de.filename, basename.c_str(), sizeof(de.filename) - 1);
        de.filename[sizeof(de.filename) - 1] = '\0';
        newDocs.push_back(de);

        cout << "  -> DocId=" << docId << " chunks=" << docChunkCount << "\n";
    }

    textOut.close();

    if (newChunks.empty() && pdfIdx == 0) {
        cout << "\n  Nothing new to index.\n";
        finalize_python();
        return 0;
    }

    cout << "\n  New chunks this run: " << newChunks.size() << "\n";

    // ── Merge and sort all chunks by segment_id ──
    // This is the key SSD-optimization: chunks in the same segment are
    // written contiguously, so a single segment lookup = one sequential SSD read.
    for (auto& pc : newChunks) allChunks.push_back(pc);
    stable_sort(allChunks.begin(), allChunks.end(), [](const PendingChunk& a, const PendingChunk& b){
        return a.seg_id < b.seg_id;
    });

    cout << "  Total chunks after merge: " << allChunks.size() << "\n";

    // ── Write chunk_store.bin (sorted by segment) ──
    cout << "  Writing chunk_store.bin...\n";
    ofstream chunkOut(chunkFile, ios::binary | ios::trunc);
    if (!chunkOut) {
        cerr << "  FATAL: Cannot open " << chunkFile << "\n";
        finalize_python();
        return 1;
    }

    // Build segment directory in memory
    SegEntry segDir[NUM_SEGMENTS] = {};
    {
        uint32_t curSeg = UINT32_MAX;
        uint64_t offset = 0;
        for (const auto& pc : allChunks) {
            if (pc.seg_id != curSeg) {
                segDir[pc.seg_id].chunk_store_offset = offset;
                curSeg = pc.seg_id;
            }
            segDir[pc.seg_id].chunk_count++;
            chunkOut.write(reinterpret_cast<const char*>(&pc.rec), sizeof(ChunkRecord));
            offset += sizeof(ChunkRecord);
        }
    }
    chunkOut.close();

    // ── Fix doc_catalog chunk_store_start offsets ──
    // For each new doc, find its first chunk record's byte position in chunk_store.bin
    for (auto& de : newDocs) {
        for (size_t ci = 0; ci < allChunks.size(); ++ci) {
            if (allChunks[ci].rec.doc_id == de.doc_id) {
                de.chunk_store_start = static_cast<uint64_t>(ci) * sizeof(ChunkRecord);
                break;
            }
        }
        // Also fix existing docs
        for (auto& existing : catalog) {
            for (size_t ci = 0; ci < allChunks.size(); ++ci) {
                if (allChunks[ci].rec.doc_id == existing.doc_id) {
                    existing.chunk_store_start = static_cast<uint64_t>(ci) * sizeof(ChunkRecord);
                    break;
                }
            }
        }
    }
    for (auto& de : newDocs) catalog.push_back(de);

    // ── Write segment_dir.bin ──
    cout << "  Writing segment_dir.bin...\n";
    ofstream segOut(segFile, ios::binary | ios::trunc);
    if (!segOut) {
        cerr << "  FATAL: Cannot open " << segFile << "\n";
        finalize_python();
        return 1;
    }
    SegDirHeader segHdr = {MAGIC, VERSION, NUM_SEGMENTS, 0};
    segOut.write(reinterpret_cast<const char*>(&segHdr), sizeof(segHdr));
    segOut.write(reinterpret_cast<const char*>(segDir), sizeof(segDir));
    segOut.close();

    // ── Write doc_catalog.bin ──
    cout << "  Writing doc_catalog.bin...\n";
    ofstream catOut(catFile, ios::binary | ios::trunc);
    if (!catOut) {
        cerr << "  FATAL: Cannot open " << catFile << "\n";
        finalize_python();
        return 1;
    }
    CatalogHeader catHdr = {static_cast<uint32_t>(catalog.size()), 0};
    catOut.write(reinterpret_cast<const char*>(&catHdr), sizeof(catHdr));
    for (const auto& de : catalog) {
        catOut.write(reinterpret_cast<const char*>(&de), sizeof(de));
    }
    catOut.close();

    // ── Summary ──
    uint64_t chunkStoreSize = allChunks.size() * sizeof(ChunkRecord);
    cout << "\n══════════════════════════════════════════════════\n";
    cout << "  BUILD COMPLETE\n";
    cout << "══════════════════════════════════════════════════\n";
    cout << "  Total documents: " << catalog.size() << "\n";
    cout << "  Total chunks:    " << allChunks.size() << "\n";
    cout << "  Segments used:   ";
    uint32_t usedSegs = 0;
    for (uint32_t i = 0; i < NUM_SEGMENTS; ++i) if (segDir[i].chunk_count > 0) usedSegs++;
    cout << usedSegs << " / " << NUM_SEGMENTS << "\n";
    cout << "  chunk_store.bin: " << chunkStoreSize / 1024 << " KB\n";
    cout << "  pdf_text.bin:    " << textWriteOffset / 1024 << " KB\n";
    cout << "  segment_dir.bin: "
         << (sizeof(SegDirHeader) + sizeof(segDir)) << " bytes (fully RAM-resident during search)\n";
    cout << "══════════════════════════════════════════════════\n";

    finalize_python();
    return 0;
}
