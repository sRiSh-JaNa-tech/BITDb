#pragma once

#include <vector>
#include <string>
#include <cstdint>

// ──────────────────────────────────────────────────────────────────
// Python bridge for embedding generation — Prototype-3
// ──────────────────────────────────────────────────────────────────

// Initialize the Python interpreter and import the vendor module.
// Must be called once at program start.
void init_python();

// Finalize the Python interpreter.
// Must be called once at program end.
void finalize_python();

// ──────────────────────────────────────────────────────────────────
// Shared structs
// ──────────────────────────────────────────────────────────────────

struct ChunkInfo {
    std::string text;
    uint64_t byteOffset;   // byte offset within the source text/file
    uint32_t byteLength;
};

struct PageText {
    int         pageNum;   // 0-indexed
    std::string text;
};

// ──────────────────────────────────────────────────────────────────
// Text chunking
// ──────────────────────────────────────────────────────────────────

// Read a .txt file and split it into sentence-level chunks via NLTK.
std::vector<std::string> chunk_file(const std::string& filepath);

// Read a .txt file and split into sentence-level chunks with byte offsets.
std::vector<ChunkInfo> chunk_file_with_offsets(const std::string& filepath);

// Chunk a raw text string (not a file path) into sentences with byte offsets.
// Used for per-page PDF text (text already extracted in memory).
std::vector<ChunkInfo> chunk_text_with_offsets(const std::string& text);

// ──────────────────────────────────────────────────────────────────
// Embedding
// ──────────────────────────────────────────────────────────────────

// Batch-embed a list of sentences into 384-dim int8 vectors.
std::vector<std::vector<int8_t>> embed_chunks(const std::vector<std::string>& sentences);

// ──────────────────────────────────────────────────────────────────
// PDF extraction (Prototype-3)
// ──────────────────────────────────────────────────────────────────

// Extract text from a PDF file page by page.
// Returns list of (pageNum, pageText) pairs. Empty pages are skipped.
std::vector<PageText> pdf_to_text_pages(const std::string& filepath);
