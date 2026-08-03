#pragma once

#include <vector>
#include <string>
#include <cstdint>

// ──────────────────────────────────────────────
// Python bridge for embedding generation
// ──────────────────────────────────────────────

// Initialize the Python interpreter and import the vendor module.
// Must be called once at program start.
void init_python();

// Finalize the Python interpreter.
// Must be called once at program end.
void finalize_python();

// Read a .txt file and split it into sentence-level chunks via NLTK.
// Returns a list of sentence strings.
std::vector<std::string> chunk_file(const std::string& filepath);

struct ChunkInfo {
    std::string text;
    uint64_t byteOffset;
    uint32_t byteLength;
};

// Read a .txt file and split it into sentence-level chunks via NLTK,
// also finding the byte offset and length of each chunk in the original file.
std::vector<ChunkInfo> chunk_file_with_offsets(const std::string& filepath);

// Batch-embed a list of sentences into 384-dim int8 vectors.
// Returns a list of embeddings (each is a vector of 384 int8 values).
std::vector<std::vector<int8_t>> embed_chunks(const std::vector<std::string>& sentences);

