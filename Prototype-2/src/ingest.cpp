#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "embed.h"
#include "lsh_vectors.h"

using namespace std;
namespace fs = std::filesystem;

static constexpr uint32_t INT8_DIMS = 384;
static constexpr uint32_t LSH_BITS = 24;

// Calculate float centroid of all embeddings in the document
vector<float> compute_document_centroid(const vector<vector<int8_t>>& embeddings) {
    vector<float> centroid(INT8_DIMS, 0.0f);
    if (embeddings.empty()) return centroid;
    
    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < INT8_DIMS; ++i) {
            centroid[i] += static_cast<float>(emb[i]);
        }
    }
    for (size_t i = 0; i < INT8_DIMS; ++i) {
        centroid[i] /= embeddings.size();
    }
    return centroid;
}

// Compute 24-bit LSH hash using the predefined hyperplanes
uint32_t compute_lsh_hash(const vector<float>& centroid) {
    uint32_t hash = 0;
    for (size_t i = 0; i < LSH_BITS; ++i) {
        float dot_product = 0.0f;
        for (size_t j = 0; j < INT8_DIMS; ++j) {
            dot_product += centroid[j] * LSH_HYPERPLANES[i][j];
        }
        if (dot_product > 0) {
            hash |= (1 << i);
        }
    }
    return hash;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <path_to_text_file>" << endl;
        return 1;
    }

    string sourceFilePath = argv[1];
    if (!fs::exists(sourceFilePath)) {
        cerr << "Error: File does not exist -> " << sourceFilePath << endl;
        return 1;
    }

    cout << "Initializing embedding model for LSH Semantic Hashing..." << endl;
    init_python();

    cout << "Chunking file: " << sourceFilePath << endl;
    vector<ChunkInfo> chunks = chunk_file_with_offsets(sourceFilePath);
    if (chunks.empty()) {
        cerr << "Error: No sentences extracted from file." << endl;
        finalize_python();
        return 1;
    }

    vector<string> sentences;
    sentences.reserve(chunks.size());
    for (const auto& c : chunks) {
        sentences.push_back(c.text);
    }

    cout << "Embedding " << sentences.size() << " chunks..." << endl;
    vector<vector<int8_t>> embeddings = embed_chunks(sentences);

    if (embeddings.empty()) {
        cerr << "Error: Failed to generate embeddings." << endl;
        finalize_python();
        return 1;
    }

    cout << "Calculating semantic centroid..." << endl;
    vector<float> centroid = compute_document_centroid(embeddings);

    cout << "Applying Locality Sensitive Hashing (LSH) projection..." << endl;
    uint32_t lsh_hash = compute_lsh_hash(centroid);

    // Format as 6 character hex
    stringstream ss;
    ss << setfill('0') << setw(6) << hex << lsh_hash;
    string hex_hash = ss.str();

    // Extract path components
    string p1 = hex_hash.substr(0, 2);
    string p2 = hex_hash.substr(2, 2);
    string p3 = hex_hash.substr(4, 2);

    string dataDir = "C:/Users/srish/Desktop/BitDB/Prototype-2/DataStorage";
    fs::path targetDir = fs::path(dataDir) / p1 / p2 / p3;
    fs::path targetFilePath = targetDir / fs::path(sourceFilePath).filename();

    cout << "\n=============================================" << endl;
    cout << "LSH Hash:       " << hex_hash << endl;
    cout << "Target Folder:  " << targetDir.string() << endl;
    
    // Create folders and copy
    fs::create_directories(targetDir);
    fs::copy_file(sourceFilePath, targetFilePath, fs::copy_options::overwrite_existing);

    cout << "File ingested:  " << targetFilePath.string() << endl;
    cout << "=============================================\n" << endl;

    finalize_python();
    return 0;
}
