#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <cstring>

#include "embed.h"
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

static constexpr uint32_t INT8_DIMS   = 384;

static vector<float> compute_centroid(const vector<vector<int8_t>>& embeddings) {
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

static vector<float> compute_centroid_float(const vector<vector<float>>& centroids) {
    vector<float> centroid(INT8_DIMS, 0.0f);
    if (centroids.empty()) return centroid;
    for (const auto& c : centroids) {
        for (size_t i = 0; i < INT8_DIMS; ++i) {
            centroid[i] += c[i];
        }
    }
    for (size_t i = 0; i < INT8_DIMS; ++i) {
        centroid[i] /= centroids.size();
    }
    return centroid;
}

// ──────────────────────────────────────────────
// Node struct (packed to 40 bytes)
// ──────────────────────────────────────────────

#pragma pack(push, 1)
struct Node {
    uint32_t nodeId;
    uint32_t parentId;
    uint64_t address;
    uint32_t firstChild;
    uint32_t childCount;
    uint32_t centroidOffset;    // byte offset in centroids.bin (folder-level BBQ)
    uint64_t leafLookupBase;    // byte offset of first file-row in leaf_centroids.bin (0 = non-leaf)
    uint32_t leafLookupCount;   // number of file-rows (0 = non-leaf)
    uint16_t level;
    uint8_t  flags;
    uint8_t  reserved;
};
#pragma pack(pop)

// ──────────────────────────────────────────────
// TempNode: working data used during tree construction
// ──────────────────────────────────────────────

struct TempNode {
    fs::path   path;
    uint32_t   id;
    uint32_t   parentId;
    uint64_t   address;
    uint16_t   level;
    vector<uint32_t> childIds;       // IDs of direct children
    bool       isLeaf;               // true if no subdirectory children

    // Computed during bottom-up pass
    vector<float> centroid;           // 384-dim float centroid (running sum/avg)
    vector<int8_t> int8Centroid;      // 384-dim int8 quantized centroid

    // Leaf-only: lookup table metadata (written into Node on disk)
    uint64_t leafLookupBase  = 0;    // byte offset of first row in leaf_centroids.bin
    uint32_t leafLookupCount = 0;    // number of file rows written
};

// ──────────────────────────────────────────────
// Helper: get the relative path from dataDir for mirroring into binDir
// ──────────────────────────────────────────────

static string get_relative(const fs::path& fullPath, const fs::path& basePath) {
    return fs::relative(fullPath, basePath).generic_string();
}

// ──────────────────────────────────────────────
// LeafRow: fixed 404-byte row in leaf_centroids.bin
// ──────────────────────────────────────────────
// Layout:
//   [384] int8_t centroid[384]     (full uncompressed file centroid)
//   [8]   uint64_t embeddingsAddr  (byte offset in global embeddings.bin)
//   [4]   uint32_t embeddingsCount (number of chunk vectors)
//   [8]   uint64_t textPathOffset  (byte offset in string table)
//   ─────────────────────────────────────────────
//   Total: 404 bytes

static constexpr uint32_t LEAF_ROW_SZ = 404;

#pragma pack(push, 1)
struct LeafRow {
    int8_t   centroid[INT8_DIMS];   // 384
    uint64_t embeddingsAddr;        //   8
    uint32_t embeddingsCount;       //   4
    uint64_t textPathOffset;        //   8
                                    // ───
                                    // 404 bytes total
};
#pragma pack(pop)

static_assert(sizeof(LeafRow) == LEAF_ROW_SZ, "LeafRow must be exactly 80 bytes");

static uint64_t append_chunk_embeddings(ofstream& out,
                                        const vector<vector<int8_t>>& embeddings,
                                        const vector<ChunkInfo>& chunks) {
    uint64_t startPos = static_cast<uint64_t>(out.tellp());
    uint32_t numChunks = static_cast<uint32_t>(embeddings.size());
    uint16_t dim = numChunks > 0 ? static_cast<uint16_t>(embeddings[0].size()) : INT8_DIMS;

    out.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));
    out.write(reinterpret_cast<const char*>(&dim),       sizeof(dim));
    for (size_t i = 0; i < numChunks; ++i) {
        // Write 384-byte embedding
        out.write(reinterpret_cast<const char*>(embeddings[i].data()), embeddings[i].size());
        
        // Write 8-byte source file offset and 4-byte chunk length
        out.write(reinterpret_cast<const char*>(&chunks[i].byteOffset), sizeof(uint64_t));
        out.write(reinterpret_cast<const char*>(&chunks[i].byteLength), sizeof(uint32_t));
    }
    return startPos;
}

// ──────────────────────────────────────────────
// Write one 404-byte LeafRow into leaf_centroids.bin.
// ──────────────────────────────────────────────

static void write_leaf_row(ofstream& out, const vector<int8_t>& centroid,
                           uint64_t embeddingsAddr, uint32_t embeddingsCount,
                           uint64_t textPathOffset) {
    LeafRow row = {};
    uint32_t toCopy = static_cast<uint32_t>(min(centroid.size(), (size_t)INT8_DIMS));
    memcpy(row.centroid, centroid.data(), toCopy);
    
    row.embeddingsAddr  = embeddingsAddr;
    row.embeddingsCount = embeddingsCount;
    row.textPathOffset  = textPathOffset;

    out.write(reinterpret_cast<const char*>(&row), sizeof(LeafRow));
}

// ──────────────────────────────────────────────
// Helper: write an int8 centroid into centroids.bin
// Format: [uint32_t num_bytes][int8_t values...]
// Returns the byte offset where this centroid was written.
// ──────────────────────────────────────────────

static uint32_t write_int8_centroid(ofstream& out, const vector<int8_t>& centroid) {
    uint32_t pos = static_cast<uint32_t>(out.tellp());

    uint32_t numBytes = static_cast<uint32_t>(centroid.size());
    out.write(reinterpret_cast<const char*>(&numBytes), sizeof(numBytes));
    out.write(reinterpret_cast<const char*>(centroid.data()), numBytes);

    return pos;
}


// ══════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════

int main() {
    const string dataDir = "C:/Users/srish/Desktop/BitDB/Prototype-1/DataStorage";
    const string binDir  = "C:/Users/srish/Desktop/BitDB/Prototype-1/BinStorage";

    // ── Clean and recreate BinStorage ──
    if (fs::exists(binDir)) {
        fs::remove_all(binDir);
    }
    fs::create_directories(binDir);

    // ══════════════════════════════════════════
    // PHASE 1: BFS — Build the tree
    // ══════════════════════════════════════════

    cout << "═══════════════════════════════════════════" << endl;
    cout << "  PHASE 1: Building directory tree (BFS)   " << endl;
    cout << "═══════════════════════════════════════════" << endl;

    vector<TempNode> tempNodes;
    vector<uint32_t> childrenBin;   // flat array of child IDs
    uint32_t nextNodeId = 0;

    queue<size_t> q;  // queue of indices into tempNodes

    // Create root
    {
        TempNode root;
        root.path     = dataDir;
        root.id       = nextNodeId++;
        root.parentId = 0;
        root.address  = 0;
        root.level    = 0;
        root.isLeaf   = true;  // will be updated if children found
        tempNodes.push_back(root);
        q.push(0);
    }

    while (!q.empty()) {
        size_t idx = q.front();
        q.pop();

        // Collect subdirectories
        vector<fs::path> subdirs;
        if (fs::exists(tempNodes[idx].path) && fs::is_directory(tempNodes[idx].path)) {
            for (const auto& entry : fs::directory_iterator(tempNodes[idx].path)) {
                if (entry.is_directory()) {
                    subdirs.push_back(entry.path());
                }
            }
        }

        // Sort for deterministic ordering
        sort(subdirs.begin(), subdirs.end());

        if (!subdirs.empty()) {
            tempNodes[idx].isLeaf = false;
        }

        uint8_t childIndex = 1;
        for (const auto& subdir : subdirs) {
            TempNode child;
            child.path     = subdir;
            child.id       = nextNodeId++;
            child.parentId = tempNodes[idx].id;
            child.address  = (tempNodes[idx].address << 8) | childIndex;
            child.level    = tempNodes[idx].level + 1;
            child.isLeaf   = true;  // will be updated when its children are scanned

            tempNodes[idx].childIds.push_back(child.id);
            childrenBin.push_back(child.id);

            size_t childIdx = tempNodes.size();
            tempNodes.push_back(child);
            q.push(childIdx);
            childIndex++;
        }
    }

    cout << "  Built tree with " << tempNodes.size() << " nodes." << endl;
    for (const auto& tn : tempNodes) {
        cout << "  [" << tn.id << "] " << tn.path.filename().string()
             << (tn.isLeaf ? " (LEAF)" : " (PARENT)")
             << " children=" << tn.childIds.size() << endl;
    }

    // ══════════════════════════════════════════
    // PHASE 2: Bottom-up — Compute embeddings & centroids
    // ══════════════════════════════════════════

    cout << endl;
    cout << "═══════════════════════════════════════════" << endl;
    cout << "  PHASE 2: Computing embeddings (bottom-up)" << endl;
    cout << "═══════════════════════════════════════════" << endl;

    // Initialize Python (loads model once)
    init_python();

    // Open global output files for leaf processing
    string globalEmbFile     = binDir + "/embeddings.bin";
    string leafCentroidsFile = binDir + "/leaf_centroids.bin";

    ofstream outEmbeddings(globalEmbFile, ios::binary);
    if (!outEmbeddings) {
        cerr << "[Node] FATAL: Cannot open " << globalEmbFile << endl;
        return 1;
    }
    ofstream outLeafCentroids(leafCentroidsFile, ios::binary);
    if (!outLeafCentroids) {
        cerr << "[Node] FATAL: Cannot open " << leafCentroidsFile << endl;
        return 1;
    }
    string stringTable;  // null-terminated file paths appended after all rows

    // Process nodes in REVERSE order (leaves first, root last)
    for (int i = static_cast<int>(tempNodes.size()) - 1; i >= 0; --i) {
        TempNode& tn = tempNodes[i];
        cout << endl << "  Processing node [" << tn.id << "] "
             << tn.path.filename().string() << "..." << endl;

        if (tn.isLeaf) {
            // ── LEAF NODE: embed text files, write leaf_centroids.bin rows ──

            // Collect all .txt files in this folder
            vector<fs::path> txtFiles;
            if (fs::exists(tn.path) && fs::is_directory(tn.path)) {
                for (const auto& entry : fs::directory_iterator(tn.path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                        txtFiles.push_back(entry.path());
                    }
                }
            }
            sort(txtFiles.begin(), txtFiles.end());

            if (txtFiles.empty()) {
                cout << "    No .txt files found. Centroid = zeros." << endl;
                tn.centroid.assign(INT8_DIMS, 0.0f);
                tn.int8Centroid.assign(INT8_DIMS, 0);
                continue;
            }

            // Record byte address of first row for this leaf node
            tn.leafLookupBase = static_cast<uint64_t>(outLeafCentroids.tellp());

            vector<vector<float>> fileCentroids;  // one centroid per file (for folder centroid)

            for (const auto& txtFile : txtFiles) {
                string filePath = txtFile.generic_string();
                cout << "    Chunking: " << txtFile.filename().string() << endl;

                // Chunk the file into sentences
                vector<ChunkInfo> chunks = chunk_file_with_offsets(filePath);
                if (chunks.empty()) {
                    cout << "      (no sentences)" << endl;
                    continue;
                }
                cout << "      " << chunks.size() << " sentences" << endl;

                // Prepare string vector for Python bridge
                vector<string> sentences;
                sentences.reserve(chunks.size());
                for (const auto& c : chunks) {
                    sentences.push_back(c.text);
                }

                // Embed all sentences for this file
                vector<vector<int8_t>> embeddings = embed_chunks(sentences);
                cout << "      " << embeddings.size() << " embeddings generated" << endl;

                if (embeddings.empty()) continue;

                // ── 1. Append this file's chunks to global embeddings.bin ──
                uint64_t embAddr  = append_chunk_embeddings(outEmbeddings, embeddings, chunks);
                uint32_t embCount = static_cast<uint32_t>(embeddings.size());

                // ── 2. Compute per-file float centroid ──
                vector<float> fileCentroid = compute_centroid(embeddings);
                fileCentroids.push_back(fileCentroid);

                // Convert file centroid to int8
                vector<int8_t> fileInt8(INT8_DIMS);
                for (size_t k = 0; k < INT8_DIMS; ++k) {
                    fileInt8[k] = static_cast<int8_t>(std::clamp(std::round(fileCentroid[k]), -128.0f, 127.0f));
                }

                // ── 3. Add file path to string table ──
                uint64_t pathOffset = static_cast<uint64_t>(stringTable.size());
                stringTable += filePath;
                stringTable += '\0';

                // ── 4. Write row to leaf_centroids.bin ──
                write_leaf_row(outLeafCentroids, fileInt8, embAddr, embCount, pathOffset);
                tn.leafLookupCount++;

                cout << "      Row written: embAddr=" << embAddr
                     << " chunks=" << embCount << endl;
            }

            // Compute folder centroid = average of file centroids
            if (!fileCentroids.empty()) {
                tn.centroid = compute_centroid_float(fileCentroids);
            } else {
                tn.centroid.assign(INT8_DIMS, 0.0f);
            }

            // Convert folder centroid back to int8
            tn.int8Centroid.assign(INT8_DIMS, 0);
            for (size_t k = 0; k < INT8_DIMS; ++k) {
                tn.int8Centroid[k] = static_cast<int8_t>(std::clamp(std::round(tn.centroid[k]), -128.0f, 127.0f));
            }

            cout << "    Folder centroid (int8: " << tn.int8Centroid.size() << " bytes)" << endl;
            cout << "    leaf_centroids rows: " << tn.leafLookupCount
                 << " (base offset=" << tn.leafLookupBase << ")" << endl;

        } else {
            // ── PARENT NODE: average child centroids ──

            vector<vector<float>> childCentroids;
            for (uint32_t childId : tn.childIds) {
                // Find child TempNode by ID
                for (const auto& cn : tempNodes) {
                    if (cn.id == childId && !cn.centroid.empty()) {
                        bool isAllZeros = true;
                        for (float val : cn.centroid) {
                            if (val != 0.0f) {
                                isAllZeros = false;
                                break;
                            }
                        }
                        if (!isAllZeros) {
                            childCentroids.push_back(cn.centroid);
                        }
                        break;
                    }
                }
            }

            if (!childCentroids.empty()) {
                tn.centroid = compute_centroid_float(childCentroids);
            } else {
                tn.centroid.assign(INT8_DIMS, 0.0f);
            }

            // Convert parent folder centroid back to int8
            tn.int8Centroid.assign(INT8_DIMS, 0);
            for (size_t k = 0; k < INT8_DIMS; ++k) {
                tn.int8Centroid[k] = static_cast<int8_t>(std::clamp(std::round(tn.centroid[k]), -128.0f, 127.0f));
            }

            cout << "    Parent centroid = avg of " << childCentroids.size()
                 << " child centroids (int8: " << tn.int8Centroid.size() << " bytes)" << endl;
        }
    }

    // Flush string table at the end of leaf_centroids.bin and close global files
    outLeafCentroids.write(stringTable.data(), static_cast<std::streamsize>(stringTable.size()));
    outLeafCentroids.close();
    outEmbeddings.close();
    cout << "  Written global embeddings -> " << globalEmbFile << endl;
    cout << "  Written lookup table    -> " << leafCentroidsFile
         << " (" << stringTable.size() << " bytes string table)" << endl;

    // Done with Python
    finalize_python();

    // ══════════════════════════════════════════
    // PHASE 3: Write binary output files
    // ══════════════════════════════════════════

    cout << endl;
    cout << "═══════════════════════════════════════════" << endl;
    cout << "  PHASE 3: Writing binary output files      " << endl;
    cout << "═══════════════════════════════════════════" << endl;

    // ── centroids.bin ──
    string centroidsFile = binDir + "/centroids.bin";
    ofstream outCentroids(centroidsFile, ios::binary);

    // Write centroids and record offsets
    vector<uint32_t> centroidOffsets(tempNodes.size());
    for (size_t i = 0; i < tempNodes.size(); ++i) {
        centroidOffsets[i] = write_int8_centroid(outCentroids, tempNodes[i].int8Centroid);
    }
    outCentroids.close();
    cout << "  Written " << centroidsFile << endl;

    // ── nodes.bin ──
    string nodesFile = binDir + "/nodes.bin";
    ofstream outNodes(nodesFile, ios::binary);

    // Build firstChild offsets by scanning childrenBin
    // childrenBin was built during BFS in order
    uint32_t childOffset = 0;
    for (size_t i = 0; i < tempNodes.size(); ++i) {
        const TempNode& tn = tempNodes[i];

        Node n = {};
        n.nodeId          = tn.id;
        n.parentId        = tn.parentId;
        n.address         = tn.address;
        n.level           = tn.level;
        n.childCount      = static_cast<uint32_t>(tn.childIds.size());
        n.firstChild      = childOffset;
        n.centroidOffset  = centroidOffsets[i];
        n.leafLookupBase  = tn.leafLookupBase;   // 0 for non-leaf
        n.leafLookupCount = tn.leafLookupCount;  // 0 for non-leaf
        n.flags           = tn.isLeaf ? 1 : 0;   // bit 0 = leaf node

        outNodes.write(reinterpret_cast<const char*>(&n), sizeof(Node));
        childOffset += n.childCount;
    }
    outNodes.close();
    cout << "  Written " << nodesFile << endl;

    // ── children.bin ──
    string childrenFile = binDir + "/children.bin";
    ofstream outChildren(childrenFile, ios::binary);
    for (uint32_t c : childrenBin) {
        outChildren.write(reinterpret_cast<const char*>(&c), sizeof(uint32_t));
    }
    outChildren.close();
    cout << "  Written " << childrenFile << endl;

    // ══════════════════════════════════════════
    // Summary
    // ══════════════════════════════════════════

    cout << endl;
    cout << "═══════════════════════════════════════════" << endl;
    cout << "  PIPELINE COMPLETE                        " << endl;
    cout << "═══════════════════════════════════════════" << endl;
    cout << "  Total nodes:     " << tempNodes.size() << endl;
    cout << "  Leaf nodes:      "
         << count_if(tempNodes.begin(), tempNodes.end(),
                     [](const TempNode& t){ return t.isLeaf; })
         << endl;
    cout << "  Parent nodes:    "
         << count_if(tempNodes.begin(), tempNodes.end(),
                     [](const TempNode& t){ return !t.isLeaf; })
         << endl;
    cout << "  Output files:    " << binDir << "/nodes.bin" << endl;
    cout << "                   " << binDir << "/children.bin" << endl;
    cout << "                   " << binDir << "/centroids.bin" << endl;
    cout << "                   " << binDir << "/embeddings.bin         (global chunk store)" << endl;
    cout << "                   " << binDir << "/leaf_centroids.bin     (404-byte row lookup)" << endl;
    cout << "═══════════════════════════════════════════" << endl;

    return 0;
}