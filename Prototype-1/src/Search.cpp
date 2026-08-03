#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <unordered_map>

#include "embed.h"

using namespace std;
namespace fs = std::filesystem;

// ──────────────────────────────────────────────
// Struct definitions matching print_nodes.cpp
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

static constexpr uint32_t INT8_DIMS   = 384;
static constexpr uint32_t LEAF_ROW_SZ = 404;

#pragma pack(push, 1)
struct LeafRow {
    int8_t   centroid[INT8_DIMS];   // 384
    uint64_t embeddingsAddr;        //   8
    uint32_t embeddingsCount;       //   4
    uint64_t textPathOffset;        //   8
};
#pragma pack(pop)

static_assert(sizeof(LeafRow) == LEAF_ROW_SZ, "LeafRow must be exactly 404 bytes");

// ──────────────────────────────────────────────
// Similarity Metrics
// ──────────────────────────────────────────────

// Dot product between two int8 vectors
static int32_t compute_similarity_int8(const std::vector<int8_t>& vecA,
                                       const std::vector<int8_t>& vecB) {
    int32_t score = 0;
    for (size_t i = 0; i < INT8_DIMS; ++i) {
        score += static_cast<int32_t>(vecA[i]) * static_cast<int32_t>(vecB[i]);
    }
    return score;
}

// Helper to read null-terminated string from string table
static string read_null_terminated_string(ifstream& in, uint64_t offset) {
    uint64_t original = in.tellg();
    in.seekg(offset, ios::beg);
    string result;
    char ch;
    while (in.get(ch) && ch != '\0') {
        result += ch;
    }
    in.seekg(original, ios::beg);
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " \"<search query text>\" [N_results]" << endl;
        return 1;
    }
    string queryText = argv[1];
    int topN = 3; // Default to 3
    if (argc >= 3) {
        try {
            topN = std::stoi(argv[2]);
        } catch (...) {
            std::cerr << "Warning: Invalid N parameter, defaulting to 3." << std::endl;
        }
    }

    const string binDir = "C:/Users/srish/Desktop/BitDB/Prototype-1/BinStorage";

    // ──────────────────────────────────────────
    // Step 1: Embed query via Python
    // ──────────────────────────────────────────
    cout << "[Search] Initializing embedding model..." << endl;
    init_python();

    cout << "[Search] Generating query vector for: \"" << queryText << "\"" << endl;
    vector<vector<int8_t>> queryEmbeddings = embed_chunks({queryText});
    if (queryEmbeddings.empty() || queryEmbeddings[0].size() != INT8_DIMS) {
        cerr << "[Search] ERROR: Failed to generate a valid 384-dimensional query vector." << endl;
        finalize_python();
        return 1;
    }
    const vector<int8_t>& queryVec = queryEmbeddings[0];

    // ──────────────────────────────────────────
    // Load Nodes index to traverse the tree
    // ──────────────────────────────────────────
    ifstream inNodes(binDir + "/nodes.bin", ios::binary);
    if (!inNodes) {
        cerr << "[Search] ERROR: Could not open " << binDir << "/nodes.bin. Run Node.exe first." << endl;
        finalize_python();
        return 1;
    }
    vector<Node> nodes;
    Node n;
    uint32_t totalLeafRows = 0;
    while (inNodes.read(reinterpret_cast<char*>(&n), sizeof(Node))) {
        nodes.push_back(n);
        if ((n.flags & 1) != 0) {
            totalLeafRows += n.leafLookupCount;
        }
    }
    inNodes.close();

    // Load parent-child mapping index
    ifstream inChildren(binDir + "/children.bin", ios::binary);
    if (!inChildren) {
        cerr << "[Search] ERROR: Could not open " << binDir << "/children.bin" << endl;
        finalize_python();
        return 1;
    }
    vector<uint32_t> childrenArr;
    uint32_t childId;
    while (inChildren.read(reinterpret_cast<char*>(&childId), sizeof(uint32_t))) {
        childrenArr.push_back(childId);
    }
    inChildren.close();

    // Open centroids binary
    ifstream inCentroids(binDir + "/centroids.bin", ios::binary);
    if (!inCentroids) {
        cerr << "[Search] ERROR: Could not open " << binDir << "/centroids.bin" << endl;
        finalize_python();
        return 1;
    }

    // ──────────────────────────────────────────
    // Step 2: Beam search (width=2) to find best leaf node(s)
    // At each level keep top-2 scoring children to avoid wrong-branch selection
    // when root centroids are over-averaged and ambiguous.
    // ──────────────────────────────────────────
    cout << "\n[Search] Traversing index tree (beam width=2)..." << endl;

    static const int BEAM_WIDTH = 2;

    // Each beam entry: {accumulated score, nodeIdx}
    vector<pair<float, uint32_t>> beam = {{0.0f, 0u}};  // start at root
    vector<uint32_t> leafBeam;   // leaves reached

    while (!beam.empty()) {
        vector<pair<float, uint32_t>> nextBeam;

        for (auto& [beamScore, nodeIdx] : beam) {
            const Node& cur = nodes[nodeIdx];

            if ((cur.flags & 1) != 0) {
                // Already a leaf — keep it
                leafBeam.push_back(nodeIdx);
                continue;
            }

            // Score all children and collect into nextBeam candidates
            cout << "  Scoring children of Node [" << cur.nodeId << "]:" << endl;
            for (uint32_t c = 0; c < cur.childCount; ++c) {
                uint32_t childArrIdx = cur.firstChild + c;
                if (childArrIdx >= childrenArr.size()) continue;
                uint32_t cid = childrenArr[childArrIdx];

                inCentroids.seekg(nodes[cid].centroidOffset, ios::beg);
                uint32_t numBytes = 0;
                inCentroids.read(reinterpret_cast<char*>(&numBytes), sizeof(numBytes));
                
                vector<int8_t> centroidBits(numBytes);
                inCentroids.read(reinterpret_cast<char*>(centroidBits.data()), numBytes);

                int32_t score = compute_similarity_int8(queryVec, centroidBits);
                cout << "    Node [" << cid << "] score=" << score << endl;
                nextBeam.push_back({static_cast<float>(score), cid});
            }
        }

        if (nextBeam.empty()) break;

        // Keep top BEAM_WIDTH candidates by score
        sort(nextBeam.begin(), nextBeam.end(),
             [](const pair<float,uint32_t>& a, const pair<float,uint32_t>& b){
                 return a.first > b.first;
             });
        if ((int)nextBeam.size() > BEAM_WIDTH)
            nextBeam.resize(BEAM_WIDTH);

        cout << "  => Beam keeps: ";
        for (auto& [s, idx] : nextBeam)
            cout << "Node [" << nodes[idx].nodeId << "] (score=" << s << ")  ";
        cout << endl;

        beam = move(nextBeam);
    }
    
    inCentroids.close();

    // Deduplicate leaves just in case different beam branches merged
    sort(leafBeam.begin(), leafBeam.end());
    leafBeam.erase(unique(leafBeam.begin(), leafBeam.end()), leafBeam.end());

    if (leafBeam.empty()) {
        cerr << "[Search] ERROR: Beam search found no leaf nodes." << endl;
        finalize_python();
        return 1;
    }

    cout << "\n  Beam reached " << leafBeam.size() << " unique leaf node(s): ";
    for (uint32_t li : leafBeam) cout << "[" << nodes[li].nodeId << "] ";
    cout << endl;

    struct MatchResult {
        int32_t score;
        uint64_t sourceOffset;
        uint32_t sourceLength;
        uint32_t fileIdx;
    };

    // ──────────────────────────────────────────
    // Step 3: Find Best Files & Sentences by checking chunks directly
    // ──────────────────────────────────────────
    cout << "\n[Search] Matching best sentences across all files in selected leaves..." << endl;
    ifstream inLeafCentroids(binDir + "/leaf_centroids.bin", ios::binary);
    ifstream inEmbeddings(binDir + "/embeddings.bin", ios::binary);
    if (!inLeafCentroids || !inEmbeddings) {
        cerr << "[Search] ERROR: Could not open leaf_centroids.bin or embeddings.bin" << endl;
        finalize_python();
        return 1;
    }

    vector<MatchResult> allMatches;
    vector<string> filePaths;

    for (uint32_t leafIdx : leafBeam) {
        const Node& leafNode = nodes[leafIdx];
        if (leafNode.leafLookupCount == 0) continue;

        // Iterate over files in this leaf
        for (uint32_t i = 0; i < leafNode.leafLookupCount; ++i) {
            uint64_t rowOffset = leafNode.leafLookupBase + (i * LEAF_ROW_SZ);
            inLeafCentroids.seekg(rowOffset, ios::beg);

            LeafRow row;
            inLeafCentroids.read(reinterpret_cast<char*>(&row), sizeof(LeafRow));

            // Get file path
            uint64_t stringTableStart = totalLeafRows * LEAF_ROW_SZ;
            string filePath = read_null_terminated_string(inLeafCentroids, stringTableStart + row.textPathOffset);
            
            uint32_t fIdx = static_cast<uint32_t>(filePaths.size());
            filePaths.push_back(filePath);

            // Read chunks for this file
            inEmbeddings.seekg(row.embeddingsAddr, ios::beg);
            uint32_t numChunks = 0;
            inEmbeddings.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
            uint16_t dim = 0;
            inEmbeddings.read(reinterpret_cast<char*>(&dim), sizeof(dim));

            vector<int8_t> chunkVec(dim);
            for (uint32_t c = 0; c < numChunks; ++c) {
                inEmbeddings.read(reinterpret_cast<char*>(chunkVec.data()), dim);
                uint64_t sourceOffset = 0;
                uint32_t sourceLength = 0;
                inEmbeddings.read(reinterpret_cast<char*>(&sourceOffset), sizeof(sourceOffset));
                inEmbeddings.read(reinterpret_cast<char*>(&sourceLength), sizeof(sourceLength));

                int32_t score = compute_similarity_int8(queryVec, chunkVec);
                allMatches.push_back({score, sourceOffset, sourceLength, fIdx});
            }
        }
    }

    inLeafCentroids.close();
    inEmbeddings.close();

    // Sort by score descending
    sort(allMatches.begin(), allMatches.end(), [](const MatchResult& a, const MatchResult& b) {
        return a.score > b.score;
    });

    // ──────────────────────────────────────────
    // Step 4: Load matching sentences text from original files and display top N
    // ──────────────────────────────────────────
    cout << "\n================== BEST MATCHES FOUND ==================" << endl;

    int numToDisplay = min((int)allMatches.size(), topN);

    for (int idx = 0; idx < numToDisplay; ++idx) {
        const auto& match = allMatches[idx];
        const string& filePath = filePaths[match.fileIdx];
        
        ifstream inFile(filePath, ios::binary);
        if (!inFile) {
            cerr << "[Search] ERROR: Could not open source file " << filePath << endl;
            continue;
        }

        inFile.seekg(match.sourceOffset, ios::beg);
        string sentenceText(match.sourceLength, '\0');
        inFile.read(&sentenceText[0], match.sourceLength);
        
        cout << "Rank " << (idx + 1) << " | Score: " << match.score 
             << " | File: " << filePath << endl;
        cout << "Sentence: \"" << sentenceText << "\"\n" << endl;
    }
    cout << "========================================================" << endl;

    // Cleanup Python
    finalize_python();
    return 0;
}
