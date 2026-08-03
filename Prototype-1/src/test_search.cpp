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
#include <chrono>
#include <unordered_map>

#include "embed.h"

using namespace std;
namespace fs = std::filesystem;

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

static int32_t compute_similarity_int8(const std::vector<int8_t>& vecA,
                                       const std::vector<int8_t>& vecB) {
    int32_t score = 0;
    for (size_t i = 0; i < INT8_DIMS; ++i) {
        score += static_cast<int32_t>(vecA[i]) * static_cast<int32_t>(vecB[i]);
    }
    return score;
}

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

void run_single_query(const string& queryText,
                      const vector<Node>& nodes,
                      const vector<uint32_t>& childrenArr,
                      uint32_t totalLeafRows,
                      const string& binDir) {
    cout << "\n======================================================\n";
    cout << "QUERY: \"" << queryText << "\"\n";
    cout << "======================================================\n";

    // 1. Embed query
    auto t_start_embed = chrono::high_resolution_clock::now();
    vector<vector<int8_t>> queryEmbeddings = embed_chunks({queryText});
    auto t_end_embed = chrono::high_resolution_clock::now();
    
    if (queryEmbeddings.empty() || queryEmbeddings[0].size() != INT8_DIMS) {
        cerr << "ERROR: Failed to generate a valid 384-dimensional query vector." << endl;
        return;
    }
    const vector<int8_t>& queryVec = queryEmbeddings[0];

    auto t_start_search = chrono::high_resolution_clock::now();

    // Open files
    ifstream inCentroids(binDir + "/centroids.bin", ios::binary);
    if (!inCentroids) {
        cerr << "ERROR: Could not open centroids.bin" << endl;
        return;
    }

    // 2. Beam search (width=2)
    static const int BEAM_WIDTH = 2;
    vector<pair<float, uint32_t>> beam = {{0.0f, 0u}};
    vector<uint32_t> leafBeam;

    while (!beam.empty()) {
        vector<pair<float, uint32_t>> nextBeam;

        for (auto& [beamScore, nodeIdx] : beam) {
            const Node& cur = nodes[nodeIdx];

            if ((cur.flags & 1) != 0) {
                leafBeam.push_back(nodeIdx);
                continue;
            }

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
                nextBeam.push_back({static_cast<float>(score), cid});
            }
        }

        if (nextBeam.empty()) break;

        sort(nextBeam.begin(), nextBeam.end(),
             [](const pair<float,uint32_t>& a, const pair<float,uint32_t>& b){
                 return a.first > b.first;
             });
        if ((int)nextBeam.size() > BEAM_WIDTH)
            nextBeam.resize(BEAM_WIDTH);

        beam = move(nextBeam);
    }
    inCentroids.close();

    sort(leafBeam.begin(), leafBeam.end());
    leafBeam.erase(unique(leafBeam.begin(), leafBeam.end()), leafBeam.end());

    if (leafBeam.empty()) {
        cerr << "ERROR: Beam search found no leaf nodes." << endl;
        return;
    }

    struct MatchResult {
        int32_t score;
        uint64_t sourceOffset;
        uint32_t sourceLength;
        uint32_t fileIdx;
    };

    // 3. Find matches by checking chunks directly
    ifstream inLeafCentroids(binDir + "/leaf_centroids.bin", ios::binary);
    ifstream inEmbeddings(binDir + "/embeddings.bin", ios::binary);
    if (!inLeafCentroids || !inEmbeddings) {
        cerr << "ERROR: Could not open leaf_centroids.bin or embeddings.bin" << endl;
        return;
    }

    vector<MatchResult> allMatches;
    vector<string> filePaths;

    for (uint32_t leafIdx : leafBeam) {
        const Node& leafNode = nodes[leafIdx];
        if (leafNode.leafLookupCount == 0) continue;

        for (uint32_t i = 0; i < leafNode.leafLookupCount; ++i) {
            uint64_t rowOffset = leafNode.leafLookupBase + (i * LEAF_ROW_SZ);
            inLeafCentroids.seekg(rowOffset, ios::beg);

            LeafRow row;
            inLeafCentroids.read(reinterpret_cast<char*>(&row), sizeof(LeafRow));

            uint64_t stringTableStart = totalLeafRows * LEAF_ROW_SZ;
            string filePath = read_null_terminated_string(inLeafCentroids, stringTableStart + row.textPathOffset);

            uint32_t fIdx = static_cast<uint32_t>(filePaths.size());
            filePaths.push_back(filePath);

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

    // 4. Load matching sentence text from file and display top 3
    int numToDisplay = min((int)allMatches.size(), 3);

    for (int idx = 0; idx < numToDisplay; ++idx) {
        const auto& match = allMatches[idx];
        const string& filePath = filePaths[match.fileIdx];
        
        ifstream inFile(filePath, ios::binary);
        if (!inFile) {
            cerr << "ERROR: Could not open source file " << filePath << endl;
            continue;
        }

        inFile.seekg(match.sourceOffset, ios::beg);
        string sentenceText(match.sourceLength, '\0');
        inFile.read(&sentenceText[0], match.sourceLength);
        
        cout << "  Rank " << (idx + 1) << " | Score: " << match.score 
             << " | File: " << fs::path(filePath).filename().string() << "\n";
        cout << "         Line:  \"" << sentenceText << "\"\n";
    }
    auto t_end_search = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed_embed = t_end_embed - t_start_embed;
    chrono::duration<double, milli> elapsed_search = t_end_search - t_start_search;
    cout << "Embedding Time: " << fixed << setprecision(2) << elapsed_embed.count() << " ms\n";
    cout << "Index Search:   " << elapsed_search.count() << " ms\n";
}

int main() {
    const string binDir = "C:/Users/srish/Desktop/BitDB/Prototype-1/BinStorage";

    // Load nodes.bin
    ifstream inNodes(binDir + "/nodes.bin", ios::binary);
    if (!inNodes) {
        cerr << "ERROR: Could not open nodes.bin. Run Node.exe first." << endl;
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
        cerr << "ERROR: Could not open children.bin" << endl;
        return 1;
    }
    vector<uint32_t> childrenArr;
    uint32_t childId;
    while (inChildren.read(reinterpret_cast<char*>(&childId), sizeof(uint32_t))) {
        childrenArr.push_back(childId);
    }
    inChildren.close();

    cout << "Initializing embedding model..." << endl;
    init_python();

    vector<string> queries = {
        "What is border gateway protocol BGP?",
        "Explain indexing strategies in relational databases.",
        "How do recurrent neural networks RNN handle sequential data?",
        "What is virtual memory and page replacement algorithms?",
        "What consensus algorithms are used in distributed databases?"
    };

    for (const auto& query : queries) {
        auto start = chrono::high_resolution_clock::now();
        run_single_query(query, nodes, childrenArr, totalLeafRows, binDir);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;
        cout << "QUERY LATENCY:  " << fixed << setprecision(2) << elapsed.count() << " ms\n";
    }

    finalize_python();
    cout << "\nAll 5 queries processed successfully.\n";
    return 0;
}
