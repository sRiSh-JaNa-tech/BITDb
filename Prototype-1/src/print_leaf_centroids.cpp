#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <algorithm>

#pragma pack(push, 1)
struct Node {
    uint32_t nodeId;
    uint32_t parentId;
    uint64_t address;
    uint32_t firstChild;
    uint32_t childCount;
    uint32_t centroidOffset;
    uint64_t leafLookupBase;
    uint32_t leafLookupCount;
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

// Read a null-terminated string from file at absolute offset
std::string read_null_terminated_string(std::ifstream& in, uint64_t offset) {
    uint64_t originalOffset = in.tellg();
    in.seekg(offset, std::ios::beg);
    
    std::string str;
    char ch;
    while (in.get(ch) && ch != '\0') {
        str += ch;
    }
    
    in.seekg(originalOffset, std::ios::beg);
    return str;
}

int main() {
    std::string binDir = "C:/Users/srish/Desktop/BitDB/BinStorage";

    // 1. Read nodes.bin to compute total rows and find ownership
    std::ifstream inNodes(binDir + "/nodes.bin", std::ios::binary);
    std::vector<Node> nodes;
    uint32_t totalRows = 0;
    if (inNodes) {
        Node n;
        while (inNodes.read(reinterpret_cast<char*>(&n), sizeof(Node))) {
            nodes.push_back(n);
            if ((n.flags & 1) != 0) { // Leaf node
                totalRows += n.leafLookupCount;
            }
        }
        inNodes.close();
    }

    // 2. Open leaf_centroids.bin
    std::ifstream inLeafCentroids(binDir + "/leaf_centroids.bin", std::ios::binary);
    if (!inLeafCentroids) {
        std::cerr << "Error: Could not open " << binDir << "/leaf_centroids.bin" << std::endl;
        return 1;
    }

    std::cout << "=== leaf_centroids.bin (File-level centroid lookup table) ===" << std::endl;
    std::cout << "Computed total rows: " << totalRows << " (Binary size of rows: " 
              << (totalRows * LEAF_ROW_SZ) << " bytes)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    uint64_t stringTableStart = totalRows * LEAF_ROW_SZ;

    for (uint32_t r = 0; r < totalRows; ++r) {
        uint64_t rowOffset = r * LEAF_ROW_SZ;
        
        LeafRow row;
        inLeafCentroids.seekg(rowOffset, std::ios::beg);
        inLeafCentroids.read(reinterpret_cast<char*>(&row), sizeof(LeafRow));

        // Read path from string table
        std::string filePath = read_null_terminated_string(inLeafCentroids, stringTableStart + row.textPathOffset);

        // Find which node owns this file row
        int ownerNodeId = -1;
        for (const auto& node : nodes) {
            if ((node.flags & 1) != 0 && 
                rowOffset >= node.leafLookupBase && 
                rowOffset < node.leafLookupBase + (node.leafLookupCount * LEAF_ROW_SZ)) {
                ownerNodeId = node.nodeId;
                break;
            }
        }

        std::cout << "Row Index      : " << r << " (Byte offset: " << rowOffset << ")" << std::endl;
        if (ownerNodeId != -1) {
            std::cout << "  Belongs to Node: " << ownerNodeId << std::endl;
        }
        std::cout << "  File Path      : " << filePath << std::endl;
        std::cout << "  Embeddings Addr: " << row.embeddingsAddr << " (in embeddings.bin)" << std::endl;
        std::cout << "  Embeddings Count: " << row.embeddingsCount << " chunks" << std::endl;
        std::cout << "  First 8 values of Centroid: ";
        for (size_t i = 0; i < std::min((size_t)8, (size_t)INT8_DIMS); ++i) {
            std::cout << (int)row.centroid[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    inLeafCentroids.close();
    return 0;
}
