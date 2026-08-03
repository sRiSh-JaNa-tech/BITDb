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

int main() {
    std::string binDir = "C:/Users/srish/Desktop/BitDB/Prototype-2/BinStorage";

    // 1. Read nodes.bin first to map centroid offsets to nodes
    std::ifstream inNodes(binDir + "/nodes.bin", std::ios::binary);
    std::vector<Node> nodes;
    if (inNodes) {
        Node n;
        while (inNodes.read(reinterpret_cast<char*>(&n), sizeof(Node))) {
            nodes.push_back(n);
        }
        inNodes.close();
    }

    // 2. Open centroids.bin
    std::ifstream inCentroids(binDir + "/centroids.bin", std::ios::binary);
    if (!inCentroids) {
        std::cerr << "Error: Could not open " << binDir << "/centroids.bin" << std::endl;
        return 1;
    }

    std::cout << "=== centroids.bin Sequential Read ===" << std::endl;
    uint32_t count = 0;
    while (true) {
        uint64_t currentOffset = inCentroids.tellg();
        uint32_t numBytes = 0;
        if (!inCentroids.read(reinterpret_cast<char*>(&numBytes), sizeof(numBytes))) {
            break; // EOF
        }

        std::vector<int8_t> centroid(numBytes);
        inCentroids.read(reinterpret_cast<char*>(centroid.data()), numBytes);

        // Find which node owns this centroid
        int ownerNodeId = -1;
        for (const auto& node : nodes) {
            if (node.centroidOffset == currentOffset) {
                ownerNodeId = node.nodeId;
                break;
            }
        }

        std::cout << "[" << count++ << "] Centroid Offset: " << currentOffset << std::endl;
        if (ownerNodeId != -1) {
            std::cout << "    Owner Node ID  : " << ownerNodeId << std::endl;
        } else {
            std::cout << "    Owner Node ID  : UNKNOWN (Orphaned centroid)" << std::endl;
        }
        std::cout << "    Num Int8 Bytes : " << numBytes << " dimensions" << std::endl;
        std::cout << "    First 8 values : ";
        for (size_t i = 0; i < std::min((size_t)8, (size_t)numBytes); ++i) {
            std::cout << (int)centroid[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    inCentroids.close();
    return 0;
}
