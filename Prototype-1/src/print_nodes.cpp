#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <string>

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

// Structure to hold the read centroid data
struct CentroidInfo {
    uint32_t numBytes;
    std::vector<int8_t> values;
};

// Helper to read centroid information from centroids.bin given an offset
CentroidInfo read_centroid(std::ifstream& inCentroids, uint32_t offset) {
    CentroidInfo info = {0, {}};
    if (!inCentroids) return info;

    inCentroids.seekg(offset, std::ios::beg);
    inCentroids.read(reinterpret_cast<char*>(&info.numBytes), sizeof(info.numBytes));
    if (inCentroids.gcount() != sizeof(info.numBytes)) return info; // EOF or error
    
    info.values.resize(info.numBytes);
    inCentroids.read(reinterpret_cast<char*>(info.values.data()), info.numBytes);
    
    return info;
}

int main() {
    std::string binDir = "C:/Users/srish/Desktop/BitDB/BinStorage";
    
    // Open nodes.bin
    std::ifstream inNodes(binDir + "/nodes.bin", std::ios::binary);
    if (!inNodes) {
        std::cerr << "Could not open " << binDir << "/nodes.bin. Did you run Node.exe first?" << std::endl;
        return 1;
    }

    // Open centroids.bin
    std::ifstream inCentroids(binDir + "/centroids.bin", std::ios::binary);
    if (!inCentroids) {
        std::cerr << "Warning: Could not open " << binDir << "/centroids.bin." << std::endl;
    }

    std::vector<Node> nodes;
    Node n;
    while (inNodes.read(reinterpret_cast<char*>(&n), sizeof(Node))) {
        nodes.push_back(n);
    }
    inNodes.close();

    std::cout << "Read " << nodes.size() << " nodes from nodes.bin." << std::endl;
    std::cout << "========================================================" << std::endl;

    for (const auto& node : nodes) {
        std::cout << "Node ID        : " << node.nodeId << std::endl;
        std::cout << "Parent ID      : " << node.parentId << std::endl;
        
        // Print the address in hex so it's easier to see the hierarchical routing
        std::cout << "Address        : 0x" << std::hex << std::setfill('0') << std::setw(16) << node.address << std::dec << std::endl;
        std::cout << "Level          : " << node.level << std::endl;
        
        bool isLeaf = (node.flags & 1) != 0;
        std::cout << "Type           : " << (isLeaf ? "LEAF" : "PARENT") << std::endl;
        
        if (isLeaf) {
            std::cout << "Leaf Lookup Base: " << node.leafLookupBase << std::endl;
            std::cout << "Leaf Lookup Count: " << node.leafLookupCount << std::endl;
        } else {
            std::cout << "Children       : " << node.childCount << " (Starts at offset " << node.firstChild << " in children.bin)" << std::endl;
        }

        // Print centroid info
        if (inCentroids) {
            CentroidInfo centroid = read_centroid(inCentroids, node.centroidOffset);
            std::cout << "Int8 Centroid  : Offset=" << node.centroidOffset 
                      << ", Dimensions=" << centroid.numBytes;
            if (centroid.numBytes > 0) {
                std::cout << ", First 5 values: ";
                for (size_t i = 0; i < std::min((size_t)5, (size_t)centroid.numBytes); ++i) {
                    std::cout << (int)centroid.values[i] << " ";
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "Int8 Centroid  : Offset=" << node.centroidOffset << std::endl;
        }
        
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    if (inCentroids) inCentroids.close();

    return 0;
}
