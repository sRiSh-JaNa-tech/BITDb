#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>

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

    // 1. Read children.bin
    std::ifstream inChildren(binDir + "/children.bin", std::ios::binary);
    if (!inChildren) {
        std::cerr << "Error: Could not open " << binDir << "/children.bin" << std::endl;
        return 1;
    }

    std::vector<uint32_t> children;
    uint32_t childId;
    while (inChildren.read(reinterpret_cast<char*>(&childId), sizeof(uint32_t))) {
        children.push_back(childId);
    }
    inChildren.close();

    // 2. Read nodes.bin for parent context
    std::ifstream inNodes(binDir + "/nodes.bin", std::ios::binary);
    std::vector<Node> nodes;
    if (inNodes) {
        Node n;
        while (inNodes.read(reinterpret_cast<char*>(&n), sizeof(Node))) {
            nodes.push_back(n);
        }
        inNodes.close();
    }

    std::cout << "=== children.bin flat array (Size: " << children.size() << " elements) ===" << std::endl;
    for (size_t i = 0; i < children.size(); ++i) {
        std::cout << "[" << i << "] Node ID: " << children[i] << std::endl;
    }
    std::cout << std::endl;

    if (!nodes.empty()) {
        std::cout << "=== Reconstructed Parent -> Child Relationships ===" << std::endl;
        for (const auto& node : nodes) {
            if (node.childCount > 0) {
                bool isLeaf = (node.flags & 1) != 0;
                std::cout << "Parent Node ID " << node.nodeId 
                          << " (Address: 0x" << std::hex << std::setfill('0') << std::setw(16) << node.address << std::dec 
                          << ", Level: " << node.level << "):" << std::endl;
                
                std::cout << "  Children count: " << node.childCount 
                          << ", starts at children.bin index: " << node.firstChild << std::endl;
                
                for (uint32_t c = 0; c < node.childCount; ++c) {
                    uint32_t index = node.firstChild + c;
                    if (index < children.size()) {
                        uint32_t cid = children[index];
                        std::cout << "    -> Child [" << c << "] Node ID: " << cid << std::endl;
                    } else {
                        std::cout << "    -> Child [" << c << "] INDEX OUT OF BOUNDS (" << index << ")" << std::endl;
                    }
                }
                std::cout << "--------------------------------------------------------" << std::endl;
            }
        }
    }

    return 0;
}
