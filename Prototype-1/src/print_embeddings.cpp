#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>

int main() {
    std::string binDir = "C:/Users/srish/Desktop/BitDB/BinStorage";

    std::ifstream inEmbeddings(binDir + "/embeddings.bin", std::ios::binary);
    if (!inEmbeddings) {
        std::cerr << "Error: Could not open " << binDir << "/embeddings.bin" << std::endl;
        return 1;
    }

    std::cout << "=== embeddings.bin (Global Chunk Embeddings Store) ===" << std::endl;
    uint32_t count = 0;
    while (true) {
        uint64_t currentOffset = inEmbeddings.tellg();
        uint32_t numChunks = 0;
        if (!inEmbeddings.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks))) {
            break; // EOF
        }

        uint16_t dim = 0;
        inEmbeddings.read(reinterpret_cast<char*>(&dim), sizeof(dim));

        std::cout << "[" << count++ << "] Block Offset: " << currentOffset << std::endl;
        std::cout << "    Num Chunks  : " << numChunks << std::endl;
        std::cout << "    Dimension   : " << dim << std::endl;

        // Read the first chunk to show a sample if there are chunks
        if (numChunks > 0 && dim > 0) {
            std::vector<int8_t> sample(dim);
            inEmbeddings.read(reinterpret_cast<char*>(sample.data()), dim);

            std::cout << "    Sample First Chunk int8 values (first 10 dims): ";
            for (int j = 0; j < std::min((int)dim, 10); ++j) {
                std::cout << (int)sample[j] << " ";
            }
            std::cout << "..." << std::endl;

            // Skip remaining chunks in this file's block to advance to the next file block
            if (numChunks > 1) {
                inEmbeddings.seekg((numChunks - 1) * dim, std::ios::cur);
            }
        }
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    inEmbeddings.close();
    return 0;
}
