#include <iostream>
#include <fstream>
#include <cstdint>
#include "../src/PathConfig.h"

using namespace std;

#pragma pack(push, 1)
struct SegDirHeader { uint32_t magic; uint32_t version; uint32_t num_segments; uint32_t reserved; };
struct SegEntry     { uint64_t chunk_store_offset; uint32_t chunk_count; uint32_t reserved; };
#pragma pack(pop)

int main() {
    const string path = PathConfig::getSegmentDirFile().string();
    ifstream f(path, ios::binary);
    if (!f) { cerr << "Cannot open " << path << "\n"; return 1; }

    SegDirHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    printf("segment_dir.bin — magic=0x%08X version=%u segments=%u\n\n",
           hdr.magic, hdr.version, hdr.num_segments);
    printf("  %-5s | %-14s | %-10s\n", "SegID", "ChunkOffset", "ChunkCount");
    printf("  ------|----------------|----------\n");

    uint32_t totalChunks = 0;
    uint32_t usedSegs    = 0;
    SegEntry se;
    for (uint32_t i = 0; i < hdr.num_segments; ++i) {
        f.read(reinterpret_cast<char*>(&se), sizeof(se));
        if (se.chunk_count > 0) {
            printf("  %5u | %14llu | %10u\n", i,
                   (unsigned long long)se.chunk_store_offset, se.chunk_count);
            totalChunks += se.chunk_count;
            usedSegs++;
        }
    }
    printf("\n  Used segments: %u / %u\n", usedSegs, hdr.num_segments);
    printf("  Total chunks : %u\n", totalChunks);
    return 0;
}
