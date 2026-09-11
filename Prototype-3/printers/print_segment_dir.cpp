#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "../src/PathConfig.h"

using namespace std;

static constexpr uint32_t EXPECTED_VERSION = 3;

#pragma pack(push, 1)
struct SegDirHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_segments;
    uint32_t reserved;
};

struct SegEntry {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t ext_chain_head;
};

struct ExtentNode {
    uint64_t chunk_store_offset;
    uint32_t chunk_count;
    uint32_t next_extent_idx;
};
#pragma pack(pop)

int main() {
    const string path = PathConfig::getSegmentDirFile().string();
    const string extPath = PathConfig::getSegmentExtentsFile().string();

    ifstream f(path, ios::binary);
    if (!f) {
        cerr << "Cannot open " << path << "\n";
        return 1;
    }

    SegDirHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f || hdr.magic != 0x42444233u || hdr.version != EXPECTED_VERSION || hdr.num_segments == 0) {
        cerr << "Incompatible or corrupt segment index. Rebuild with Prototype-3 Build.exe --rebuild.\n";
        return 1;
    }
    printf("segment_dir.bin — magic=0x%08X version=%u segments=%u\n\n",
           hdr.magic, hdr.version, hdr.num_segments);

    vector<ExtentNode> extents;
    ifstream extIn(extPath, ios::binary);
    if (extIn) {
        ExtentNode node;
        while (extIn.read(reinterpret_cast<char*>(&node), sizeof(node))) {
            extents.push_back(node);
        }
    }

    printf("  %-5s | %-14s | %-10s | %-12s\n", "SegID", "ChunkOffset", "PrimaryCnt", "TotalWithExt");
    printf("  ------|----------------|------------|-------------\n");

    vector<uint32_t> totalPerSeg(hdr.num_segments, 0);
    uint32_t totalChunks = 0;
    uint32_t usedSegs    = 0;
    uint32_t minCount    = UINT32_MAX;
    uint32_t maxCount    = 0;

    for (uint32_t i = 0; i < hdr.num_segments; ++i) {
        SegEntry se;
        f.read(reinterpret_cast<char*>(&se), sizeof(se));

        uint32_t total = se.chunk_count;
        uint32_t extIdx = se.ext_chain_head;
        while (extIdx > 0 && extIdx <= extents.size()) {
            total += extents[extIdx - 1].chunk_count;
            extIdx = extents[extIdx - 1].next_extent_idx;
        }
        totalPerSeg[i] = total;

        if (total > 0) {
            printf("  %5u | %14llu | %10u | %10u\n", i,
                   (unsigned long long)se.chunk_store_offset, se.chunk_count, total);
            totalChunks += total;
            usedSegs++;
            minCount = min(minCount, total);
            maxCount = max(maxCount, total);
        }
    }
    if (usedSegs == 0) minCount = 0;

    // Statistical Analysis (Shortcoming 5)
    double mean = totalChunks / static_cast<double>(hdr.num_segments);
    double varSum = 0.0;
    for (uint32_t i = 0; i < hdr.num_segments; ++i) {
        double diff = totalPerSeg[i] - mean;
        varSum += diff * diff;
    }
    double stddev = sqrt(varSum / hdr.num_segments);
    double maxMeanRatio = mean > 0 ? (maxCount / mean) : 0.0;

    // Gini coefficient
    vector<uint32_t> sortedCounts = totalPerSeg;
    sort(sortedCounts.begin(), sortedCounts.end());
    double diffSum = 0.0;
    for (size_t i = 0; i < hdr.num_segments; ++i) {
        for (size_t j = 0; j < hdr.num_segments; ++j) {
            diffSum += abs(static_cast<double>(sortedCounts[i]) - static_cast<double>(sortedCounts[j]));
        }
    }
    double gini = (totalChunks > 0) ? (diffSum / (2.0 * hdr.num_segments * totalChunks)) : 0.0;

    // Percentiles
    uint32_t p50 = sortedCounts[hdr.num_segments / 2];
    uint32_t p95 = sortedCounts[static_cast<size_t>(hdr.num_segments * 0.95)];
    uint32_t p99 = sortedCounts[static_cast<size_t>(hdr.num_segments * 0.99)];

    printf("\n══════════════════════════════════════════════════\n");
    printf("  SEGMENT BALANCE & SKEW REPORT (Shortcoming 5)\n");
    printf("══════════════════════════════════════════════════\n");
    printf("  Total Chunks   : %u\n", totalChunks);
    printf("  Chained Extents: %zu\n", extents.size());
    printf("  Used Segments  : %u / %u (%.1f%%)\n", usedSegs, hdr.num_segments, (usedSegs * 100.0) / hdr.num_segments);
    printf("  Min Chunks/Seg : %u\n", minCount);
    printf("  Max Chunks/Seg : %u\n", maxCount);
    printf("  Mean Chunks/Seg: %.2f\n", mean);
    printf("  Std Deviation  : %.2f\n", stddev);
    printf("  Max/Mean Ratio : %.2fx\n", maxMeanRatio);
    printf("  Gini Index     : %.4f (0 = perfect equality)\n", gini);
    printf("  Percentiles    : p50=%u, p95=%u, p99=%u\n", p50, p95, p99);
    printf("══════════════════════════════════════════════════\n");

    return 0;
}
