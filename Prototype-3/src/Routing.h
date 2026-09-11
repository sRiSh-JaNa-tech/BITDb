#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include "probe_vectors.h"

// ─────────────────────────────────────────────────────────────────────────────
// Routing.h — Prototype-3 Segment Routing & Signature Math
//
// Resolves:
//   Shortcoming 1: Full 32-bit signature exploitation via avalanche bit-mixing
//   Shortcoming 3: Margin/confidence-adaptive multi-segment probing
// ─────────────────────────────────────────────────────────────────────────────

namespace BitDB {

static constexpr uint32_t NUM_SEGMENTS = 256;
static constexpr uint32_t DIMS         = 384;
static constexpr uint32_t BINARY_CODE_BYTES = DIMS / 8;

// ── MurmurHash3 32-bit finalizer (fmix32) ──
// Provides full bit avalanche: every single input bit of the 32-bit signature
// influences every bit of the output with ~50% probability.
inline uint32_t hash32_avalanche(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Map full 32-bit signature to a segment ID [0 .. NUM_SEGMENTS - 1]
inline uint32_t signature_to_segment(uint32_t mask) {
    return hash32_avalanche(mask) & (NUM_SEGMENTS - 1);
}

// Compute 32-bit probe bitmask from an int8 embedding
inline uint32_t compute_probe_bitmask(const int8_t* emb) {
    uint32_t mask = 0;
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < DIMS; ++j) {
            dot += static_cast<float>(emb[j]) * PROBE_VECTORS[i][j];
        }
        if (dot > 0.0f) {
            mask |= (1u << i);
        }
    }
    return mask;
}

// Compute 32-bit probe bitmask AND margin (absolute dot product) for each probe bit.
// Margin represents distance from the decision hyperplane (confidence).
inline uint32_t compute_probe_bitmask_and_margins(const int8_t* emb, float* out_margins) {
    uint32_t mask = 0;
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < DIMS; ++j) {
            dot += static_cast<float>(emb[j]) * PROBE_VECTORS[i][j];
        }
        if (dot > 0.0f) {
            mask |= (1u << i);
        }
        out_margins[i] = std::abs(dot);
    }
    return mask;
}

// Adaptive multi-probe segment selection:
// Sorts probe bits by confidence margin (ascending: lowest margin = closest to boundary).
// Flips the most uncertain bits in priority order to explore nearest neighbor segments.
inline std::vector<uint32_t> get_adaptive_probe_segments(uint32_t base_mask,
                                                         const float* margins,
                                                         size_t max_segments) {
    std::vector<uint32_t> segments;
    segments.reserve(max_segments);

    // Primary segment
    uint32_t primSeg = signature_to_segment(base_mask);
    segments.push_back(primSeg);
    if (max_segments <= 1) return segments;

    // Rank probe bits by margin (smallest margin first = closest to boundary)
    struct ProbeMargin {
        int   bit_index;
        float margin;
    };
    std::vector<ProbeMargin> ranked(P3_NUM_PROBES);
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        ranked[i] = {i, margins[i]};
    }
    std::sort(ranked.begin(), ranked.end(), [](const ProbeMargin& a, const ProbeMargin& b) {
        return a.margin < b.margin;
    });

    // 1-bit flips in order of lowest margin
    for (int i = 0; i < P3_NUM_PROBES && segments.size() < max_segments; ++i) {
        uint32_t flipped_mask = base_mask ^ (1u << ranked[i].bit_index);
        uint32_t seg = signature_to_segment(flipped_mask);
        if (std::find(segments.begin(), segments.end(), seg) == segments.end()) {
            segments.push_back(seg);
        }
    }

    // If more segments requested, do 2-bit flips on top 4 uncertain bits
    if (segments.size() < max_segments) {
        int topK_bits = std::min(4, P3_NUM_PROBES);
        for (int i = 0; i < topK_bits && segments.size() < max_segments; ++i) {
            for (int j = i + 1; j < topK_bits && segments.size() < max_segments; ++j) {
                uint32_t flipped_mask = base_mask ^ (1u << ranked[i].bit_index) ^ (1u << ranked[j].bit_index);
                uint32_t seg = signature_to_segment(flipped_mask);
                if (std::find(segments.begin(), segments.end(), seg) == segments.end()) {
                    segments.push_back(seg);
                }
            }
        }
    }

    return segments;
}

// Fast Hamming distance between two 32-bit signatures
inline int hamming_distance_32(uint32_t a, uint32_t b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(a ^ b);
#else
    uint32_t v = a ^ b;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}

inline void compute_binary_code(const int8_t* emb, uint8_t* out_code) {
    std::fill(out_code, out_code + BINARY_CODE_BYTES, uint8_t{0});
    for (uint32_t dim = 0; dim < DIMS; ++dim) {
        if (emb[dim] >= 0) {
            out_code[dim >> 3] |= static_cast<uint8_t>(1u << (dim & 7));
        }
    }
}

inline uint32_t hamming_distance_code(const uint8_t* a, const uint8_t* b) {
    uint32_t distance = 0;
    for (uint32_t i = 0; i < BINARY_CODE_BYTES; ++i) {
        const uint8_t x = static_cast<uint8_t>(a[i] ^ b[i]);
#if defined(__GNUC__) || defined(__clang__)
        distance += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned int>(x)));
#else
        uint8_t v = x;
        while (v) { distance += v & 1u; v >>= 1; }
#endif
    }
    return distance;
}

} // namespace BitDB
