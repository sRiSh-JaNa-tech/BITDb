#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include "probe_vectors.h"

// ─────────────────────────────────────────────────────────────────────────────
// Routing.h — Prototype-4 Multi-Index Hashing (MIH) & ADC Distance Math
// ─────────────────────────────────────────────────────────────────────────────

namespace BitDB {

static constexpr uint32_t NUM_SEGMENTS = 256;
static constexpr uint32_t DIMS         = 384;
static constexpr uint32_t BINARY_CODE_BYTES = DIMS / 8;

// ── Multi-Index Hashing (MIH) Substring Extraction ──
// Splits a 32-bit signature into 4 disjoint 8-bit substrings.
inline std::vector<uint8_t> extract_mih_substrings(uint32_t signature) {
    std::vector<uint8_t> substrings(4);
    substrings[0] = static_cast<uint8_t>(signature & 0xFF);
    substrings[1] = static_cast<uint8_t>((signature >> 8) & 0xFF);
    substrings[2] = static_cast<uint8_t>((signature >> 16) & 0xFF);
    substrings[3] = static_cast<uint8_t>((signature >> 24) & 0xFF);
    return substrings;
}

// Map signature to primary segment (uses first 8 bits, which are highest variance in SH/ITQ)
inline uint32_t signature_to_segment(uint32_t mask) {
    return mask & (NUM_SEGMENTS - 1);
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

// AVX2 vpshufb (SSSE3) Popcount for 48-byte codes
inline uint32_t hamming_distance_code(const uint8_t* a, const uint8_t* b) {
#if defined(__AVX2__)
    // Lookup table for 4-bit popcounts
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    uint32_t total_distance = 0;

    // Process first 32 bytes (256 bits)
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    __m256i vxor = _mm256_xor_si256(va, vb);

    __m256i lo = _mm256_and_si256(vxor, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vxor, 4), low_mask);

    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i popcnt = _mm256_add_epi8(popcnt1, popcnt2);

    // Sum the popcounts across the 256-bit register
    // _mm256_sad_epu8 sums 8-byte chunks
    __m256i sum = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
    total_distance += _mm256_extract_epi64(sum, 0);
    total_distance += _mm256_extract_epi64(sum, 1);
    total_distance += _mm256_extract_epi64(sum, 2);
    total_distance += _mm256_extract_epi64(sum, 3);

    // Process remaining 16 bytes manually
    for (uint32_t i = 32; i < BINARY_CODE_BYTES; ++i) {
        const uint8_t x = static_cast<uint8_t>(a[i] ^ b[i]);
#if defined(__GNUC__) || defined(__clang__)
        total_distance += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned int>(x)));
#else
        uint8_t v = x;
        while (v) { total_distance += v & 1u; v >>= 1; }
#endif
    }
    return total_distance;
#else
    // Fallback if AVX2 is not enabled
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
#endif
}

// ── Asymmetric Distance Computation (ADC) ──
// Inner product between unquantized query vector (int8_t[384])
// and binary-quantized candidate code (48 bytes, 384 bits).
// Each bit represents sign: bit=1 -> +1, bit=0 -> -1.
inline int32_t adc_score(const int8_t* queryVec, const uint8_t* binaryCode) {
    int32_t score = 0;
    for (uint32_t byte_idx = 0; byte_idx < BINARY_CODE_BYTES; ++byte_idx) {
        uint8_t byte = binaryCode[byte_idx];
        uint32_t dim_base = byte_idx * 8;
        for (uint32_t b = 0; b < 8; ++b) {
            int32_t q = static_cast<int32_t>(queryVec[dim_base + b]);
            score += (byte & (1u << b)) ? q : -q;
        }
    }
    return score;
}

} // namespace BitDB
