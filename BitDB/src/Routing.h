#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "probe_vectors.h"

// ─────────────────────────────────────────────────────────────────────────────
// Routing.h — Prototype-4 Multi-Index Hashing (MIH) & ADC Distance Math
// ─────────────────────────────────────────────────────────────────────────────

namespace BitDB {

static constexpr uint32_t NUM_SEGMENTS = 256;
static constexpr uint32_t DIMS         = 384;
static constexpr uint32_t BINARY_CODE_BYTES = DIMS / 8;

// ── CPU Capability Detection ──
inline bool cpu_supports_avx2() {
#if defined(__AVX2__)
    return true;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5: AVX2
    }
    return false;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return __builtin_cpu_supports("avx2") > 0;
#else
    return false;
#endif
}

inline bool cpu_supports_popcnt() {
#if defined(__POPCNT__)
    return true;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 1) {
        __cpuidex(cpuInfo, 1, 0);
        return (cpuInfo[2] & (1 << 23)) != 0; // ECX bit 23: POPCNT
    }
    return false;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    return __builtin_cpu_supports("popcnt") > 0;
#else
    return false;
#endif
}

// ── Portable Popcount Utilities ──
inline int popcount32(uint32_t x) {
#if defined(_MSC_VER)
    #if defined(_M_X64) || defined(_M_IX86)
    return static_cast<int>(__popcnt(x));
    #else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
#endif
}

// Fast Hamming distance between two 32-bit signatures
inline int hamming_distance_32(uint32_t a, uint32_t b) {
    return popcount32(a ^ b);
}

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

// ── Normalised Probe Vector Cache ──
// The raw PROBE_VECTORS are Halton low-discrepancy sequences and are NOT
// unit-normalised. The sign test (dot > 0) is scale-invariant so routing is
// unaffected. However, the absolute margin values used for multi-probe ranking
// must be comparable across probes, which requires equal-norm vectors.
//
// We cache a normalised copy once (lazy-static via a local bool).
namespace detail {
inline const float (&get_normalised_probes())[P3_NUM_PROBES][P3_DIMS] {
    static float norm_probes[P3_NUM_PROBES][P3_DIMS];
    static bool initialised = false;
    if (!initialised) {
        for (int i = 0; i < P3_NUM_PROBES; ++i) {
            float norm_sq = 0.0f;
            for (int j = 0; j < P3_DIMS; ++j) {
                norm_sq += PROBE_VECTORS[i][j] * PROBE_VECTORS[i][j];
            }
            float inv_norm = (norm_sq > 1e-12f) ? (1.0f / std::sqrt(norm_sq)) : 1.0f;
            for (int j = 0; j < P3_DIMS; ++j) {
                norm_probes[i][j] = PROBE_VECTORS[i][j] * inv_norm;
            }
        }
        initialised = true;
    }
    return norm_probes;
}
} // namespace detail

// Compute 32-bit probe bitmask from an int8 embedding
inline uint32_t compute_probe_bitmask(const int8_t* emb) {
    uint32_t mask = 0;
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < DIMS; ++j) {
            float centered_val = (static_cast<float>(emb[j]) / 127.0f) - CENTROID_VECTOR[j];
            dot += centered_val * PROBE_VECTORS[i][j];
        }
        if (dot >= 0.0f) {
            mask |= (1u << i);
        }
    }
    return mask;
}

// Compute 32-bit probe bitmask AND normalised margin (absolute dot product)
// for each probe bit. Margins are now directly comparable across probes because
// the normalised probe vectors all have unit L2 norm.
inline uint32_t compute_probe_bitmask_and_margins(const int8_t* emb, float* out_margins) {
    const float (&np)[P3_NUM_PROBES][P3_DIMS] = detail::get_normalised_probes();
    uint32_t mask = 0;
    for (int i = 0; i < P3_NUM_PROBES; ++i) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < DIMS; ++j) {
            float centered_val = (static_cast<float>(emb[j]) / 127.0f) - CENTROID_VECTOR[j];
            dot += centered_val * np[i][j];
        }
        if (dot >= 0.0f) {
            mask |= (1u << i);
        }
        out_margins[i] = std::abs(dot);
    }
    return mask;
}

inline void compute_binary_code(const int8_t* emb, uint8_t* out_code) {
    std::fill(out_code, out_code + BINARY_CODE_BYTES, uint8_t{0});
    for (uint32_t dim = 0; dim < DIMS; ++dim) {
        if (emb[dim] >= 0) {
            out_code[dim >> 3] |= static_cast<uint8_t>(1u << (dim & 7));
        }
    }
}

// ── Scalar Fallback Hamming Distance for 48-byte codes ──
// Operates on 64-bit words, portable to all platforms without AVX2.
inline uint32_t hamming_distance_code_scalar(const uint8_t* a, const uint8_t* b) {
    uint32_t total_distance = 0;
    for (uint32_t i = 0; i < BINARY_CODE_BYTES / 8; ++i) {
        uint64_t wa = 0, wb = 0;
        std::memcpy(&wa, a + i * 8, 8);
        std::memcpy(&wb, b + i * 8, 8);
        uint64_t diff = wa ^ wb;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        #if defined(_M_X64)
        total_distance += static_cast<uint32_t>(__popcnt64(diff));
        #else
        total_distance += static_cast<uint32_t>(__popcnt(static_cast<uint32_t>(diff)) + __popcnt(static_cast<uint32_t>(diff >> 32)));
        #endif
#elif defined(__GNUC__) || defined(__clang__)
        total_distance += static_cast<uint32_t>(__builtin_popcountll(diff));
#else
        diff = diff - ((diff >> 1) & 0x5555555555555555ULL);
        diff = (diff & 0x3333333333333333ULL) + ((diff >> 2) & 0x3333333333333333ULL);
        diff = (diff + (diff >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        total_distance += static_cast<uint32_t>((diff * 0x0101010101010101ULL) >> 56);
#endif
    }
    return total_distance;
}

// ── AVX2 Vectorized Popcount for 48-byte codes ──
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,ssse3")))
#endif
inline uint32_t hamming_distance_code_avx2(const uint8_t* a, const uint8_t* b) {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
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
    total_distance += static_cast<uint32_t>(_mm256_extract_epi64(sum, 0));
    total_distance += static_cast<uint32_t>(_mm256_extract_epi64(sum, 1));
    total_distance += static_cast<uint32_t>(_mm256_extract_epi64(sum, 2));
    total_distance += static_cast<uint32_t>(_mm256_extract_epi64(sum, 3));

    // Process remaining 16 bytes using 64-bit scalar popcount (2 words of 8 bytes)
    for (uint32_t i = 4; i < 6; ++i) {
        uint64_t wa = 0, wb = 0;
        std::memcpy(&wa, a + i * 8, 8);
        std::memcpy(&wb, b + i * 8, 8);
        uint64_t diff = wa ^ wb;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        #if defined(_M_X64)
        total_distance += static_cast<uint32_t>(__popcnt64(diff));
        #else
        total_distance += static_cast<uint32_t>(__popcnt(static_cast<uint32_t>(diff)) + __popcnt(static_cast<uint32_t>(diff >> 32)));
        #endif
#elif defined(__GNUC__) || defined(__clang__)
        total_distance += static_cast<uint32_t>(__builtin_popcountll(diff));
#else
        diff = diff - ((diff >> 1) & 0x5555555555555555ULL);
        diff = (diff & 0x3333333333333333ULL) + ((diff >> 2) & 0x3333333333333333ULL);
        diff = (diff + (diff >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        total_distance += static_cast<uint32_t>((diff * 0x0101010101010101ULL) >> 56);
#endif
    }
    return total_distance;
#else
    return hamming_distance_code_scalar(a, b);
#endif
}

// ── Dynamic Dispatcher ──
// Uses AVX2 when compiled with AVX2 or detected on the CPU at runtime,
// otherwise transparently falls back to scalar implementation.
inline uint32_t hamming_distance_code(const uint8_t* a, const uint8_t* b) {
#if defined(__AVX2__)
    return hamming_distance_code_avx2(a, b);
#else
    static const bool has_avx2 = cpu_supports_avx2();
    if (has_avx2) {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
        return hamming_distance_code_avx2(a, b);
#else
        return hamming_distance_code_scalar(a, b);
#endif
    }
    return hamming_distance_code_scalar(a, b);
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
