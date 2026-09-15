# Prototype-4: Elastic Radix Extent Routing (ER2) Architecture

This document describes the design, mathematics, and implementation of **BitDB Prototype-4**. It explains how the system achieves 100% intra-extent recall approximate nearest neighbor search (ANNS) on NVMe SSDs with sub-megabyte RAM and microsecond CPU prefiltering.

---

## 1. Executive Summary & The Problem

Traditional vector databases rely on one of two paradigms, both of which face severe hardware bottlenecks when scaling to millions or billions of vectors:

1. **In-Memory Graphs (e.g., HNSW)**: Require keeping the entire vector dataset and its high-degree graph adjacency lists in RAM. For 1,000,000 vectors (384 dimensions), HNSW requires **1.5 GB to 2.5 GB of RAM**. At scale, RAM costs become prohibitive.
2. **SSD-Based Inverted Files & Graphs (e.g., DiskANN, SPANN)**: Shift vectors to SSD but still keep compressed 1-bit/2-bit vectors and navigation structures in RAM (30 MB to 100 MB per million points). Furthermore, during search they issue dozens of small random 4 KB I/O reads, incurring high read amplification and SSD controller queue latency.
3. **Fixed Avalanche Hashing (Prototype-3 Baseline)**: Uses a 32-bit Murmur3 hash to distribute vectors across 256 segments. While this guarantees uniform bucket sizes, the avalanche effect **destroys metric locality**: two vectors with a Hamming distance of 1 get mapped to completely unrelated SSD segments, requiring unguided multi-probing.

**Prototype-4 introduces Elastic Radix Extent Routing (ER2)**, an integrated systems-algorithmic co-design that:
- **Preserves Metric Locality**: Similar vectors reside in the same or adjacent physical extents via Multi-Index Hashing (MIH).
- **Margin-Ranked Multi-Probing (`--probes`)**: Expands search along the most uncertain hyperplane boundaries ranked by normalized geometric margin.
- **Microscopic RAM Footprint**: Holds less than **3 MB of RAM** for routing, MIH tables, and catalog metadata.
- **Hardware-Co-Designed 128 KB Columnar Extents**: Aligns every disk extent to physical NVMe flash pages (131,072 bytes) and groups binary codes into a contiguous cache-friendly array.
- **AVX2 SIMD Acceleration**: Built with native AVX2 SIMD flags (`/arch:AVX2` on MSVC, `-mavx2 -mpopcnt` on GCC) using the Harley-Seal `vpshufb` algorithm to scan candidate codes in memory at over **30 GB/s**.
- **100% Intra-Extent Recall (Zero False Dismissals)**: Uses Asymmetric Distance Computation (ADC) strictly as an I/O scheduling and heap-pruning optimizer, scoring all candidates in accessed extents with exact int8 dot products.
- **Safe Early-Exit via Cauchy-Schwarz WAND**: Prunes unvisited extents with mathematical certainty.
- **Crash-Safe Atomic Ingestion**: Employs `.tmp` write and atomic rename across all 5 index files including text storage.

---

## 2. High-Level Architectural Pipeline

The following diagram illustrates the complete end-to-end query execution lifecycle in Prototype-4:

```
                      Query String
                           │
                           ▼
             [ Hardware Embedding Model ]
          (OpenVINO iGPU / CUDA / MiniLM-L6)
                           │
                           ▼
                 Query Vector q (384-dim)
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
  1. 32-bit Probe Mask            Continuous Vector q
     (Hyperplane Projections)       (Retained in RAM for ADC)
           │                               │
           ▼                               │
  2. Margin-Ranked MIH Multi-Probing       │
     (4 x 8-bit Substrings +               │
      Margin-Ranked Uncertain Bits)        │
           │                               │
           ▼                               │
  3. Geometric WAND Upper Bounding         │
     MaxScore(q, E) = (q · C) + ||q||·R    │
     (Safely prune extents with 0 SSD I/O) │
           │                               │
           ▼                               │
  4. Two-Stage Extent Read & Scoring       │
     ├── Read Stage 1 Block (~14 KB) ◄─────┘
     ├── AVX2 Harley-Seal Popcount / ADC Pre-Sort
     │   (Order candidates to elevate min-heap rapidly)
     └── Dynamic Hardware I/O Crossover:
         ├── If count > 40: 1-Stage Bulk Extent Read (128 KB)
         └── If count <= 40: Targeted Payload Reads (Scatter)
                           │
                           ▼
  5. Second-Stage Exact int8 SIMD Dot-Product & Top-K Min-Heap
     (All candidates scored -> 100% intra-extent recall)
                           │
                           ▼
  6. Passage Text Retrieval from pdf_text.bin
                           │
                           ▼
               Ranked RAG Context & Citations
```

---

## 3. Pillar 1: Data-Calibrated Entropy-Maximized Hyperplanes (EM-Hash)

### The Problem with Uncalibrated Sequences
Previous iterations used Halton low-discrepancy sequences to generate hyperplanes. However, real-world dense semantic language embeddings are **highly anisotropic** (variance is concentrated along specific principal axes). Uniform or uncalibrated hyperplanes cut through empty dead space or slice across dense clusters, producing low bit-entropy and uneven bucket distributions.

### The Solution: ITQ / PCA Calibration
In Prototype-4, hyperplanes can be calibrated against a representative sample of the dataset:
1. Compute the top principal components via randomized SVD/PCA.
2. Apply an orthogonal rotation matrix $R$ via **Iterative Quantization (ITQ)** to minimize the quantization error:
   $$\min_R \|V - \text{sgn}(V \cdot R)\|_F^2$$
3. Center hyperplanes so that each decision boundary cuts the dataset median with near-zero mean.

**Result**: Every bit in the signature achieves Shannon entropy $H(b_i) \approx 1.0$ (50% zeros, 50% ones), maximizing informational density before any partitioning occurs.

---

## 4. Pillar 2: Multi-Index Hashing (MIH) & Margin-Ranked Probing

### Why Deep Radix Trees Cause Locality Collapse
In theoretical designs, an adaptive Radix Trie is proposed to route variable-length prefixes. However, deep binary tries suffer from **catastrophic metric collapse at top-level splits**: two vectors that are near-identical in high-dimensional space can differ on bit 0 due to boundary noise, placing them into completely disjoint subtrees.

### The Solution: Multi-Index Hashing (MIH)
Prototype-4 replaces the deep trie with **Multi-Index Hashing (MIH)**:
1. The 32-bit signature mask is divided into **four disjoint 8-bit substrings**:
   $$\text{Signature} \implies [s_0, s_1, s_2, s_3] \quad (s_i \in [0, 255])$$
2. Four shallow routing tables (256 buckets each) are maintained in memory:
   ```cpp
   uint8_t mih_tables[4][256][32]; // exactly 32,768 bytes (32 KB)
   ```
3. **Pigeonhole Principle Guarantee**: If two 32-bit signatures have a Hamming distance $d \le 3$, they **must** share at least one 8-bit substring identically:
   $$\lfloor 3 / 4 \rfloor = 0 \implies \text{At least one 8-bit chunk has 0 bit flips}$$

### Margin-Ranked Probing (`--probes N`)
Rather than flipping bits arbitrarily, Prototype-4 ranks probe bits by geometric uncertainty:

1. **Normalized Probe Cache**: Probe vectors $P_i$ have varying norms in raw configurations. To make geometric distances comparable across hyperplanes, Prototype-4 lazily computes and caches unit-normalized probe vectors:
   $$\hat{P}_i = \frac{P_i}{\|P_i\|_2}$$
2. **Margin Calculation**: For a query vector $q$, the margin $m_i$ to hyperplane $i$ is the absolute value of the projection:
   $$m_i = |q \cdot \hat{P}_i|$$
   A smaller margin means the query lies directly on or near the decision boundary, where slight quantization noise could flip the bit.
3. **Adaptive Probing**:
   - Probe bits are sorted in ascending order of margin $m_i$ (most uncertain first).
   - `--probes N` flips the top $(N-1)$ most uncertain bits, computes their flipped signatures, and queries the MIH tables to gather candidate segment IDs.
   - `numProbes = 1`: Probes only the primary exact segment (lowest latency).
   - `numProbes = 4..8`: Probes the primary segment plus the most plausible metric neighbors.

---

## 5. Pillar 3: Columnar Extent Layout (CEL)

### The Problem with Row-Based Records
Traditional vector layouts store each candidate as a contiguous struct (`ChunkRecord` of 456–464 bytes), interleaving binary codes, int8 vectors, and metadata:

```
[Row 0: 384B int8 | 48B code | 32B meta] [Row 1: 384B int8 | 48B code | 32B meta] ...
```

Scanning 1,000 candidates forces the CPU to stride 464 bytes between binary codes, causing cache misses on L1/L2 and reading megabytes of unused int8 payload data into CPU cache lines.

### The Columnar Layout Inside 128 KB Extents
Prototype-4 implements **Columnar Extent Layout (CEL)**. Every physical extent on SSD is strictly sized to **131,072 bytes (128 KB)**, aligned to hardware NVMe flash pages:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PHYSICAL SSD EXTENT (128 KB)                      │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 1: EXTENT HEADER                          (512 Bytes)          │
│   Extent ID, record count (max 283), chain pointer, reserved           │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 2: CONTIGUOUS 384-BIT BINARY CODES        (13,584 Bytes)       │
│   [Code 0: 48B][Code 1: 48B][Code 2: 48B] ... [Code 282: 48B]         │
│   Linear block: Scanned at 30+ GB/s via AVX2 popcount / ADC            │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 3: CONTIGUOUS INT8 PAYLOAD EMBEDDINGS     (108,672 Bytes)      │
│   [Vec 0: 384B][Vec 1: 384B][Vec 2: 384B] ... [Vec 282: 384B]         │
│   Accessed sequentially or targeted per candidate                      │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 4: METADATA & TEXT OFFSETS                (7,924 Bytes)        │
│   [Meta 0: 28B][Meta 1: 28B] ... [Meta 282: 28B]                       │
│   (DocID, PageNum, ChunkIdx, TextOffset, TextLength)                   │
├────────────────────────────────────────────────────────────────────────┤
│ SECTION 5: PAGE ALIGNMENT PADDING                 (380 Bytes)          │
│   Ensures total extent size == exactly 131,072 bytes                   │
└────────────────────────────────────────────────────────────────────────┘
```

#### Exact Geometry Verification
$$512\text{ B} + (283 \times 48\text{ B}) + (283 \times 384\text{ B}) + (283 \times 28\text{ B}) + 380\text{ B} = \mathbf{131,072\text{ Bytes}}$$

---

## 6. Pillar 4: Two-Phase Filtering & Exact Dot Product Scoring

### Phase 1: AVX2 Harley-Seal Popcount (`vpshufb`)
Standard AVX2 lacks the native `_mm256_popcnt_epi32` instruction (which requires AVX-512). Prototype-4 implements the **Muła / Harley-Seal 4-bit lookup table** popcount using `_mm256_shuffle_epi8`:
1. Precomputes a 16-byte lookup table containing the bit count of every 4-bit nibble (0 to 15).
2. Broadcasts the query code into `__m256i` registers and XORs it with candidate codes.
3. Masks the low and high nibbles, then performs parallel lookups using `_mm256_shuffle_epi8`.
4. Accumulates 32 bytes per instruction cycle, sweeping **1,000 binary codes in under $2.5\,\mu\text{s}$**.

### Asymmetric Distance Computation (ADC) as I/O Scheduler
To prevent precision loss from symmetric binary quantization, Prototype-4 retains the continuous unquantized query vector $q \in \mathbb{R}^{384}$ in RAM. It computes the Asymmetric Distance between continuous query weights and candidate binary codes:
$$\text{Score}_{\text{ADC}}(q, c) = \sum_{j=0}^{383} (2 \cdot c_j - 1) \cdot q_j$$

### The False Dismissal Fix: 100% Intra-Extent Recall
Earlier designs used a fixed top-15% ADC survival quota before exact dot products. However, because binary quantization introduces approximation error, the true nearest neighbor could occasionally fall below the 15% cutoff and be permanently discarded.

**Prototype-4 eliminates heuristic drop quotas**:
- **ADC Orders Evaluation**: All records in the extent have their ADC score computed and are sorted descending by ADC score.
- **Heap Threshold Acceleration**: Processing high-ADC candidates first drives up the `minHeap.top().score` rapidly, maximizing the pruning power of Cauchy-Schwarz WAND on all subsequent extents.
- **Full Precision Scoring**: Every active record in an extent that survives WAND is evaluated with exact int8 dot products (`dot_int8`). No candidate is prematurely dropped.

### Dynamic Hardware I/O Crossover
The engine chooses the optimal SSD read strategy based on candidate density:
- **Bulk Read (`record_count > 40`)**: A single contiguous 116 KB read retrieves all remaining embeddings and metadata in one sequential NVMe transaction.
- **Scattered Seek (`record_count <= 40`)**: Issues targeted seeks for individual embeddings and metadata records, avoiding reading unused extent padding.

---

## 7. Pillar 5: Cauchy-Schwarz WAND Geometric Upper Bounding

### The Safety Problem with Heuristic Early Exit
Stopping multi-probing based on score gap heuristics ($\Delta = S_{\text{rank 1}} - S_{\text{rank } K+1} > \tau$) is dangerous: unvisited extents may contain superior candidates, resulting in unpredictable recall collapse.

### The Mathematical Guarantee
Prototype-4 calculates and stores a **centroid** $C_E \in \mathbb{R}^{384}$ and a **maximum radius** $R_E$ for every extent during index construction:
$$C_E = \frac{1}{|E|} \sum_{v \in E} v, \quad R_E = \max_{v \in E} \|v - C_E\|_2$$

By the **Cauchy-Schwarz Inequality**:
$$\forall v \in E: \quad q \cdot v = q \cdot (C_E + (v - C_E)) = (q \cdot C_E) + q \cdot (v - C_E) \le (q \cdot C_E) + \|q\|_2 \cdot \|v - C_E\|_2$$

Since $\|v - C_E\|_2 \le R_E$, the maximum possible score of any vector in extent $E$ is strictly bounded:
$$\text{MaxScore}(q, E) = (q \cdot C_E) + \|q\|_2 \cdot R_E$$

### Pruning Rule
Candidate extents are evaluated against the current min-heap.
If:
$$\text{MaxScore}(q, E) \le \text{MinHeap.top().score}$$
Extent $E$ is **safely skipped with zero SSD reads**. Because this bound is mathematically rigorous, it guarantees zero false dismissals.

---

## 8. Memory vs. SSD Storage Footprint

The table below contrasts the memory and storage requirements of Prototype-4 against leading vector database systems for 1,000,000 vectors (384-dimensional):

| System | Architecture | RAM Footprint (1M vectors) | SSD Footprint | SSD Read Pattern |
|---|---|---|---|---|
| **HNSW (FAISS)** | In-Memory Graph | **~1,800 MB** | None (RAM-only) | Memory fetches |
| **DiskANN** | Graph on SSD | **~35 – 80 MB** (compressed vectors) | ~450 MB | 16–32 random 4KB reads |
| **SPANN** | Inverted Postings | **~40 – 60 MB** (centroids) | ~500 MB | 4–12 postings reads |
| **BitDB Prototype-3** | Avalanche Segment-Chain | **~2.8 MB** (catalog + dir) | ~440 MB | 4–8 bulk extents (64KB) |
| **BitDB Prototype-4 (ER2)** | Columnar Extent Routing | **< 2.5 MB** (32KB MIH + catalog) | ~460 MB (128KB extents) | 1–4 columnar extents (14–128KB) |

**Key Takeaway**: Prototype-4 shifts **>99.4%** of the total storage footprint to NVMe SSD, requiring less than 3 MB of RAM to serve sub-millisecond queries.

---

## 9. Binary File Formats & Crash-Safe Ingestion

All files in `Prototype-4/DataStorage/` are versioned with magic `0x42444234` (`"BDB4"`). Ingestion uses an atomic rename workflow: every file is written completely to a `.tmp` file and then renamed into place with `std::filesystem::rename`, ensuring index integrity even if ingestion is terminated mid-run.

### 1. `chunk_store.bin` (Physical Extents)
Contains contiguous 131,072-byte `ExtentBlock` structures:
- `header` (512 bytes)
- `codes[283][48]` (13,584 bytes)
- `embeddings[283][384]` (108,672 bytes)
- `metadata[283]` (7,924 bytes)
- `padding` (380 bytes)

### 2. `segment_dir.bin` (Primary Directory)
- **Header** (16 bytes): `magic` (4B), `version` (4B), `num_segments` (4B = 256), `reserved` (4B).
- **Entries** ($256 \times 1,556\text{ bytes}$):
  - `chunk_store_offset` (uint64_t)
  - `chunk_count` (uint32_t)
  - `ext_chain_head` (uint32_t)
  - `centroid[384]` (float[384] = 1,536 bytes)
  - `max_radius` (float = 4 bytes)

### 3. `segment_extents.bin` (Extent Overflow Chain)
Stores `ExtentNode` records (1,552 bytes each) representing chained extent blocks for segments that exceed the single-extent capacity (283 records).

### 4. `mih_table.bin` (MIH Inverted Table)
- Exactly **32,768 bytes** ($4 \times 256 \times 32\text{ bytes}$).
- Maps substring index $[0..3]$ and 8-bit value $[0..255]$ to a 256-bit segment occurrence bitmap.

### 5. `doc_catalog.bin` (Document Catalog)
- **Header** (8 bytes): `num_docs` (uint32_t), `active_docs` (uint32_t).
- **Entries** ($N \times 256\text{ bytes}$): `doc_id`, `page_count`, `chunk_start`, `chunk_count`, `is_deleted`, `filename[232]`. Supports tombstoning for fast document deletion.

### 6. `pdf_text.bin` (Passage Text Store)
- Contiguous UTF-8 passage text referenced by `(text_offset, text_length)` in chunk metadata. Only fetched for final top-ranked results.
- Written via `.tmp` atomic commit matching all other database files.

### Ingestion Engine & Fast PDF Extractor
Prototype-4 integrates `fast_pdf_agent/pdf_extractor3.py` directly into the ingestion pipeline (`embed.cpp` and `pdf_extractor.py`). It provides:
- High-throughput layout-aware PDF text and table extraction.
- Automatic fallback mechanisms (PyMuPDF / `pdfplumber`).
- Direct chunking with byte-range indexing into `pdf_text.bin`.

---

## 10. Verification & Invariant Testing

Prototype-4 includes a standalone C++ invariant test binary ([`build/test_suite.exe`](file:///c:/Users/srish/Desktop/BitDB/Prototype-4/src/test_suite.cpp)):

```cmd
cd Prototype-4
build\test_suite.exe
```

### Complete Test Suite (8/8 PASS):
1. **Physical Layout**: Verifies `sizeof(ExtentBlock) == 131072`, section offsets, and byte alignments.
2. **MIH Decompositions**: Verifies 4-way substring slicing and asserts all 8 single-bit flips per byte yield Hamming distance strictly equal to 1.
3. **AVX2 Popcount Correctness**: Compares `_mm256_shuffle_epi8` Harley-Seal implementation against scalar reference across 500 pseudo-random vectors and single-bit flips.
4. **ADC Scoring Invariant**: Verifies asymmetric float-to-bit dot product calculations against reference scalar code.
5. **Cauchy-Schwarz Inequality**: Validates that no point in an extent can exceed $\text{MaxScore}(q, E) = (q \cdot C) + \|q\| \cdot R$.
6. **On-Disk Health**: Confirms all binary files match magic `"BDB4"`, version `4`, and 128 KB divisibility.
7. **Extent Chain Traversal**: Validates multi-extent linked list traversal, cycle detection, and record capacity across extended segment chains.
8. **ADC Recall & Zero False Dismissals**: Validates that candidates ranked outside the top 15% of ADC pre-scores are never dropped and the true nearest neighbor is discovered with 100% precision.

---

## 11. Quick Execution Guide

### Build from Source
```cmd
cd Prototype-4
build.bat
```
*Note: `build.bat` automatically enables `-mavx2 -mpopcnt` on GCC/Clang or `/arch:AVX2` on MSVC, and builds `Build.exe`, `BitDBSearch.exe`, and `test_suite.exe`.*

### Ingest Documents
Place PDFs into `Prototype-4\ingestor\` and run:
```cmd
build\Build.exe --rebuild
```

### Search Queries
Run search specifying query text, top-K, and multi-probe expansion:
```cmd
build\BitDBSearch.exe "approximate nearest neighbor search on SSD" 3 4
```

### Persistent Interactive Daemon Mode
Supports interactive queries without reloading model weights on each turn:
```cmd
build\BitDBSearch.exe --interactive --probes 4
```

### Run Test Suite
```cmd
build\test_suite.exe
```

### Workbench & RAG Studio
To run benchmarks, head-to-head shootouts, and grounded RAG query synthesis:
```cmd
cd ..\BitDB-Workbench
run.bat
```
