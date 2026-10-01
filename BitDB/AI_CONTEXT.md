# 🧠 BitDB: Complete AI Context & Architecture Codex

> **Audience**: AI Agents, LLM Pair Programmers, and Autonomous Coding Systems.  
> **Purpose**: Ingest this document once to immediately understand the entire BitDB codebase, mathematical foundations, on-disk binary layouts, multi-prototype evolution, benchmarking pipelines, and operational conventions.

---

## 1. Executive Summary & Core Mission

**BitDB** is a hardware-co-designed, SSD-first vector database and semantic retrieval engine engineered specifically for cost-effective Retrieval-Augmented Generation (RAG).

### The Fundamental Problem
Traditional vector databases rely on in-memory graph indices (such as HNSW in FAISS, Milvus, or Pinecone). For 1,000,000 vectors (384 dimensions), HNSW requires **1.5 GB to 2.5 GB of RAM**. At enterprise scale (100M+ vectors), RAM costs become economically and operationally prohibitive. Conversely, standard disk-based approaches (DiskANN, SPANN) still require tens of megabytes of RAM per million vectors and suffer from severe read amplification due to dozens of random 4 KB SSD read operations per query.

### BitDB's Solution: Elastic Radix Extent Routing (ER2)
BitDB moves **>99.4%** of the vector corpus directly to NVMe SSDs in flash-aligned **128 KB Columnar Extents (CEL)**, holding **less than 3.0 MB of RAM** for routing structures, and achieves sub-millisecond to low-millisecond retrieval with **100% intra-extent recall (zero false dismissals)**.

```
+-----------------------------------------------------------------------------------------+
|                                    QUERY STRING                                         |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
                   [ Embedding Engine: all-MiniLM-L6-v2 (384-dim) ]
                   OpenVINO iGPU / PyTorch CPU-GPU / ONNX Runtime
                                           |
                                           v
                              Dense Query Vector q in R^384
                                           |
            +------------------------------+------------------------------+
            |                                                             |
            v                                                             v
+-------------------------------+                         +-------------------------------+
|  32-bit ITQ Sign Signature    |                         |  Continuous Normalized q      |
|  (Centered Hyperplane Signs)  |                         |  (Kept in RAM for Exact ADC)  |
+-------------------------------+                         +-------------------------------+
            |                                                             |
            v                                                             |
+-------------------------------+                                         |
| Margin-Ranked Multi-Probing   |                                         |
| 4x8-bit MIH Substrings        |                                         |
| Ranked by Geometric Margin    |                                         |
+-------------------------------+                                         |
            |                                                             |
            v                                                             |
+-------------------------------+                                         |
| Geometric WAND Upper Bound    |                                         |
| MaxScore(q, E) = (q.C) + ||q||R                                         |
| Prune Extents with ZERO I/O   |                                         |
+-------------------------------+                                         |
            |                                                             |
            v                                                             |
+-------------------------------------------------------------------------+
| Two-Stage Extent Read & Scoring (128 KB Columnar Extents on NVMe)       |
|  1. Read Stage-1 Extent Header (~14 KB)                                 |
|  2. AVX2 Harley-Seal SIMD Popcount (vpshufb nibble LUT at 30+ GB/s)     |
|  3. Dynamic I/O Crossover:                                              |
|     - Candidate count > 40: Single 128 KB Bulk Extent Read              |
|     - Candidate count <= 40: Targeted Payload Scatter Reads             |
|  4. Exact int8 SIMD Dot-Product Scoring (100% Intra-Extent Recall)      |
+-------------------------------------------------------------------------+
                                           |
                                           v
+-------------------------------------------------------------------------+
| Passage Retrieval from pdf_text.bin -> Ranked Citations & RAG Context   |
+-------------------------------------------------------------------------+
```

---

## 2. Monorepo Organization & Directory Map

```text
c:\Users\srish\Desktop\BitDB/
├── PROJECT_AI_CONTEXT.md          # Master AI orientation codex (root)
├── README.md                      # High-level user overview & quickstart
├── proposal.md                    # Research thesis & systems motivation
├── Short-Commings.md              # Historical post-mortems of earlier prototypes
├── run_tests.bat                  # Top-level integration test suite (5 stages)
├── fetch_papers.py                # Academic paper ingestion orchestrator
├── search_topics.txt              # Research queries for paper acquisition
│
├── BitDB/                         # Core C++17 Systems Engine & Data Infrastructure
│   ├── AI_CONTEXT.md              # THIS FILE: Engine-level AI orientation codex
│   ├── CMakeLists.txt             # Unified build system (7 native executable targets)
│   ├── build.bat / build.sh       # Native compiler build scripts (MSVC / GCC)
│   ├── run_tests.bat / .sh        # Comprehensive C++ & Python test harness
│   ├── architecture-4.md          # Exhaustive architectural & mathematical spec
│   ├── stress_test_prompts_250.txt# 250 domain-authentic queries from ingestor corpus
│   │
│   ├── src/                       # High-Performance C++17 Source Code
│   │   ├── Search.cpp             # BitDBSearch.exe (CLI, benchmark, --interactive daemon)
│   │   ├── Build.cpp              # Build.exe (Ingestion & binary index constructor)
│   │   ├── Routing.h              # SIMD Harley-Seal popcount, MIH hashing, WAND math
│   │   ├── PathConfig.h           # Multi-environment runtime directory discovery
│   │   ├── Watchdog.cpp           # Native C++17 zero-overhead filesystem sync daemon
│   │   ├── HaltonProbes.cpp       # Native C++ Halton space-filling probe generator
│   │   ├── embed.cpp / embed.h    # Local embedding inference bridge & token chunker
│   │   ├── probe_vectors.h        # Calibrated 32x384 ITQ/PCA hyperplane matrices
│   │   └── test_suite.cpp         # 11-test mathematical invariant verification suite
│   │
│   ├── scripts/                   # Data Science, Ingestion, Calibration & EDA Tools
│   │   ├── eda_analysis.py        # 12-chart comprehensive EDA suite & telemetry generator
│   │   ├── stress_test_segments.py# 250-query daemon stress tester & access profiler
│   │   ├── plot_storage_vs_access_heatmap.py # Workload snapshot storage heatmaps
│   │   ├── plot_hyperplane_variance.py       # PCA variance & hyperplane alignment
│   │   ├── calibrate_hyperplanes.py          # Centroid + PCA Whitening + ITQ pipeline
│   │   ├── vendor.py              # OpenVINO / PyTorch bridge & Rust tokenizer
│   │   └── pdf_extractor.py       # Layout-aware PDF text extraction utility
│   │
│   ├── DataStorage/               # Production On-Disk Binary NVMe Storage Files
│   │   ├── chunk_store.bin        # 128 KB physical columnar extent blocks (~67 MB)
│   │   ├── segment_dir.bin        # 256-segment catalog & bounding spheres (~398 KB)
│   │   ├── segment_extents.bin    # Overflow extent chain pointers (~396 KB)
│   │   ├── mih_table.bin          # 32 KB Multi-Index Hashing inverted table
│   │   ├── doc_catalog.bin        # Document metadata, chunk ranges & tombstone flags
│   │   └── pdf_text.bin           # Contiguous UTF-8 passage text store (~7.1 MB)
│   │
│   ├── ingestor/                  # Ingested PDF research corpus (ANN, SSD, RAG papers)
│   ├── eda_output/                # Generated analytical charts (1-12) & telemetry JSONs
│   │   ├── latest/                # Mirrored current production EDA assets
│   │   └── run_YYYYMMDD_HHMMSS/   # Versioned timestamped EDA runs
│   └── models/                    # Model weights (all-MiniLM-L6-v2 OpenVINO/PyTorch)
│
└── BitDB-Workbench/               # Interactive Terminal Studio & RAG Evaluation Suite
    ├── workbench.py               # Rich interactive TUI workspace entrypoint
    ├── config.py                  # Dynamic prototype path & capability detector
    ├── run.bat / run.ps1          # Automatic virtualenv bootstrapper
    ├── test_workbench.py          # Automated Workbench unit & integration test
    ├── adapters/                  # Polymorphic Database Engine Adapters
    │   ├── base.py                # BasePrototypeAdapter abstract class
    │   ├── prototype4_adapter.py  # Prototype-4 (ER2 Columnar, MIH, ADC, WAND)
    │   ├── prototype3_adapter.py  # Prototype-3 (Avalanche Hash, Row Extents)
    │   └── legacy_adapter.py      # Prototypes 1 & 2 (Hierarchical BBQ Tree)
    └── rag/                       # Grounded RAG Generation Engines
        ├── engine.py              # RAG query, context assembly & execution manager
        └── synthesizer.py         # Dual synthesis: Local Extractive vs Google Gemini API
```

---

## 3. Prototype Evolution & Architectural Shootout

BitDB evolved through 4 distinct generations:

| Dimension | Prototype-1 & 2 | Prototype-3 (Baseline) | Prototype-4 (Production ER2) |
| :--- | :--- | :--- | :--- |
| **Index Topology** | Hierarchical BBQ Tree | Flat Avalanche Hash Buckets | **Elastic Radix Extent Routing (ER2)** |
| **Locality Preservation** | Moderate (Tree depth skew) | **Zero (Avalanche destroyed metric locality)** | **Guaranteed via Multi-Index Hashing (MIH)** |
| **Hyperplane Design** | Random Hyperplanes | Halton Quasi-Random Planes | **Centroid-Centered + PCA + ITQ Calibrated** |
| **Segment Skew** | Heavy leaf skew | Severe left-skew (80% in Segments 0–127) | **100% Active (256/256), Max/Mean 1.90x, Gini 0.1785** |
| **Disk Format** | Row-oriented chunk records | Unaligned row extents | **128 KB Flash-Aligned Columnar Extents (CEL)** |
| **Popcount Kernel** | Scalar loop `__popcnt` | Scalar hardware `_mm_popcnt_u64` | **AVX2 Harley-Seal SIMD (`vpshufb` 16-way)** |
| **I/O Strategy** | Multiple random 4KB reads | Fixed extent reads | **Two-Stage Read with Dynamic Crossover (>40)** |
| **Pruning Technique** | Branch pruning | Linear scan of segment | **Cauchy-Schwarz Geometric WAND Bounding** |
| **Recall Guarantee** | Approximate (~82-88%) | Approximate (~85-91%) | **100% Intra-Extent Recall (Zero False Dismissals)** |
| **RAM Footprint** | ~12 MB | ~8 MB | **< 3.0 MB** |

---

## 4. Key Mathematical Pillars & Algorithms

### Pillar 1: Data-Calibrated Entropy-Maximized Hyperplanes
High-dimensional dense embeddings (`all-MiniLM-L6-v2`, 384 dimensions) do not distribute uniformly on a hypersphere; they form a narrow semantic cone with non-zero mean $C = \mathbb{E}[x] \neq \mathbf{0}$.
- **Uncalibrated Flaw**: Synthetic hyperplanes cause up to 80.7% of vectors to evaluate to bit `0` on Bit 7, forcing nearly all chunks into the lower 128 segments.
- **ITQ Pipeline (`scripts/calibrate_hyperplanes.py`)**:
  1. *Centroid Centering*: Subtract global corpus mean $C \in \mathbb{R}^{384}$: $x' = (x / 127.0) - C$.
  2. *PCA Whitening*: Align coordinate axes with principal directions of data variance: $V = \text{top-32 eigenvectors}({x'}^T x')$.
  3. *Iterative Quantization (ITQ)*: Solve orthogonal rotation matrix $R \in \mathbb{R}^{32 \times 32}$ minimizing quantization loss $\| x' V R - B \|_F^2$.
  4. Yields balanced bit splits ($\sim 50\%/50\%$) across all 32 hyperplanes and 0 chained overflow extents.

### Pillar 2: Multi-Index Hashing (MIH) & Margin-Ranked Probing
- A 32-bit sign vector is decomposed into $m = 4$ sub-codes of $b = 8$ bits each.
- By the **Pigeonhole Principle**, if two vectors differ by Hamming distance $d \le 3$, they **must be identical** in at least one 8-bit sub-code:
  $$\lfloor d / m \rfloor = 0$$
- Probe ranking: Hyperplanes with projection values $|x \cdot w_i|$ closest to 0 (highest geometric uncertainty) are flipped first (1-bit and 2-bit combinations). If additional probe slots are requested or empty segments are encountered, fallback segment routing ranks unvisited segments by their true **geometric centroid proximity / Cauchy-Schwarz upper bound in $\mathbb{R}^{384}$**, completely avoiding arbitrary integer segment XORs.

### Pillar 3: Cauchy-Schwarz Geometric WAND Extent Pruning
Every segment $S$ maintains an in-memory bounding sphere defined by centroid $C_S \in \mathbb{R}^{384}$ and maximum radius $R_S = \max_{x \in S} \|x - C_S\|_2$.
By the Cauchy-Schwarz inequality:
$$\text{MaxScore}(q, S) = \max_{x \in S} (q \cdot x) \le (q \cdot C_S) + \|q\|_2 \cdot R_S$$
If $\text{MaxScore}(q, S) < \text{current } k\text{-th best score in min-heap}$, the entire segment and its SSD extents are **bypassed with zero disk I/O**. Because the bound is mathematically exact ($\alpha = 1.0$), it guarantees zero false dismissals.

### Pillar 4: AVX2 Harley-Seal SIMD Popcount
Rather than computing bit counts sequentially, `Routing.h` executes the Harley-Seal vectorized popcount using AVX2 instructions (`_mm256_shuffle_epi8` / `vpshufb`):
- Slices 256-bit registers into 4-bit nibbles.
- Uses a precomputed 16-entry LUT in a SIMD register.
- Processes 1,000 candidate signatures in under $2.5\,\mu\text{s}$ (>30 GB/s throughput).

### Pillar 5: Dynamic Two-Stage I/O Crossover & Cosine Similarity Metric
- **Metric Specification**: BitDB implements **Cosine Similarity via Fixed-Point Scaled Embeddings on the Unit Hypersphere**. The neural encoder strictly $L_2$-unit normalizes continuous vectors ($\|\mathbf{u}\|_2 = 1.0$) prior to symmetric fixed-point quantization into $[-127, 127]$. Exact int8 dot products $\mathbf{q} \cdot \mathbf{x} \approx 127^2 \cos(\theta)$ directly compute scaled cosine similarity, and exact normalized cosine similarity $\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{x}}{\|\mathbf{q}\|_2 \|\mathbf{x}\|_2}$ is reported for all top-ranked hits.
- **I/O Crossover**:
  - **Stage 1**: Read the first ~14 KB to perform SIMD popcount and pre-filter candidates.
  - **Stage 2 Crossover**:
    - If promising candidates $\ge 40$: Issue a single sequential **128 KB bulk read** (maximizes NVMe sequential read bandwidth, avoiding queue overhead).
    - If promising candidates $< 40$: Issue targeted random scatter reads for only the qualifying chunks.

---

## 5. Storage Layout & Binary File Specifications

All binary storage resides in `BitDB/DataStorage/`:

| Binary File | Size (Current) | Sector Alignment | Purpose & Internal Structure |
| :--- | :--- | :--- | :--- |
| `manifest.bin` | 64 bytes | Structured struct | Atomic Storage Manifest (`magic=0x4244424D` 'BDBM', version 4). Tracks generation ID, commit timestamp, document count, active document count, total extents, and CRC32 checksum. Acts as single durable commit point. |
| `chunk_store.bin` | ~67.0 MB | **131,072 bytes (128 KB)** | Physical columnar extents. Contains extent header, 256-bit signatures array, int8 normalized embeddings (384-dim), chunk metadata, and text byte offsets. |
| `segment_dir.bin` | ~398 KB | Structured structs | Directory of all 256 segments: primary extent offset, extent count, chunk count, centroid vector $C$ (384 floats), and bounding radius $R$. |
| `segment_extents.bin`| ~396 KB | Extent pointers | Singly-linked overflow extent chain records (allocated only if a segment exceeds 283 chunks; currently 0 in calibrated P4). |
| `mih_table.bin` | 32,768 bytes | Contiguous array | Inverted index for 4 Multi-Index Hashing substring tables ($4 \times 256 \times 32$ bytes). |
| `doc_catalog.bin` | ~16.6 KB | Fixed-size records | Registered PDF documents: file path, content SHA256, total chunks, extent index range, and deletion tombstone flags. |
| `pdf_text.bin` | ~7.1 MB | Stream / Byte offsets | Append-only raw UTF-8 passage text store referenced by file offsets in `chunk_store.bin`. |

---

## 6. Native C++ Executables & Usage

Compiled via `BitDB/build.bat` (MSVC) or `BitDB/build.sh` (GCC/Clang) into `BitDB/build/`:

### 1. `BitDBSearch.exe` (The Retrieval Engine)
- **Single Query Mode**:
  ```cmd
  BitDBSearch.exe "SSD approximate nearest neighbor vector search" 5 4
  # Arguments: <query_string> <top_k> <num_probes>
  # Note: num_probes specifies the number of segment partition IDs probed (1-256), not signature bit dimensions.
  ```
- **Interactive Daemon Mode (Crucial for Stress Testing & Workbench)**:
  ```cmd
  BitDBSearch.exe --interactive --probes 4
  ```
  Keeps OpenVINO and PyTorch models resident in memory. Listens on `stdin` for queries and outputs JSON/formatted lines to `stdout`. Reduces query latency from ~450ms (process spawn + model load) to **15–25ms**.
- **Benchmark Battery Mode**:
  ```cmd
  BitDBSearch.exe --benchmark
  ```

### 2. `Build.exe` (Index Constructor)
Ingests PDFs from `BitDB/ingestor/`, invokes `scripts/vendor.py` for tokenization and embeddings, flushes all index and payload data to `.tmp` files, and atomically commits `manifest.bin` with an incremented generation ID.

### 3. `Watchdog.exe` (Native Filesystem Sync Daemon)
C++17 `ReadDirectoryChangesW` daemon monitoring `ingestor/`. When a PDF is added, modified, or deleted, triggers automated incremental index re-builds with zero Python runtime overhead. Retains snapshot state on failure to guarantee retry.

### 4. `HaltonProbes.exe`
Fast native generator for quasi-random space-filling hyperplane vectors.

### 5. `test_suite.exe`
Executes 18 native unit tests validating SIMD Harley-Seal popcount, MIH correctness, exact Cauchy-Schwarz WAND bounds, binary extent alignment, transactional storage manifest integrity, calibrated centroid application, modified PDF detection, and watchdog failure state handling.

---

## 7. Python Scripts & EDA Analytics Pipeline

The `BitDB/scripts/` directory contains tools for verification, calibration, and exploratory data analysis:

### Ingestion & Model Utilities
- `scripts/vendor.py`: Embedded inference script supporting OpenVINO iGPU, DirectML, and PyTorch CPU/CUDA with model `all-MiniLM-L6-v2`. Provides tokenizer sliding window chunking with cross-page carryover.
- `scripts/calibrate_hyperplanes.py`: Executes Centroid Centering, PCA Whitening, and Iterative Quantization (ITQ), outputting `BitDB/src/probe_vectors.h`.

### 12-Plot EDA Suite (`scripts/eda_analysis.py`)
Run via:
```cmd
cd BitDB
python scripts/eda_analysis.py
```
Generates 12 visualizations and telemetry manifests saved to `eda_output/run_<timestamp>/` and mirrored to `eda_output/latest/`:

| Graph | Filename | What It Analyzes |
| :---: | :--- | :--- |
| **1** | `1_segment_distribution.png` | Segment chunk occupancy (Target: uniform across 256 segments) |
| **2** | `2_chunk_size_distribution.png` | Token length distribution & sliding window consistency |
| **3** | `3_workload_scaling.png` | Latency & memory scaling across query volume |
| **4** | `4_latency_breakdown.png` | Sub-millisecond latency profile (Model vs SSD vs SIMD) |
| **5** | `5_pruning_efficiency.png` | Candidate bypass rate vs scored candidates (WAND + MIH) |
| **6** | `6_storage_compression.png` | Extent compression ratio vs uncompressed float32 vectors |
| **7** | `7_storage_footprint.png` | Breakdown of bytes across index files |
| **8** | `8_lsh_bit_distribution.png` | Hyperplane bit balance (Target: 50% ones / 50% zeros) |
| **9** | `9_extent_radii_distribution.png` | Bounding sphere radii distribution for WAND pruning |
| **10**| `10_uncalibrated_hyperplane_variance.png` | PCA eigenvalue decay and variance retention |
| **11**| `11_query_segment_access_distribution.png` | Segment probe frequencies across 50, 100, 175, 250 queries |
| **12**| `12_storage_vs_query_access_heatmap.png` | Storage density vs query access frequency heatmap |

### 250-Query Stress Test (`scripts/stress_test_segments.py`)
- Reads strictly 250 genuine domain queries from `stress_test_prompts_250.txt`.
- Spawns `BitDBSearch.exe --interactive --probes 4`.
- Logs probe distributions across checkpoints `[50, 100, 175, 250]`.
- Emits `stress_test_telemetry.json`.

---

## 8. BitDB-Workbench (Interactive Terminal Studio)

Located in `BitDB-Workbench/`, launched via `run.bat` or `python workbench.py`:
- **Architecture**: Terminal UI powered by `rich` and `prompt_toolkit`.
- **Dynamic Fleet Switcher**:
  - Seamlessly hot-swaps between `Prototype-4` (ER2 Columnar Extents), `Prototype-3` (Avalanche Hash), and `Legacy Prototypes 1 & 2` (BBQ Tree).
  - Inspects active disk extent counts, binary index files, and SIMD support in real time.
- **RAG Synthesizer Modes**:
  1. *Offline Extractive Synthesizer*: Zero external dependencies; extracts top-scoring sentences and key claims.
  2. *Google Gemini API (`gemini-2.5-flash`)*: Fully grounded generation with inline citations (`[Source 1]`, `[Source 2]`).
- **Benchmark & Shootout Suites**:
  - Runs standard 8-query performance battery across technical domains.
  - Computes p50, p90, p95, p99 latencies, bulk I/O volume, and WAND candidate bypass rate.
  - Head-to-head comparison between Prototype-3 and Prototype-4.

---

## 9. Developer & AI Engineering Rules

When reading, modifying, or extending BitDB code, adhere strictly to these principles:

1. **NVMe Sector Alignment**:
   - Every extent in `chunk_store.bin` **MUST** be an exact multiple of 131,072 bytes (128 KB). Never introduce unaligned offsets or arbitrary byte padding.
2. **Low-Memory Philosophy**:
   - Never load entire vector arrays or large index structures into RAM. RAM usage for routing and catalogs must strictly stay **under 3.0 MB**. All candidate traversal must happen via streaming or two-stage extent reads.
3. **Interactive Mode for Multi-Query Workloads**:
   - When running batch evaluations, tests, or stress tests, **never** execute `BitDBSearch.exe <query>` in a loop. Always use `BitDBSearch.exe --interactive` to avoid reloading transformer model weights on every query.
4. **Preserve 100% Intra-Extent Recall**:
   - Binary signatures and MIH are used strictly for extent routing, candidate pre-sorting, and pruning. Within accessed extents, every candidate chunk must be scored with exact int8 dot-products to guarantee zero false dismissals.
5. **Path Portability**:
   - Always resolve runtime paths through `PathConfig.h` (in C++) or `find_prototype_base_dir()` (in Python). Never hardcode absolute `C:\...` paths into core engine logic.
6. **Cross-Platform Compilation**:
   - Support both MSVC (`/arch:AVX2 /O2`) and GCC/Clang (`-mavx2 -mpopcnt -O3`). Do not write MSVC-only intrinsics without standard `#ifdef _MSC_VER` guards.
