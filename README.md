# BitDB: SSD-First Semantic Retrieval for Cost-Effective RAG

BitDB is an ultra-low-memory, hardware-co-designed semantic search engine and vector database designed for document-centric Retrieval-Augmented Generation (RAG). Its central objective is to eliminate the severe RAM bottleneck of traditional in-memory vector databases (such as HNSW) by shifting **>99.4%** of the vector corpus directly to NVMe SSDs, holding **less than 3 MB of RAM** while delivering sub-millisecond retrieval latency.

---

## Prototype-4: Elastic Radix Extent Routing (ER2)

BitDB Prototype-4 introduces an integrated systems-algorithmic co-design that achieves **100% intra-extent recall with zero false dismissals**:

- **Hardware-Co-Designed 128 KB Columnar Extents (CEL)**: Physical extents strictly aligned to NVMe flash page boundaries (131,072 bytes). Contiguous binary sign codes enable cache-friendly sequential scans at **30+ GB/s**.
- **Data-Calibrated Hyperplanes (Centroid + PCA + ITQ)**: Solves high-dimensional anisotropic embedding bias through Iterative Quantization. Achieves **100% segment utilization (256/256)**, **0 chained overflow extents**, a **1.90x Max/Mean ratio** (down from 11.4x), and a Gini inequality index of **0.1785**.
- **Multi-Index Hashing (MIH) & Margin-Ranked Probing**: Slices 32-bit signatures into four 8-bit substrings. Leverages the pigeonhole principle to guarantee near-neighbor collision and ranks probes by normalized geometric hyperplane margin.
- **AVX2 Harley-Seal SIMD Popcount (`vpshufb`)**: Employs parallel nibble-lookup tables to scan 1,000 binary codes in under $2.5\,\mu\text{s}$.
- **Cauchy-Schwarz WAND Upper Bounding**: Prunes unpromising candidate extents with mathematical certainty ($\text{MaxScore}(q, E) = q \cdot C + \|q\| R$), skipping unneeded SSD I/O.
- **Tokenizer-Native Sliding Window with Cross-Page Buffering**: Direct Rust subword tokenization with continuous carry-over buffering across PDF page breaks, eliminating sentence fragmentation and chunk length skew.
- **Native C++ Zero-Overhead Subsystems**: Native C++17 filesystem watchdog (`Watchdog.exe`) and Halton space-filling probe generator (`HaltonProbes.exe`), removing Python runtime latency and saving ~150 MB RAM per daemon.

---

## Storage & Memory Profile (1M 384-dim Vectors)

| System | Architecture | RAM Footprint (1M vectors) | SSD Footprint | Read Pattern |
|---|---|---|---|---|
| **HNSW (FAISS)** | In-Memory Graph | **~1,800 MB** | None (RAM-only) | Memory fetches |
| **DiskANN** | Graph on SSD | **~35 – 80 MB** | ~450 MB | 16–32 random 4KB reads |
| **SPANN** | Inverted Postings | **~40 – 60 MB** | ~500 MB | 4–12 postings reads |
| **BitDB Prototype-4 (ER2)** | Columnar Extent Routing | **< 3.0 MB** | ~460 MB (128KB extents) | 1–4 bulk extents (14–128KB) |

---

## Directory Structure

```text
BitDB/
├── CMakeLists.txt                 # Unified CMake build configuration (7 targets)
├── build.bat                      # Windows build script (AVX2 + POPCNT)
├── build.sh                       # Linux/Unix build script
├── run_tests.bat                  # 5-stage automated test runner (Windows)
├── run_tests.sh                   # 5-stage automated test runner (Unix)
├── architecture-4.md              # Complete Prototype-4 architectural specification
├── src/
│   ├── Build.cpp                  # PDF ingestion and extent index construction
│   ├── Search.cpp                 # Columnar search engine and passage retrieval
│   ├── Watchdog.cpp               # Native C++17 directory watcher and auto-sync daemon
│   ├── HaltonProbes.cpp           # Native C++ Halton space-filling probe generator
│   ├── test_suite.cpp             # 11-test mathematical invariant verification suite
│   ├── Routing.h                  # AVX2 Harley-Seal popcount, MIH, and ADC distance math
│   ├── embed.cpp / embed.h        # Hardware-accelerated embedding bridge & native text chunker
│   ├── PathConfig.h               # Portable runtime directory discovery
│   └── probe_vectors.h            # Generated ITQ/Halton calibrated hyperplanes
├── scripts/
│   ├── vendor.py                  # OpenVINO iGPU / PyTorch inference & Rust token windowing
│   ├── pdf_extractor.py           # High-throughput layout-aware PDF text extractor
│   ├── calibrate_hyperplanes.py   # Centroid + PCA Whitening + ITQ hyperplane calibration
│   └── stress_test_segments.py    # Segment workload simulation and access analysis
├── ingestor/                      # Input PDFs to be indexed
├── DataStorage/                   # NVMe binary index and columnar extent files
│   ├── segment_dir.bin            # Primary 256-segment directory and bounding spheres
│   ├── chunk_store.bin            # 128 KB physical columnar extent blocks
│   ├── segment_extents.bin        # Overflow extent chain pointers
│   ├── mih_table.bin              # Multi-Index Hashing inverted occurrence tables (32 KB)
│   ├── doc_catalog.bin            # Document metadata, chunk offsets, and deletion tombstones
│   └── pdf_text.bin               # Contiguous UTF-8 passage text store
├── models/                        # Local transformer & OpenVINO model weights
└── printers/                      # Low-level binary index diagnostic utilities
```

---

## Quick Start

### 1. Build All Binaries
```cmd
cd BitDB
.\build.bat
```
Compiles all 7 native targets (`Build.exe`, `BitDBSearch.exe`, `Watchdog.exe`, `HaltonProbes.exe`, `print_catalog.exe`, `print_segment_dir.exe`, `test_suite.exe`) with AVX2 and hardware POPCNT optimizations.

### 2. Ingest PDF Documents
Place PDFs into `BitDB\ingestor\` and run:
```cmd
build\Build.exe
```

### 3. Query Semantic Index
```cmd
build\BitDBSearch.exe "approximate nearest neighbor search on SSD" 3 4
```

### 4. Interactive Search Daemon
```cmd
build\BitDBSearch.exe --interactive --probes 4
```

### 5. Automated Verification Suite
Run all 11 invariant tests and end-to-end benchmarks:
```cmd
.\run_tests.bat
```
*(On Linux/macOS: `./run_tests.sh`)*

---

## Documentation
For complete mathematical derivations, SIMD kernel details, and I/O layout proofs, refer to [BitDB/architecture-4.md](BitDB/architecture-4.md).
