# Prototype-3: Segment-Chain Architecture

This document details the architectural evolution from Prototype-2 to Prototype-3, highlighting the new techniques implemented to achieve a high-speed, SSD-first, PDF-native semantic search database.

## 1. Core Paradigm Shift: From Hierarchical Tree to Flat Segment-Chain

**Prototype-2** relied on a hierarchical tree structure for index routing. It clustered embeddings into nodes, requiring beam search during querying. This meant traversing down a tree, loading multiple small binary node files from nested directories into RAM at each step, which introduced significant disk seek latency and RAM overhead.

**Prototype-3** abandons the tree entirely in favor of a **flat Segment-Chain architecture**. It partitions the vector space into a fixed number of segments (buckets). During search, the query is mapped directly to a specific segment, and only that segment's contiguous block of data is read from the SSD. 

This achieves the primary goal: **shifting 80% of the load to the SSD while maintaining a microscopic RAM footprint.**

## 2. New Technique: Halton Sequence Probe Vectors

Instead of using random Gaussian projections (standard LSH) or K-Means centroids for routing, Prototype-3 introduces **Halton low-discrepancy sequences** to generate its probe vectors.

- **How it works:** We generate 32 fixed, deterministic probe vectors (384 dimensions each) based on the Halton sequence. The Halton sequence provides a much more uniform and evenly distributed coverage of the high-dimensional space compared to pseudorandom numbers.
- **Routing:** When an embedding is generated, we compute its dot product against all 32 probe vectors. If the dot product is positive, we set a bit to 1; otherwise, 0. This creates a highly descriptive **32-bit signature mask** for every chunk.
- **Segment Mapping:** This 32-bit mask is then mapped to one of 256 segments (via a simple modulo operation). Chunks with similar semantic meaning share similar bitmasks and fall into the same segment.

## 3. SSD-Optimized Storage Layout

Prototype-2 created a complex web of directories and small binary files for each node. Prototype-3 consolidates the entire database into just **four flat binary files** optimized for sequential SSD reads:

1. **`segment_dir.bin` (RAM-Resident):** A tiny 4KB routing table. It stores the byte offset and chunk count for each of the 256 segments in the chunk store. This is the *only* routing data kept in RAM.
2. **`doc_catalog.bin` (RAM-Resident):** A small catalog tracking indexed PDFs and metadata.
3. **`chunk_store.bin` (SSD-Resident):** The core database. All chunks belonging to the same segment are grouped and written **contiguously**. When searching, the engine reads a segment's entire block of chunks in one fast, sequential SSD read.
4. **`pdf_text.bin` (SSD-Resident):** The raw UTF-8 passage text, only accessed for the final Top-K results via sparse seeks.

## 4. Native PDF Ingestion

Unlike Prototype-2, which only accepted pre-processed `.txt` files, Prototype-3 is **PDF-native**. 

- It implements a Python bridge utilizing `pypdf` to extract text page-by-page directly from `.pdf` files dropped into the `ingestor/` directory.
- Metadata is enhanced: search results now return the specific **source file and page number** where the passage was found.
- The ingestion engine (`Build.exe`) processes this entirely in-memory, chunking the extracted text into sentences and batch-embedding them before flushing to the SSD.

## 5. Hardware Acceleration (OpenVINO iGPU & CUDA)

Prototype-3 embeds an intelligent hardware-acceleration layer:
- **Intel Iris Xe iGPU Acceleration (OpenVINO)**: Utilizes native OpenCL / Level Zero execution units on Intel iGPUs, reducing embedding latency to **~30ms** (over **7.5x faster** than CPU).
- **NVIDIA GPU (CUDA)**: Automatically detected and prioritized if an NVIDIA discrete GPU is available.
- **CPU Fallback**: Automatic fallback if no hardware accelerator is present.

## 6. Portable Path Management & Cross-Platform CMake Build System

To ensure seamless execution across different machines and operating systems without hardcoded paths:
- **`PathConfig.h`**: Discovers the project root at runtime by checking environment variables (`BITDB_ROOT`), executable location (`GetModuleFileName` / `/proc/self/exe`), and upward directory traversal. All paths (`DataStorage/`, `ingestor/`, `scripts/`, `models/`) are resolved dynamically.
- **`CMakeLists.txt`**: Standardized cross-platform build configuration automatically discovering Python 3 headers and libraries, compiling all targets into `build/`.
- **`build.bat` / `build.sh`**: Turnkey scripts for one-click compilation across Windows, Linux, and macOS.

## Summary of Execution Flow

**Ingestion (`Build.exe`):**
1. Read `.pdf` -> Extract text pages in memory.
2. Chunk text into sentences/lines -> Batch embed via OpenVINO iGPU.
3. Compute 32-bit Halton mask for each chunk -> Assign to Segment ID.
4. Sort all chunks by Segment ID -> Write contiguously to `chunk_store.bin`.

**Search (`BitDBSearch.exe`):**
1. Embed Query via iGPU -> Compute 32-bit Halton mask -> Get Target Segment ID.
2. Lookup Segment ID in 4KB `segment_dir.bin` (in RAM).
3. Seek to that segment's offset in `chunk_store.bin` -> Read block into memory.
4. Fast int8 dot-product scoring against candidates -> Fetch text from `pdf_text.bin` for Top-K.

**Result:** Sub-millisecond disk search times with virtually zero RAM dependency for index traversal and portable cross-platform deployment.
