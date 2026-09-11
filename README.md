# BitDB: SSD-First Semantic Retrieval for Cost-Effective RAG

BitDB is an experimental semantic-search database designed for document-centric Retrieval-Augmented Generation (RAG). Its central objective is to reduce dependence on expensive main memory by storing the large vector corpus on SSD while keeping only small routing and metadata structures in RAM.

The project evolved from Prototype-2 to Prototype-3. Prototype-2 uses a hierarchical centroid tree and beam search. Prototype-3 replaces that tree with a flat segment-chain layout: embedded document chunks are assigned to segments, stored contiguously, and searched using sequential SSD reads. The intended trade-off is lower RAM usage and lower infrastructure cost while retaining useful retrieval recall and interactive latency.

Prototype-3 is currently a research prototype, not a production-ready vector database. Claims about latency, memory reduction, and recall must be validated with controlled benchmarks.

## Project objectives

- Store most vector data on SSD rather than in RAM.
- Reduce random disk access through contiguous segment storage.
- Support PDF-native ingestion with document and page metadata.
- Use compact int8 embeddings and fast dot-product scoring.
- Exploit OpenVINO, CUDA, or CPU embedding backends.
- Investigate adaptive retrieval, incremental updates, compression, and cost-aware RAG.

The intended research question is:

> Can an adaptive, incremental, SSD-resident retrieval architecture reduce RAM usage and RAG cost while preserving retrieval quality and acceptable tail latency?

## Prototype-3 architecture

### Ingestion

1. PDFs are read from `ingestor/`.
2. `pypdf` extracts text page by page.
3. Text is split into sentence or line-level chunks.
4. Chunks are embedded in batches using the Python embedding bridge.
5. Each 384-dimensional embedding is quantized to int8 and receives a 48-byte sign code for prefiltering.
6. A probe signature is computed and mapped to a segment.
7. Chunks are sorted by segment and written to the binary data files.

### Search

1. The query is embedded using the same model.
2. The query receives a probe signature and primary segment.
3. The segment directory identifies the relevant byte ranges.
4. Candidate chunk records are read from `chunk_store.bin`.
5. Int8 dot products rank the candidates.
6. Text for the final Top-K results is read from `pdf_text.bin`.

## Prototype-3 file structure

```text
Prototype-3/
├── CMakeLists.txt                 # CMake build configuration
├── build.bat                      # Windows build helper
├── build.sh                       # Linux/macOS build helper
├── architecture-3.md              # Initial architecture notes
├── src/
│   ├── Build.cpp                  # PDF ingestion and index construction
│   ├── Search.cpp                 # Query embedding, segment probing, ranking
│   ├── embed.cpp                  # C++/Python embedding bridge
│   ├── embed.h                    # Embedding bridge declarations
│   ├── PathConfig.h               # Portable project/data path discovery
│   └── probe_vectors.h            # Generated probe-vector constants
├── scripts/
│   ├── vendor.py                  # Chunking, embeddings, and backend selection
│   ├── pdf_extractor.py           # Page-level PDF text extraction
│   └── gen_probes.py              # Generates Halton-based probe vectors
├── ingestor/                      # Input PDFs to be indexed
├── DataStorage/                   # Generated binary index and text files
│   ├── segment_dir.bin            # Segment offsets and counts
│   ├── chunk_store.bin            # Fixed-size vector records
│   ├── pdf_text.bin               # Retrieved passage text
│   └── doc_catalog.bin            # Document and page metadata
├── models/
│   ├── local_minilm/              # Local sentence-transformer model
│   └── openvino_minilm/           # OpenVINO model and cache
├── printers/                      # Binary index inspection utilities
└── build/                         # Compiled executables
```

The four main data files are:

| File | Purpose | Intended residence |
|---|---|---|
| `segment_dir.bin` | Maps each segment to an offset and chunk count | RAM |
| `doc_catalog.bin` | Stores document IDs, filenames, and page metadata | Small RAM structure |
| `chunk_store.bin` | Stores 384-byte int8 embeddings, a 48-byte binary code, and metadata | SSD |
| `pdf_text.bin` | Stores original chunk text | SSD |

## Research context

DiskANN demonstrates that SSD-resident approximate nearest-neighbor indexes can achieve high recall with substantially lower DRAM requirements ([DiskANN](https://www.microsoft.com/en-us/research/?p=634449)). SPANN explores a memory–disk hybrid design with query-aware posting-list pruning ([SPANN](https://www.microsoft.com/en-us/research/publication/spann-highly-efficient-billion-scale-approximate-nearest-neighbor-search/)). FreshDiskANN and SPFresh address the cost of maintaining indexes under updates ([FreshDiskANN](https://www.microsoft.com/en-us/research/?p=905277), [SPFresh](https://arxiv.org/abs/2410.14452)).

BitDB investigates a different integrated design based on flat segment storage, compact metadata, PDF retrieval, adaptive probing, and RAG-level cost measurement. The proposed system must be evaluated against these methods rather than assumed to outperform them.

## Planned evaluation

The evaluation should compare Prototype-2, Prototype-3, exact search, HNSW, IVF-PQ, SPANN, DiskANN, and an update-capable baseline. It should report Recall@K, nDCG, p50/p95/p99 latency, peak RAM, SSD bytes read, index size, ingestion/update cost, write amplification, and end-to-end RAG answer quality and token cost.
