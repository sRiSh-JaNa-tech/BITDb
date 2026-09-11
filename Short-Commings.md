# Shortcomings of Prototype-3

Prototype-3 establishes a useful SSD-first design, but the current implementation has algorithmic, systems, evaluation, and RAG limitations. These limitations define the work required before making publication-level performance claims.

## 1. The 32-bit signature currently behaves as an 8-bit signature

`Build.cpp` and `Search.cpp` compute a 32-bit probe mask, but map it using:

```cpp
mask & (NUM_SEGMENTS - 1)
```

With `NUM_SEGMENTS = 256`, only the lowest eight bits affect the segment ID. Probes 8 through 31 do not affect routing. The implementation therefore does not currently exploit the full 32-bit signature described in the architecture document.

This should be treated as a primary correctness and novelty issue. Candidate alternatives include a learned hash over all bits, multiple segment assignments, Hamming-distance routing, or a partitioning scheme trained on the corpus.

## 2. The Halton routing assumption is unvalidated

Prototype-3 assumes that deterministic Halton probe vectors distribute semantically similar embeddings into useful segments. No experiment currently demonstrates:

- Segment balance.
- Locality of semantically similar chunks.
- Recall compared with random hyperplanes.
- Recall compared with k-means/IVF centroids.
- Stability under domain shift.

Halton vectors may distribute points uniformly in probe space without necessarily preserving the nearest-neighbor structure needed for retrieval.

## 3. Fixed four-segment probing may miss relevant results

Search probes the primary segment and three low-bit variants. This is a heuristic, not a demonstrated approximation guarantee. A relevant neighbor whose signature differs in another bit may never be examined.

The system needs experiments comparing one, four, eight, and more segments against adaptive probing based on score gaps, segment size, or query difficulty.

## 4. Ingestion rewrites the complete vector store

The builder loads existing chunks, merges them with new chunks, sorts all chunks by segment, and truncates and rewrites `chunk_store.bin`. This creates:

- High peak RAM during ingestion.
- Full rebuild cost for small updates.
- SSD write amplification.
- Long update pauses as the corpus grows.
- No true incremental insertion path.
- No document deletion or replacement protocol.
- No crash-safe transaction or recovery mechanism.

FreshDiskANN and SPFresh show that update-aware vector indexes are an important production requirement ([FreshDiskANN](https://www.microsoft.com/en-us/research/?p=905277), [SPFresh](https://arxiv.org/abs/2410.14452)).

## 5. Segment imbalance can cause unpredictable latency

The system uses 256 fixed segments, but does not rebalance them. If the embedding distribution is skewed, a few segments may contain most chunks. Queries routed to those segments will scan many more candidates than other queries.

Required measurements include segment occupancy, maximum-to-average size ratio, standard deviation, candidate-count distribution, and p95/p99 latency.

## 6. Candidate results are accumulated in memory

Search reserves space for all candidates and stores a result object for every scored chunk before selecting Top-K. A large segment can therefore create a significant temporary RAM allocation.

The implementation should use a bounded Top-K heap or streaming selection so memory remains proportional to K rather than to the number of candidates.

## 7. The fixed-size record is storage-heavy

Each record contains a 384-byte int8 embedding plus metadata, for a total of 416 bytes. One million chunks require roughly 416 MB for the chunk store alone.

The project should evaluate scalar quantization, product quantization, binary codes, residual vectors, and two-stage reranking. Compression must be measured against recall and final RAG answer quality, not storage size alone. Product-quantized search is established in systems such as FAISS ([FAISS research](https://arxiv.org/abs/1702.08734)).

## 8. Sequential storage does not guarantee one physical SSD read

The logical segment is contiguous in the file, but the current implementation uses ordinary stream seeks and record-by-record reads. Operating-system caching, filesystem allocation, SSD queueing, and segment size can all change actual I/O behavior.

The paper must report measured SSD bytes read and page-cache behavior under both warm-cache and cold-cache workloads.

## 9. The embedding path may dominate query latency

The system reports embedding, disk-search, and total time separately, but model loading, Python bridging, tokenization, and accelerator transfers may dominate short-query latency. A faster SSD search does not necessarily produce a faster end-to-end RAG response.

Persistent model processes, batching, model warm-up, and backend-specific measurements are required.

## 10. Prototype-3 is retrieval, not yet a complete RAG system

The current code retrieves passages but does not implement a complete generator pipeline with:

- Reranking.
- Context deduplication.
- Prompt construction.
- Token-budget management.
- Citation verification.
- Evidence sufficiency checks.
- Abstention.
- End-to-end answer evaluation.

Therefore, “cost-effective RAG” cannot be claimed from vector-search measurements alone. Retrieval cost, reranking cost, generation cost, token usage, and answer quality must be measured together.

## 11. Performance claims are not yet supported by a benchmark

The architecture notes describe sub-millisecond search and an 80% SSD load shift, but the repository does not yet provide a controlled comparison with exact search, HNSW, IVF-PQ, SPANN, DiskANN, or update-capable baselines.

Publication-quality evaluation requires public datasets, multiple corpus sizes, fixed hardware, repeated trials, warm- and cold-cache tests, confidence intervals, and p50/p95/p99 latency.

## 12. Portability and reliability remain incomplete

Although `PathConfig.h` removes several hard-coded paths, the Python environment, model availability, tokenizer dependencies, OpenVINO device behavior, and generated binary formats still require validation across machines.

The index format also needs version validation, checksums, atomic replacement, corruption detection, and recovery after interrupted builds.

