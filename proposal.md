# Proposal: Adaptive SSD-Resident Retrieval for Cost-Effective RAG

## Proposed title

**Adaptive SSD-Resident Semantic Retrieval with Incremental Compaction for Low-Memory RAG Systems**

## Abstract

Retrieval-Augmented Generation systems depend on vector indexes to locate relevant evidence, but large embedding collections can consume substantial DRAM and increase infrastructure cost. Existing systems address parts of this problem: DiskANN provides SSD-resident approximate nearest-neighbor search, SPANN uses memory–disk hybrid posting lists, FreshDiskANN and SPFresh improve index freshness, and recent adaptive RAG methods control retrieval depth and generation cost. These approaches do not provide one unified evaluation of SSD traffic, RAM usage, retrieval quality, update cost, and end-to-end RAG cost for PDF-centric workloads.

This project proposes an adaptive SSD-resident semantic retrieval architecture. The system stores compact vector records in contiguous SSD segments, keeps only compact routing metadata in RAM, dynamically selects the number of segments to probe, supports append-only updates with background compaction, and optionally compresses vectors before a higher-precision reranking stage. A query controller uses early retrieval signals to allocate SSD reads and reranking effort according to query difficulty. The system will be evaluated against exact search, Prototype-2, Prototype-3, HNSW, IVF-PQ, SPANN, DiskANN, and update-aware baselines.

The goal is not to maximize one isolated metric. The goal is to identify whether the proposed architecture provides a better Pareto trade-off among peak RAM, SSD bytes read, p95 latency, recall, update cost, and RAG answer quality.

## Problem statement

Production RAG systems face several connected costs:

1. Dense vector indexes may require large amounts of RAM.
2. Moving the index to SSD can reduce RAM but increase I/O latency.
3. Fixed probing and fixed Top-K settings waste resources on easy queries and underserve difficult queries.
4. Full index rebuilding makes frequently changing document collections expensive to maintain.
5. Vector compression saves storage but can reduce retrieval quality.
6. Retrieval quality does not necessarily translate directly to answer quality.

The research problem is to design and measure a retrieval architecture that jointly controls these trade-offs.

## Research questions

- Can SSD-resident segment routing reduce peak RAM while preserving Recall@K?
- Does full-signature routing outperform the current lowest-eight-bit mapping?
- Can adaptive segment probing reduce SSD traffic while maintaining recall?
- Can append-only updates and background compaction reduce rebuild time and write amplification?
- How much compression is acceptable before retrieval and RAG quality degrade?
- Can query difficulty control retrieval depth, reranking depth, and SSD reads?
- Which stage dominates total RAG cost: embedding, retrieval, reranking, or generation?

## Proposed architecture

### 1. Full-signature routing

The current Prototype-3 implementation computes 32 probe bits but uses only the lowest eight bits when assigning one of 256 segments. The proposed system will compare several full-signature routing methods:

- Learned hashing of the complete probe signature.
- Multi-assignment to neighboring segments.
- Hamming-distance probing.
- Balanced corpus-aware partitioning.
- A hybrid learned router with a bounded fallback search.

The router will be selected using measured recall, segment balance, SSD bytes read, and tail latency.

### 2. SSD-first segment storage

Vector records remain grouped into contiguous segment regions. The segment directory stores offsets and counts. Search streams candidate records and maintains a bounded Top-K structure so memory does not grow with the number of candidates.

The implementation will measure actual SSD traffic under cold-cache and warm-cache conditions instead of assuming that logical contiguity always produces one physical read.

### 3. Adaptive probing

The query first probes a small primary set. It then examines retrieval signals such as:

- Score distribution.
- Gap between the best and next-best candidates.
- Segment occupancy.
- Query embedding confidence or norm statistics.
- Diversity of early results.

If evidence is weak, the controller probes additional segments. If evidence is strong, it stops early. This creates a controllable latency–recall trade-off.

### 4. Incremental ingestion and compaction

New documents are written to append-only segment runs rather than forcing an immediate global rewrite. Background compaction merges runs, rebalances segments, removes deleted records, and produces a new version atomically.

The system will expose searchable freshness, update throughput, compaction cost, write amplification, and recall during updates.

### 5. Compression and reranking

The project will compare int8, binary, scalar-quantized, and product-quantized representations. A compressed scan will produce a larger candidate set, followed by optional higher-precision reranking of only the most promising candidates.

### 6. RAG cost controller

The final pipeline will allow the controller to select:

- Number of segments to probe.
- Candidate budget.
- Reranking depth.
- Number of passages passed to the generator.
- Whether to abstain or request more evidence.

The controller will optimize a declared quality–latency–cost objective rather than using a fixed retrieval depth for every query.

## Why the idea is unique

The individual components are related to existing research, so the contribution must not be presented as “the first SSD vector index” or “the first adaptive RAG system.” DiskANN already demonstrates SSD-resident ANN search ([DiskANN](https://www.microsoft.com/en-us/research/?p=634449)); SPANN demonstrates memory–disk hybrid search with query-aware pruning ([SPANN](https://www.microsoft.com/en-us/research/publication/spann-highly-efficient-billion-scale-approximate-nearest-neighbor-search/)); FreshDiskANN and SPFresh address updates ([FreshDiskANN](https://www.microsoft.com/en-us/research/?p=905277), [SPFresh](https://arxiv.org/abs/2410.14452)); and adaptive RAG work addresses retrieval or generation budgets ([SAGE](https://arxiv.org/abs/2608.08237)).

The proposed uniqueness is the integrated, measurable operating point:

> A document-centric retrieval system that jointly adapts routing breadth, SSD reads, vector precision, reranking effort, and index freshness under a RAM–latency–recall–cost objective.

The novelty claim will be valid only if experiments demonstrate that this integration produces a better Pareto frontier for the target workload than the individual baselines.

## Experimental design

### Baselines

- Exact brute-force search for ground truth.
- Prototype-2 hierarchical beam search.
- Current Prototype-3 fixed segment probing.
- Improved Prototype-3 full-signature routing.
- HNSW.
- IVF-PQ.
- SPANN.
- DiskANN.
- FreshDiskANN or SPFresh for update workloads.

### Workloads

- Public ANN benchmark data.
- BEIR-style text retrieval tasks.
- Research-paper PDFs.
- Synthetic corpora at increasing scales.
- Static, append-heavy, delete-heavy, and distribution-shift workloads.

### Metrics

Retrieval:

- Recall@1, Recall@5, and Recall@10.
- MRR and nDCG@10.
- Candidate recall.

Systems:

- Peak RAM.
- Index size.
- SSD bytes read per query.
- SSD bytes written per update.
- p50, p95, and p99 latency.
- Queries per second.
- Ingestion and update throughput.
- Write amplification.
- Time until a document becomes searchable.

RAG:

- Answer exact match or F1 where available.
- Faithfulness and answer relevance.
- Context precision and context recall.
- Citation correctness.
- Token cost.
- End-to-end latency.
- Abstention or evidence-sufficiency accuracy.

## Expected contributions

1. A corrected and evaluated full-signature segment-routing method.
2. An SSD-first storage layout with bounded-memory streaming Top-K search.
3. An adaptive probing policy that responds to query difficulty.
4. An incremental update and background-compaction mechanism.
5. A compression–reranking trade-off study for document RAG.
6. A reproducible benchmark covering memory, SSD traffic, latency, recall, updates, and answer quality.

## Success criteria

The project will be considered successful if it demonstrates, under a clearly defined workload, a statistically supported Pareto improvement such as:

- Lower peak RAM than in-memory baselines.
- Comparable recall to SSD-based baselines at the target operating point.
- Lower p95 latency or SSD traffic than fixed-probing Prototype-3.
- Lower update cost than global rebuilding.
- No unacceptable degradation in final RAG answer quality.

The paper will report weaknesses as well as improvements. If the proposed system is better only under low-RAM or update-heavy conditions, that operating region will be stated explicitly rather than claiming universal superiority.

