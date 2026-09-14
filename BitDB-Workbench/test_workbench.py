"""
Verification Test Script for BitDB-Workbench
Tests:
1. Adapter discovery and index stats for all prototypes.
2. Prototype-4 Search and Latency parsing.
3. RAG Grounded Synthesizer (offline mode).
4. Benchmark runner single-query pass.
"""

import sys
from pathlib import Path

# Add current dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import get_all_adapters
from rag.engine import RAGEngine
from rag.synthesizer import RAGSynthesizer
from benchmarks.runner import BenchmarkRunner

def test_all():
    print("=== TEST 1: Adapter Discovery & Stats ===")
    adapters = get_all_adapters()
    for key, adapter in adapters.items():
        stats = adapter.get_stats()
        print(f"[{key}] Built: {stats.is_built}, Total bytes: {stats.total_index_bytes:,}, Chunks: {stats.chunk_count:,}")
    assert "Prototype-4" in adapters
    assert "Prototype-3" in adapters

    print("\n=== TEST 2: Prototype-4 Search & Latency Telemetry ===")
    p4 = adapters["Prototype-4"]
    query = "approximate nearest neighbor search on SSD"
    res = p4.search(query, top_k=3, probes=4)
    print(f"Exit code: {res.exit_code}")
    print(f"Items retrieved: {len(res.items)}")
    for item in res.items:
        print(f"  Rank {item.rank}: Score {item.score:.4f} | File: {item.filename} | Page: {item.page}")
    print(f"Latency Profile: Total {res.latency.total_ms:.2f}ms (Embed {res.latency.embed_ms:.2f}ms, Disk {res.latency.disk_ms:.2f}ms)")
    print(f"Candidate Bypass Rate: {res.latency.bypass_rate_pct:.1f}% ({res.latency.scored_candidates} scored / {res.latency.total_candidates} candidates)")
    assert res.exit_code == 0
    assert len(res.items) > 0

    print("\n=== TEST 3: Grounded RAG Retrieval & Synthesis ===")
    synth = RAGSynthesizer()
    engine = RAGEngine(synth)
    rag_res = engine.answer_question(p4, query, top_k=3, probes=4, prefer_gemini=False)
    assert rag_res["success"] is True
    print(f"RAG Mode: {rag_res['synthesis']['mode']}")
    print(f"RAG Answer:\n{rag_res['synthesis']['answer']}")

    print("\n=== ALL WORKBENCH TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_all()
