"""
Automated Latency & I/O Benchmark Runner for BitDB Prototypes
Executes a technical query battery, collects fine-grained telemetry, and computes percentiles.
"""

import time
import statistics
from typing import List, Dict, Any, Optional
from adapters.base import BasePrototypeAdapter, SearchResult

BENCHMARK_QUERIES = [
    "approximate nearest neighbor search on SSD",
    "iterative quantization ITQ spectral hashing",
    "multi index hashing for fast search",
    "weak AND WAND dynamic query pruning algorithm",
    "io_uring scatter gather NVMe random read latency",
    "SIMD popcount Harley Seal vpshufb vector filtering",
    "graph-based disk resident vector similarity search DiskANN",
    "product quantization inverted file IVF PQ index"
]

class BenchmarkRunner:
    def __init__(self, queries: Optional[List[str]] = None):
        self.queries = queries or BENCHMARK_QUERIES

    def run_benchmark(
        self,
        adapter: BasePrototypeAdapter,
        top_k: int = 5,
        probes: int = 4,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Runs the query battery and aggregates performance statistics."""
        results: List[Dict[str, Any]] = []
        total_queries = len(self.queries)

        for i, q in enumerate(self.queries, 1):
            if progress_callback:
                progress_callback(i, total_queries, q)

            t0 = time.perf_counter()
            res: SearchResult = adapter.search(q, top_k=top_k, probes=probes)
            t_wall = (time.perf_counter() - t0) * 1000.0

            prof = res.latency
            total_lat = prof.total_ms if prof.total_ms > 0 else t_wall

            top_item = res.items[0] if res.items else None

            results.append({
                "query": q,
                "success": res.exit_code == 0,
                "total_ms": total_lat,
                "embed_ms": prof.embed_ms,
                "disk_ms": prof.disk_ms,
                "io_kb": prof.bulk_read_kb,
                "segments": prof.segments_probed,
                "scored_candidates": prof.scored_candidates,
                "total_candidates": prof.total_candidates,
                "bypass_rate_pct": prof.bypass_rate_pct,
                "top_score": top_item.score if top_item else 0.0,
                "top_file": top_item.filename if top_item else "None",
                "results_count": len(res.items)
            })

        # Calculate statistics
        latencies = [r["total_ms"] for r in results if r["success"]]
        disk_times = [r["disk_ms"] for r in results if r["success"] and r["disk_ms"] > 0]
        io_kbs = [r["io_kb"] for r in results if r["success"]]
        bypass_rates = [r["bypass_rate_pct"] for r in results if r["success"]]

        summary = {
            "prototype_key": adapter.key,
            "prototype_name": adapter.info["name"],
            "total_queries": len(self.queries),
            "successful_queries": len(latencies),
            "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
            "latency_p90_ms": (sorted(latencies)[int(len(latencies)*0.9)] if latencies else 0.0),
            "latency_p95_ms": (sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0.0),
            "latency_p99_ms": (sorted(latencies)[-1] if latencies else 0.0),
            "mean_disk_ms": statistics.mean(disk_times) if disk_times else 0.0,
            "mean_io_kb": statistics.mean(io_kbs) if io_kbs else 0.0,
            "mean_bypass_pct": statistics.mean(bypass_rates) if bypass_rates else 0.0,
            "query_results": results
        }

        return summary
