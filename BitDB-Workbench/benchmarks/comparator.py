"""
Side-by-Side Benchmark Shootout
Compares two prototypes on identical queries and quantifies speedup, I/O reduction, and candidate filtering efficacy.
"""

from typing import Dict, Any, List, Optional
from adapters.base import BasePrototypeAdapter
from .runner import BenchmarkRunner, BENCHMARK_QUERIES

class PrototypeComparator:
    def __init__(self, queries: Optional[List[str]] = None):
        self.runner = BenchmarkRunner(queries)

    def compare(
        self,
        adapter_a: BasePrototypeAdapter,
        adapter_b: BasePrototypeAdapter,
        top_k: int = 5,
        probes: int = 4,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Runs side-by-side shootout between Prototype A and Prototype B."""
        summary_a = self.runner.run_benchmark(
            adapter_a, top_k=top_k, probes=probes,
            progress_callback=(lambda i, n, q: progress_callback("A", i, n, q) if progress_callback else None)
        )
        summary_b = self.runner.run_benchmark(
            adapter_b, top_k=top_k, probes=probes,
            progress_callback=(lambda i, n, q: progress_callback("B", i, n, q) if progress_callback else None)
        )

        pairs = []
        for ra, rb in zip(summary_a["query_results"], summary_b["query_results"]):
            pairs.append({
                "query": ra["query"],
                "lat_a": ra["total_ms"],
                "lat_b": rb["total_ms"],
                "disk_a": ra["disk_ms"],
                "disk_b": rb["disk_ms"],
                "io_a": ra["io_kb"],
                "io_b": rb["io_kb"],
                "bypass_a": ra["bypass_rate_pct"],
                "bypass_b": rb["bypass_rate_pct"],
                "top_file_a": ra["top_file"],
                "top_file_b": rb["top_file"],
                "score_a": ra["top_score"],
                "score_b": rb["top_score"]
            })

        return {
            "prototype_a": summary_a,
            "prototype_b": summary_b,
            "pairs": pairs
        }
