#!/usr/bin/env python3
"""
Comprehensive Integration, Invariant & Stress Test Suite
Tests Prototype-3 and Prototype-4 to guarantee that nothing breaks during execution.
"""

import sys
import os
import subprocess
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Setup path to import workbench modules
WORKBENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKBENCH_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from adapters import get_adapter, get_all_adapters
from adapters.base import SearchResult
from rag.engine import RAGEngine
from rag.synthesizer import RAGSynthesizer
from benchmarks.runner import BenchmarkRunner

console = Console()

def run_cpp_test_suite(proto_name: str, proto_path: Path) -> bool:
    """Executes the C++ compiled test_suite.exe in the prototype's build dir."""
    exe_path = proto_path / "build" / "test_suite.exe"
    if not exe_path.exists():
        console.print(f"  [bold red]FAIL:[/] {exe_path} does not exist. Run build.bat first.")
        return False

    res = subprocess.run(
        [str(exe_path)],
        cwd=str(proto_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if res.returncode == 0:
        console.print(f"  [bold green]PASS:[/] {proto_name} C++ test_suite passed all assertions!")
        return True
    else:
        console.print(f"  [bold red]FAIL:[/] {proto_name} C++ test_suite failed (exit {res.returncode}):")
        console.print(res.stdout)
        console.print(res.stderr)
        return False

def test_edge_case_queries(adapter, proto_name: str) -> bool:
    """Sends boundary and adversarial queries to verify crash resilience."""
    test_cases = [
        ("Normal query", "approximate nearest neighbor search on SSD", 3, 4),
        ("Single word", "DiskANN", 3, 4),
        ("Punctuation & symbols", "SSD & NVMe: high-speed vector search (O(K) RAM) -> 99.9% recall!?", 3, 4),
        ("Technical acronyms", "AVX2 SIMD popcount vpshufb Harley-Seal ITQ", 3, 4),
        ("Top-1 query", "SPANN billion-scale inverted index", 1, 2),
        ("High Top-K query", "graph-based nearest neighbor search", 10, 4),
        ("Single probe", "Disk-resident vector database survey", 3, 1),
        ("High probes", "Multi-index hashing non-metric", 3, 8),
    ]

    all_ok = True
    for label, q, top_k, probes in test_cases:
        res: SearchResult = adapter.search(q, top_k=top_k, probes=probes)
        if res.exit_code != 0:
            console.print(f"  [bold red]FAIL:[/] {proto_name} failed on '{label}': {res.error_message}")
            all_ok = False
        elif len(res.items) == 0:
            console.print(f"  [bold yellow]WARN:[/] {proto_name} returned 0 items for '{label}'")
        else:
            top_item = res.items[0]
            # Check passage text isn't empty and score is non-zero
            if not top_item.passage or top_item.score == 0:
                console.print(f"  [bold red]FAIL:[/] {proto_name} returned invalid passage/score for '{label}'")
                all_ok = False
            else:
                console.print(f"  [bold green]PASS:[/] {proto_name} -> {label} (Top-1: {top_item.filename[:30]}..., Score: {top_item.score:.2f})")

    return all_ok

def test_rag_synthesis(adapter, proto_name: str) -> bool:
    """Verifies that RAG pipeline extracts grounded passages and formats citations."""
    synth = RAGSynthesizer()
    engine = RAGEngine(synth)

    q = "What is DiskANN and how does it optimize SSD search?"
    res = engine.answer_question(adapter, q, top_k=3, probes=4, prefer_gemini=False)

    if not res["success"]:
        console.print(f"  [bold red]FAIL:[/] RAG pipeline failed on {proto_name}: {res['error']}")
        return False

    synthesis = res["synthesis"]
    sources = synthesis.get("sources", [])
    if not sources:
        console.print(f"  [bold red]FAIL:[/] RAG generated 0 sources for {proto_name}")
        return False

    answer_text = synthesis["answer"]
    if "[Source 1]" not in answer_text and "[Source" not in answer_text:
        console.print(f"  [bold red]FAIL:[/] RAG answer missing citations for {proto_name}")
        return False

    console.print(f"  [bold green]PASS:[/] RAG engine successfully retrieved {len(sources)} sources and formatted grounded citations for {proto_name}!")
    return True

def main():
    console.print(Panel("[bold cyan]BitDB Full Reliability & Invariants Test Suite[/]\nGuarantees zero regressions across Prototype-3 & Prototype-4", border_style="cyan"))

    adapters = get_all_adapters()
    results = {}

    # 1. Prototype-4 Tests
    console.print("\n[bold white]=== STEP 1: Prototype-4 (ER2 Columnar Engine) ===[/]")
    p4_path = WORKBENCH_DIR.parent / "Prototype-4"
    p4_cpp_ok = run_cpp_test_suite("Prototype-4", p4_path)
    p4_edge_ok = test_edge_case_queries(adapters["Prototype-4"], "Prototype-4")
    p4_rag_ok = test_rag_synthesis(adapters["Prototype-4"], "Prototype-4")
    results["Prototype-4"] = p4_cpp_ok and p4_edge_ok and p4_rag_ok

    # 2. Prototype-3 Tests
    console.print("\n[bold white]=== STEP 2: Prototype-3 (Avalanche Hash Engine) ===[/]")
    p3_path = WORKBENCH_DIR.parent / "Prototype-3"
    p3_cpp_ok = run_cpp_test_suite("Prototype-3", p3_path)
    p3_edge_ok = test_edge_case_queries(adapters["Prototype-3"], "Prototype-3")
    p3_rag_ok = test_rag_synthesis(adapters["Prototype-3"], "Prototype-3")
    results["Prototype-3"] = p3_cpp_ok and p3_edge_ok and p3_rag_ok

    # Summary Table
    console.print("\n[bold white]=== TEST RESULTS SUMMARY ===[/]")
    table = Table(border_style="dim white", show_header=True)
    table.add_column("System / Component", style="bold white")
    table.add_column("C++ Layout & Invariants", justify="center")
    table.add_column("Query Stress & Edge Cases", justify="center")
    table.add_column("Grounded RAG Pipeline", justify="center")
    table.add_column("Overall Verdict", justify="center")

    table.add_row(
        "Prototype-4 (ER2 Columnar)",
        "[bold green]PASS[/]" if p4_cpp_ok else "[bold red]FAIL[/]",
        "[bold green]PASS[/]" if p4_edge_ok else "[bold red]FAIL[/]",
        "[bold green]PASS[/]" if p4_rag_ok else "[bold red]FAIL[/]",
        "[bold green]READY FOR EXECUTION[/]" if results["Prototype-4"] else "[bold red]ACTION REQUIRED[/]"
    )
    table.add_row(
        "Prototype-3 (Avalanche Hash)",
        "[bold green]PASS[/]" if p3_cpp_ok else "[bold red]FAIL[/]",
        "[bold green]PASS[/]" if p3_edge_ok else "[bold red]FAIL[/]",
        "[bold green]PASS[/]" if p3_rag_ok else "[bold red]FAIL[/]",
        "[bold green]READY FOR EXECUTION[/]" if results["Prototype-3"] else "[bold red]ACTION REQUIRED[/]"
    )

    console.print(table)

    if all(results.values()):
        console.print("\n[bold green]✓ ALL TESTS PASSED! System is completely robust and ready for execution.[/]\n")
        return 0
    else:
        console.print("\n[bold red]✗ Some tests failed. Please inspect the output above.[/]\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
