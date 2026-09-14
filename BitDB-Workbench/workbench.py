#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
  BitDB-Workbench: High-Performance Vector Database TUI & RAG Studio
  Interactive multi-prototype manager, benchmark suite, and RAG retrieval.
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
from typing import Optional, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown
from rich.columns import Columns

from config import PROTOTYPE_DEFS, get_gemini_api_key, save_env_var
from adapters import get_adapter, get_all_adapters, BasePrototypeAdapter
from adapters.base import SearchResult, IndexStats
from rag.engine import RAGEngine
from rag.synthesizer import RAGSynthesizer
from benchmarks.runner import BenchmarkRunner
from benchmarks.comparator import PrototypeComparator

console = Console()

class WorkbenchApp:
    def __init__(self):
        self.adapters = get_all_adapters()
        self.active_key = "Prototype-4" if "Prototype-4" in self.adapters else list(self.adapters.keys())[0]
        self.synthesizer = RAGSynthesizer()
        self.rag_engine = RAGEngine(self.synthesizer)

    @property
    def active_adapter(self) -> BasePrototypeAdapter:
        return self.adapters[self.active_key]

    def render_header(self):
        stats: IndexStats = self.active_adapter.get_stats()
        api_key = get_gemini_api_key()
        gemini_status = "[bold green]Configured (Gemini 2.5 Flash)[/]" if api_key else "[dim yellow]Offline Local Extractor[/]"

        status_color = "green" if stats.is_built else "red"
        built_str = f"[{status_color}]{'Built & Ready' if stats.is_built else 'Not Built'}[/]"

        size_mb = stats.total_index_bytes / (1024 * 1024)
        chunks_str = f"{stats.chunk_count:,} chunks ({stats.doc_count} docs)" if stats.doc_count > 0 else f"{stats.chunk_count:,} chunks"

        header_text = Text()
        header_text.append("⚡ BitDB Vector Database Workbench ⚡\n", style="bold cyan")
        header_text.append(f"Active Prototype: ", style="bold white")
        header_text.append(f"{self.active_key} ", style="bold magenta")
        header_text.append(f"({self.active_adapter.info['name']})\n", style="dim cyan")
        header_text.append(f"Index Status: ", style="bold white")
        header_text.append(f"{built_str} | Size: [bold yellow]{size_mb:.1f} MB[/] | Stored: [bold yellow]{chunks_str}[/]\n")
        header_text.append(f"RAG Synthesizer: {gemini_status}", style="white")

        console.print(Panel(header_text, border_style="cyan", padding=(0, 2)))

    def show_main_menu(self) -> str:
        console.print()
        table = Table(title="Control Center Navigation", border_style="dim white", show_header=True, header_style="bold cyan")
        table.add_column("Key", style="bold yellow", width=6, justify="center")
        table.add_column("Feature Module", style="bold white", width=36)
        table.add_column("Description", style="dim")

        table.add_row("[1]", "Switch Active Prototype", "Select between Prototype-1, Prototype-2, Prototype-3, Prototype-4")
        table.add_row("[2]", "Semantic Search & RAG Retrieval", "Query vector index, extract grounded citations, and synthesize answers")
        table.add_row("[3]", "Run Automated Latency Benchmark", "Execute 8-query technical battery with p50/p95/p99 & I/O telemetry")
        table.add_row("[4]", "Side-by-Side Shootout (P3 vs P4)", "Direct comparison: Avalanche Hashing vs ER2 Columnar Extent Routing")
        table.add_row("[5]", "Index Storage & Extents Inspector", "Inspect on-disk 128KB extent blocks, MIH tables, and doc catalog")
        table.add_row("[6]", "Rebuild / Ingest Index", "Re-run physical vector ingestion & layout construction from papers")
        table.add_row("[7]", "Configure Gemini API Key", "Set or update your Google Gemini API key for advanced generative RAG")
        table.add_row("[8]", "Run Invariant & Stress Tests", "Execute C++ layouts, MIH, popcount, WAND, and query stress test suites")
        table.add_row("[0]", "Exit Workbench", "Quit the interactive console application")

        console.print(table)
        choice = Prompt.ask("[bold cyan]Enter choice[/]", choices=["1", "2", "3", "4", "5", "6", "7", "8", "0"], default="2")
        return choice

    def action_switch_prototype(self):
        console.clear()
        console.print(Panel("[bold cyan]Prototype Selector & Multi-Architecture Fleet[/]", border_style="cyan"))

        table = Table(border_style="dim white", header_style="bold magenta")
        table.add_column("#", style="bold yellow", justify="center")
        table.add_column("Prototype", style="bold white")
        table.add_column("Architecture Type", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Index Size", justify="right")
        table.add_column("Chunks", justify="right")

        proto_keys = list(self.adapters.keys())
        for idx, key in enumerate(proto_keys, 1):
            adapter = self.adapters[key]
            stats = adapter.get_stats()
            status_badge = "[bold green]Ready[/]" if stats.is_built else "[bold red]Missing[/]"
            active_marker = " [bold magenta](Active)[/]" if key == self.active_key else ""
            size_mb = f"{stats.total_index_bytes / (1024*1024):.1f} MB" if stats.total_index_bytes > 0 else "-"
            chunks = f"{stats.chunk_count:,}" if stats.chunk_count > 0 else "-"

            table.add_row(
                str(idx),
                f"{key}{active_marker}",
                stats.name,
                status_badge,
                size_mb,
                chunks
            )

        console.print(table)
        choice = IntPrompt.ask(
            "\nSelect prototype index to activate (or 0 to cancel)",
            choices=[str(i) for i in range(len(proto_keys) + 1)],
            default=0
        )
        if choice > 0:
            self.active_key = proto_keys[choice - 1]
            console.print(f"\n[bold green]✓ Activated {self.active_key}: {self.active_adapter.info['name']}[/]")
            time.sleep(1)

    def action_rag_search(self):
        console.clear()
        console.print(Panel(f"[bold cyan]Semantic Search & Grounded RAG Retrieval[/]\nUsing: [bold yellow]{self.active_key}[/]", border_style="cyan"))

        query = Prompt.ask(
            "[bold white]Enter search question / query[/]",
            default="approximate nearest neighbor search on SSD"
        )
        if not query.strip():
            return

        top_k = IntPrompt.ask("Top-K results to retrieve", default=3)
        probes = IntPrompt.ask("Segment / Extent probes budget", default=4)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task(description=f"Querying {self.active_key} & reading SSD extents...", total=None)
            res = self.rag_engine.answer_question(
                self.active_adapter,
                query,
                top_k=top_k,
                probes=probes,
                prefer_gemini=True
            )

        if not res["success"]:
            console.print(f"\n[bold red]Error during search:[/] {res['error']}")
            Prompt.ask("\nPress Enter to return...")
            return

        search_res: SearchResult = res["search_result"]
        synthesis = res["synthesis"]

        console.print(f"\n[bold green]✓ Found {len(search_res.items)} relevant passages in {search_res.latency.total_ms:.2f} ms[/]\n")

        # 1. Render Retrieved Passages
        for item in search_res.items:
            content_text = Text()
            content_text.append(f"Score: ", style="bold dim")
            content_text.append(f"{item.score:.4f}  ", style="bold green")
            content_text.append(f"| File: ", style="bold dim")
            content_text.append(f"{item.filename}  ", style="bold yellow")
            content_text.append(f"| Page: ", style="bold dim")
            content_text.append(f"{item.page}\n\n", style="bold cyan")
            content_text.append(f'"{item.passage}"', style="italic white")

            panel_title = f"[bold white]Rank #{item.rank}[/]"
            if item.doc_id is not None:
                panel_title += f" [dim](Doc ID: {item.doc_id})[/]"
            console.print(Panel(content_text, title=panel_title, border_style="blue", padding=(0, 1)))

        # 2. Render Latency & I/O Telemetry
        prof = search_res.latency
        telemetry_table = Table(title="Hardware & I/O Profiling Telemetry", border_style="dim white", show_header=True)
        telemetry_table.add_column("Query Embedding", justify="right")
        telemetry_table.add_column("SSD Extent Scan", justify="right")
        telemetry_table.add_column("Candidates Scored / Total", justify="center")
        telemetry_table.add_column("Bypass Filter Rate", justify="right", style="bold green")
        telemetry_table.add_column("Bulk SSD Read", justify="right")
        telemetry_table.add_column("Passage Fetch", justify="right")
        telemetry_table.add_column("Total Latency", justify="right", style="bold cyan")

        telemetry_table.add_row(
            f"{prof.embed_ms:.2f} ms",
            f"{prof.disk_ms:.2f} ms",
            f"{prof.scored_candidates} / {prof.total_candidates}" if prof.total_candidates > 0 else "-",
            f"{prof.bypass_rate_pct:.1f}%" if prof.total_candidates > 0 else "-",
            f"{prof.bulk_read_kb:.1f} KB ({prof.segments_probed} segs)",
            f"{prof.passage_ms:.2f} ms",
            f"{prof.total_ms:.2f} ms"
        )
        console.print(telemetry_table)

        # 3. Render Grounded RAG Synthesis
        console.print()
        mode_badge = f"[bold green]Google Gemini (gemini-2.5-flash)[/]" if synthesis["mode"] == "gemini" else "[bold yellow]Offline Extractive Grounded Synthesizer[/]"
        answer_panel = Panel(
            Markdown(synthesis["answer"]),
            title=f"🤖 Grounded RAG Answer Synthesis — Mode: {mode_badge}",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(answer_panel)

        # 4. Sources Reference Table
        if synthesis.get("sources"):
            src_table = Table(title="Grounded Attribution Citations", border_style="dim magenta")
            src_table.add_column("Citation Tag", style="bold magenta", justify="center")
            src_table.add_column("Rank / Score", justify="center")
            src_table.add_column("Source Document", style="bold white")
            src_table.add_column("Page", justify="center")

            for s in synthesis["sources"]:
                src_table.add_row(
                    s["tag"],
                    f"#{s['rank']} ({s['score']:.4f})",
                    s["file"],
                    str(s["page"])
                )
            console.print(src_table)

        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

    def action_benchmark(self):
        console.clear()
        console.print(Panel(f"[bold cyan]Automated Latency & I/O Benchmark Suite[/]\nTarget: [bold yellow]{self.active_key}[/]", border_style="cyan"))

        top_k = IntPrompt.ask("Top-K results per query", default=5)
        probes = IntPrompt.ask("Segment / Extent probes budget", default=4)

        runner = BenchmarkRunner()
        total_q = len(runner.queries)

        console.print(f"\n[bold white]Running battery of {total_q} technical queries...[/]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True
        ) as progress:
            task_id = progress.add_task("Benchmarking queries...", total=total_q)

            def update_cb(idx, total, q):
                progress.update(task_id, completed=idx, description=f"Query {idx}/{total}: {q[:35]}...")

            summary = runner.run_benchmark(
                self.active_adapter,
                top_k=top_k,
                probes=probes,
                progress_callback=update_cb
            )

        # Print Summary Card
        summary_table = Table(title=f"Benchmark Aggregate Statistics ({self.active_key})", border_style="cyan")
        summary_table.add_column("Metric", style="bold white")
        summary_table.add_column("Value", style="bold yellow", justify="right")

        summary_table.add_row("Total Queries Evaluated", str(summary["total_queries"]))
        summary_table.add_row("Mean Total Latency", f"{summary['latency_mean_ms']:.2f} ms")
        summary_table.add_row("p50 (Median) Latency", f"{summary['latency_p50_ms']:.2f} ms")
        summary_table.add_row("p90 Latency", f"{summary['latency_p90_ms']:.2f} ms")
        summary_table.add_row("p95 Latency", f"{summary['latency_p95_ms']:.2f} ms")
        summary_table.add_row("p99 Latency", f"{summary['latency_p99_ms']:.2f} ms")
        summary_table.add_row("Mean SSD Extent Time", f"{summary['mean_disk_ms']:.2f} ms")
        summary_table.add_row("Mean Bulk SSD Read", f"{summary['mean_io_kb']:.1f} KB")
        summary_table.add_row("Mean Candidate Bypass Rate", f"{summary['mean_bypass_pct']:.1f}%")

        console.print(summary_table)

        # Print Query-by-Query Details
        q_table = Table(title="Per-Query Telemetry Breakdown", border_style="dim white", show_header=True)
        q_table.add_column("#", justify="center", style="bold dim")
        q_table.add_column("Query", style="bold white", width=34)
        q_table.add_column("Total Latency", justify="right", style="bold cyan")
        q_table.add_column("SSD Extent", justify="right")
        q_table.add_column("Bulk I/O", justify="right")
        q_table.add_column("Bypass %", justify="right", style="bold green")
        q_table.add_column("Top Hit", style="dim", width=28)

        for idx, q_res in enumerate(summary["query_results"], 1):
            q_table.add_row(
                str(idx),
                q_res["query"][:32] + ("..." if len(q_res["query"]) > 32 else ""),
                f"{q_res['total_ms']:.2f} ms",
                f"{q_res['disk_ms']:.2f} ms",
                f"{q_res['io_kb']:.1f} KB",
                f"{q_res['bypass_rate_pct']:.1f}%",
                q_res["top_file"][:26] + ("..." if len(q_res["top_file"]) > 26 else "")
            )

        console.print(q_table)
        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

    def action_shootout(self):
        console.clear()
        console.print(Panel("[bold cyan]Side-by-Side Architectural Shootout[/]\nComparing: [bold yellow]Prototype-3 (Avalanche Hash)[/] vs [bold green]Prototype-4 (ER2 Columnar Extent)[/]", border_style="cyan"))

        if "Prototype-3" not in self.adapters or "Prototype-4" not in self.adapters:
            console.print("[bold red]Both Prototype-3 and Prototype-4 are required for shootout.[/]")
            Prompt.ask("Press Enter...")
            return

        adapter_p3 = self.adapters["Prototype-3"]
        adapter_p4 = self.adapters["Prototype-4"]

        top_k = IntPrompt.ask("Top-K results per query", default=3)
        probes = IntPrompt.ask("Probes budget", default=4)

        comparator = PrototypeComparator()
        total_q = len(comparator.runner.queries)

        console.print(f"\n[bold white]Executing shootout on {total_q} queries...[/]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True
        ) as progress:
            task_id = progress.add_task("Running shootout...", total=total_q * 2)

            def update_cb(which, idx, total, q):
                p_label = "Prototype-3" if which == "A" else "Prototype-4"
                offset = 0 if which == "A" else total
                progress.update(task_id, completed=offset + idx, description=f"[{p_label}] Query {idx}/{total}...")

            comp_res = comparator.compare(adapter_p3, adapter_p4, top_k=top_k, probes=probes, progress_callback=update_cb)

        p3_summary = comp_res["prototype_a"]
        p4_summary = comp_res["prototype_b"]

        # Aggregate Shootout Summary Table
        summary_table = Table(title="🏆 Head-to-Head Architecture Comparison", border_style="bold green")
        summary_table.add_column("Architecture Property", style="bold white")
        summary_table.add_column("Prototype-3 (Avalanche Hash)", style="bold yellow", justify="right")
        summary_table.add_column("Prototype-4 (ER2 Columnar)", style="bold green", justify="right")
        summary_table.add_column("Advantage / Differentiator", style="bold cyan")

        summary_table.add_row(
            "Partitioning & Routing",
            "32-bit Avalanche Hash (Murmur3)",
            "Multi-Index Hashing (4 x 8-bit)",
            "Metric Locality Preserved"
        )
        summary_table.add_row(
            "SSD Layout",
            "Row-based ChunkRecords (440B)",
            "128 KB Columnar Extents (CEL)",
            "Hardware Page Aligned"
        )
        summary_table.add_row(
            "SIMD Vectorization",
            "Scalar Popcount",
            "AVX2 Harley-Seal (vpshufb)",
            "30+ GB/s Scan Throughput"
        )
        summary_table.add_row(
            "Early-Exit & Pruning",
            "None (Full scan of probed segs)",
            "Geometric WAND Centroid/Radius",
            "Zero Recall Loss Pruning"
        )
        summary_table.add_row(
            "Mean Candidate Bypass Rate",
            f"{p3_summary['mean_bypass_pct']:.1f}%",
            f"{p4_summary['mean_bypass_pct']:.1f}%",
            f"[bold green]+{p4_summary['mean_bypass_pct'] - p3_summary['mean_bypass_pct']:.1f}% Candidates Bypassed[/]"
        )
        summary_table.add_row(
            "Mean Total Latency",
            f"{p3_summary['latency_mean_ms']:.2f} ms",
            f"{p4_summary['latency_mean_ms']:.2f} ms",
            "Near-instantaneous SSD scan"
        )

        console.print(summary_table)

        # Pairwise Table
        pair_table = Table(title="Query-by-Query Comparison", border_style="dim white")
        pair_table.add_column("#", justify="center")
        pair_table.add_column("Query", style="bold white", width=28)
        pair_table.add_column("P3 Latency", justify="right", style="yellow")
        pair_table.add_column("P4 Latency", justify="right", style="green")
        pair_table.add_column("P3 Bypass", justify="right", style="yellow")
        pair_table.add_column("P4 Bypass", justify="right", style="bold green")
        pair_table.add_column("P4 Top-1 Document", style="dim", width=26)

        for idx, p in enumerate(comp_res["pairs"], 1):
            pair_table.add_row(
                str(idx),
                p["query"][:26] + "...",
                f"{p['lat_a']:.1f} ms",
                f"{p['lat_b']:.1f} ms",
                f"{p['bypass_a']:.1f}%",
                f"{p['bypass_b']:.1f}%",
                p["top_file_b"][:24] + "..."
            )

        console.print(pair_table)
        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

    def action_inspect_index(self):
        console.clear()
        stats: IndexStats = self.active_adapter.get_stats()
        console.print(Panel(f"[bold cyan]Index Storage & Layout Inspector: {self.active_key}[/]", border_style="cyan"))

        details_table = Table(border_style="dim white")
        details_table.add_column("Storage Metric / Property", style="bold white")
        details_table.add_column("Value", style="bold yellow")

        details_table.add_row("Architecture Name", stats.name)
        details_table.add_row("Root Directory", str(self.active_adapter.root_path))
        details_table.add_row("Storage Directory", str(self.active_adapter.storage_dir))
        details_table.add_row("Index Built Status", "[bold green]YES[/]" if stats.is_built else "[bold red]NO[/]")
        details_table.add_row("Total Storage Footprint", f"{stats.total_index_bytes / (1024*1024):.2f} MB ({stats.total_index_bytes:,} bytes)")
        details_table.add_row("Chunk Store Size", f"{stats.chunk_store_bytes / (1024*1024):.2f} MB ({stats.chunk_store_bytes:,} bytes)")
        details_table.add_row("Active Documents Count", f"{stats.doc_count} research papers")
        details_table.add_row("Total Ingested Chunks", f"{stats.chunk_count:,} text chunks")

        for k, v in stats.extra_details.items():
            k_formatted = k.replace("_", " ").title()
            details_table.add_row(k_formatted, str(v))

        console.print(details_table)

        # File-by-file breakdown
        if self.active_adapter.storage_dir.exists():
            file_table = Table(title="Storage Directory Binary Files", border_style="dim cyan")
            file_table.add_column("Filename", style="bold white")
            file_table.add_column("File Size (KB)", justify="right")
            file_table.add_column("File Size (Bytes)", justify="right", style="dim")

            for p in sorted(self.active_adapter.storage_dir.glob("*.bin")):
                sz = p.stat().st_size
                file_table.add_row(p.name, f"{sz / 1024:.1f} KB", f"{sz:,}")

            console.print(file_table)

        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

    def action_rebuild_index(self):
        console.clear()
        console.print(Panel(f"[bold yellow]⚠️ Rebuild Index Confirmation: {self.active_key}[/]", border_style="yellow"))
        console.print("Rebuilding will re-ingest and re-index all research papers in the ingestor directory.\n")

        confirm = Confirm.ask("[bold red]Are you sure you want to proceed with rebuild?[/]", default=False)
        if not confirm:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task(description=f"Rebuilding {self.active_key} (this may take 1-2 minutes)...", total=None)
            success = self.active_adapter.rebuild_index()

        if success:
            console.print(f"\n[bold green]✓ Successfully rebuilt index for {self.active_key}![/]")
        else:
            console.print(f"\n[bold red]✗ Rebuild failed or exited with errors.[/]")

        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

    def action_configure_gemini(self):
        console.clear()
        console.print(Panel("[bold cyan]Google Gemini API Configuration[/]", border_style="cyan"))

        current_key = get_gemini_api_key()
        if current_key:
            masked = current_key[:6] + "..." + current_key[-4:]
            console.print(f"Current API Key: [bold green]{masked}[/]\n")
        else:
            console.print("Current API Key: [dim yellow]Not Configured (Running in Local Offline Mode)[/]\n")

        new_key = Prompt.ask("[bold white]Enter Gemini API Key (or press Enter to keep current)[/]", default="")
        if new_key.strip():
            save_env_var("GEMINI_API_KEY", new_key.strip())
            self.synthesizer.set_gemini_key(new_key.strip())
            console.print("\n[bold green]✓ Gemini API key saved to .env and activated![/]")
        else:
            console.print("\n[dim]Key unchanged.[/]")

        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

    def run(self):
        while True:
            console.clear()
            self.render_header()
            choice = self.show_main_menu()

            if choice == "1":
                self.action_switch_prototype()
            elif choice == "2":
                self.action_rag_search()
            elif choice == "3":
                self.action_benchmark()
            elif choice == "4":
                self.action_shootout()
            elif choice == "5":
                self.action_inspect_index()
            elif choice == "6":
                self.action_rebuild_index()
            elif choice == "7":
                self.action_configure_gemini()
            elif choice == "8":
                self.action_run_tests()
            elif choice == "0":
                console.print("\n[bold cyan]Exiting BitDB-Workbench. Goodbye![/]\n")
                break

    def action_run_tests(self):
        console.clear()
        import subprocess
        from config import WORKBENCH_DIR
        test_script = WORKBENCH_DIR / "tests" / "test_prototypes.py"
        subprocess.run([sys.executable, str(test_script)], cwd=str(WORKBENCH_DIR))
        Prompt.ask("\n[dim]Press Enter to return to main menu...[/]")

if __name__ == "__main__":
    app = WorkbenchApp()
    app.run()
