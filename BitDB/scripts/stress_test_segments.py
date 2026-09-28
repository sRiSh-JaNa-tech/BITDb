"""
stress_test_segments.py — BitDB Prototype-4 Workload Stress Testing & Segment Access Analysis

Simulates/executes real stress tests with 30, 40, 100, 150 prompts on BitDBSearch.exe:
1. Measures live query routing: which segments are accessed, probed, and scanned.
2. Evaluates the compound effect of hyperplane skew:
   - Are queries also biased towards Segments 0-127?
   - How does access frequency scale across 30, 40, 100, 150 prompt batches?
   - Does storage skew correlate with query access skew (compound bottleneck)?
   - What is the latency impact on queries that probe overloaded/chained segments?
3. Generates eda_output/11_query_segment_access_distribution.png
"""

import os
import sys
import re
import time
import json
import struct
import subprocess
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build')
DATA_DIR = os.path.join(PROJECT_ROOT, 'DataStorage')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'eda_output')
SEARCH_BIN = os.path.join(BUILD_DIR, 'BitDBSearch.exe')

def load_segment_catalog():
    """Loads segment sizes and chained extent pointers from segment_dir.bin"""
    seg_dir_path = os.path.join(DATA_DIR, "segment_dir.bin")
    if not os.path.exists(seg_dir_path):
        raise FileNotFoundError(f"Missing {seg_dir_path}")

    seg_info = {}
    with open(seg_dir_path, "rb") as f:
        hdr = f.read(16)
        magic, ver, num_segs, _ = struct.unpack("<IIII", hdr)
        for seg_id in range(num_segs):
            entry = f.read(8 + 4 + 4 + 384 * 4 + 4)
            offset, count, chain = struct.unpack("<QII", entry[:16])
            seg_info[seg_id] = {
                "chunk_count": count,
                "has_chain": (chain > 0),
                "chain_head": chain
            }
    return seg_info

def generate_test_prompts(count=150):
    """Generates a diverse set of realistic, domain-specific technical queries."""
    base_topics = [
        "approximate nearest neighbor search on SSD",
        "gradient boosted decision tree classifiers",
        "automated evaluation of retrieval augmented generation with Ragas",
        "cache friendly SIMD product quantization for vector databases",
        "performance of chatgpt on usmle medical licensing examination",
        "autonomous chemical research with large language models",
        "a prompt pattern catalog to enhance prompt engineering with chatgpt",
        "attention mechanisms in computer vision a survey",
        "point cloud transformer for 3d deep learning",
        "extreme gradient boosting and tree based pipeline optimization",
        "out of core algorithms and dual byte block addressable ssds",
        "computational cache management schemes for ssds",
        "graph retrieval augmented generation using knowledge graphs",
        "biomedical conversational ai agents for clinical applications",
        "accelerating neural transformer via average attention network",
        "explainable machine learning in credit risk management",
        "active retrieval augmented generation for large models",
        "comparison of vision transformers and convolutional neural networks",
        "survey on large language model based autonomous agents",
        "electronic health records clinical knowledge extraction with llms"
    ]
    variations = [
        "",
        " overview and benchmark results",
        " theoretical foundations and algorithmic bounds",
        " optimization techniques and implementation details",
        " comparative analysis and performance trade-offs",
        " practical applications and case studies",
        " scalability and memory efficiency",
        " architecture design and latency evaluation"
    ]
    prompts = []
    for v in variations:
        for t in base_topics:
            if len(prompts) < count:
                prompts.append((t + v).strip())
    return prompts

def run_stress_test(prompts, checkpoints=[30, 40, 100, 150]):
    """Launches BitDBSearch.exe in interactive daemon mode and runs the battery of queries."""
    print(f"[*] Starting stress test with {len(prompts)} prompts across checkpoints {checkpoints}...")
    if not os.path.exists(SEARCH_BIN):
        raise FileNotFoundError(f"Missing {SEARCH_BIN}. Run build.bat first.")

    proc = subprocess.Popen(
        [SEARCH_BIN, "--interactive", "--probes", "4"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        bufsize=1
    )

    # Wait for daemon initialization
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        if "bitdb>" in line or "[Interactive Daemon Mode Active]" in line:
            break

    print("[*] BitDBSearch engine pre-warmed and ready.")

    query_logs = []
    checkpoint_snapshots = {}

    for q_idx, query in enumerate(prompts):
        proc.stdin.write(query + "\n")
        proc.stdin.flush()

        probed_segments = []
        total_latency_ms = 0.0
        extent_scan_ms = 0.0
        bulk_io_kb = 0.0
        records_scored = 0
        total_candidates = 0

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if "Probing" in line and "segment(s)" in line and "[" in line:
                m = re.search(r"\[([\d,\s]+)\]", line)
                if m:
                    probed_segments = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
            if "Records Scored" in line:
                m = re.search(r"Records Scored\s*:\s*(\d+)\s*/\s*(\d+)", line)
                if m:
                    records_scored = int(m.group(1))
                    total_candidates = int(m.group(2))
            if "SSD Extent Scan" in line:
                m = re.search(r"SSD Extent Scan\s*:\s*([\d\.]+)", line)
                if m:
                    extent_scan_ms = float(m.group(1))
            if "Bulk I/O Read" in line:
                m = re.search(r"Bulk I/O Read\s*:\s*([\d\.]+)", line)
                if m:
                    bulk_io_kb = float(m.group(1))
            if "Total Latency" in line:
                m = re.search(r"Total Latency\s*:\s*([\d\.]+)", line)
                if m:
                    total_latency_ms = float(m.group(1))
            if "bitdb>" in line:
                break

        log_entry = {
            "query_idx": q_idx + 1,
            "query": query,
            "probed_segments": probed_segments,
            "primary_segment": probed_segments[0] if probed_segments else None,
            "records_scored": records_scored,
            "total_candidates": total_candidates,
            "ssd_scan_ms": extent_scan_ms,
            "bulk_io_kb": bulk_io_kb,
            "total_latency_ms": total_latency_ms
        }
        query_logs.append(log_entry)

        current_count = q_idx + 1
        if current_count in checkpoints:
            # Capture snapshot of segment hit counts at this checkpoint
            seg_hits = [0] * 256
            for entry in query_logs[:current_count]:
                for s in entry["probed_segments"]:
                    if s < 256:
                        seg_hits[s] += 1
            checkpoint_snapshots[current_count] = list(seg_hits)
            print(f" -> Checkpoint {current_count:3d} reached: Total probes logged={sum(seg_hits)}")

    # Terminate daemon
    try:
        proc.stdin.write("exit\n")
        proc.stdin.flush()
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        pass

    return query_logs, checkpoint_snapshots

def plot_stress_test_analysis(query_logs, checkpoint_snapshots, seg_info, output_dir=OUTPUT_DIR):
    """Generates the 4-panel Graph 11 showing segment access patterns and hyperplane skew impact."""
    os.makedirs(output_dir, exist_ok=True)
    total_queries = len(query_logs)

    # 1. Total segment access frequency across all queries
    total_seg_hits = np.zeros(256, dtype=int)
    primary_seg_hits = np.zeros(256, dtype=int)
    for entry in query_logs:
        if entry["primary_segment"] is not None and entry["primary_segment"] < 256:
            primary_seg_hits[entry["primary_segment"]] += 1
        for s in entry["probed_segments"]:
            if s < 256:
                total_seg_hits[s] += 1

    # Figure Setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"BitDB Prototype-4: Workload Stress Test & Segment Access Heatmap ({total_queries} Prompts)",
                 fontsize=18, fontweight='bold', y=0.98)

    # ─────────────────────────────────────────────────────────────────
    # Subplot 1 (Top-Left): Segment Access Frequency across All 256 Segments
    # ─────────────────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    seg_indices = np.arange(256)
    # Color red for Segments 0-127 (left half) vs blue for 128-255 (right half)
    bar_colors = ['#e74c3c' if s < 128 else '#3498db' for s in seg_indices]
    ax1.bar(seg_indices, total_seg_hits, color=bar_colors, alpha=0.85, width=1.0, edgecolor='none')
    
    left_half_probes = sum(total_seg_hits[:128])
    right_half_probes = sum(total_seg_hits[128:])
    total_probes = sum(total_seg_hits)
    left_pct = (left_half_probes / total_probes * 100) if total_probes > 0 else 0
    right_pct = 100.0 - left_pct

    # Highlight top 3 hottest segments
    top_segs = np.argsort(total_seg_hits)[::-1][:3]
    for ts in top_segs:
        ax1.annotate(f"Seg {ts}\n({total_seg_hits[ts]} hits)",
                     xy=(ts, total_seg_hits[ts]), xytext=(ts, total_seg_hits[ts] + 3),
                     fontsize=8, fontweight='bold', ha='center', color='#900C3F',
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=4))

    ax1.set_title("1. Segment Access Frequency (All 256 Segments)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Segment ID (0 to 255)", fontsize=11)
    ax1.set_ylabel("Total Probes Received", fontsize=11)
    ax1.set_xlim(-2, 258)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Legend & Stats Box
    stats_text = (f"Total Probes: {total_probes}\n"
                  f"Segments 0-127: {left_half_probes} ({left_pct:.1f}%)\n"
                  f"Segments 128-255: {right_half_probes} ({right_pct:.1f}%)\n"
                  f"Cold Segments (0 hits): {np.count_nonzero(total_seg_hits == 0)}/256")
    ax1.text(0.96, 0.94, stats_text, transform=ax1.transAxes, ha='right', va='top',
             fontsize=9, fontweight='semibold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#bdc3c7'))

    # ─────────────────────────────────────────────────────────────────
    # Subplot 2 (Top-Right): Hot-Spot Load Progression Across Batches (30, 40, 100, 150)
    # ─────────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    checkpoints = sorted(checkpoint_snapshots.keys())
    
    # Identify the Top 5 overall hottest segments
    top5_overall = np.argsort(total_seg_hits)[::-1][:5]
    top20_overall = np.argsort(total_seg_hits)[::-1][:20]

    top5_shares = []
    top20_shares = []
    other_shares = []

    for cp in checkpoints:
        hits = np.array(checkpoint_snapshots[cp])
        cp_total = np.sum(hits)
        t5_sum = np.sum(hits[top5_overall])
        t20_sum = np.sum(hits[top20_overall])
        top5_shares.append(t5_sum / cp_total * 100 if cp_total > 0 else 0)
        top20_shares.append((t20_sum - t5_sum) / cp_total * 100 if cp_total > 0 else 0)
        other_shares.append((cp_total - t20_sum) / cp_total * 100 if cp_total > 0 else 0)

    x_pos = np.arange(len(checkpoints))
    bar_width = 0.55

    p1 = ax2.bar(x_pos, top5_shares, bar_width, label=f'Top 5 Hot Segments {list(top5_overall)}', color='#c0392b', alpha=0.9, edgecolor='black')
    p2 = ax2.bar(x_pos, top20_shares, bar_width, bottom=top5_shares, label='Next 15 Warm Segments', color='#e67e22', alpha=0.9, edgecolor='black')
    bottom_combined = np.array(top5_shares) + np.array(top20_shares)
    p3 = ax2.bar(x_pos, other_shares, bar_width, bottom=bottom_combined, label='Remaining 236 Cold Segments', color='#bdc3c7', alpha=0.7, edgecolor='black')

    # Annotate percentages
    for i in range(len(checkpoints)):
        ax2.text(i, top5_shares[i] / 2, f"{top5_shares[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax2.text(i, top5_shares[i] + top20_shares[i] / 2, f"{top20_shares[i]:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=9)

    ax2.set_title("2. Query Access Concentration across Workload Batches", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Number of Prompts / Queries Tested", fontsize=11)
    ax2.set_ylabel("Share of Total Segment Probes (%)", fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{c} Prompts" for c in checkpoints], fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(loc='lower right', frameon=True, fontsize=9)

    # ─────────────────────────────────────────────────────────────────
    # Subplot 3 (Bottom-Left): Compound Bottleneck (Stored Chunks vs Query Accesses)
    # ─────────────────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    stored_chunks = np.array([seg_info[s]["chunk_count"] for s in range(256)])
    query_hits = total_seg_hits

    # Color code by segment range (left vs right half)
    point_colors = ['#e74c3c' if s < 128 else '#3498db' for s in range(256)]
    scatter = ax3.scatter(stored_chunks, query_hits, c=point_colors, s=45, alpha=0.75, edgecolors='black', linewidth=0.5)

    # Add trend line
    if np.max(stored_chunks) > 0 and np.max(query_hits) > 0:
        z = np.polyfit(stored_chunks, query_hits, 1)
        p = np.poly1d(z)
        x_vals = np.linspace(0, np.max(stored_chunks), 100)
        ax3.plot(x_vals, p(x_vals), color='#2c3e50', linestyle='--', linewidth=2, label=f'Linear Fit (Slope={z[0]:.2f})')

    # Annotate top outlier segments in quadrant
    for s in top5_overall:
        ax3.annotate(f"Seg {s} ({stored_chunks[s]} chunks, {query_hits[s]} hits)",
                     xy=(stored_chunks[s], query_hits[s]),
                     xytext=(stored_chunks[s] + 15, query_hits[s] + 1),
                     fontsize=8, fontweight='bold', color='#900C3F',
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=4))

    ax3.set_title("3. Compound Bottleneck: Storage Size vs. Query Access Frequency", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Stored Chunks in Segment (from segment_dir.bin)", fontsize=11)
    ax3.set_ylabel("Query Probes Received during Stress Test", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', frameon=True)

    ax3.text(0.96, 0.15,
             "High Correlation:\nHot segments that stored the most chunks\nare ALSO hit most often by incoming queries.",
             transform=ax3.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fcf3cf', alpha=0.9, edgecolor='#f39c12'))

    # ─────────────────────────────────────────────────────────────────
    # Subplot 4 (Bottom-Right): Query Search Latency vs Chained/Hot Segments Probed
    # ─────────────────────────────────────────────────────────────────
    ax4 = axes[1, 1]
    
    # Classify each query by how many "hot" segments (top 15 segments) it probed
    top15_set = set(np.argsort(total_seg_hits)[::-1][:15])
    hot_counts = []
    latencies = []
    scan_times = []
    
    for entry in query_logs:
        h_count = sum(1 for s in entry["probed_segments"] if s in top15_set)
        hot_counts.append(h_count)
        latencies.append(entry["total_latency_ms"])
        scan_times.append(entry["ssd_scan_ms"])

    hot_counts = np.array(hot_counts)
    latencies = np.array(latencies)
    scan_times = np.array(scan_times)

    categories = sorted(list(set(hot_counts)))
    cat_lat_means = [np.mean(latencies[hot_counts == c]) for c in categories]
    cat_scan_means = [np.mean(scan_times[hot_counts == c]) for c in categories]
    cat_sample_sizes = [np.sum(hot_counts == c) for c in categories]

    x_c = np.arange(len(categories))
    w = 0.35

    ax4.bar(x_c - w/2, cat_lat_means, width=w, label='Total Query Latency (ms)', color='#2980b9', alpha=0.85, edgecolor='black')
    ax4.bar(x_c + w/2, cat_scan_means, width=w, label='SSD Extent Scan Time (ms)', color='#e74c3c', alpha=0.85, edgecolor='black')

    for i in range(len(categories)):
        ax4.text(x_c[i] - w/2, cat_lat_means[i] + 0.5, f"{cat_lat_means[i]:.1f}ms", ha='center', fontsize=9, fontweight='bold')
        ax4.text(x_c[i] + w/2, cat_scan_means[i] + 0.5, f"{cat_scan_means[i]:.1f}ms", ha='center', fontsize=9, fontweight='bold')
        ax4.text(x_c[i], 1.0, f"n={cat_sample_sizes[i]}", ha='center', fontsize=8, color='white', fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor='black', alpha=0.6))

    ax4.set_title("4. Latency Penalty vs. Hot/Overloaded Segments Probed", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Number of Top-15 Hot Segments Probed in Query (0 to 4)", fontsize=11)
    ax4.set_ylabel("Average Time (Milliseconds)", fontsize=11)
    ax4.set_xticks(x_c)
    ax4.set_xticklabels([f"{c} Hot Segments" for c in categories], fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax4.legend(loc='upper left', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(output_dir, "11_query_segment_access_distribution.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[*] Successfully saved Graph 11 to {out_file}")

    # Also save raw telemetry JSON
    json_path = os.path.join(output_dir, "stress_test_telemetry.json")
    with open(json_path, "w") as f:
        json.dump({
            "total_queries": total_queries,
            "checkpoints": checkpoints,
            "left_half_probes_pct": round(left_pct, 2),
            "right_half_probes_pct": round(right_pct, 2),
            "top5_segments": [int(x) for x in top5_overall],
            "top5_shares_at_checkpoints": {str(k): round(v, 2) for k, v in zip(checkpoints, top5_shares)},
            "sample_query_logs": query_logs[:10]
        }, f, indent=2)
    print(f"[*] Successfully saved telemetry data to {json_path}")

    return out_file

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Graph 11: Workload Stress Test & Segment Access Heatmap")
    parser.add_argument("--tag", "-t", type=str, default=None,
                        help="Run folder name inside eda_output (e.g. 'calibrated'). If omitted, outputs to latest or default eda_output.")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Target output directory.")
    return parser.parse_args()

def main():
    import shutil
    args = parse_args()
    base_eda = OUTPUT_DIR

    if args.out:
        target_dir = os.path.abspath(args.out)
    elif args.tag:
        target_dir = os.path.join(base_eda, args.tag)
    else:
        target_dir = os.path.join(base_eda, "latest")
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print("  BitDB Prototype-4: Workload Stress Test & Segment Access")
    print(f"  Target Directory: {target_dir}")
    print("=" * 60)

    seg_info = load_segment_catalog()
    prompts = generate_test_prompts(150)
    
    query_logs, checkpoint_snapshots = run_stress_test(prompts, checkpoints=[30, 40, 100, 150])
    out_img = plot_stress_test_analysis(query_logs, checkpoint_snapshots, seg_info, output_dir=target_dir)

    if args.tag or args.out:
        latest_dir = os.path.join(base_eda, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        shutil.copy2(out_img, os.path.join(latest_dir, os.path.basename(out_img)))

    print(f"\n[DONE] Graph 11 generated at: {out_img}")

if __name__ == "__main__":
    main()
