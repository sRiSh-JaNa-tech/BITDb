"""
plot_storage_vs_access_heatmap.py — Graph 12: Storage Size vs Query Access Frequency Heatmaps

Generates 2D Heatmaps comparing Segment Storage Size (stored chunk count from segment_dir.bin)
against Query Access Frequency (probes received) across each prompt workload batch:
- 30 Prompts
- 40 Prompts
- 100 Prompts
- 150 Prompts
"""

import os
import sys
import json
import struct
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'DataStorage')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'eda_output')

def load_segment_storage_sizes():
    """Loads stored chunks for all 256 segments from segment_dir.bin"""
    seg_dir_path = os.path.join(DATA_DIR, "segment_dir.bin")
    if not os.path.exists(seg_dir_path):
        raise FileNotFoundError(f"Missing {seg_dir_path}")

    stored_chunks = np.zeros(256, dtype=int)
    with open(seg_dir_path, "rb") as f:
        hdr = f.read(16)
        magic, ver, num_segs, _ = struct.unpack("<IIII", hdr)
        for seg_id in range(num_segs):
            entry = f.read(8 + 4 + 4 + 384 * 4 + 4)
            offset, count, chain = struct.unpack("<QII", entry[:16])
            stored_chunks[seg_id] = count
    return stored_chunks

def get_or_run_workload_snapshots(checkpoints=[30, 40, 100, 150]):
    """Loads snapshots from stress_test_segments or executes the stress test."""
    from stress_test_segments import load_segment_catalog, generate_test_prompts, run_stress_test
    
    seg_info = load_segment_catalog()
    prompts = generate_test_prompts(150)
    query_logs, checkpoint_snapshots = run_stress_test(prompts, checkpoints=checkpoints)
    return checkpoint_snapshots, query_logs

def generate_heatmaps(stored_chunks, checkpoint_snapshots, output_dir=OUTPUT_DIR):
    """
    Plots a 4-panel 2D Heatmap (Graph 12) for each set of prompts:
    - 30 Prompts
    - 40 Prompts
    - 100 Prompts
    - 150 Prompts
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoints = sorted(checkpoint_snapshots.keys())

    from matplotlib.colors import PowerNorm

    # Define balanced bins for Segment Storage Size (X-Axis) based on actual corpus percentiles
    size_bins = [0, 25, 45, 65, 85, 200]
    size_bin_labels = ["0-25\n(Cold)", "26-45\n(Low)", "46-65\n(Medium)", "66-85\n(High)", "86+\n(Peak)"]
    num_x_bins = len(size_bin_labels)

    # 4-panel figure (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("BitDB: Storage Size vs. Query Access Frequency Heatmaps", 
                 fontsize=18, fontweight='bold', y=0.98)

    # Custom access frequency bins for each checkpoint scale
    access_bin_configs = {
        30:  ([0, 1, 2, 3, 5, 10], ["0 hits", "1 hit", "2 hits", "3-4 hits", "5+ hits"]),
        40:  ([0, 1, 2, 4, 7, 12], ["0 hits", "1 hit", "2-3 hits", "4-6 hits", "7+ hits"]),
        100: ([0, 1, 3, 6, 10, 20], ["0 hits", "1-2 hits", "3-5 hits", "6-9 hits", "10+ hits"]),
        150: ([0, 1, 4, 8, 14, 30], ["0 hits", "1-3 hits", "4-7 hits", "8-13 hits", "14+ hits"])
    }

    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, cp in enumerate(checkpoints):
        r, c = subplot_positions[idx]
        ax = axes[r, c]

        hits = np.array(checkpoint_snapshots[cp])
        acc_edges, acc_labels = access_bin_configs[cp]
        num_y_bins = len(acc_labels)

        # Build 2D matrix: shape (num_y_bins, num_x_bins)
        # Rows = Access Frequency (inverted so highest access is on top), Cols = Storage Size
        matrix = np.zeros((num_y_bins, num_x_bins), dtype=int)

        for s in range(256):
            s_size = stored_chunks[s]
            s_hits = hits[s]

            # Find x bin index
            x_bin = num_x_bins - 1
            for b_i in range(num_x_bins):
                if s_size < size_bins[b_i + 1]:
                    x_bin = b_i
                    break

            # Find y bin index
            y_bin = num_y_bins - 1
            for b_j in range(num_y_bins):
                if s_hits < acc_edges[b_j + 1]:
                    y_bin = b_j
                    break

            # We invert Y so top row is highest access frequency
            matrix[num_y_bins - 1 - y_bin, x_bin] += 1

        # Plot Heatmap using imshow with PowerNorm for dynamic range
        max_val = max(int(np.max(matrix)), 1)
        norm = PowerNorm(gamma=0.55, vmin=0, vmax=max_val)
        im = ax.imshow(matrix, cmap="YlOrRd", norm=norm, aspect="auto", interpolation="nearest")

        # Set tick marks and labels
        ax.set_xticks(np.arange(num_x_bins))
        ax.set_xticklabels(size_bin_labels, fontsize=10, fontweight='bold')
        
        y_labels_inverted = acc_labels[::-1]
        ax.set_yticks(np.arange(num_y_bins))
        ax.set_yticklabels(y_labels_inverted, fontsize=10, fontweight='bold')

        # Pearson correlation and segment metrics
        corr = np.corrcoef(stored_chunks, hits)[0, 1] if np.std(hits) > 0 else 0.0
        large_segs_mask = (stored_chunks > 40)
        hits_in_large = np.sum(hits[large_segs_mask])
        total_hits = np.sum(hits)
        pct_in_large = (hits_in_large / total_hits * 100) if total_hits > 0 else 0
        cold_count = int(np.count_nonzero(hits == 0))

        # Title includes clean telemetry summary without overlapping any heatmap cells
        ax.set_title(
            f"Workload Set: {cp} Prompts ({total_hits} Total Probes)\n"
            f"Correlation (r): {corr:.3f}   |   >40 Chunks: {pct_in_large:.1f}% hits   |   Cold (0 hits): {cold_count}/256 segs", 
            fontsize=11, fontweight='bold', pad=10
        )
        ax.set_xlabel("Segment Storage Size (Stored Chunks)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Query Access Frequency (Hits)", fontsize=11, fontweight='bold')

        # Add cell text annotations (number of segments in each cell)
        for i in range(num_y_bins):
            for j in range(num_x_bins):
                val = matrix[i, j]
                if val == 0:
                    ax.text(j, i, "0", ha="center", va="center",
                            color="#95a5a6", fontsize=10, fontweight="normal")
                else:
                    text_color = "white" if val > (max_val * 0.45) else "black"
                    ax.text(j, i, f"{val}", ha="center", va="center",
                            color=text_color, fontsize=11, fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Segment Count", rotation=-90, va="bottom", fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(output_dir, "12_storage_vs_query_access_heatmap.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[*] Successfully saved Graph 12 to {out_file}")
    return out_file

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Graph 12: Storage vs Query Access Heatmaps")
    parser.add_argument("--tag", "-t", type=str, default=None,
                        help="Run folder name inside eda_output (e.g. 'calibrated'). If omitted, outputs to latest or timestamped folder.")
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

    print("=" * 65)
    print("  BitDB: Generating Graph 12 (Storage vs Access Heatmaps)")
    print(f"  Target Output: {target_dir}")
    print("=" * 65)

    stored_chunks = load_segment_storage_sizes()
    checkpoint_snapshots, query_logs = get_or_run_workload_snapshots([30, 40, 100, 150])
    out_img = generate_heatmaps(stored_chunks, checkpoint_snapshots, output_dir=target_dir)

    # Mirror to eda_output/latest if target was a custom tag
    if args.tag or args.out:
        latest_dir = os.path.join(base_eda, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        shutil.copy2(out_img, os.path.join(latest_dir, os.path.basename(out_img)))

    print(f"\n[SUCCESS] Graph 12 generated at: {out_img}")

if __name__ == "__main__":
    main()
