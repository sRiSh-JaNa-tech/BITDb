"""
plot_hyperplane_variance.py — Visualize Pure Uncalibrated Hyperplane Variance & Statistics

Analyzes the mathematical variance and distribution generated purely by BitDB's 
existing 32 uncalibrated Halton hyperplanes (without PCA or calibration references):
1. Continuous projection variance: Var(X · w_i) on a natural linear scale
2. Projection spread: Mean (μ) ± 1 Standard Deviation (σ) relative to threshold 0.0
3. Binary bit balance: % Ones vs % Zeros (showing directional skew)
4. Cumulative variance accumulation across the 32 uncalibrated planes
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'DataStorage')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

def load_chunk_embeddings(chunk_store_path=None):
    if chunk_store_path is None:
        chunk_store_path = os.path.join(DATA_DIR, "chunk_store.bin")
    print("[*] Loading embeddings from chunk_store.bin...")
    embeddings = []
    if not os.path.exists(chunk_store_path):
        raise FileNotFoundError(f"Cannot find {chunk_store_path}")

    with open(chunk_store_path, "rb") as f:
        while True:
            block = f.read(131072)
            if not block or len(block) < 131072:
                break
            rec_count = struct.unpack("<I", block[4:8])[0]
            # Embeddings offset: 512 + (283 * 48) = 14096
            for i in range(rec_count):
                v_offset = 14096 + (i * 384)
                vec = np.frombuffer(block[v_offset : v_offset + 384], dtype=np.int8).astype(np.float32)
                embeddings.append(vec)

    X = np.array(embeddings)
    print(f"[*] Loaded {len(X)} vector embeddings (shape: {X.shape})")
    return X

def load_probe_vectors(probe_header_path=None):
    if probe_header_path is None:
        probe_header_path = os.path.join(SRC_DIR, "probe_vectors.h")
    print(f"[*] Loading probe vectors from {probe_header_path}...")
    probes = []
    with open(probe_header_path, "r") as f:
        in_probes = False
        curr_probe = []
        for line in f:
            line = line.strip()
            if "PROBE_VECTORS" in line:
                in_probes = True
                continue
            if in_probes:
                if line.startswith("};"):
                    break
                if line.startswith("{") or line.startswith("//"):
                    continue
                if line.endswith("},"):
                    parts = line.replace("},", "").replace("f", "").split(",")
                    for p in parts:
                        p = p.strip()
                        if p: curr_probe.append(float(p))
                    probes.append(curr_probe)
                    curr_probe = []
                else:
                    parts = line.replace("f", "").split(",")
                    for p in parts:
                        p = p.strip()
                        if p: curr_probe.append(float(p))

    W = np.array(probes)
    print(f"[*] Loaded {len(W)} probe vectors (shape: {W.shape})")
    return W

def generate_uncalibrated_plots(X, W, output_dir="eda_output"):
    os.makedirs(output_dir, exist_ok=True)
    num_probes = len(W)

    # Calculate projections purely for the uncalibrated hyperplanes
    projections = np.dot(X, W.T)  # shape: (N, 32)
    variances = np.var(projections, axis=0)
    means = np.mean(projections, axis=0)
    stds = np.std(projections, axis=0)

    # Binary partitions
    bits = (projections > 0).astype(int)
    p_ones = np.mean(bits, axis=0) * 100.0
    p_zeros = 100.0 - p_ones
    bit_vars = (p_ones / 100.0) * (p_zeros / 100.0)

    # Multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("BitDB Prototype-4: Uncalibrated Hyperplane Variance & Distribution Analysis", 
                 fontsize=18, fontweight='bold', y=0.98)

    bits_idx = np.arange(num_probes)

    # ─────────────────────────────────────────────────────────────────
    # Subplot 1: Continuous Projection Variance (Linear Scale)
    # ─────────────────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    colors = ['#e74c3c' if i < 8 else '#3498db' for i in range(num_probes)]
    bars1 = ax1.bar(bits_idx, variances, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    mean_var = np.mean(variances)
    ax1.axhline(mean_var, color='#2c3e50', linestyle='--', linewidth=2, 
                label=f'Mean Variance ({mean_var:.2f})')
    
    # Custom legend for segment bits vs probe bits
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Bits 0-7: Primary Segment Routing'),
        Patch(facecolor='#3498db', edgecolor='black', label='Bits 8-31: Multi-Index Hashing'),
        plt.Line2D([0], [0], color='#2c3e50', linestyle='--', linewidth=2, label=f'Average Variance ({mean_var:.2f})')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    ax1.set_title("1. Continuous Projection Variance per Hyperplane: Var(X · w_i)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Hyperplane / Bit Index (0 to 31)", fontsize=11)
    ax1.set_ylabel("Variance of Projected Values", fontsize=11)
    ax1.set_xticks(bits_idx)
    ax1.set_ylim(0, 48)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ─────────────────────────────────────────────────────────────────
    # Subplot 2: Projection Spread: Mean (μ) ± 1 Standard Deviation (σ)
    # ─────────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    # Plot error bars representing mean ± std
    ax2.errorbar(bits_idx, means, yerr=stds, fmt='o', color='#2980b9', ecolor='#bdc3c7',
                 elinewidth=2, capsize=4, capthick=1.5, markersize=6, label='Mean μ ± 1 StdDev σ')
    
    # Highlight mean points for bits with large offset
    for i in range(num_probes):
        if abs(means[i]) > 2.0:
            ax2.plot(i, means[i], 'ro', markersize=7)

    ax2.axhline(0.0, color='black', linestyle='-', linewidth=1.5, label='Decision Boundary (Threshold = 0.0)')
    ax2.set_title("2. Projection Spread & Centering: Mean (μ) ± 1 StdDev (σ)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Hyperplane / Bit Index (0 to 31)", fontsize=11)
    ax2.set_ylabel("Projected Dot Product (z_i = x · w_i)", fontsize=11)
    ax2.set_xticks(bits_idx)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True)

    # Callout on Bit 7
    ax2.annotate(f"Bit 7: μ={means[7]:.2f}, σ={stds[7]:.2f}\n(Shifted below 0 -> 80.7% Zeros)",
                 xy=(7, means[7]), xytext=(7, -13),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight='bold', color='#c0392b', ha='center')

    # ─────────────────────────────────────────────────────────────────
    # Subplot 3: Binary Bit Partition Balance (% Ones vs % Zeros)
    # ─────────────────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.bar(bits_idx, p_ones, color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8, label='% Ones (Bit = 1)')
    ax3.bar(bits_idx, p_zeros, bottom=p_ones, color='#95a5a6', alpha=0.4, edgecolor='black', linewidth=0.8, label='% Zeros (Bit = 0)')
    ax3.axhline(50.0, color='#e74c3c', linestyle='--', linewidth=2, label='Ideal 50% / 50% Balance')
    
    ax3.set_title("3. Partition Bit Balance: % Ones vs % Zeros", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Hyperplane / Bit Index (0 to 31)", fontsize=11)
    ax3.set_ylabel("Percentage (%)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.set_xticks(bits_idx)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', frameon=True)

    # Highlight skewed bits
    for b in [0, 2, 5, 7, 11, 26, 29]:
        ax3.annotate(f"{p_ones[b]:.1f}%",
                     xy=(b, p_ones[b]), xytext=(b, p_ones[b] + (5 if p_ones[b] < 50 else -10)),
                     fontsize=8, fontweight='bold', ha='center',
                     color='#c0392b' if p_ones[b] < 35 or p_ones[b] > 65 else '#2c3e50')

    # ─────────────────────────────────────────────────────────────────
    # Subplot 4: Cumulative Variance of Uncalibrated Hyperplanes
    # ─────────────────────────────────────────────────────────────────
    ax4 = axes[1, 1]
    cum_vars = np.cumsum(variances)
    ax4.plot(bits_idx + 1, cum_vars, marker='o', linewidth=2.5, markersize=6, color='#8e44ad', label='Cumulative Variance')
    ax4.fill_between(bits_idx + 1, cum_vars, color='#8e44ad', alpha=0.15)

    for x_val in [8, 16, 24, 32]:
        y_val = cum_vars[x_val - 1]
        ax4.annotate(f"{y_val:.1f}", (x_val, y_val), textcoords="offset points", 
                     xytext=(-10, 10), fontsize=9, fontweight='bold', color='#6c3483')

    ax4.set_title("4. Cumulative Projection Variance across 32 Hyperplanes", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Number of Hyperplanes Included (1 to 32)", fontsize=11)
    ax4.set_ylabel("Cumulative Sum of Variances", fontsize=11)
    ax4.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper left', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(output_dir, "10_uncalibrated_hyperplane_variance.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[*] Successfully saved plot to {out_file}")
    return out_file

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Graph 10: Hyperplane Variance Analysis")
    parser.add_argument("--tag", "-t", type=str, default=None,
                        help="Run folder name inside eda_output (e.g. 'calibrated'). If omitted, outputs to latest or default eda_output.")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Target output directory.")
    return parser.parse_args()

def main():
    import shutil
    args = parse_args()
    base_eda = os.path.join(PROJECT_ROOT, "eda_output")
    if args.out:
        target_dir = os.path.abspath(args.out)
    elif args.tag:
        target_dir = os.path.join(base_eda, args.tag)
    else:
        target_dir = os.path.join(base_eda, "latest")
    os.makedirs(target_dir, exist_ok=True)

    X = load_chunk_embeddings()
    W = load_probe_vectors()
    out = generate_uncalibrated_plots(X, W, output_dir=target_dir)

    if args.tag or args.out:
        latest_dir = os.path.join(base_eda, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        shutil.copy2(out, os.path.join(latest_dir, os.path.basename(out)))

    print(f"[Done] Generated: {out}")

if __name__ == "__main__":
    main()
