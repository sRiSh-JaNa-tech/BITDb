# ⚡ BitDB-Workbench: High-Performance Vector Database Studio & RAG Suite

An interactive terminal workspace built with **Rich Console** for managing, benchmarking, and testing Retrieval-Augmented Generation (RAG) across all BitDB prototype generations (**Prototype-1** through **Prototype-4**).

---

## 🌟 Key Capabilities

1. **Multi-Prototype Dynamic Fleet Management**:
   - Seamlessly toggle between **Prototype-1**, **Prototype-2**, **Prototype-3** (Avalanche Hash), and **Prototype-4** (ER2 Columnar Extents).
   - Real-time disk inspection: physical extent count, chunk inventory, binary index file sizes, and SIMD support detection.

2. **Grounded RAG Retrieval Engine**:
   - Query vectors via the active prototype's C++ search engine.
   - Dual-mode grounded answer synthesis:
     - **Offline Extractive Grounded Synthesizer**: High-precision key-claim extraction with zero external API dependencies.
     - **Google Gemini Generative AI (`gemini-2.5-flash`)**: Academic synthesis with inline attribution citations (`[Source 1]`, `[Source 2]`).
   - Detailed hardware & I/O telemetry table per query (query embedding time, SSD extent scan time, candidates scored vs bypassed, bulk SSD bytes read).

3. **Automated Latency & I/O Benchmark Suite**:
   - Evaluates a technical 8-query battery across vector similarity search, SSD random access, quantization, and systems storage papers.
   - Computes statistical percentiles: **p50**, **p90**, **p95**, **p99**, mean latency, mean bulk I/O, and candidate bypass rates.

4. **Side-by-Side Architectural Shootout**:
   - Pit **Prototype-3** (Avalanche Hash Baseline) against **Prototype-4** (ER2 Columnar Extents).
   - Directly measures the empirical impact of Multi-Index Hashing (MIH), 128KB Columnar Extents (CEL), WAND Centroid/Radius Pruning, and 2-Stage ADC Crossover.

5. **Index Inspector & Extent Storage**:
   - Deep inspection of on-disk binary structures (`chunk_store.bin`, `mih_table.bin`, `segment_dir.bin`, `doc_catalog.bin`).

---

## 🚀 Quickstart

### Launch with Windows Batch
Double-click `run.bat` or run:
```cmd
cd BitDB-Workbench
run.bat
```

### Launch with PowerShell
```powershell
cd BitDB-Workbench
.\run.ps1
```

*(The launcher automatically initializes the `.venv` virtual environment and installs all dependencies from `requirements.txt` on first launch).*

---

## 📂 Project Architecture

```
BitDB-Workbench/
├── .venv/                         # Dedicated Python 3.11 virtual environment
├── requirements.txt               # rich, prompt_toolkit, google-genai, requests
├── run.bat                        # Batch launcher
├── run.ps1                        # PowerShell launcher
├── workbench.py                   # Rich interactive TUI entrypoint
├── config.py                      # Dynamic root & prototype path configuration
├── adapters/
│   ├── base.py                    # BasePrototypeAdapter abstract class
│   ├── prototype4_adapter.py      # Prototype-4 (ER2 Columnar, MIH, ADC, WAND)
│   ├── prototype3_adapter.py      # Prototype-3 (Avalanche Hash, Row Extents)
│   ├── legacy_adapter.py          # Prototypes 1 & 2 (Hierarchical BBQ Tree)
│   └── __init__.py                # Adapter factory
├── rag/
│   ├── synthesizer.py             # Dual answering (Offline + Gemini API)
│   └── engine.py                  # RAG orchestrator
└── benchmarks/
    ├── runner.py                  # 8-query latency and I/O battery
    ├── comparator.py              # Side-by-side shootout runner
    └── __init__.py
```
