"""
BitDB-Workbench Configuration and Path Resolver
Dynamic discovery of project root, prototypes, and environmental settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Root of BitDB-Workbench and parent workspace root
WORKBENCH_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = WORKBENCH_DIR.parent

ENV_FILE = WORKBENCH_DIR / ".env"

def resolve_proto_dir(candidates):
    """Returns the first existing candidate directory, or the first entry if none exist."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

# Dynamic prototype path resolution: BitDB is the consolidated active Prototype-4 engine
P4_PATH = resolve_proto_dir([WORKSPACE_ROOT / "BitDB", WORKSPACE_ROOT / "Prototype-4"])
P3_PATH = resolve_proto_dir([WORKSPACE_ROOT / "Prototype-3", WORKSPACE_ROOT / "historical" / "Prototype-3"])
P2_PATH = resolve_proto_dir([WORKSPACE_ROOT / "Prototype-2"])
P1_PATH = resolve_proto_dir([WORKSPACE_ROOT / "Prototype-1"])

PROTOTYPE_DEFS = {
    "Prototype-4": {
        "name": "Prototype-4 / BitDB (ER2 Columnar Extents + MIH + WAND)",
        "description": "Elastic Radix Extent Routing with Multi-Index Hashing, 128KB Columnar Extents, AVX2 Popcount, and Asymmetric Distance Computation.",
        "path": P4_PATH,
        "search_bin": "BitDBSearch.exe",
        "build_bin": "Build.exe",
        "test_bin": "test_suite.exe",
        "watchdog_bin": "Watchdog.exe",
        "print_catalog_bin": "print_catalog.exe",
        "print_segment_dir_bin": "print_segment_dir.exe",
        "storage_dir": "DataStorage",
        "version": 4,
        "type": "columnar_er2"
    },
    "Prototype-3": {
        "name": "Prototype-3 (Avalanche Hash + Segment Chains)",
        "description": "Fixed 256-segment inverted file using 32-bit Avalanche Hash, confidence-margin multi-probing, and row-based extent blocks.",
        "path": P3_PATH,
        "search_bin": "BitDBSearch.exe",
        "build_bin": "Build.exe",
        "storage_dir": "DataStorage",
        "version": 3,
        "type": "avalanche_chain"
    },
    "Prototype-2": {
        "name": "Prototype-2 (Hierarchical BBQ Tree + Beam Search)",
        "description": "Hierarchical K-means tree with beam search navigation and leaf centroids.",
        "path": P2_PATH,
        "search_bin": "Search.exe",
        "build_bin": "Node.exe",
        "storage_dir": "BinStorage",
        "version": 2,
        "type": "hierarchical_tree"
    },
    "Prototype-1": {
        "name": "Prototype-1 (Hierarchical BBQ Tree Baseline)",
        "description": "Foundational binary-balanced quantization hierarchical tree.",
        "path": P1_PATH,
        "search_bin": "Search.exe",
        "build_bin": "Node.exe",
        "storage_dir": "BinStorage",
        "version": 1,
        "type": "hierarchical_tree"
    }
}

def load_env_file() -> Dict[str, str]:
    """Reads simple key=value pairs from .env if present."""
    env_vars = {}
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip()
        except Exception:
            pass
    return env_vars

def save_env_var(key: str, value: str) -> None:
    """Saves or updates a key in .env file."""
    env_vars = load_env_file()
    env_vars[key] = value
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")

def get_gemini_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY from environment or local .env."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    env_vars = load_env_file()
    return env_vars.get("GEMINI_API_KEY", "").strip() or None
