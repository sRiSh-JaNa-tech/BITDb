"""
Legacy Adapter for Prototype-1 & Prototype-2
Hierarchical BBQ Tree Search Engines
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BasePrototypeAdapter, SearchResult, SearchItem, LatencyProfile, IndexStats

class LegacyTreeAdapter(BasePrototypeAdapter):
    def __init__(self, key: str, info: Dict[str, Any]):
        super().__init__(key, info)
        self.build_dir = self.root_path / "build"
        self.search_bin = self.build_dir / info["search_bin"]
        self.build_bin = self.build_dir / info["build_bin"]
        self.storage_dir = self.root_path / info["storage_dir"]

    def get_stats(self) -> IndexStats:
        has_binaries = self.search_bin.exists() and self.build_bin.exists()
        nodes_bin = self.storage_dir / "nodes.bin"
        embeddings_bin = self.storage_dir / "embeddings.bin"

        is_built = has_binaries and nodes_bin.exists() and embeddings_bin.exists()

        total_bytes = 0
        chunk_store_bytes = 0
        node_count = 0
        chunk_count = 0

        if self.storage_dir.exists():
            for p in self.storage_dir.glob("*.bin"):
                size = p.stat().st_size
                total_bytes += size
                if p.name == "embeddings.bin":
                    chunk_store_bytes = size
                    chunk_count = size // 384
                elif p.name == "nodes.bin":
                    node_count = size // 40

        extra_details = {
            "index_structure": "Hierarchical Tree",
            "nodes_count": node_count,
            "beam_width": 2 if self.key == "Prototype-2" else 1
        }

        return IndexStats(
            key=self.key,
            name=self.info["name"],
            description=self.info["description"],
            is_built=is_built,
            has_binaries=has_binaries,
            total_index_bytes=total_bytes,
            chunk_store_bytes=chunk_store_bytes,
            doc_count=0,
            chunk_count=chunk_count,
            extra_details=extra_details
        )

    def search(self, query: str, top_k: int = 5, probes: int = 4) -> SearchResult:
        if not self.search_bin.exists():
            return SearchResult(
                query=query,
                top_k=top_k,
                probes=probes,
                exit_code=-1,
                error_message=f"Search binary not found at {self.search_bin}"
            )

        cmd = [str(self.search_bin), query, str(top_k)]
        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.root_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60
            )
            raw_stdout = res.stdout
            items = self._parse_items(raw_stdout)
            return SearchResult(
                query=query,
                top_k=top_k,
                probes=probes,
                items=items,
                latency=LatencyProfile(),
                raw_stdout=raw_stdout,
                exit_code=res.returncode,
                error_message=None if res.returncode == 0 else "Legacy search encountered error"
            )
        except Exception as e:
            return SearchResult(
                query=query,
                top_k=top_k,
                probes=probes,
                exit_code=-1,
                error_message=str(e)
            )

    def _parse_items(self, output: str) -> list:
        items = []
        # Matches lines like:
        # Rank 1 | Score: 2450 | File: C:/...
        # Sentence: "..."
        pattern = r"Rank\s+(\d+)\s*\|\s*Score:\s*([\d\.\-]+)\s*\|\s*File:\s*(.*?)\n\s*Sentence:\s*\"(.*?)\""
        matches = re.findall(pattern, output, re.DOTALL)
        for m in matches:
            items.append(SearchItem(
                rank=int(m[0]),
                score=float(m[1]),
                filename=Path(m[2]).name,
                page=0,
                passage=m[3].strip()
            ))
        return items

    def rebuild_index(self) -> bool:
        if not self.build_bin.exists():
            return False
        try:
            res = subprocess.run([str(self.build_bin)], cwd=str(self.root_path), timeout=600)
            return res.returncode == 0
        except Exception:
            return False
