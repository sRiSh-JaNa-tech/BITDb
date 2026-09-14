"""
Adapter for Prototype-4:
Elastic Radix Extent Routing (ER2) Columnar Storage Engine
Features: Multi-Index Hashing (MIH), 128 KB Columnar Extents, WAND Centroid/Radius Pruning, AVX2 Harley-Seal Popcount, and ADC 2-Stage I/O.
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BasePrototypeAdapter, SearchResult, SearchItem, LatencyProfile, IndexStats

class Prototype4Adapter(BasePrototypeAdapter):
    def __init__(self, key: str, info: Dict[str, Any]):
        super().__init__(key, info)
        self.build_dir = self.root_path / "build"
        self.search_bin = self.build_dir / info["search_bin"]
        self.build_bin = self.build_dir / info["build_bin"]
        self.storage_dir = self.root_path / info["storage_dir"]

    def get_stats(self) -> IndexStats:
        has_binaries = self.search_bin.exists() and self.build_bin.exists()
        chunk_store = self.storage_dir / "chunk_store.bin"
        seg_dir = self.storage_dir / "segment_dir.bin"
        mih_table = self.storage_dir / "mih_table.bin"
        doc_catalog = self.storage_dir / "doc_catalog.bin"

        is_built = has_binaries and chunk_store.exists() and seg_dir.exists() and mih_table.exists()

        total_bytes = 0
        chunk_store_bytes = 0
        doc_count = 0
        chunk_count = 0
        extra_details = {
            "extent_size_kb": 128,
            "extent_count": 0,
            "mih_table_size_bytes": 0,
            "segments_count": 256,
            "simd_vectorization": "AVX2 Harley-Seal (_mm256_shuffle_epi8)",
            "pruning_model": "Geometric WAND (Cauchy-Schwarz Upper Bound)",
            "io_model": "2-Stage ADC Crossover (Scatter vs Contiguous)"
        }

        if self.storage_dir.exists():
            for p in self.storage_dir.glob("*.bin"):
                size = p.stat().st_size
                total_bytes += size
                if p.name == "chunk_store.bin":
                    chunk_store_bytes = size
                    extra_details["extent_count"] = size // 131072
                elif p.name == "mih_table.bin":
                    extra_details["mih_table_size_bytes"] = size

        if doc_catalog.exists():
            # In doc_catalog.bin: 8-byte header (magic+count) + entries
            try:
                with open(doc_catalog, "rb") as f:
                    data = f.read(8)
                    if len(data) >= 8:
                        import struct
                        _, count = struct.unpack("<II", data[:8])
                        doc_count = count
            except Exception:
                pass

        if seg_dir.exists():
            try:
                # SegDirHeader: magic(4), version(4), num_segments(4), reserved(4) = 16 bytes
                # SegEntry: chunk_store_offset(8), chunk_count(4), ext_chain_head(4), centroid(384*4=1536), max_radius(4) = 1556 bytes
                with open(seg_dir, "rb") as f:
                    hdr = f.read(16)
                    import struct
                    total_chunks = 0
                    entry_sz = 1556
                    while True:
                        buf = f.read(entry_sz)
                        if len(buf) < entry_sz:
                            break
                        _, cnt, _, _, _ = struct.unpack("<QII1536s4s", buf[:1556])
                        total_chunks += cnt
                    chunk_count = total_chunks
            except Exception:
                pass

        return IndexStats(
            key=self.key,
            name=self.info["name"],
            description=self.info["description"],
            is_built=is_built,
            has_binaries=has_binaries,
            total_index_bytes=total_bytes,
            chunk_store_bytes=chunk_store_bytes,
            doc_count=doc_count,
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

        cmd = [str(self.search_bin), query, str(top_k), str(probes)]
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
            latency = self._parse_latency(raw_stdout)
            return SearchResult(
                query=query,
                top_k=top_k,
                probes=probes,
                items=items,
                latency=latency,
                raw_stdout=raw_stdout,
                exit_code=res.returncode
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
        # Matches blocks like:
        # ┌─ Rank 1 ─────────────────────────────────────────
        # │  Score   : 0.730335
        # │  Doc ID  : 16 (Chunk #888, Page 1) [Optional]
        # │  File    : ...
        # │  Page    : 1
        # │  Passage : "..."
        # └──────────────────────────────────────────────────
        rank_blocks = re.split(r"┌─ Rank\s+(\d+)", output)
        for i in range(1, len(rank_blocks), 2):
            rank = int(rank_blocks[i])
            block = rank_blocks[i+1]

            score_match = re.search(r"Score\s*:\s*([\d\.\-]+)", block)
            score = float(score_match.group(1)) if score_match else 0.0

            file_match = re.search(r"File\s*:\s*(.+)", block)
            filename = file_match.group(1).strip() if file_match else "Unknown"

            page_match = re.search(r"Page\s*:\s*(\d+)", block)
            page = int(page_match.group(1)) if page_match else 0

            passage_match = re.search(r'Passage\s*:\s*["\']?(.*?)["\']?\s*(?=\n\s*└|\n\s*│|\Z)', block, re.DOTALL)
            passage = passage_match.group(1).strip() if passage_match else ""

            doc_id_match = re.search(r"Doc ID\s*:\s*(\d+)", block)
            doc_id = int(doc_id_match.group(1)) if doc_id_match else None

            chunk_match = re.search(r"Chunk #(\d+)", block)
            chunk_idx = int(chunk_match.group(1)) if chunk_match else None

            items.append(SearchItem(
                rank=rank,
                score=score,
                filename=filename,
                page=page,
                passage=passage,
                doc_id=doc_id,
                chunk_idx=chunk_idx
            ))
        return items

    def _parse_latency(self, output: str) -> LatencyProfile:
        prof = LatencyProfile()
        m_embed = re.search(r"Query Embedding\s*:\s*([\d\.]+)\s*ms", output)
        if m_embed:
            prof.embed_ms = float(m_embed.group(1))

        m_disk = re.search(r"SSD Extent Scan\s*:\s*([\d\.]+)\s*ms\s*\((?:(\d+)\s*scored\s*/\s*(\d+)\s*candidates)?\)", output)
        if m_disk:
            prof.disk_ms = float(m_disk.group(1))
            if m_disk.group(2) and m_disk.group(3):
                prof.scored_candidates = int(m_disk.group(2))
                prof.total_candidates = int(m_disk.group(3))
                if prof.total_candidates > 0:
                    prof.bypass_rate_pct = round(100.0 * (1.0 - prof.scored_candidates / prof.total_candidates), 2)

        m_io = re.search(r"Bulk I/O Read\s*:\s*([\d\.]+)\s*KB\s*in\s*(\d+)\s*segments", output)
        if m_io:
            prof.bulk_read_kb = float(m_io.group(1))
            prof.segments_probed = int(m_io.group(2))

        m_pass = re.search(r"Passage Fetch\s*:\s*([\d\.]+)\s*ms", output)
        if m_pass:
            prof.passage_ms = float(m_pass.group(1))

        m_tot = re.search(r"Total Latency\s*:\s*([\d\.]+)\s*ms", output)
        if m_tot:
            prof.total_ms = float(m_tot.group(1))

        return prof

    def rebuild_index(self) -> bool:
        if not self.build_bin.exists():
            return False
        cmd = [str(self.build_bin), "--rebuild"]
        try:
            res = subprocess.run(cmd, cwd=str(self.root_path), timeout=600)
            return res.returncode == 0
        except Exception:
            return False
