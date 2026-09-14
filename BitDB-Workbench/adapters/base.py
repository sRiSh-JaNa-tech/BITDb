"""
Base Abstract Adapter for BitDB Prototypes
Defines standard interfaces and data contracts for index status, search, and rebuilds.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class SearchItem:
    rank: int
    score: float
    filename: str
    page: int
    passage: str
    doc_id: Optional[int] = None
    chunk_idx: Optional[int] = None

@dataclass
class LatencyProfile:
    embed_ms: float = 0.0
    disk_ms: float = 0.0
    scored_candidates: int = 0
    total_candidates: int = 0
    bypass_rate_pct: float = 0.0
    bulk_read_kb: float = 0.0
    segments_probed: int = 0
    passage_ms: float = 0.0
    total_ms: float = 0.0

@dataclass
class SearchResult:
    query: str
    top_k: int
    probes: int
    items: List[SearchItem] = field(default_factory=list)
    latency: LatencyProfile = field(default_factory=LatencyProfile)
    raw_stdout: str = ""
    exit_code: int = 0
    error_message: Optional[str] = None

@dataclass
class IndexStats:
    key: str
    name: str
    description: str
    is_built: bool
    has_binaries: bool
    total_index_bytes: int = 0
    chunk_store_bytes: int = 0
    doc_count: int = 0
    chunk_count: int = 0
    extra_details: Dict[str, Any] = field(default_factory=dict)

class BasePrototypeAdapter(ABC):
    def __init__(self, key: str, info: Dict[str, Any]):
        self.key = key
        self.info = info
        self.root_path = Path(info["path"])

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Inspects disk layout and returns health and footprint stats."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, probes: int = 4) -> SearchResult:
        """Executes a search query and returns structured results and telemetry."""
        pass

    @abstractmethod
    def rebuild_index(self) -> bool:
        """Executes index rebuild."""
        pass
