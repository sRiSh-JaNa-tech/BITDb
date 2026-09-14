"""
High-Level RAG Retrieval & Answering Engine
Coordinates query routing, search execution, source extraction, and synthesis.
"""

from typing import Dict, Any, Optional
from adapters.base import BasePrototypeAdapter, SearchResult
from .synthesizer import RAGSynthesizer

class RAGEngine:
    def __init__(self, synthesizer: Optional[RAGSynthesizer] = None):
        self.synthesizer = synthesizer or RAGSynthesizer()

    def answer_question(
        self,
        adapter: BasePrototypeAdapter,
        question: str,
        top_k: int = 5,
        probes: int = 4,
        prefer_gemini: bool = True
    ) -> Dict[str, Any]:
        """Runs end-to-end RAG pipeline on the given prototype adapter."""
        # 1. Retrieve candidate chunks from the vector database
        search_res: SearchResult = adapter.search(question, top_k=top_k, probes=probes)

        if search_res.exit_code != 0 or not search_res.items:
            return {
                "success": False,
                "error": search_res.error_message or "No matching passages found or search failed.",
                "search_result": search_res,
                "synthesis": None
            }

        # 2. Synthesize answer with citations
        synthesis = self.synthesizer.synthesize(question, search_res.items, prefer_gemini=prefer_gemini)

        return {
            "success": True,
            "error": None,
            "question": question,
            "search_result": search_res,
            "synthesis": synthesis
        }
