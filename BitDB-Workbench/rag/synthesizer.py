"""
RAG Synthesizer for BitDB
Handles dual-mode grounded answering:
1. High-precision Offline Extractive Synthesizer (Zero external dependencies or API keys needed)
2. Advanced Google Gemini Generative Synthesizer (using gemini-2.5-flash if API key provided)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from adapters.base import SearchItem
from config import get_gemini_api_key

class RAGSynthesizer:
    def __init__(self):
        self.gemini_key = get_gemini_api_key()

    def set_gemini_key(self, key: Optional[str]) -> None:
        self.gemini_key = key

    def format_grounded_context(self, items: List[SearchItem]) -> Tuple[str, List[Dict[str, Any]]]:
        """Prepares numbered sources with exact attribution headers."""
        context_blocks = []
        sources = []

        for idx, item in enumerate(items, 1):
            src_tag = f"[Source {idx}]"
            sources.append({
                "tag": src_tag,
                "rank": item.rank,
                "score": item.score,
                "file": item.filename,
                "page": item.page,
                "passage": item.passage
            })
            block = (
                f"{src_tag} Document: {item.filename} (Page {item.page}, Match Score: {item.score:.4f})\n"
                f'Content: "{item.passage}"'
            )
            context_blocks.append(block)

        full_context = "\n\n".join(context_blocks)
        return full_context, sources

    def synthesize(self, query: str, items: List[SearchItem], prefer_gemini: bool = True) -> Dict[str, Any]:
        """Synthesizes a response using Gemini if available, or offline extractive reasoning."""
        if not items:
            return {
                "mode": "none",
                "answer": "No relevant passages were retrieved for this query.",
                "sources": []
            }

        context_text, sources = self.format_grounded_context(items)

        # Attempt Gemini if key is present and requested
        if prefer_gemini and self.gemini_key:
            try:
                gemini_res = self._call_gemini(query, context_text)
                if gemini_res:
                    return {
                        "mode": "gemini",
                        "model": "gemini-2.5-flash",
                        "answer": gemini_res,
                        "sources": sources
                    }
            except Exception as e:
                # Graceful fallback to offline synthesis
                pass

        # Offline Extractive & Grounded Synthesis
        offline_res = self._synthesize_offline(query, sources)
        return {
            "mode": "offline",
            "model": "BitDB Grounded Extractive Engine",
            "answer": offline_res,
            "sources": sources
        }

    def _call_gemini(self, query: str, context_text: str) -> Optional[str]:
        """Calls Google GenAI Gemini model using google-genai or requests fallback."""
        prompt = f"""You are an expert AI research assistant querying the BitDB vector storage database.
Use ONLY the provided context passages to answer the question accurately and concisely.
If the context does not contain enough information, clearly say so.
Cite sources using the exact tags like [Source 1], [Source 2] at the end of relevant facts.

Context:
{context_text}

Question:
{query}

Answer:"""

        # 1. Try google-genai SDK
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass

        # 2. Try REST API via requests
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2}
            }
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
        except Exception:
            pass

        return None

    def _synthesize_offline(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Heuristic semantic extractor and fact synthesizer."""
        query_words = set(re.findall(r"\w+", query.lower())) - {"the", "a", "an", "in", "on", "of", "and", "or", "to", "for", "is", "are"}

        scored_sentences = []
        for src in sources:
            passage = src["passage"]
            tag = src["tag"]
            # Split into approximate sentences
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 20:
                    continue
                s_words = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > 0:
                    scored_sentences.append({
                        "text": s_clean,
                        "tag": tag,
                        "score": overlap + (src["score"] if src["score"] < 2.0 else src["score"] / 10000.0)
                    })

        scored_sentences.sort(key=lambda x: x["score"], reverse=True)

        # Build grounded markdown summary
        lines = []
        lines.append("### Grounded Findings")
        
        if scored_sentences:
            chosen = scored_sentences[:4]
            seen_texts = set()
            for item in chosen:
                txt = item['text']
                if txt in seen_texts:
                    continue
                seen_texts.add(txt)
                lines.append(f"• {txt} {item['tag']}")
        else:
            # Fallback to top passages excerpts
            for src in sources[:2]:
                snippet = src['passage'][:250].strip()
                lines.append(f"• \"{snippet}...\" {src['tag']}")

        lines.append("\n### Technical Context")
        top_src = sources[0]
        lines.append(
            f"The primary relevant literature is **{top_src['file']}** (Page {top_src['page']}) "
            f"with similarity score **{top_src['score']:.4f}**. Retrieved text directly matches query semantic features."
        )

        return "\n".join(lines)
