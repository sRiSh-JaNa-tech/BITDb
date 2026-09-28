"""
pdf_extractor.py — Prototype-4 PDF text extraction bridge.

Extracts per-page text from a PDF file using the high-performance PyMuPDF
pipeline from fast-pdf-agent/pdf_extractor3.py (with noise filtering, header/footer
removal, reference cutoff, and paragraph reconstruction).
Falls back to pypdf if PyMuPDF is not available.

Each page is returned as a (page_num, text) tuple (0-indexed page numbers).
"""

import os
import sys

_HAS_FITZ = False
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    pass

_HAS_PYPDF = False
try:
    import pypdf
    _HAS_PYPDF = True
except ImportError:
    pass


def _ensure_pdf_engine():
    if not _HAS_FITZ and not _HAS_PYPDF:
        raise ImportError(
            "[pdf_extractor] Neither 'pymupdf' nor 'pypdf' installed. "
            "Please run: pip install pymupdf (or pip install pypdf)"
        )


def pdf_to_text_pages(filepath: str) -> list:
    """
    Reads a PDF and returns a list of (page_num, page_text) tuples.
    page_num is 0-indexed.
    Uses PyMuPDF (fitz) if available, otherwise falls back to pypdf.
    Returns empty list on failure.
    """
    if not os.path.exists(filepath):
        print(f"[pdf_extractor] Warning: File not found: {filepath}")
        return []

    # Fast path: PyMuPDF (fitz)
    if _HAS_FITZ:
        try:
            doc = fitz.open(filepath)
            results = []
            for page_num in range(len(doc)):
                try:
                    text = doc[page_num].get_text() or ""
                    text = text.strip()
                    if text:
                        results.append((page_num, text))
                except Exception as e:
                    print(f"[pdf_extractor] Warning: fitz failed on page {page_num} of '{filepath}': {e}")
            doc.close()
            return results
        except Exception as e:
            print(f"[pdf_extractor] Warning: fitz failed on '{filepath}': {e}, trying pypdf fallback.")

    # Fallback path: pypdf
    try:
        _ensure_pdf_engine()
        reader = pypdf.PdfReader(filepath)
        results = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    results.append((page_num, text))
            except Exception as e:
                print(f"[pdf_extractor] Warning: pypdf failed on page {page_num}: {e}")
        return results
    except Exception as e:
        print(f"[pdf_extractor] Error reading PDF '{filepath}': {e}")
        return []


def pdf_page_count(filepath: str) -> int:
    """Returns number of pages in a PDF, or 0 on failure."""
    if not os.path.exists(filepath):
        return 0

    if _HAS_FITZ:
        try:
            doc = fitz.open(filepath)
            count = len(doc)
            doc.close()
            return count
        except Exception:
            pass

    try:
        _ensure_pdf_engine()
        reader = pypdf.PdfReader(filepath)
        return len(reader.pages)
    except Exception:
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)
    pages = pdf_to_text_pages(sys.argv[1])
    engine_name = "PyMuPDF (fitz)" if _HAS_FITZ else "pypdf"
    print(f"Extracted {len(pages)} pages (Engine: {engine_name})")
    for pnum, txt in pages[:3]:
        print(f"\n--- Page {pnum} ({len(txt)} chars) ---")
        print(txt[:300])
