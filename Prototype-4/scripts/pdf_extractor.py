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

# Ensure fast_pdf_agent is in sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_p = os.path.join(_PROJECT_ROOT, 'fast_pdf_agent')
if os.path.exists(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

_USE_FAST_AGENT = False
try:
    import pdf_extractor3
    _USE_FAST_AGENT = True
except ImportError:
    pass


def _ensure_pypdf():
    try:
        import pypdf
        return pypdf
    except ImportError:
        raise ImportError(
            "[pdf_extractor] Neither 'pymupdf' nor 'pypdf' installed. "
            "Run: pip install pymupdf"
        )


def pdf_to_text_pages(filepath: str) -> list:
    """
    Reads a PDF and returns a list of (page_num, page_text) tuples.
    page_num is 0-indexed.
    Uses pdf_extractor3 (PyMuPDF) if available, otherwise pypdf.
    Returns empty list on failure.
    """
    if not os.path.exists(filepath):
        print(f"[pdf_extractor] Warning: File not found: {filepath}")
        return []

    if _USE_FAST_AGENT:
        try:
            return pdf_extractor3.pdf_to_text_pages(filepath)
        except Exception as e:
            print(f"[pdf_extractor] Warning: pdf_extractor3 failed on '{filepath}': {e}, trying fallback.")

    # Fallback to pypdf
    try:
        pypdf = _ensure_pypdf()
        reader = pypdf.PdfReader(filepath)
        results = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    results.append((page_num, text))
            except Exception as e:
                print(f"[pdf_extractor] Warning: Failed to extract page {page_num}: {e}")
                continue
        return results
    except Exception as e:
        print(f"[pdf_extractor] Error reading PDF '{filepath}': {e}")
        return []


def pdf_page_count(filepath: str) -> int:
    """Returns number of pages in a PDF, or 0 on failure."""
    if not os.path.exists(filepath):
        return 0

    if _USE_FAST_AGENT:
        try:
            return pdf_extractor3.pdf_page_count(filepath)
        except Exception:
            pass

    try:
        pypdf = _ensure_pypdf()
        reader = pypdf.PdfReader(filepath)
        return len(reader.pages)
    except Exception:
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)
    pages = pdf_to_text_pages(sys.argv[1])
    print(f"Extracted {len(pages)} pages (Engine: {'pdf_extractor3' if _USE_FAST_AGENT else 'pypdf'})")
    for pnum, txt in pages[:3]:
        print(f"\n--- Page {pnum} ({len(txt)} chars) ---")
        print(txt[:300])
