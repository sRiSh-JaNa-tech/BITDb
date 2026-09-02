"""
pdf_extractor.py — Prototype-3 PDF text extraction bridge.

Extracts per-page text from a PDF file using pypdf.
Each page is returned as a (page_num, text) tuple (0-indexed page numbers).
"""

import os

def _ensure_pypdf():
    try:
        import pypdf
        return pypdf
    except ImportError:
        raise ImportError(
            "[pdf_extractor] 'pypdf' not installed. "
            "Run: pip install pypdf"
        )


def pdf_to_text_pages(filepath: str) -> list:
    """
    Reads a PDF and returns a list of (page_num, page_text) tuples.
    page_num is 0-indexed.
    Returns empty list on failure.
    """
    pypdf = _ensure_pypdf()

    if not os.path.exists(filepath):
        print(f"[pdf_extractor] Warning: File not found: {filepath}")
        return []

    results = []
    try:
        reader = pypdf.PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    results.append((page_num, text))
            except Exception as e:
                print(f"[pdf_extractor] Warning: Failed to extract page {page_num}: {e}")
                continue
    except Exception as e:
        print(f"[pdf_extractor] Error reading PDF '{filepath}': {e}")
        return []

    return results


def pdf_page_count(filepath: str) -> int:
    """Returns number of pages in a PDF, or 0 on failure."""
    pypdf = _ensure_pypdf()
    try:
        reader = pypdf.PdfReader(filepath)
        return len(reader.pages)
    except Exception:
        return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)
    pages = pdf_to_text_pages(sys.argv[1])
    print(f"Extracted {len(pages)} pages")
    for pnum, txt in pages[:3]:
        print(f"\n--- Page {pnum} ({len(txt)} chars) ---")
        print(txt[:300])
