"""Rename research PDFs using their embedded or first-page titles.

Supports scanning multiple directories (such as papers/, pdfs/, close_topic_papers/)
and records progress in ``.pdf_rename_state.json`` so operations can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STATE_FILENAME = ".pdf_rename_state.json"
INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE = re.compile(r"\s+")
MAX_FILENAME_LENGTH = 180

DEFAULT_DIRECTORIES = ["papers", "pdfs", "close_topic_papers"]


def load_pdf_reader() -> tuple[Any, str]:
    """Load an installed PDF reader, preferring PyMuPDF's text extraction."""
    try:
        import fitz

        return fitz, "fitz"
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            return PdfReader, "pypdf2"
        except ImportError as exc:
            raise RuntimeError(
                "Install a PDF library first: python -m pip install pymupdf"
            ) from exc


def load_papers_json_mapping(base_dir: Path) -> dict[str, str]:
    """Load title mappings (by paper_id and text/title) from papers.json if available."""
    candidates = [
        base_dir / "papers.json",
        base_dir.parent / "papers.json",
        Path("papers.json"),
    ]
    mapping: dict[str, str] = {}
    for candidate in candidates:
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                for paper in data.get("papers", []):
                    pid = paper.get("paper_id")
                    title = paper.get("title")
                    if pid and title:
                        mapping[str(pid).strip()] = str(title).strip()
            except Exception:
                pass
            if mapping:
                break
    return mapping


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            raise ValueError("state file has an invalid structure")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc


import time


def save_state(path: Path, state: dict[str, Any]) -> None:
    content = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    # Attempt atomic replacement with retries for Windows file-locking
    written = False
    for attempt in range(5):
        try:
            temporary_path.write_text(content, encoding="utf-8")
            os.replace(temporary_path, path)
            written = True
            break
        except (PermissionError, OSError):
            time.sleep(0.05 * (attempt + 1))

    # Fallback to direct write if replace fails
    if not written:
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as exc:
            print(f"Warning: Could not save checkpoint to {path}: {exc}", file=sys.stderr)

    # Clean up orphan temporary file if it exists
    try:
        if temporary_path.exists():
            temporary_path.unlink()
    except Exception:
        pass


def clean_title(title: str) -> str:
    """Convert extracted title text into a valid, readable Windows filename."""
    title = unicodedata.normalize("NFKC", title)
    title = title.replace("\u00ad", "")
    # Remove LaTeX / BibTeX curly brackets if present
    title = title.replace("{", "").replace("}", "")
    title = WHITESPACE.sub(" ", title).strip(" .")
    title = INVALID_WINDOWS_CHARS.sub("-", title)
    title = re.sub(r"-{2,}", "-", title).strip(" .")
    if title.upper() in {"CON", "PRN", "AUX", "NUL"}:
        title = f"{title}-paper"
    for prefix in ("COM", "LPT"):
        if title.upper().startswith(prefix) and title[len(prefix) :].isdigit():
            title = f"{title}-paper"
    return title[:MAX_FILENAME_LENGTH].rstrip(" .")


def looks_like_title(line: str) -> bool:
    line = WHITESPACE.sub(" ", line).strip()
    lowered = line.lower()
    if len(line) < 4 or len(line) > 240:
        return False
    if lowered.startswith(("http://", "https://", "www.", "doi:")):
        return False
    if re.fullmatch(r"[\d\W_]+", line):
        return False
    # Filter out arXiv headers and references
    if re.match(r"^arxiv[:\s\d\.\/v]+", lowered) or "[cs." in lowered or "arxiv:" in lowered:
        return False
    # Filter out copyright, license, and repository portal headers
    if (
        "journal homepage" in lowered
        or "copyright" in lowered
        or "all rights reserved" in lowered
        or "permission from ieee" in lowered
        or "personal use of this material" in lowered
        or "reprinting-republishing" in lowered
    ):
        return False
    if lowered in {
        "untitled",
        "loading",
        "unknown",
        "title",
        "document",
        "cover page",
        "vu research portal",
        "research portal",
        "open access",
    }:
        return False
    # Filter out bot check and captcha pages
    if (
        "making sure you're not a bot" in lowered
        or "robot check" in lowered
        or "attention required" in lowered
        or "cloudflare" in lowered
    ):
        return False
    # Filter out raw file names and software templates
    if lowered.endswith((".pdf", ".eps", ".ps", ".doc", ".docx", ".indd")):
        return False
    if "microsoft word" in lowered or "google drive" in lowered or "powerpoint" in lowered:
        return False
    return sum(character.isalpha() for character in line) >= 4


def first_page_title(text: str) -> str | None:
    lines = [WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    candidates: list[str] = []
    for line in lines[:35]:
        if not looks_like_title(line):
            continue
        if re.search(
            r"\b(authors?|abstract|keywords?|arxiv|downloaded from|published in)\b",
            line,
            re.I,
        ):
            continue
        candidates.append(line)
    if not candidates:
        return None
    return max(candidates[:8], key=len)


def first_page_title_from_layout(page: Any) -> str | None:
    """Choose the largest meaningful text block near the top of a page."""
    candidates: list[tuple[float, float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue
        lines = block["lines"]
        for line_index, line in enumerate(lines):
            text = " ".join(
                span["text"].strip() for span in line["spans"] if span["text"].strip()
            )
            if not looks_like_title(text):
                continue
            if re.search(
                r"\b(authors?|abstract|keywords?|article info|copyright|received|accepted|"
                r"university|department|repository|journal|issn|license|published in|downloaded from)\b",
                text,
                re.I,
            ):
                continue
            if text.isupper() and len(text) <= 20:
                continue
            if re.match(r"^\d+[\s,.)-]", text):
                continue
            font_size = max(
                (span["size"] for span in line["spans"] if span["text"].strip()),
                default=0.0,
            )
            title_lines = [text]
            for continuation in lines[line_index + 1 :]:
                continuation_text = " ".join(
                    span["text"].strip()
                    for span in continuation["spans"]
                    if span["text"].strip()
                )
                continuation_size = max(
                    (
                        span["size"]
                        for span in continuation["spans"]
                        if span["text"].strip()
                    ),
                    default=0.0,
                )
                if (
                    continuation_size < font_size * 0.85
                    or not looks_like_title(continuation_text)
                    or re.search(
                        r"\b(authors?|abstract|keywords?|university|department|repository)\b",
                        continuation_text,
                        re.I,
                    )
                ):
                    break
                title_lines.append(continuation_text)
            text = " ".join(title_lines)
            candidates.append((font_size, -line["bbox"][1], text))
    if not candidates:
        return None
    return max(candidates)[2]


def extract_title(
    path: Path,
    reader_api: Any,
    reader_kind: str,
    papers_map: dict[str, str] | None = None,
    size_map: dict[int, str] | None = None,
) -> str:
    # 1. Check if file stem matches a known paper_id in papers.json
    if papers_map and path.stem in papers_map:
        return papers_map[path.stem]

    # 2. Check if file size matches a known paper_id from state/papers.json
    if size_map:
        sz = path.stat().st_size
        if sz in size_map:
            return size_map[sz]

    if reader_kind == "fitz":
        document = reader_api.open(path)
        try:
            metadata_title = (document.metadata or {}).get("title", "")
            if metadata_title and looks_like_title(metadata_title):
                return metadata_title
            if document.page_count:
                # Check page 0 layout
                title = first_page_title_from_layout(document[0])
                if title and looks_like_title(title):
                    return title

                # If page 0 is a repository cover page, check page 1 layout
                if document.page_count > 1:
                    title_p1 = first_page_title_from_layout(document[1])
                    if title_p1 and looks_like_title(title_p1):
                        return title_p1

                text = document[0].get_text("text")
            else:
                text = ""
        finally:
            document.close()
    else:
        reader = reader_api(str(path))
        metadata = reader.metadata or {}
        metadata_title = str(metadata.get("/Title") or "").strip()
        if metadata_title and looks_like_title(metadata_title):
            return metadata_title
        text = reader.pages[0].extract_text() if reader.pages else ""

    title = first_page_title(text or "")
    if not title and reader_kind == "fitz":
        # Fallback to page 1 text if page 0 had no title
        document = reader_api.open(path)
        try:
            if document.page_count > 1:
                title = first_page_title(document[1].get_text("text") or "")
        finally:
            document.close()

    if not title:
        raise ValueError("could not find a title in PDF metadata or first page")
    return title


def unique_target(
    directory: Path, title: str, source: Path, reserved: set[str]
) -> Path:
    base = clean_title(title) or source.stem
    candidate = directory / f"{base}.pdf"
    number = 2
    while (
        candidate.name.casefold() in reserved
        and candidate.name.casefold() != source.name.casefold()
    ) or (
        candidate.exists()
        and candidate.resolve() != source.resolve()
    ):
        candidate = directory / f"{base} ({number}).pdf"
        number += 1
    reserved.add(candidate.name.casefold())
    return candidate


def safe_rename(source: Path, target: Path) -> None:
    """Safely rename a file on Windows, handling same-name and case-only changes."""
    if source.resolve() == target.resolve():
        if source.name == target.name:
            return  # Exact same name, no-op
        # Case change on Windows: rename via intermediate temp file
        temp_name = source.parent / f"__temp_{os.getpid()}_{source.name}"
        source.rename(temp_name)
        temp_name.rename(target)
        return

    source.rename(target)


def fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def process_directory(
    directory: Path,
    state_path: Path,
    state: dict[str, Any],
    papers_map: dict[str, str],
    size_map: dict[int, str],
    dry_run: bool,
    retry_failed: bool,
) -> int:
    files = state["files"]
    reader_api, reader_kind = load_pdf_reader()
    pdf_files = sorted(directory.glob("*.pdf"), key=lambda item: item.name.casefold())

    if not pdf_files:
        print(f"Directory {directory.name}/: No PDF files found.")
        return 0

    print(f"\n--- Processing directory: {directory.name}/ ({len(pdf_files)} PDF files) ---")
    reserved = {path.name.casefold() for path in pdf_files}
    failures = 0

    for source in pdf_files:
        key = f"{directory.name}/{source.name}"
        record = files.get(key, {})
        current_fingerprint = fingerprint(source)

        if record.get("status") == "failed" and not retry_failed:
            print(f"SKIP  {source.name} (failed previously; use --retry-failed)")
            failures += 1
            continue

        try:
            title = extract_title(source, reader_api, reader_kind, papers_map, size_map)
            target = unique_target(directory, title, source, reserved)

            if source.name == target.name:
                files[key] = {
                    "status": "renamed",
                    "fingerprint": current_fingerprint,
                    "title": title,
                    "target": target.name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                print(f"KEEP  {source.name} (already correctly named)")
            else:
                item_record = {
                    "status": "planned" if dry_run else "renamed",
                    "fingerprint": current_fingerprint,
                    "title": title,
                    "target": target.name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                if dry_run:
                    print(f"DRY   {source.name} -> {target.name}")
                else:
                    safe_rename(source, target)
                    # Update target record in state
                    new_key = f"{directory.name}/{target.name}"
                    new_record = dict(item_record)
                    new_record["fingerprint"] = fingerprint(target)
                    files[new_key] = new_record
                    print(f"OK    {source.name} -> {target.name}")
                files[key] = item_record
        except Exception as exc:
            files[key] = {
                "status": "failed",
                "fingerprint": current_fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            failures += 1
            print(f"FAIL  {source.name}: {exc}", file=sys.stderr)

        if not dry_run:
            save_state(state_path, state)

    return failures


def build_size_map(base_dir: Path, state: dict[str, Any], papers_map: dict[str, str]) -> dict[int, str]:
    """Map file sizes to known titles from papers.json and state."""
    size_map: dict[int, str] = {}
    for k, v in state.get("files", {}).items():
        if "fingerprint" in v and "size" in v["fingerprint"]:
            sz = v["fingerprint"]["size"]
            pid = Path(k).stem
            if pid in papers_map:
                size_map[sz] = papers_map[pid]
    return size_map


def process_all(
    base_dir: Path,
    target_dirs: list[str],
    dry_run: bool,
    retry_failed: bool,
    reset: bool,
) -> int:
    state_path = base_dir / STATE_FILENAME
    state = {"version": 1, "files": {}} if reset else load_state(state_path)
    papers_map = load_papers_json_mapping(base_dir)
    size_map = build_size_map(base_dir, state, papers_map)

    total_failures = 0
    for dir_name in target_dirs:
        target_path = base_dir / dir_name if not Path(dir_name).is_absolute() else Path(dir_name)
        if not target_path.exists():
            print(f"Directory {dir_name}/ does not exist, skipping.")
            continue
        if not target_path.is_dir():
            print(f"Path {dir_name} is not a directory, skipping.")
            continue

        failures = process_directory(
            directory=target_path,
            state_path=state_path,
            state=state,
            papers_map=papers_map,
            size_map=size_map,
            dry_run=dry_run,
            retry_failed=retry_failed,
        )
        total_failures += failures

    if not dry_run:
        save_state(state_path, state)
        print(f"\nState saved to {state_path}")

    return 1 if total_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRECTORIES,
        help="Directories to scan for PDFs (default: papers, pdfs, close_topic_papers)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show planned names without renaming"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="retry files recorded as failed"
    )
    parser.add_argument(
        "--reset", action="store_true", help="discard the checkpoint and start over"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    return process_all(
        base_dir=base_dir,
        target_dirs=args.dirs,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        reset=args.reset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
