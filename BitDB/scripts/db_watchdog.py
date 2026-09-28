#!/usr/bin/env python3
"""
db_watchdog.py — Dynamic Background Auto-Sync Watchdog for BitDB Prototype-4.

Monitors './ingestor' for PDF additions, modifications, or deletions
and automatically triggers BitDB incremental ingestion or compaction.
Cross-platform (Windows, Linux, macOS) with dynamic path resolution.
"""

import os
import sys
import time
import subprocess
import argparse


def find_project_root() -> str:
    """Dynamically discover project root across environments and working directories."""
    # 1. Environment variable override
    env_root = os.environ.get("BITDB_ROOT")
    if env_root and os.path.exists(env_root):
        return os.path.abspath(env_root)

    # 2. Upward traversal from current working directory and script location
    search_starts = [os.getcwd()]
    if "__file__" in globals():
        search_starts.append(os.path.dirname(os.path.abspath(__file__)))

    for start in search_starts:
        cur = os.path.abspath(start)
        for _ in range(6):
            if (
                os.path.isdir(os.path.join(cur, "ingestor"))
                or os.path.exists(os.path.join(cur, "CMakeLists.txt"))
                or os.path.exists(os.path.join(cur, "build.bat"))
                or (os.path.isdir(os.path.join(cur, "src")) and os.path.isdir(os.path.join(cur, "DataStorage")))
            ):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    return os.path.abspath(".")


PROJECT_ROOT = find_project_root()


def resolve_ingestor_dir(custom_path: str = None) -> str:
    """Resolve the ingestor directory dynamically, defaulting to ./ingestor."""
    if custom_path:
        return os.path.abspath(custom_path)
    
    # Priority 1: Check ./ingestor relative to current working directory
    if os.path.isdir("./ingestor"):
        return os.path.abspath("./ingestor")
        
    # Priority 2: Ingestor folder in discovered project root
    root_ingestor = os.path.join(PROJECT_ROOT, "ingestor")
    if os.path.isdir(root_ingestor):
        return root_ingestor
        
    # Default fallback: ./ingestor
    return os.path.abspath("./ingestor")


def find_build_executable() -> str:
    """Locate the BitDB Build binary across platforms and build directories."""
    candidate_names = ["Build.exe", "Build"]
    candidate_dirs = [
        os.path.join(PROJECT_ROOT, "build"),
        os.path.join(PROJECT_ROOT, "build", "Release"),
        os.path.join(PROJECT_ROOT, "build_cmake"),
        os.path.join(PROJECT_ROOT, "build_cmake", "Release"),
        PROJECT_ROOT,
        os.path.join(os.getcwd(), "build"),
        os.getcwd(),
    ]
    
    for d in candidate_dirs:
        for name in candidate_names:
            full_path = os.path.join(d, name)
            if os.path.isfile(full_path):
                return os.path.abspath(full_path)
                
    # Fallback to standard platform default
    default_exe = "Build.exe" if sys.platform == "win32" else "Build"
    return os.path.join(PROJECT_ROOT, "build", default_exe)


def get_pdf_snapshot(ingestor_dir: str) -> dict:
    """Return a mapping of relative_path -> (mtime, size) for all PDFs in ingestor_dir."""
    snapshot = {}
    if not os.path.exists(ingestor_dir):
        return snapshot
        
    for root, _, files in os.walk(ingestor_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, ingestor_dir)
                try:
                    stat = os.stat(full_path)
                    snapshot[rel_path] = (stat.st_mtime, stat.st_size)
                except OSError:
                    continue
    return snapshot


def main():
    parser = argparse.ArgumentParser(
        description="BitDB Dynamic Auto-Sync Watchdog: Monitors ./ingestor and triggers incremental ingestion."
    )
    parser.add_argument(
        "--dir", "-d",
        default="./ingestor",
        help="Path to the ingestor directory to monitor (default: ./ingestor)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.5,
        help="Polling interval in seconds (default: 2.5s)"
    )
    args = parser.parse_args()

    ingestor_dir = resolve_ingestor_dir(args.dir)
    build_exe = find_build_executable()
    
    # Ensure ingestor folder exists
    os.makedirs(ingestor_dir, exist_ok=True)

    print("==================================================")
    print("  BitDB Dynamic Background Auto-Sync Watchdog")
    print("==================================================")
    print(f"[*] Project root:       {PROJECT_ROOT}")
    print(f"[*] Monitoring dir:     ./ingestor -> ({ingestor_dir})")
    print(f"[*] Engine executable:  {build_exe}")
    print(f"[*] Polling interval:   {args.interval}s")
    print("[*] Ready. Listening for file additions, edits, or deletions...")
    print("    (Press Ctrl+C to stop)\n")

    current_snapshot = get_pdf_snapshot(ingestor_dir)

    while True:
        try:
            time.sleep(args.interval)
            new_snapshot = get_pdf_snapshot(ingestor_dir)

            if new_snapshot != current_snapshot:
                current_keys = set(current_snapshot.keys())
                new_keys = set(new_snapshot.keys())

                added = new_keys - current_keys
                deleted = current_keys - new_keys
                modified = {k for k in current_keys & new_keys if current_snapshot[k] != new_snapshot[k]}

                if added:
                    print(f"[Watchdog] Detected {len(added)} new file(s): {', '.join(added)}")
                if deleted:
                    print(f"[Watchdog] Detected {len(deleted)} deleted file(s): {', '.join(deleted)}")
                if modified:
                    print(f"[Watchdog] Detected {len(modified)} modified file(s): {', '.join(modified)}")

                print("[Watchdog] Synchronizing database...")
                # Allow disk I/O to flush completely
                time.sleep(1.0)

                if not os.path.exists(build_exe):
                    print(f"[Watchdog] ERROR: Engine executable not found at '{build_exe}'.")
                    print("           Please build the project first using build.bat or build.sh.")
                else:
                    try:
                        # Run Build engine in the project root context
                        proc = subprocess.run([build_exe], cwd=PROJECT_ROOT, check=False)
                        if proc.returncode == 0:
                            print("[Watchdog] Database sync completed successfully.\n")
                        else:
                            print(f"[Watchdog] Engine exited with return code {proc.returncode}.\n")
                    except Exception as e:
                        print(f"[Watchdog] Failed to execute engine: {e}\n")

                current_snapshot = new_snapshot

        except KeyboardInterrupt:
            print("\n[*] Watchdog stopped gracefully.")
            break


if __name__ == "__main__":
    main()
