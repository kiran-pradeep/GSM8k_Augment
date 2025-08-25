# io_utils.py
"""
Utility functions for file I/O.

- ensure_dir: create directory if missing
- dump_json: save a dict as pretty JSON
- append_jsonl: append a dict as one line of JSON to a .jsonl file
- log_error: log error details to a JSONL file (safe for multiprocessing)
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict

from filelock import FileLock


def ensure_dir(path: Path):
    """
    Ensure that a directory exists (mkdir -p).
    Accepts either a Path or a file path (will create its parent).
    """
    if path.suffix:  # looks like a file (has extension)
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj: Dict[str, Any]):
    """
    Save an object as JSON with pretty formatting.
    """
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: Dict[str, Any]):
    """Safely append one JSON object per line with a file lock."""
    ensure_dir(path.parent)
    lock_path = str(path) + ".lock"
    with FileLock(lock_path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_error(outdir: Path, idx: int, error: str, tb: str):
    """Log error details into out/errors.jsonl (safe for multiprocessing)."""
    ensure_dir(outdir)
    log_path = outdir / "errors.jsonl"
    record = {
        "time": datetime.now().isoformat(),
        "index": idx,
        "error": error,
        "traceback": tb
    }
    append_jsonl(log_path, record)
