# agent/utils_llama.py
from __future__ import annotations
import csv, time
from pathlib import Path

# separate directory for llama tokens
TOK_DIR = Path("tokens_count_llama")
USAGE_CSV = TOK_DIR / "usage_log.csv"
TOTAL_FILE = TOK_DIR / "total_tokens.txt"

def _ensure_files() -> None:
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    if not USAGE_CSV.exists():
        with USAGE_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","kind","model","tag","prompt_tokens","completion_tokens","total_tokens"])
    if not TOTAL_FILE.exists():
        TOTAL_FILE.write_text("0", encoding="utf-8")

def _safe_int(x) -> int:
    try:
        return int(x or 0)
    except Exception:
        return 0

def append_usage_llama(kind: str, model: str, tag: str,
                       prompt_tokens: int|None,
                       completion_tokens: int|None,
                       total_tokens: int|None) -> None:
    _ensure_files()
    pt = _safe_int(prompt_tokens)
    ct = _safe_int(completion_tokens)
    tt = _safe_int(total_tokens if total_tokens is not None else (pt + ct))

    with USAGE_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), kind, model, tag, pt, ct, tt])

    cur = _safe_int(TOTAL_FILE.read_text(encoding="utf-8").strip())
    TOTAL_FILE.write_text(str(cur + tt), encoding="utf-8")