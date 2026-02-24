#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import requests

LIST_URL = "http://molgpu02.mit.edu:8999/list"
CORPUS_DIR = Path("/nfs/ccoleylab001/bmahjour/corpus/documents")
BASE_DIR = Path("integrated_20260206")
PAPER_IDS_CACHE = BASE_DIR / "paper_ids.json"
FAILED_OUT = BASE_DIR / "failed_paper_ids.jsonl"
CONSOLIDATED_JSONL = BASE_DIR / "results.jsonl"
MAX_RETRIES = 2
SLEEP_BETWEEN = 1  # seconds
OUTPUT_DIR = BASE_DIR

# Thread-safe lock for writing to the shared failed_paper_ids file
_fail_lock = threading.Lock()


def fetch_paper_ids() -> list[str]:
    """Fetch paper IDs from server, caching locally to avoid re-fetching."""
    if PAPER_IDS_CACHE.exists():
        with PAPER_IDS_CACHE.open("r") as f:
            ids = json.load(f)
        print(f"Loaded {len(ids)} paper IDs from cache ({PAPER_IDS_CACHE})")
        return ids

    print(f"Fetching paper list from {LIST_URL} ...")
    resp = requests.get(LIST_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("Expected list of paper_ids from list endpoint.")
    ids = [str(x) for x in data]
    with PAPER_IDS_CACHE.open("w") as f:
        json.dump(ids, f)
    print(f"Fetched and cached {len(ids)} paper IDs to {PAPER_IDS_CACHE}")
    return ids


def load_failed_ids() -> set[str]:
    if not FAILED_OUT.exists():
        return set()
    failed: set[str] = set()
    with FAILED_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("paper_id")
                if pid is not None:
                    failed.add(str(pid))
            except json.JSONDecodeError:
                continue
    return failed


def record_failure(paper_id: str) -> None:
    """Thread-safe append to the failed paper IDs file."""
    with _fail_lock:
        FAILED_OUT.parent.mkdir(parents=True, exist_ok=True)
        with FAILED_OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"paper_id": paper_id}) + "\n")


def run_one(paper_id: str, model: str, base_url: str) -> bool:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/gpt5_paper_classifier.py",
        paper_id,
        "--output",
        str(OUTPUT_DIR / f"{paper_id}_classification.json"),
        "--model",
        model,
        "--base_url",
        base_url,
        "--local-dir",
        str(CORPUS_DIR / paper_id),
    ]
    try:
        completed = subprocess.run(cmd, check=True)
        return completed.returncode == 0
    except subprocess.CalledProcessError:
        return False


def process_paper(paper_id: str, model: str, base_url: str) -> tuple[str, bool]:
    """Process a single paper with retries. Returns (paper_id, success)."""
    for attempt in range(1, MAX_RETRIES + 1):
        if run_one(paper_id, model=model, base_url=base_url):
            return paper_id, True
        time.sleep(SLEEP_BETWEEN)
    record_failure(paper_id)
    return paper_id, False


def fmt_seconds(s: Optional[float]) -> str:
    if s is None:
        return "unknown"
    s = max(0, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classification on all papers")
    parser.add_argument(
        "--model", type=str, default="openai/gpt-oss-120b",
        help="Model name to pass to the classifier (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:9001/v1",
        help="vLLM API base URL (default: http://localhost:9001/v1)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Number of concurrent papers to process (default: 8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        paper_ids = fetch_paper_ids()
    except Exception as e:
        print(f"Failed to fetch paper list: {e}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed_ids = load_failed_ids()

    total = len(paper_ids)
    if total == 0:
        print("No paper_ids returned.")
        return 0

    # Filter to only papers that need processing
    to_process: list[str] = []
    skipped = 0
    for pid in paper_ids:
        if pid in failed_ids:
            skipped += 1
            continue
        output_path = OUTPUT_DIR / f"{pid}_classification.json"
        if output_path.exists():
            try:
                json.loads(output_path.read_text(encoding="utf-8"))
                skipped += 1
                continue
            except (json.JSONDecodeError, OSError):
                output_path.unlink(missing_ok=True)
        to_process.append(pid)

    print(f"Model: {args.model}  |  Base URL: {args.base_url}  |  Workers: {args.max_workers}")
    print(f"Total: {total}  |  Skipped: {skipped}  |  To process: {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        consolidate_results()
        return 0

    failures: list[str] = []
    start_time = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_paper, pid, args.model, args.base_url): pid
            for pid in to_process
        }
        for future in as_completed(futures):
            pid, ok = future.result()
            done += 1
            elapsed = time.time() - start_time
            eta = (elapsed / done) * (len(to_process) - done)
            status = "OK" if ok else "FAIL"
            print(
                f"[{done}/{len(to_process)}] {pid} {status} "
                f"| elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}"
            )
            if not ok:
                failures.append(pid)

    if failures:
        print(f"Failed {len(failures)} paper_ids. See {FAILED_OUT}")

    # Consolidate all individual JSONs into a single JSONL
    consolidate_results()

    return 2 if failures else 0


def consolidate_results() -> None:
    """Merge all individual classification JSONs into a single results.jsonl."""
    json_files = sorted(OUTPUT_DIR.glob("*_classification.json"))
    if not json_files:
        print("No classification JSONs found to consolidate.")
        return
    with CONSOLIDATED_JSONL.open("w", encoding="utf-8") as out:
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                paper_id = jf.stem.replace("_classification", "")
                data["paper_id"] = paper_id
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read {jf.name}: {e}")
    print(f"Consolidated {len(json_files)} results → {CONSOLIDATED_JSONL}")


if __name__ == "__main__":
    raise SystemExit(main())
