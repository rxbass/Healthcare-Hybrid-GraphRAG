"""Build phase orchestrator: fetch -> parse -> extract -> load graph -> embed.

Runs the offline pipeline end to end. Each stage is its own script and is
idempotent, so this just sequences them and stops at the first failure.

Usage:
    python ingestion/build_index.py                 # full build, merge into graph
    python ingestion/build_index.py --reset         # wipe Neo4j first (scheduled refresh)
    python ingestion/build_index.py --skip-fetch    # reuse data/raw/
    python ingestion/build_index.py --skip-extract  # reuse data/processed/triplets.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = [
    ("fetch", "ingestion/fetch_openfda.py"),
    ("parse", "ingestion/parse_labels.py"),
    ("extract", "ingestion/extract_triplets.py"),
    ("load", "ingestion/load_graph.py"),
    ("embed", "ingestion/build_vectors.py"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="wipe the graph before loading")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for name, script in STAGES:
        if (name == "fetch" and args.skip_fetch) or (name == "extract" and args.skip_extract):
            print(f"== {name}: skipped")
            continue
        cmd = [sys.executable, str(ROOT / script)]
        if name == "load" and args.reset:
            cmd.append("--reset")
        if name == "embed" and args.reset:
            cmd.append("--force")
        print(f"== {name}: {' '.join(cmd[1:])}")
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"stage '{name}' failed with exit code {rc}", file=sys.stderr)
            return rc
    print(f"\nbuild complete in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
