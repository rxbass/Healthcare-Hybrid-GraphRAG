"""Scheduled data refresh (DESIGN.md section 7): rebuild everything, then gate.

    fetch openFDA -> parse -> extract (cache keyed by label id, so only changed
    labels cost LLM calls) -> load graph (--reset) -> embed -> Stage 1 eval

If Stage 1 fails on the rebuilt graph the script exits non-zero so a scheduler
(GitHub Actions, cron) treats the refresh as failed. The BuildInfo stamp in the
graph is what the app shows as "graph built on {date}".

Usage:
    python scripts/refresh.py            # full refresh + gate
    python scripts/refresh.py --no-gate  # rebuild only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402


def run(cmd: list[str]) -> int:
    print(f"== {' '.join(cmd)}")
    return subprocess.call([sys.executable, *cmd], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rc = run(["ingestion/build_index.py", "--reset"])
    if rc != 0:
        print("refresh FAILED at build", file=sys.stderr)
        return rc
    gate = None
    if not args.no_gate:
        gate = run(["eval/run_eval.py", "--stage", "1", "--min-pass", "1.0"])

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
              "build_ok": True, "stage1_rc": gate}
    with (settings.LOG_DIR / "refresh.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"refresh {'OK' if not gate else 'FAILED GATE'} in {record['seconds']}s")
    return gate or 0


if __name__ == "__main__":
    raise SystemExit(main())
