"""Cost & latency report from the per-query log (DESIGN.md section 7).

Reads $LOG_DIR/queries.jsonl (written by observability/logger.py on every
guarded query, including every Stage 2 eval run) and prints p50/p95 latency
per stage, cost per query, status mix, rail blocks, gate drop rate and
degradation events. Optionally writes a markdown snapshot.

Usage:
    python eval/report.py                 # print
    python eval/report.py --md eval/results/ops_latest.md
    python eval/report.py --last 200      # only the most recent N queries
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from observability.logger import log_path, read_log  # noqa: E402

STAGES = ("input_rails_ms", "link_ms", "graph_ms", "vector_ms", "generate_ms", "output_rails_ms", "total_ms")


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    idx = min(len(v) - 1, max(0, round(p / 100 * len(v) + 0.5) - 1))
    return v[idx]


def build(records: list[dict]) -> dict:
    n = len(records)
    answered = [r for r in records if r.get("status") not in ("refuse",) and not r.get("blocked_by")]
    lat = {s: [r["latency_ms"].get(s) for r in records if r.get("latency_ms", {}).get(s) is not None] for s in STAGES}
    lat_answered = [r["latency_ms"]["total_ms"] for r in answered if r.get("latency_ms", {}).get("total_ms") is not None]
    costs = [r.get("cost_usd", 0.0) for r in records]
    gate_kept = sum(r["gate"].get("kept") or 0 for r in answered)
    gate_dropped = sum(r["gate"].get("dropped") or 0 for r in answered)
    return {
        "queries": n,
        "status_mix": dict(Counter(r.get("status") for r in records)),
        "blocked_by": dict(Counter(r.get("blocked_by") for r in records if r.get("blocked_by"))),
        "pii_masked": sum(1 for r in records if r.get("pii_masked")),
        "latency_ms": {s: {"p50": pct(v, 50), "p95": pct(v, 95), "n": len(v)} for s, v in lat.items() if v},
        "latency_answered_total_ms": {"p50": pct(lat_answered, 50), "p95": pct(lat_answered, 95)},
        "cost_usd": {"total": round(sum(costs), 4), "per_query": round(sum(costs) / n, 6) if n else None,
                     "per_answered_query": round(sum(r.get("cost_usd", 0) for r in answered) / len(answered), 6) if answered else None},
        "tokens_per_answered_query": {
            "input": round(sum(r["tokens"]["input"] for r in answered) / len(answered)) if answered else None,
            "output": round(sum(r["tokens"]["output"] for r in answered) / len(answered)) if answered else None,
        },
        "gate": {"claims_kept": gate_kept, "claims_dropped": gate_dropped,
                 "drop_rate": round(gate_dropped / (gate_kept + gate_dropped), 3) if gate_kept + gate_dropped else None,
                 "retries": sum(r["gate"].get("retries") or 0 for r in answered),
                 "failed_closed": sum(1 for r in answered if r["gate"].get("passed") is False)},
        "degraded_events": sum(1 for r in records if r.get("degraded") or r.get("llm_error")),
    }


def to_markdown(rep: dict, source: Path) -> str:
    L = [f"# Operational report — {rep['queries']} queries from `{source.name}`", ""]
    L += [f"- status mix: {rep['status_mix']}", f"- blocked by rails: {rep['blocked_by']}", f"- PII masked on input: {rep['pii_masked']}",
          f"- degraded events (graph/vector/LLM down): {rep['degraded_events']}", ""]
    L += ["## Latency (ms)", "", "| stage | p50 | p95 | n |", "|---|---|---|---|"]
    for s, v in rep["latency_ms"].items():
        L.append(f"| {s} | {v['p50']} | {v['p95']} | {v['n']} |")
    la = rep["latency_answered_total_ms"]
    L += ["", f"Answered questions end-to-end: **p50 {la['p50']} ms, p95 {la['p95']} ms**", ""]
    c, t = rep["cost_usd"], rep["tokens_per_answered_query"]
    L += ["## Cost", "", f"- total: ${c['total']}", f"- per query: ${c['per_query']}", f"- per answered query: ${c['per_answered_query']}",
          f"- tokens per answered query: {t['input']} in / {t['output']} out", ""]
    g = rep["gate"]
    L += ["## Grounding gate", "", f"- claims kept / dropped: {g['claims_kept']} / {g['claims_dropped']} (drop rate {g['drop_rate']})",
          f"- retries: {g['retries']}", f"- failed closed (answer -> not found): {g['failed_closed']}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--md", type=Path, default=None, help="also write a markdown snapshot here")
    args = ap.parse_args()
    src = args.log or log_path()
    records = read_log(src)
    if args.last:
        records = records[-args.last:]
    if not records:
        print(f"no records in {src}")
        return 1
    rep = build(records)
    md = to_markdown(rep, src)
    print(md)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
        print(f"-> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
