"""Per-query observability (DESIGN.md section 7).

Every query that goes through `guardrails.rails.guarded_answer` is appended as
one JSON line to $LOG_DIR/queries.jsonl with: the question *as the pipeline saw
it* (PII already masked by the input rail — raw user text is never logged),
linked entities, graph/vector hit counts, per-stage latency, grounding-gate
outcome, which rail blocked (if any), degradation notes, token usage and an
estimated cost. eval/report.py turns the log into p50/p95 latency and
cost-per-query numbers.

Prices are USD per 1M tokens and are only an estimate; override with the
PRICE_TABLE_JSON env var (e.g. '{"gpt-4o-mini": [0.15, 0.6]}') when they change.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

# model -> (input $/1M, output $/1M). Verify against current OpenAI pricing.
DEFAULT_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
}
_lock = threading.Lock()


def prices() -> dict[str, tuple[float, float]]:
    table = dict(DEFAULT_PRICES)
    override = os.environ.get("PRICE_TABLE_JSON")
    if override:
        try:
            table.update({k: tuple(v) for k, v in json.loads(override).items()})
        except (ValueError, TypeError):
            pass
    return table


def estimate_cost(usage: list[dict], question_chars: int = 0) -> float:
    """USD for the LLM calls in `usage` plus the query embedding (estimated at 4 chars/token)."""
    table = prices()
    total = 0.0
    for u in usage:
        pin, pout = table.get(u.get("model", ""), (0.0, 0.0))
        total += u.get("input_tokens", 0) / 1e6 * pin + u.get("output_tokens", 0) / 1e6 * pout
    pin, _ = table.get(settings.EMBEDDING_MODEL, (0.0, 0.0))
    total += (question_chars / 4) / 1e6 * pin
    return round(total, 6)


def log_path() -> Path:
    return settings.LOG_DIR / "queries.jsonl"


def record(resp, extra: dict | None = None) -> dict:
    """Build the log record from a pipeline/guarded Response and append it. Returns the record."""
    stats = resp.stats or {}
    usage = stats.get("usage", [])
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "question": resp.question,                       # masked by the input rail before it got here
        "status": resp.status,
        "blocked_by": stats.get("blocked_by"),
        "pii_masked": stats.get("pii_masked", False),
        "linked_drugs": stats.get("linked_drugs", []),
        "linked_conditions": stats.get("linked_conditions", []),
        "focus": stats.get("focus", []),
        "graph_facts": stats.get("graph_facts", 0),
        "no_interaction_pairs": stats.get("no_interaction_pairs", 0),
        "vector_chunks": stats.get("vector_chunks", 0),
        "degraded": stats.get("degraded", []),
        "llm_error": stats.get("llm_error"),
        "gate": {
            "status": stats.get("gate_status"),
            "passed": stats.get("gate_passed"),
            "kept": stats.get("claims_kept"),
            "dropped": stats.get("claims_dropped"),
            "retries": stats.get("gate_retries"),
        },
        "latency_ms": {
            k: stats.get(k) for k in ("input_rails_ms", "link_ms", "graph_ms", "vector_ms", "generate_ms", "output_rails_ms", "total_ms")
        },
        "tokens": {
            "input": sum(u.get("input_tokens", 0) for u in usage),
            "output": sum(u.get("output_tokens", 0) for u in usage),
            "llm_calls": len(usage),
        },
        "model": settings.LLM_MODEL,
        "cost_usd": estimate_cost(usage, len(resp.question or "")),
    }
    if extra:
        rec.update(extra)
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_log(path: Path | None = None) -> list[dict]:
    p = path or log_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
