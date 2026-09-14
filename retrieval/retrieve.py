"""Hybrid retrieval entry point: link -> run BOTH retrievers -> merge.

No router (DESIGN.md section 5): both retrievers always run. Either side may
fail independently; the merged context records what degraded so the answer can
say so instead of crashing.

Usage:
    python retrieval/retrieve.py "Does warfarin interact with aspirin?"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import entity_linker, graph_retriever, vector_retriever  # noqa: E402
from retrieval.merge import MergedContext, merge  # noqa: E402


def retrieve(question: str) -> tuple[MergedContext, dict]:
    """Return the merged context and per-stage timings/counts for observability."""
    timings: dict = {}
    t = time.perf_counter()
    link = entity_linker.link(question)
    timings["link_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    graph = graph_retriever.retrieve(link, question)
    timings["graph_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    vector = vector_retriever.retrieve(question, link.drug_ids)
    timings["vector_ms"] = round((time.perf_counter() - t) * 1000)

    ctx = merge(link, graph, vector)
    stats = {
        **timings,
        "linked_drugs": link.drug_ids,
        "linked_classes": [e.id for e in link.classes],
        "linked_conditions": link.condition_keywords,
        "focus": sorted(graph.focus),
        "graph_facts": len(ctx.facts),
        "no_interaction_pairs": len(graph.no_interaction_pairs),
        "vector_chunks": len(ctx.chunks),
        "degraded": ctx.degraded,
    }
    return ctx, stats


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "I take warfarin and need something for pain. What should I avoid?"
    ctx, stats = retrieve(q)
    print(f"Q: {q}\n")
    print(ctx.text)
    print("\n--- stats:", stats)
