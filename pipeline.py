"""End-to-end query pipeline: link -> retrieve both -> merge -> generate -> gate -> cite.

    from pipeline import answer
    resp = answer("Does warfarin interact with aspirin?")
    print(resp.text)

Graceful degradation:
  graph down   -> vectors-only context; gate then fails closed to not_found
                  (no facts to ground on) but the user sees why.
  vectors down -> graph-only context; answers normally.
  LLM down     -> raw graph facts with citations, no prose (status="degraded").

Guardrails (NeMo) wrap this in the next step; observability logging in the
production step. Both hook `answer()` without changing it.

Usage:
    python pipeline.py "Does warfarin interact with aspirin?"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generation import citations, generate, grounding_gate  # noqa: E402
from retrieval.merge import MergedContext  # noqa: E402
from retrieval.retrieve import retrieve  # noqa: E402

MAX_GATE_RETRIES = 1


@dataclass
class Response:
    question: str
    status: str                 # answer | not_found | refuse | degraded
    text: str
    context: MergedContext | None = None
    claims: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _built_on() -> str | None:
    try:
        from graph import queries
        info = queries.build_info()
        return info["built_at"] if info else None
    except Exception:
        return None


def answer(question: str) -> Response:
    t0 = time.perf_counter()
    ctx, stats = retrieve(question)
    built_on = _built_on()
    stats["gate_retries"] = 0
    stats["usage"] = []

    problems = ""
    report = None
    summary = ""
    best: tuple | None = None  # (report, summary) of the attempt with the most verified claims
    for attempt in range(MAX_GATE_RETRIES + 1):
        t = time.perf_counter()
        try:
            parsed, usage = generate.generate(ctx.text, question, problems)
        except Exception as exc:  # LLM unavailable -> facts only
            stats["llm_error"] = f"{type(exc).__name__}: {exc}"
            stats["total_ms"] = round((time.perf_counter() - t0) * 1000)
            return Response(question, "degraded", citations.render_facts_only(ctx, built_on), ctx, stats=stats)
        stats["generate_ms"] = round((time.perf_counter() - t) * 1000)
        stats["usage"].append(usage)
        summary = parsed.summary
        report = grounding_gate.check(parsed, ctx)
        if best is None or len(report.kept) > len(best[0].kept):
            best = (report, summary)
        if attempt == MAX_GATE_RETRIES:
            break
        if parsed.status == "not_found" and citations.NO_INTERACTION_WORDING.search(summary):
            # recoverable wording miss: "not in my data" was phrased as "no interaction"
            problems = ("the not_found explanation was worded as if no interaction exists; say instead that the "
                        "unrecognised drug/product is not among the FDA labels held, naming it")
        elif parsed.status == "answer" and report.dropped:
            # recoverable grounding miss: tell the model what failed and let it try once more
            problems = report.problems
        else:
            break
        stats["gate_retries"] += 1

    # a retry that verified fewer claims than the first pass is not an improvement
    if best is not None:
        report, summary = best
    stats["gate_status"] = report.status
    stats["gate_passed"] = report.passed
    stats["claims_kept"] = len(report.kept)
    stats["claims_dropped"] = len(report.dropped)
    stats["total_ms"] = round((time.perf_counter() - t0) * 1000)
    text = citations.render(summary, report, ctx, built_on)
    return Response(question, report.status, text, ctx, report.kept, report.dropped, stats)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "I take warfarin and need something for pain. What should I avoid?"
    r = answer(q)
    print(f"Q: {q}\n\n{r.text}\n")
    print("--- status:", r.status, "| stats:", {k: v for k, v in r.stats.items() if k not in ("usage",)},
          "| tokens:", [ (u.get("input_tokens"), u.get("output_tokens")) for u in r.stats.get("usage", [])])
    if r.dropped:
        print("--- dropped by gate:", [(c.statement[:70], why) for c, why in r.dropped])
