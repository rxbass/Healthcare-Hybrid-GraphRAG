"""Merge both retrievers into one labelled context block (DESIGN.md section 5).

The block has two clearly labelled parts:

  GRAPH FACTS (source of truth)      numbered [F1]..[Fn], each with label id
  BACKGROUND TEXT (context only)     numbered [C1]..[Cn], each with label id

plus explicit "[N#] GRAPH SAYS: no documented interaction between A and B"
lines. The generation prompt cites [F#]/[N#]/[C#] ids, and the grounding gate
checks that every factual claim maps to an [F#] or [N#]. The labelling *is* the division of labour:
the model is told which part may establish a fact and which may only phrase it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.entity_linker import LinkResult  # noqa: E402
from retrieval.graph_retriever import Fact, GraphResult  # noqa: E402
from retrieval.vector_retriever import Chunk, VectorResult  # noqa: E402

MAX_CHUNK_CHARS = 900

PREDICATE_TEXT = {
    "INTERACTS_WITH": "interacts with",
    "TREATS": "is indicated for",
    "CONTRAINDICATED_FOR": "is contraindicated in",
    "CAUSES_SIDE_EFFECT": "has reported adverse reaction",
    "BELONGS_TO_CLASS": "belongs to class",
    "HAS_INGREDIENT": "contains",
}


@dataclass
class MergedContext:
    link: LinkResult
    graph: GraphResult
    vector: VectorResult
    facts: list[Fact] = field(default_factory=list)     # [F1].. in order
    nones: list[tuple[str, str]] = field(default_factory=list)  # [N1].. drug-name pairs with no interaction
    chunks: list[Chunk] = field(default_factory=list)   # [C1].. in order
    text: str = ""
    degraded: list[str] = field(default_factory=list)   # which side failed, if any

    @property
    def has_facts(self) -> bool:
        return bool(self.facts) or bool(self.graph.no_interaction_pairs)

    def citations(self) -> dict[str, dict]:
        """Citation id -> {label_id, set_id, drug, field, quote}."""
        out = {}
        for i, f in enumerate(self.facts, 1):
            out[f"F{i}"] = {"label_id": f.source_label_id, "set_id": f.set_id, "drug": f.subject, "field": f.source_field,
                            "label_drug": (f.source_drug or f.subject).replace("_", " "),
                            "quote": f.evidence, "predicate": f.predicate, "object": f.object}
        for i, (a, b) in enumerate(self.nones, 1):
            out[f"N{i}"] = {"label_id": None, "set_id": None, "drug": a, "field": "graph",
                            "quote": f"no documented interaction between {a} and {b} in the FDA labels held",
                            "predicate": "NO_INTERACTION", "object": b}
        for i, c in enumerate(self.chunks, 1):
            out[f"C{i}"] = {"label_id": c.label_id, "set_id": c.set_id, "drug": c.drug_name, "field": c.field, "quote": c.text[:200]}
        return out


def _fact_line(i: int, f: Fact) -> str:
    verb = PREDICATE_TEXT.get(f.predicate, f.predicate.lower())
    via = "" if f.via == "direct" else f" (via {f.via})"
    return (f"[F{i}] {f.subject} {verb} {f.object}{via}\n"
            f"      evidence: \"{f.evidence}\"  (FDA label {f.source_label_id}, section {f.source_field})")


def merge(link: LinkResult, graph: GraphResult, vector: VectorResult) -> MergedContext:
    ctx = MergedContext(link=link, graph=graph, vector=vector)
    if not graph.ok:
        ctx.degraded.append(f"graph unavailable: {graph.error}")
    if not vector.ok:
        ctx.degraded.append(f"vector index unavailable: {vector.error}")

    ctx.facts = list(graph.facts)
    ctx.chunks = list(vector.chunks)

    lines: list[str] = []
    lines.append("=== LINKED ENTITIES ===")
    if link.drugs:
        for e in link.drugs:
            status = "FDA label in corpus" if e.in_corpus else "mentioned in another label only; no label of its own in corpus"
            lines.append(f"- drug: {e.name} ({status})")
    else:
        lines.append("- no known drug names found in the question")
    for e in link.classes:
        lines.append(f"- drug class: {e.name}")
    for e in link.conditions:
        lines.append(f"- condition keyword: {e.name}")

    lines.append("\n=== GRAPH FACTS (source of truth — the ONLY basis for factual claims) ===")
    if not graph.ok:
        lines.append("(graph unavailable — no facts could be retrieved)")
    else:
        for a, b in graph.no_interaction_pairs:
            na = graph.drugs.get(a, {}).get("name", a)
            nb = graph.drugs.get(b, {}).get("name", b)
            ctx.nones.append((na, nb))
            lines.append(f"[N{len(ctx.nones)}] GRAPH SAYS: no documented interaction between {na} and {nb} in the FDA labels held.")
        if ctx.facts:
            for i, f in enumerate(ctx.facts, 1):
                lines.append(_fact_line(i, f))
        elif not graph.no_interaction_pairs:
            lines.append("(no graph facts found for this question)")

    lines.append("\n=== BACKGROUND TEXT (context and phrasing only — may NOT establish or override a fact) ===")
    if not vector.ok:
        lines.append("(vector index unavailable)")
    elif not ctx.chunks:
        lines.append("(no background text found)")
    else:
        for i, c in enumerate(ctx.chunks, 1):
            text = c.text if len(c.text) <= MAX_CHUNK_CHARS else c.text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + " …"
            lines.append(f"[C{i}] {c.drug_name} / {c.field} (FDA label {c.label_id}):\n      \"{text}\"")

    ctx.text = "\n".join(lines)
    return ctx
