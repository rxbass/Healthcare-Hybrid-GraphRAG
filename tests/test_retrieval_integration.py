"""Hybrid retrieval against the live graph + vector index. Skipped without credentials."""

import os

import pytest

from config import settings  # noqa: F401  (loads .env so the skip check sees the credentials)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("NEO4J_URI") and os.environ.get("OPENAI_API_KEY")),
    reason="needs NEO4J_* and OPENAI_API_KEY (integration)",
)


@pytest.fixture(scope="module")
def retrieve():
    from retrieval.retrieve import retrieve
    return retrieve


def test_demo_question_multi_hop(retrieve):
    ctx, stats = retrieve("I take warfarin and need something for pain. What should I avoid?")
    avoid = {f.object for f in ctx.facts if f.predicate == "INTERACTS_WITH"}
    assert {"aspirin", "ibuprofen", "naproxen"} <= avoid
    assert stats["linked_conditions"] == ["pain"]
    assert not ctx.degraded


def test_pair_no_interaction_is_explicit(retrieve):
    ctx, _ = retrieve("Does levothyroxine interact with lisinopril?")
    assert ("levothyroxine", "lisinopril") in ctx.graph.no_interaction_pairs
    assert "GRAPH SAYS: no documented interaction" in ctx.text
    assert not any(f.predicate == "INTERACTS_WITH" for f in ctx.facts)


def test_pair_interaction_cited_from_label(retrieve):
    ctx, _ = retrieve("Does warfarin interact with aspirin?")
    pair = [f for f in ctx.facts if f.predicate == "INTERACTS_WITH" and {f.subject, f.object} == {"warfarin", "aspirin"}]
    assert pair and all(f.source_label_id and f.evidence for f in pair)


def test_both_retrievers_always_run(retrieve):
    ctx, stats = retrieve("What is warfarin generally used for, in plain language?")
    assert stats["graph_facts"] > 0 and stats["vector_chunks"] > 0
    assert all(c.drug_id == "warfarin" for c in ctx.chunks[:3])
