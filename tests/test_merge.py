"""Merge/labelling rules and graceful degradation — offline, no graph or API."""

from retrieval.entity_linker import Entity, LinkResult
from retrieval.graph_retriever import Fact, GraphResult, focus
from retrieval.merge import merge
from retrieval.vector_retriever import Chunk, VectorResult


def _link(*drug_ids):
    return LinkResult(drugs=[Entity(d, "drug", d, d, 0, len(d)) for d in drug_ids])


def _fact(**kw):
    base = dict(subject="warfarin", predicate="INTERACTS_WITH", object="aspirin", object_type="Drug",
                source_label_id="LBL1", source_field="drug_interactions", evidence="Aspirin increases bleeding risk.")
    return Fact(**{**base, **kw})


def _chunk(text="Warfarin is an anticoagulant."):
    return Chunk("warfarin/description/0", "warfarin", "warfarin", "description", text, "LBL1", "SET1", 0.9)


def test_graph_facts_are_labelled_source_of_truth_and_cited():
    ctx = merge(_link("warfarin", "aspirin"), GraphResult(facts=[_fact()]), VectorResult(chunks=[_chunk()]))
    assert "GRAPH FACTS (source of truth" in ctx.text
    assert "[F1] warfarin interacts with aspirin" in ctx.text
    assert "FDA label LBL1" in ctx.text
    assert "BACKGROUND TEXT (context and phrasing only" in ctx.text
    assert "[C1] warfarin / description" in ctx.text
    assert ctx.citations()["F1"]["label_id"] == "LBL1"
    assert ctx.has_facts


def test_no_interaction_is_stated_explicitly():
    g = GraphResult(no_interaction_pairs=[("levothyroxine", "lisinopril")],
                    drugs={"levothyroxine": {"name": "levothyroxine"}, "lisinopril": {"name": "lisinopril"}})
    ctx = merge(_link("levothyroxine", "lisinopril"), g, VectorResult())
    assert "GRAPH SAYS: no documented interaction between levothyroxine and lisinopril" in ctx.text
    assert ctx.has_facts  # an explicit "none" is a graph fact, not an empty result


def test_graph_down_degrades_to_vectors_only():
    ctx = merge(_link("warfarin"), GraphResult(error="ServiceUnavailable: x"), VectorResult(chunks=[_chunk()]))
    assert ctx.degraded == ["graph unavailable: ServiceUnavailable: x"]
    assert "(graph unavailable" in ctx.text
    assert not ctx.has_facts
    assert "[C1]" in ctx.text


def test_vectors_down_degrades_to_graph_only():
    ctx = merge(_link("warfarin", "aspirin"), GraphResult(facts=[_fact()]), VectorResult(error="boom"))
    assert ctx.degraded == ["vector index unavailable: boom"]
    assert "[F1]" in ctx.text and "(vector index unavailable)" in ctx.text


def test_unlinked_question_is_marked():
    ctx = merge(LinkResult(), GraphResult(), VectorResult())
    assert "no known drug names found" in ctx.text
    assert not ctx.has_facts


def test_focus_selection():
    assert focus("Does warfarin interact with aspirin?") == {"INTERACTS_WITH"}
    assert focus("What are the contraindications for lisinopril?") == {"CONTRAINDICATED_FOR"}
    assert "CAUSES_SIDE_EFFECT" in focus("What side effects are reported for amiodarone?")
    assert focus("Give me an overview of what kind of drug amiodarone is.") >= {"TREATS", "BELONGS_TO_CLASS"}
    assert focus("Tell me about warfarin.") == {"TREATS", "BELONGS_TO_CLASS"}  # descriptive default
