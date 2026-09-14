"""The custom grounding gate must fail closed. Offline: hand-built context, no LLM."""

from generation.citations import render
from generation.generate import Answer, Claim
from generation.grounding_gate import check
from retrieval.entity_linker import Entity, LinkResult
from retrieval.graph_retriever import Fact, GraphResult
from retrieval.merge import merge
from retrieval.vector_retriever import Chunk, VectorResult


def _ctx(facts=(), none_pairs=(), chunks=()):
    drugs = {}
    for f in facts:
        drugs[f.subject] = {"name": f.subject}
    for a, b in none_pairs:
        drugs[a] = {"name": a}
        drugs[b] = {"name": b}
    link = LinkResult(drugs=[Entity(d, "drug", d, d, 0, 1) for d in drugs])
    g = GraphResult(facts=list(facts), no_interaction_pairs=list(none_pairs), drugs=drugs)
    v = VectorResult(chunks=list(chunks))
    return merge(link, g, v)


F_ASPIRIN = Fact("warfarin", "INTERACTS_WITH", "aspirin", "Drug", "LBL-W", "drug_interactions",
                 "Examples of drugs known to increase the risk of bleeding are presented in Table 3.", set_id="SET-W")
F_PREG = Fact("warfarin", "CONTRAINDICATED_FOR", "pregnancy", "Condition", "LBL-W", "contraindications",
              "Warfarin sodium is contraindicated in women who are pregnant.", set_id="SET-W")
C_DESC = Chunk("warfarin/description/0", "warfarin", "warfarin", "description",
               "Warfarin is an anticoagulant. Ibuprofen and warfarin together raise bleeding risk.", "LBL-W", "SET-W", 0.9)


def _answer(*claims, status="answer", summary="s"):
    return Answer(status=status, summary=summary, claims=list(claims))


def test_supported_claim_passes():
    ctx = _ctx([F_ASPIRIN])
    r = check(_answer(Claim(statement="Warfarin interacts with aspirin, raising bleeding risk.", fact_ids=["F1"])), ctx)
    assert r.passed and r.status == "answer" and len(r.kept) == 1 and not r.dropped


def test_hallucinated_citation_is_dropped_and_fails_closed():
    ctx = _ctx([F_ASPIRIN])
    r = check(_answer(Claim(statement="Warfarin interacts with aspirin.", fact_ids=["F7"])), ctx)
    assert not r.passed and r.status == "not_found" and r.dropped[0][1].startswith("unknown citation")


def test_background_only_cannot_establish_a_fact():
    ctx = _ctx([F_ASPIRIN], chunks=[C_DESC])
    r = check(_answer(
        Claim(statement="Warfarin interacts with aspirin.", fact_ids=["F1"]),
        Claim(statement="Ibuprofen and warfarin together raise bleeding risk.", context_ids=["C1"]),  # vector-only fact
        Claim(statement="Warfarin is an anticoagulant.", context_ids=["C1"]),                         # descriptive, ok
    ), ctx)
    assert [c.statement for c in r.kept] == ["Warfarin interacts with aspirin."]
    assert len(r.background) == 1 and r.background[0].statement == "Warfarin is an anticoagulant."
    assert r.dropped and "background text" in r.dropped[0][1]


def test_vector_text_cannot_override_graph_says_none():
    ctx = _ctx(none_pairs=[("levothyroxine", "lisinopril")], chunks=[C_DESC])
    r = check(_answer(
        Claim(statement="Levothyroxine and lisinopril may interact according to the label text.", context_ids=["C1"]),
        Claim(statement="There is no documented interaction between levothyroxine and lisinopril.", fact_ids=["N1"]),
    ), ctx)
    assert len(r.kept) == 1 and r.kept[0].fact_ids == ["N1"]
    assert r.dropped and "graph says is not documented" in r.dropped[0][1]


def test_citing_an_unrelated_fact_does_not_launder_a_claim():
    ctx = _ctx([F_PREG])
    r = check(_answer(Claim(statement="Metformin causes lactic acidosis.", fact_ids=["F1"])), ctx)
    assert not r.passed and "not about this statement" in r.dropped[0][1]


def test_dosing_leak_is_dropped():
    ctx = _ctx([F_ASPIRIN])
    r = check(_answer(Claim(statement="Warfarin with aspirin 81 mg daily increases bleeding.", fact_ids=["F1"])), ctx)
    assert not r.passed and r.dropped[0][1] == "contains dosing"


def test_refusal_and_not_found_carry_no_claims():
    ctx = _ctx([F_ASPIRIN])
    for status in ("refuse", "not_found"):
        r = check(_answer(Claim(statement="Warfarin interacts with aspirin.", fact_ids=["F1"]), status=status), ctx)
        assert r.status == status and not r.kept and r.dropped


def test_render_not_found_never_says_no_interaction():
    ctx = _ctx([F_ASPIRIN])
    r = check(_answer(status="not_found", summary="The labels do not document any interaction between warfarin and semaglutide."), ctx)
    text = render("The labels do not document any interaction between warfarin and semaglutide.", r, ctx)
    assert "couldn't find that" in text and "recognised in your question: warfarin" in text
    assert "not document any interaction" not in text


def test_render_cites_label_and_quote():
    ctx = _ctx([F_ASPIRIN])
    r = check(_answer(Claim(statement="Warfarin interacts with aspirin.", fact_ids=["F1"])), ctx)
    text = render("Summary.", r, ctx, built_on="2026-09-14T00:00:00")
    assert "[F1]" in text and "setid=SET-W" in text and "Table 3" in text and "Graph built on 2026-09-14" in text


def test_lab_threshold_is_not_a_dose():
    f = Fact("metformin", "CONTRAINDICATED_FOR", "severe renal impairment", "Condition", "LBL-M", "contraindications",
             "Metformin is contraindicated in patients with severe renal impairment (eGFR below 30 mL/min/1.73 m2).", set_id="SET-M")
    ctx = _ctx([f])
    r = check(_answer(Claim(statement="Metformin is contraindicated in severe renal impairment (eGFR below 30 mL/min).", fact_ids=["F1"])), ctx)
    assert r.passed and len(r.kept) == 1


def test_drug_name_alone_does_not_support_a_claim():
    """'Amiodarone is a class III antiarrhythmic' cited an indication fact: laundering a vector-only statement."""
    f = Fact("amiodarone", "TREATS", "ventricular fibrillation", "Condition", "LBL-A", "indications_and_usage",
             "Pacerone is an antiarrhythmic indicated for recurrent ventricular fibrillation.", set_id="SET-A")
    ctx = _ctx([f])
    r = check(_answer(
        Claim(statement="Amiodarone is indicated for ventricular fibrillation.", fact_ids=["F1"]),
        Claim(statement="Amiodarone is considered a class III antiarrhythmic drug that blocks potassium channels.", fact_ids=["F1"]),
    ), ctx)
    assert [c.statement for c in r.kept] == ["Amiodarone is indicated for ventricular fibrillation."]
    assert "not about this statement" in r.dropped[0][1]


def test_evidence_overlap_supports_a_claim_beyond_the_object():
    """'rhabdomyolysis' is in the evidence sentence of the myopathy fact, so it is supported."""
    f = Fact("atorvastatin", "CAUSES_SIDE_EFFECT", "myopathy", "SideEffect", "LBL-S", "warnings_and_cautions",
             "Cases of myopathy and rhabdomyolysis with acute renal failure have been reported with statins.", set_id="SET-S")
    ctx = _ctx([f])
    r = check(_answer(Claim(statement="Atorvastatin has reported rhabdomyolysis with acute renal failure.", fact_ids=["F1"])), ctx)
    assert r.passed


def test_speculative_relevance_clause_is_stripped_not_dropped():
    f = Fact("metformin", "CONTRAINDICATED_FOR", "hypersensitivity", "Condition", "LBL-M", "contraindications",
             "Known hypersensitivity to metformin hydrochloride.", set_id="SET-M")
    ctx = _ctx([f])
    r = check(_answer(Claim(statement="Metformin is contraindicated in hypersensitivity, which may occur in patients with kidney issues.", fact_ids=["F1"])), ctx)
    assert r.passed and r.kept[0].statement == "Metformin is contraindicated in hypersensitivity."
    # evidence-backed consequence clauses are untouched
    g = Fact("warfarin", "INTERACTS_WITH", "amiodarone", "Drug", "LBL-A", "drug_interactions",
             "Potentiates anticoagulant response and can result in serious or fatal bleeding.", set_id="SET-A")
    r = check(_answer(Claim(statement="Warfarin interacts with amiodarone, which can result in serious or fatal bleeding.", fact_ids=["F1"])), _ctx([g]))
    assert r.kept[0].statement.endswith("serious or fatal bleeding.")
