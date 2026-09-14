"""Streamlit demo UI for the drug-label reference assistant.

    streamlit run app.py

Everything goes through guardrails.rails.guarded_answer, so the UI has exactly
the same behaviour (PII masking, scope refusals, grounding gate, citations,
logging) as the evaluation harness. The sidebar shows the graph build stamp,
per-stage latency, token cost and what the grounding gate did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings  # noqa: E402

EXAMPLES = [
    "I take warfarin and need something for pain. What should I avoid?",
    "Does warfarin interact with aspirin?",
    "Which antibiotics are documented to interact with warfarin?",
    "Does levothyroxine interact with lisinopril?",
    "What is amiodarone used for?",
    "Is warfarin contraindicated in pregnancy?",
    "Does warfarin interact with semaglutide?",
    "How many milligrams of warfarin should I take per day?",
]

st.set_page_config(page_title="Drug Label Reference (Hybrid GraphRAG)", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Loading graph, vector index and guardrails…")
def _load():
    from graph import queries
    from guardrails.rails import get_rails, guarded_answer

    get_rails()  # warm NeMo + spaCy once
    return guarded_answer, queries.build_info()


try:
    guarded_answer, build_info = _load()
    load_error = None
except Exception as exc:  # missing credentials, graph down, etc.
    guarded_answer, build_info, load_error = None, None, f"{type(exc).__name__}: {exc}"

st.title("💊 Drug Label Reference — hybrid graph + vector RAG")
st.caption(
    "Reports what official FDA drug labels document — interactions, indications, contraindications, adverse "
    "reactions — with a citation for every fact. **Informational only. Not medical advice.**"
)

with st.sidebar:
    st.header("System")
    if build_info:
        st.metric("Graph built on", build_info["built_at"][:10])
        st.caption(f"{build_info['n_labels']} FDA labels · {build_info['n_triplets']} extracted facts · model `{settings.LLM_MODEL}`")
    elif load_error:
        st.error(f"Not ready: {load_error}")
    st.markdown(
        "**How an answer is made**\n\n"
        "1. NeMo input rails: PII masked, scope check, jailbreak check\n"
        "2. Entity linking → graph facts (**source of truth**) + dense vector context\n"
        "3. LLM drafts claims → **grounding gate** drops anything the graph doesn't support\n"
        "4. Citations back to the FDA label"
    )
    st.markdown("**Refuses:** dosing · diagnosis · personal treatment decisions")

st.subheader("Ask about a medication")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Question", value=st.session_state.get("question", ""), placeholder="e.g. Does warfarin interact with ibuprofen?")

if question and guarded_answer:
    with st.spinner("Checking rails, querying the graph, grounding the answer…"):
        resp = guarded_answer(question)
    s = resp.stats

    badge = {"answer": "✅ Answered from FDA labels", "not_found": "🔍 Not in the labels held",
             "refuse": "⛔ Out of scope", "degraded": "⚠️ Degraded (raw facts)"}.get(resp.status, resp.status)
    st.markdown(f"**{badge}**" + (f" · blocked by rail `{s['blocked_by']}`" if s.get("blocked_by") else ""))
    if s.get("pii_masked"):
        st.info(f"Personal identifiers were masked before processing. The question as processed: _{resp.question}_")
    st.markdown(resp.text.replace("\n", "  \n"))

    with st.expander("Retrieval & grounding details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total latency", f"{s.get('total_ms', 0) / 1000:.1f} s")
        c2.metric("Graph facts", s.get("graph_facts", 0))
        c3.metric("Vector chunks", s.get("vector_chunks", 0))
        kept, dropped = s.get("claims_kept"), s.get("claims_dropped")
        c4.metric("Claims kept / dropped", f"{kept if kept is not None else '-'} / {dropped if dropped is not None else '-'}")
        st.json({k: v for k, v in s.items() if k in (
            "linked_drugs", "linked_conditions", "linked_classes", "focus", "no_interaction_pairs", "degraded",
            "gate_status", "gate_retries", "input_rails_ms", "link_ms", "graph_ms", "vector_ms", "generate_ms", "output_rails_ms")})
        if resp.dropped:
            st.markdown("**Dropped by the grounding gate**")
            for claim, why in resp.dropped:
                st.markdown(f"- ~~{claim.statement}~~ — _{why}_")
        usage = s.get("usage", [])
        if usage:
            st.caption("Tokens: " + ", ".join(f"{u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out ({u.get('model')})" for u in usage))
        if resp.context is not None:
            st.markdown("**Context block given to the model**")
            st.code(resp.context.text, language="text")
elif question and not guarded_answer:
    st.error("The system is not ready — check credentials in .env and that the graph is built (see CLAUDE.md).")
