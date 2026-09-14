# Healthcare Hybrid GraphRAG

A drug-interaction reference assistant: a **hybrid graph + vector RAG** over
public FDA drug-label data. The graph is the source of truth for drug
relationships, vectors supply descriptive context, a hand-written grounding
gate refuses anything the graph doesn't support, and a golden-set evaluation
runs in CI.

**Informational only.** It reports documented facts with citations and refuses
dosing, diagnosis and personal treatment advice — by design, not by disclaimer.

> *"I take warfarin and need something for pain — what should I avoid?"*
> → aspirin, ibuprofen, naproxen — each with the label sentence that documents
> the interaction and the label sentence that documents the indication.
> One Cypher traversal; not possible with plain vector RAG.

## How it works

```
question
  │  NeMo Guardrails input rails: PII masked (Presidio) → scope check (deterministic) → jailbreak check (LLM)
  ▼
entity linking  ── exact/alias match against openFDA names + synonyms (the precision layer)
  ├──► graph retrieval (Neo4j, Cypher)   exact facts w/ provenance:  warfarin INTERACTS_WITH aspirin [label …]
  └──► vector retrieval (dense-only)     descriptive label prose for phrasing — never a fact
  ▼
merged, labelled context   [F#] graph facts (source of truth) · [N#] "no documented interaction" · [C#] background
  ▼
LLM drafts claims (LCEL, structured output)
  ▼
grounding gate (custom, fails closed): unknown citation / vector-only fact / dose / contradiction of [N#] → dropped
  ▼
answer with citations to the FDA label (DailyMed link + evidence sentence) + "graph built on {date}"
```

Design and rationale: [DESIGN.md](DESIGN.md). Always-true rules for contributors: [CLAUDE.md](CLAUDE.md).

### Data & schema
openFDA drug-label API → 24 warfarin-centric labels (single-ingredient,
prescription where one exists). Nodes `Drug` `DrugClass` `Condition`
`SideEffect` `Ingredient`; edges `TREATS` `INTERACTS_WITH`
`CAUSES_SIDE_EFFECT` `CONTRAINDICATED_FOR` `BELONGS_TO_CLASS` `HAS_INGREDIENT`.
Structured openFDA fields become nodes without an LLM; free-text sections are
extracted into triplets by an LLM offline, and **every edge stores the label
id and the evidence sentence** — verified verbatim against the label text
before it is allowed into the graph (unverifiable evidence is refused).

## Results (golden set, 42 questions — `eval/results/`)

| | result |
|---|---|
| Stage 1 — deterministic retrieval (no LLM, ~9 s) | **42/42**, entity exact-match 1.00, fact recall 1.00 |
| Stage 2 — full stack + LLM judge (`gpt-4o` judging `gpt-4o-mini`) | **42/42**, grounded 1.00, safe 1.00, status match 1.00 |
| Latency, answered questions | p50 3.1 s · p95 5.8 s |
| Cost | ≈ $0.0004 per answered query (generation); ≈ $0.05 per full Stage 2 run |

Categories: interaction, treats, contraindication, side effect, class,
multi-hop, descriptive, **no-interaction controls**, **not-in-data**, dosing /
diagnosis / treatment refusals, jailbreak, PII, off-topic.

## Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg          # PII rail model, once
cp .env.example .env                             # OPENAI_API_KEY, NEO4J_URI / USERNAME / PASSWORD (Aura free tier works)

python ingestion/build_index.py                  # fetch → parse → extract → load Neo4j → embed  (~10 min, ~$0.10)
python graph/queries.py --demo                   # the multi-hop demo query
python eval/run_eval.py --stage 1                # deterministic check, free
streamlit run app.py
```

All credentials come from the environment; nothing is hardcoded and `.env` is
git-ignored.

## Repository map

| path | what |
|---|---|
| `ingestion/` | build phase: `fetch_openfda` → `parse_labels` → `extract_triplets` → `load_graph` → `build_vectors` (`build_index.py` runs all) |
| `graph/` | schema + constraints, Cypher queries incl. the multi-hop demo |
| `retrieval/` | entity linker, graph retriever, dense vector retriever, merge |
| `generation/` | prompt, LCEL chain, **grounding gate**, citations |
| `guardrails/` | NeMo Guardrails config (Colang 1.0, pinned) + deterministic scope action + wrapper |
| `pipeline.py` | end-to-end query; `guardrails/rails.py` wraps it |
| `eval/` | two-stage harness, cost/latency report, results |
| `observability/` | per-query JSONL logging (masked question, hits, latency, gate, tokens, cost) |
| `scripts/refresh.py` · `.github/workflows/` | scheduled data refresh with Stage 1 as release gate; CI eval |
| `docs/DEPLOYMENT.md` | deployment & scaling plan — written, deliberately not built |
| `data/` | golden set, seed drugs, raw labels, processed triplets |

## What's deliberately not here
BM25 and a reranker (the graph is the precision layer, so dense-only is
enough for background prose); a retrieval router (both retrievers always
run); auth, multi-tenancy, Kubernetes, a live deployment (see the deployment
doc for how they would be done).
