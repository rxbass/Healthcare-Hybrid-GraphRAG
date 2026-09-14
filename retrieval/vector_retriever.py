"""Vector retriever: question -> descriptive label chunks (dense-only).

Embeds the question with the same model used at build time and queries the
Neo4j native vector index. No BM25, no reranker (DESIGN.md sections 2/5): the
graph is the precision layer; this side only supplies background prose.

When the entity linker found drugs, a second query restricted to those drugs'
chunks is run and merged in, so the most relevant prose for the named drug is
present even if a generic question embedding drifts toward another label.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from graph import db  # noqa: E402
from graph.schema import VECTOR_DIMENSIONS, VECTOR_INDEX_NAME  # noqa: E402

TOP_K = 6
PER_DRUG_K = 3

_SEARCH = f"""
CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $k, $vec) YIELD node, score
RETURN node.id AS id, node.drug_id AS drug_id, node.drug_name AS drug_name, node.field AS field,
       node.text AS text, node.label_id AS label_id, node.set_id AS set_id, score
"""

# Restrict by drug: over-fetch then filter (the vector index has no metadata filter).
_SEARCH_BY_DRUG = f"""
CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $k, $vec) YIELD node, score
WHERE node.drug_id IN $drug_ids
RETURN node.id AS id, node.drug_id AS drug_id, node.drug_name AS drug_name, node.field AS field,
       node.text AS text, node.label_id AS label_id, node.set_id AS set_id, score
LIMIT $limit
"""


@dataclass
class Chunk:
    id: str
    drug_id: str
    drug_name: str
    field: str
    text: str
    label_id: str
    set_id: str
    score: float


@dataclass
class VectorResult:
    chunks: list[Chunk] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.openai_api_key(), dimensions=VECTOR_DIMENSIONS)


def embed_query(question: str) -> list[float]:
    return _embedder().embed_query(question)


def retrieve(question: str, drug_ids: list[str] | None = None, k: int = TOP_K) -> VectorResult:
    result = VectorResult()
    try:
        vec = embed_query(question)
        rows = db.run(_SEARCH, k=k, vec=vec)
        if drug_ids:
            rows += db.run(_SEARCH_BY_DRUG, k=max(k * 20, 100), vec=vec, drug_ids=drug_ids,
                           limit=PER_DRUG_K * len(drug_ids))
        seen = set()
        for r in sorted(rows, key=lambda r: -r["score"]):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            result.chunks.append(Chunk(r["id"], r["drug_id"], r["drug_name"], r["field"], r["text"],
                                       r["label_id"], r["set_id"], float(r["score"])))
    except Exception as exc:  # vectors down -> caller degrades to graph-only
        result.error = f"{type(exc).__name__}: {exc}"
    return result
