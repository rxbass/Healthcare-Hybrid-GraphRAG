"""Step 4b of the build phase: dense-only vector index over label prose.

Splits each label section in data/processed/labels.jsonl into ~1,200-char
chunks on sentence boundaries, embeds them with OpenAI text-embedding-3-small,
and stores them as (:Drug)-[:HAS_CHUNK]->(:Chunk) nodes under a Neo4j native
cosine vector index — one store for graph and vectors.

Design rule (DESIGN.md section 5): chunks are *context*, never facts. They carry
label_id/set_id so a citation can point at the label, but nothing here creates
an edge.

Idempotent: chunk ids are deterministic (drug/field/index) and a sha1 of the
text is stored; re-running only embeds chunks whose text changed. No BM25, no
reranker — dense-only by design.

Usage:
    python ingestion/build_vectors.py            # embed new/changed chunks
    python ingestion/build_vectors.py --force    # re-embed everything
    python ingestion/build_vectors.py --only warfarin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from graph.schema import CONSTRAINTS, VECTOR_DIMENSIONS, VECTOR_INDEX, VECTOR_INDEX_NAME  # noqa: E402

CHUNK_CHARS = 1200
CHUNK_OVERLAP_SENTENCES = 1
EMBED_BATCH = 64

# Which sections are worth embedding as background prose. Tables of adverse
# reaction percentages embed poorly and are already covered by graph edges.
VECTOR_FIELDS = (
    "indications_and_usage", "description", "clinical_pharmacology", "mechanism_of_action",
    "drug_interactions", "contraindications", "warnings_and_cautions", "warnings", "boxed_warning",
    "adverse_reactions",
)


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[•])", text) if s.strip()]


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_SENTENCES) -> list[str]:
    sents = split_sentences(text)
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c) > 40]


def build_chunks(drug: dict) -> list[dict]:
    out = []
    for field in VECTOR_FIELDS:
        text = drug["sections"].get(field)
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            out.append({
                "id": f"{drug['id']}/{field}/{i}",
                "drug_id": drug["id"],
                "drug_name": drug["name"],
                "field": field,
                "chunk_index": i,
                "text": chunk,
                "text_hash": hashlib.sha1(chunk.encode("utf-8")).hexdigest(),
                "label_id": drug["label_id"],
                "set_id": drug["set_id"],
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-embed all chunks")
    ap.add_argument("--only", help="drug id")
    args = ap.parse_args()

    drugs = [json.loads(l) for l in (settings.PROCESSED_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        drugs = [d for d in drugs if d["id"] == args.only]
    chunks = [c for d in drugs for c in build_chunks(d)]
    print(f"{len(drugs)} drugs -> {len(chunks)} chunks | embedding model {settings.EMBEDDING_MODEL} | index {VECTOR_INDEX_NAME}")

    embedder = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.openai_api_key(), dimensions=VECTOR_DIMENSIONS)

    with GraphDatabase.driver(settings.neo4j_uri(), auth=(settings.neo4j_username(), settings.neo4j_password())) as driver:
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE) as s:
            for c in CONSTRAINTS:
                s.run(c)
            s.run(VECTOR_INDEX)

            # upsert chunk text/metadata; find which need (re)embedding
            existing = {r["id"]: r["h"] for r in s.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.id AS id, c.text_hash AS h")}
            s.run("""
                UNWIND $rows AS row
                MATCH (d:Drug {id: row.drug_id})
                MERGE (c:Chunk {id: row.id})
                SET c.drug_id = row.drug_id, c.drug_name = row.drug_name, c.field = row.field,
                    c.chunk_index = row.chunk_index, c.text = row.text, c.text_hash = row.text_hash,
                    c.label_id = row.label_id, c.set_id = row.set_id
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, rows=chunks)
            # drop chunks that no longer exist for these drugs (label text changed on refresh)
            s.run("""
                MATCH (c:Chunk) WHERE c.drug_id IN $drug_ids AND NOT c.id IN $ids
                DETACH DELETE c
            """, drug_ids=[d["id"] for d in drugs], ids=[c["id"] for c in chunks])

            todo = [c for c in chunks if args.force or existing.get(c["id"]) != c["text_hash"]]
            print(f"  {len(chunks) - len(todo)} up to date, {len(todo)} to embed")
            t0 = time.time()
            for i in range(0, len(todo), EMBED_BATCH):
                batch = todo[i:i + EMBED_BATCH]
                vectors = embedder.embed_documents([c["text"] for c in batch])
                s.run("""
                    UNWIND $rows AS row
                    MATCH (c:Chunk {id: row.id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)
                """, rows=[{"id": c["id"], "vec": v} for c, v in zip(batch, vectors)])
                print(f"  embedded {min(i + EMBED_BATCH, len(todo))}/{len(todo)}")
            s.run("MATCH (b:BuildInfo {id:'current'}) SET b.n_chunks = $n, b.embedding_model = $m",
                  n=len(chunks), m=settings.EMBEDDING_MODEL)
            n = s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n").single()["n"]
            print(f"\n{n} embedded chunks in index '{VECTOR_INDEX_NAME}' ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
