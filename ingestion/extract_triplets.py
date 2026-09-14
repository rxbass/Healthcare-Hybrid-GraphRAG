"""Step 3 of the build phase: LLM extraction of graph edges from label free text.

For every (drug, section) in data/processed/labels.jsonl the section is split
into chunks and an LCEL chain (prompt | chat model with structured output)
returns triplets:

    subject drug -> predicate -> object   + the verbatim evidence sentence

The predicate set allowed per section comes from graph.schema.EXTRACTION_FIELDS
so the model cannot emit, say, TREATS from adverse_reactions.

Provenance is enforced, not trusted: an emitted evidence sentence must actually
occur in the source chunk — exactly, or at >= 85% similarity once bracketed
cross-references and punctuation are stripped (the model reliably drops
"[see Warnings and Precautions (5.7)]"). Triplets whose evidence cannot be
found are kept in the output with `evidence_verified: false` and the graph
loader refuses to load them. Fail closed.

Output: data/processed/triplets.jsonl (one triplet per line). The file is a
cache keyed by (drug, label id, section, chunk, model): re-running skips work
already done, so the build is idempotent and a partial run resumes; when a
refresh fetches a *different* label for a drug its old triplets are dropped and
the drug is re-extracted. Use --force to re-extract everything.

Usage:
    python ingestion/extract_triplets.py            # all drugs, all sections
    python ingestion/extract_triplets.py --only warfarin
    python ingestion/extract_triplets.py --force
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from graph.schema import EXTRACTION_FIELDS  # noqa: E402

CHUNK_CHARS = 7000  # ~1.7k tokens; sections are split on sentence boundaries near this size
MAX_CHUNKS_PER_SECTION = 8

Predicate = Literal["TREATS", "INTERACTS_WITH", "CAUSES_SIDE_EFFECT", "CONTRAINDICATED_FOR"]
ObjectType = Literal["drug", "drug_class", "condition", "side_effect"]

OBJECT_TYPE_FOR = {
    "TREATS": ("condition",),
    "INTERACTS_WITH": ("drug", "drug_class"),
    "CAUSES_SIDE_EFFECT": ("side_effect",),
    "CONTRAINDICATED_FOR": ("condition",),
}


class Triplet(BaseModel):
    predicate: Predicate
    object: str = Field(description="Canonical short name of the object: a generic drug name, a drug class, a condition, or an adverse reaction. Lower-case, singular, no dose or qualifiers.")
    object_type: ObjectType
    evidence: str = Field(description="One sentence copied VERBATIM from the text that states this fact.")


class Extraction(BaseModel):
    triplets: list[Triplet]


SYSTEM_PROMPT = """You extract facts from an FDA drug label section into graph triplets.
The subject of every triplet is the label's drug: {drug}.
This section is `{field}`; you may ONLY emit these predicates: {predicates}.

Rules:
- TREATS: object is a condition/indication the drug is indicated for (object_type=condition).
- INTERACTS_WITH: object is another drug (object_type=drug, generic name) or a drug class such as
  "nsaids", "cyp3a4 inhibitors", "anticoagulants" (object_type=drug_class). When the text names a
  class AND gives specific example drugs, emit the class AND each example drug separately.
  Do not emit the label's own drug as the object. Foods/alcohol/herbals are not drugs: skip them
  unless they are a named drug product.
- CONTRAINDICATED_FOR: object is a condition, patient state, or hypersensitivity (object_type=condition).
  If the contraindication is co-administration with a specific drug, emit INTERACTS_WITH with that
  drug instead (only if INTERACTS_WITH is allowed here).
- CAUSES_SIDE_EFFECT: object is an adverse reaction (object_type=side_effect). Prefer clinically
  significant and common reactions; skip percentages, study design, and placebo comparisons.
  At most 25 per chunk.
- evidence must be ONE sentence copied exactly as it appears in the text. Never paraphrase.
- Object names: lower-case, singular, canonical ("atrial fibrillation", not "AF (atrial fibrillation)").
- Emit nothing that the text does not state. If the chunk contains no facts of the allowed types,
  return an empty list."""

USER_PROMPT = "TEXT (chunk {chunk_idx} of {n_chunks}):\n\n{text}"

_WS = re.compile(r"\s+")
_XREF = re.compile(r"\[\s*see[^\]]*\]|\(\s*\d+(\.\d+)?\s*\)", re.I)  # "[see Warnings (5.7)]", "( 7 )"
_PUNCT = re.compile(r"[^a-z0-9 ]+")
MAX_PIECES = 3
MIN_PIECE = 15
MIN_COVERAGE = 0.9


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _norm_for_match(s: str) -> str:
    """Lower-case, drop label cross-references and punctuation, collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub(" ", _XREF.sub(" ", s.lower()))).strip()


def verify_evidence(evidence: str, source: str) -> str:
    """Return 'exact', 'fuzzy' or 'none' for how well the evidence sentence is found in source.

    exact  - the normalised evidence is a substring of the normalised source.
    fuzzy  - the evidence is a composite of at most MAX_PIECES fragments, each of
             which occurs verbatim in the source and is at least MIN_PIECE chars,
             together covering >= MIN_COVERAGE of the evidence. This accepts the
             two things the model reliably does — drop a "[see ...]" cross-ref,
             or stitch a list header onto one bullet — while a paraphrase cannot
             produce long verbatim fragments and still fails.
    """
    ev, src = _norm_for_match(evidence), _norm_for_match(source)
    if not ev:
        return "none"
    if ev in src:
        return "exact"
    covered, remaining, pieces = 0, [ev], 0
    while remaining and pieces < MAX_PIECES:
        frag = max(remaining, key=len)
        remaining.remove(frag)
        m = difflib.SequenceMatcher(None, src, frag, autojunk=False).find_longest_match(0, len(src), 0, len(frag))
        if m.size < MIN_PIECE:
            break
        pieces += 1
        covered += m.size
        left, right = frag[: m.b].strip(), frag[m.b + m.size :].strip()
        remaining += [x for x in (left, right) if len(x) >= MIN_PIECE]
    leftover = sum(len(x) for x in remaining)
    return "fuzzy" if (covered / len(ev)) >= MIN_COVERAGE and leftover / len(ev) <= (1 - MIN_COVERAGE) else "none"


def split_chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks[:MAX_CHUNKS_PER_SECTION]


def build_chain(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=settings.openai_api_key())
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])
    return prompt | llm.with_structured_output(Extraction)


def extract_chunk(chain, drug: dict, field: str, chunk_idx: int, n_chunks: int, text: str, model_name: str) -> list[dict]:
    predicates = EXTRACTION_FIELDS[field]
    result: Extraction = chain.invoke({
        "drug": drug["name"],
        "field": field,
        "predicates": ", ".join(predicates),
        "chunk_idx": chunk_idx + 1,
        "n_chunks": n_chunks,
        "text": text,
    })
    out = []
    for t in result.triplets:
        if t.predicate not in predicates or t.object_type not in OBJECT_TYPE_FOR[t.predicate]:
            continue
        obj = _norm(t.object)
        if not obj or obj == drug["name"]:
            continue
        out.append({
            "subject": drug["id"],
            "subject_name": drug["name"],
            "predicate": t.predicate,
            "object": obj,
            "object_type": t.object_type,
            "evidence": _WS.sub(" ", t.evidence).strip(),
            "evidence_match": (match := verify_evidence(t.evidence, text)),
            "evidence_verified": match != "none",
            "source_label_id": drug["label_id"],
            "set_id": drug["set_id"],
            "source_field": field,
            "chunk_idx": chunk_idx,
            "extracted_by": model_name,
        })
    return out


def _write_all(path: Path, triplets: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for t in triplets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="drug id to extract (e.g. warfarin)")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    ap.add_argument("--model", default=settings.LLM_MODEL)
    args = ap.parse_args()

    labels_path = settings.PROCESSED_DIR / "labels.jsonl"
    if not labels_path.exists():
        print("run ingestion/parse_labels.py first", file=sys.stderr)
        return 1
    drugs = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        drugs = [d for d in drugs if d["id"] == args.only]

    out_path = settings.PROCESSED_DIR / "triplets.jsonl"
    current_label = {d["id"]: d["label_id"] for d in drugs}
    all_labels = {json.loads(l)["id"]: json.loads(l)["label_id"] for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    done: set[tuple] = set()
    existing: list[dict] = []
    stale = 0
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t["source_label_id"] != all_labels.get(t["subject"]):
                stale += 1  # label changed on refresh -> discard
                continue
            if args.force and (not args.only or t["subject"] == args.only):
                continue
            existing.append(t)
            done.add((t["subject"], t["source_label_id"], t["source_field"], t["chunk_idx"], t["extracted_by"]))
    # A chunk that produced zero triplets is still "done": record it in a sidecar.
    sidecar = settings.PROCESSED_DIR / "triplets.done.json"
    if sidecar.exists():
        for x in json.loads(sidecar.read_text(encoding="utf-8")):
            x = tuple(x)
            if len(x) == 5 and x[1] == all_labels.get(x[0]) and not (args.force and (not args.only or x[0] == args.only)):
                done.add(x)
    if stale:
        print(f"  dropped {stale} triplets from superseded labels")
    rewrite = args.force or stale > 0

    chain = build_chain(args.model)
    print(f"model={args.model} | {len(drugs)} drugs | cached chunks={len(done)}")
    new_triplets: list[dict] = []
    calls = 0
    t0 = time.time()
    for drug in drugs:
        for field in EXTRACTION_FIELDS:
            text = drug["sections"].get(field)
            if not text:
                continue
            chunks = split_chunks(text)
            for i, chunk in enumerate(chunks):
                key = (drug["id"], drug["label_id"], field, i, args.model)
                if key in done:
                    continue
                trips = extract_chunk(chain, drug, field, i, len(chunks), chunk, args.model)
                calls += 1
                done.add(key)
                new_triplets.extend(trips)
                unverified = sum(1 for t in trips if not t["evidence_verified"])
                print(f"  {drug['id']:34s} {field:22s} chunk {i+1}/{len(chunks)} -> {len(trips):3d} triplets"
                      + (f" ({unverified} unverified evidence)" if unverified else ""))
                # persist after every chunk so a crash keeps progress. A plain append is
                # only safe when nothing on disk was discarded; otherwise rewrite atomically.
                if rewrite:
                    _write_all(out_path, existing + new_triplets)
                else:
                    with out_path.open("a", encoding="utf-8") as fh:
                        for t in trips:
                            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
                sidecar.write_text(json.dumps(sorted(done)), encoding="utf-8")

    if rewrite:
        _write_all(out_path, existing + new_triplets)

    all_trips = existing + new_triplets
    verified = sum(1 for t in all_trips if t["evidence_verified"])
    print(f"\n{calls} LLM calls in {time.time()-t0:.0f}s | total triplets {len(all_trips)} "
          f"({verified} evidence-verified, {len(all_trips)-verified} unverified) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
