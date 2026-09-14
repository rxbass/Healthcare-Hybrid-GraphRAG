"""Cypher queries over the drug-label graph. The graph retriever is built on these.

Every query returns facts *with provenance* (source_label_id, evidence) so the
generation layer can cite them. Drug arguments are graph ids (the entity
linker's job is to turn question text into these ids).

The demo query (DESIGN.md section 4):
    "I take [drug A] and need something for [condition B] — what should I avoid?"
    drug A -> INTERACTS_WITH -> drug X  where  X -> TREATS -> condition B
    (either a direct A~X edge, or A~class<-X via BELONGS_TO_CLASS)

Usage:
    python graph/queries.py --demo                         # warfarin + pain
    python graph/queries.py --demo --drug amiodarone --condition cholesterol
    python graph/queries.py --interactions warfarin
    python graph/queries.py --pair warfarin aspirin
    python graph/queries.py --facts metronidazole
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import db  # noqa: E402

PROV = "source_label_id: e.source_label_id, set_id: e.set_id, source_drug: e.source_drug, source_field: e.source_field, evidence: e.evidence"

# ---------------------------------------------------------------------------
# The multi-hop demo query
# ---------------------------------------------------------------------------
DEMO_QUERY = """
MATCH (a:Drug {id: $drug})
MATCH (x:Drug)-[t:TREATS]->(c:Condition)
WHERE x <> a AND toLower(c.name) CONTAINS toLower($condition)
WITH a, x, collect(DISTINCT {condition: c.name, source_label_id: t.source_label_id, set_id: t.set_id, source_drug: t.source_drug, evidence: t.evidence}) AS treats
// hop 1: a direct interaction edge in either direction
OPTIONAL MATCH (a)-[e:INTERACTS_WITH]-(x)
WITH a, x, treats, collect(DISTINCT {via: 'direct', """ + PROV + """}) AS direct
// hop 2: a interacts with a class that x belongs to (a -> class <- x)
OPTIONAL MATCH (a)-[e:INTERACTS_WITH]->(cls:DrugClass)<-[:BELONGS_TO_CLASS]-(x)
WITH x, treats, direct, collect(DISTINCT {via: 'class:' + cls.name, """ + PROV + """}) AS via_class
WITH x, treats, [i IN direct + via_class WHERE i.source_label_id IS NOT NULL] AS interactions
WHERE size(interactions) > 0
RETURN x.id AS drug_id, x.name AS drug, x.in_corpus AS in_corpus,
       treats, interactions
ORDER BY drug
"""

INTERACTIONS_QUERY = """
MATCH (a:Drug {id: $drug})-[e:INTERACTS_WITH]-(o)
RETURN a.name AS subject, type(e) AS predicate,
       CASE WHEN o:Drug THEN o.name ELSE o.name END AS object,
       CASE WHEN o:Drug THEN 'Drug' ELSE 'DrugClass' END AS object_type,
       startNode(e) = a AS from_subject_label,
       e.source_label_id AS source_label_id, e.source_field AS source_field, e.evidence AS evidence
ORDER BY object
"""

PAIR_QUERY = """
MATCH (a:Drug {id: $a}), (b:Drug {id: $b})
OPTIONAL MATCH (a)-[e:INTERACTS_WITH]-(b)
WITH a, b, collect(DISTINCT {via: 'direct', """ + PROV + """}) AS direct
OPTIONAL MATCH (a)-[e:INTERACTS_WITH]->(cls:DrugClass)<-[:BELONGS_TO_CLASS]-(b)
WITH a, b, direct, collect(DISTINCT {via: 'class:' + cls.name, """ + PROV + """}) AS via_ab
OPTIONAL MATCH (b)-[e:INTERACTS_WITH]->(cls:DrugClass)<-[:BELONGS_TO_CLASS]-(a)
WITH a, b, direct, via_ab, collect(DISTINCT {via: 'class:' + cls.name, """ + PROV + """}) AS via_ba
RETURN a.name AS drug_a, b.name AS drug_b,
       [i IN direct + via_ab + via_ba WHERE i.source_label_id IS NOT NULL] AS interactions
"""

FACTS_QUERY = """
// INTERACTS_WITH is symmetric: an edge extracted from the *other* drug's label still counts.
MATCH (a:Drug {id: $drug})-[e]-(o)
WHERE type(e) IN $predicates AND (startNode(e) = a OR type(e) = 'INTERACTS_WITH')
RETURN a.name AS subject, type(e) AS predicate, o.name AS object, labels(o)[0] AS object_type,
       e.source_label_id AS source_label_id, e.set_id AS set_id, e.source_drug AS source_drug,
       e.source_field AS source_field, e.evidence AS evidence
// corpus drugs first (we hold their labels), then classes, then stub drugs
ORDER BY predicate,
         CASE WHEN o:Drug AND o.in_corpus THEN 0 WHEN o:DrugClass THEN 1 ELSE 2 END,
         object
"""

DRUG_QUERY = """
MATCH (d:Drug {id: $drug})
RETURN d.id AS id, d.name AS name, d.in_corpus AS in_corpus, d.label_id AS label_id,
       d.set_id AS set_id, d.effective_time AS effective_time, d.brand_names AS brand_names
"""

BUILD_INFO_QUERY = "MATCH (b:BuildInfo {id: 'current'}) RETURN b.built_at AS built_at, b.n_labels AS n_labels, b.n_triplets AS n_triplets"


def demo(drug: str, condition: str) -> list[dict]:
    return db.run(DEMO_QUERY, drug=drug, condition=condition)


def interactions(drug: str) -> list[dict]:
    return db.run(INTERACTIONS_QUERY, drug=drug)


def pair(a: str, b: str) -> dict | None:
    rows = db.run(PAIR_QUERY, a=a, b=b)
    return rows[0] if rows else None


def facts(drug: str, predicates: tuple[str, ...] = ("TREATS", "CONTRAINDICATED_FOR", "CAUSES_SIDE_EFFECT", "BELONGS_TO_CLASS", "INTERACTS_WITH")) -> list[dict]:
    return db.run(FACTS_QUERY, drug=drug, predicates=list(predicates))


def drug(drug_id: str) -> dict | None:
    rows = db.run(DRUG_QUERY, drug=drug_id)
    return rows[0] if rows else None


def build_info() -> dict | None:
    rows = db.run(BUILD_INFO_QUERY)
    return rows[0] if rows else None


def _print_demo(drug_id: str, condition: str) -> None:
    rows = demo(drug_id, condition)
    print(f'DEMO: "I take {drug_id} and need something for {condition} — what should I avoid?"')
    print(f"  {len(rows)} drug(s) treat '{condition}' AND have a documented interaction with {drug_id}:\n")
    for r in rows:
        print(f"  AVOID {r['drug']}  (in_corpus={r['in_corpus']})")
        for t in r["treats"][:3]:
            print(f"     treats: {t['condition']}   [label {t['source_label_id'][:8]}…] \"{t['evidence'][:110]}…\"")
        for i in r["interactions"][:3]:
            print(f"     interacts ({i['via']}) [label {i['source_label_id'][:8]}…, {i['source_field']}] \"{i['evidence'][:110]}…\"")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--drug", default="warfarin")
    ap.add_argument("--condition", default="pain")
    ap.add_argument("--interactions", metavar="DRUG_ID")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--facts", metavar="DRUG_ID")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of a summary")
    args = ap.parse_args()

    info = build_info()
    print(f"graph built on {info['built_at'] if info else '<never>'}\n")
    if args.demo:
        rows = demo(args.drug, args.condition)
        print(json.dumps(rows, indent=2)) if args.json else _print_demo(args.drug, args.condition)
    if args.interactions:
        rows = interactions(args.interactions)
        print(json.dumps(rows, indent=2)) if args.json else print(
            f"{args.interactions} interacts with {len(rows)}:", sorted({r['object'] for r in rows}))
    if args.pair:
        print(json.dumps(pair(*args.pair), indent=2))
    if args.facts:
        rows = facts(args.facts)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                print(f"  {r['predicate']:20s} {r['object']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
