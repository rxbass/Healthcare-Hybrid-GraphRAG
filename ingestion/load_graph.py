"""Step 4a of the build phase: load nodes and edges into Neo4j.

Inputs (data/processed/): labels.jsonl, aliases.json, triplets.jsonl.

  * Drug nodes for every label we hold (`in_corpus: true`), with identity
    properties for citations (label_id, set_id, effective_time, brands).
  * Structured edges straight from openfda fields — no LLM involved:
      BELONGS_TO_CLASS  (openfda.pharm_class_epc)
      HAS_INGREDIENT    (openfda.substance_name)
  * Extracted edges from triplets.jsonl — only those with evidence_verified.
    Objects are resolved through the shared alias table: a drug mention that
    matches a corpus drug becomes an edge to that node; an unknown drug becomes
    a stub Drug node (`in_corpus: false`) so the fact is kept and auditable.

Every edge carries source_label_id, set_id, source_field, evidence,
extracted_by. MERGE keys include (source_label_id, evidence) so the load is
idempotent: re-running does not duplicate edges.

A single (:BuildInfo {id:'current'}) node records when the graph was built
and from what; the app surfaces it as "graph built on {date}".

Usage:
    python ingestion/load_graph.py            # merge into the existing graph
    python ingestion/load_graph.py --reset    # wipe the database first (refresh)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from graph.schema import CONSTRAINTS, PROVENANCE_KEYS  # noqa: E402

# Class-name normalisation so an extracted "nsaids" links to the EPC class node
# "nonsteroidal anti-inflammatory drug" that BELONGS_TO_CLASS edges use.
CLASS_ALIASES = {
    "nsaids": "nonsteroidal anti-inflammatory drug",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    "nonsteroidal anti-inflammatory drugs": "nonsteroidal anti-inflammatory drug",
    "non-steroidal anti-inflammatory drugs": "nonsteroidal anti-inflammatory drug",
    "non-steroidal anti-inflammatory drug": "nonsteroidal anti-inflammatory drug",
    "ssris": "selective serotonin reuptake inhibitor",
    "ssri": "selective serotonin reuptake inhibitor",
    "selective serotonin reuptake inhibitors": "selective serotonin reuptake inhibitor",
    "snris": "serotonin and norepinephrine reuptake inhibitor",
    "statins": "hmg-coa reductase inhibitor",
    "hmg-coa reductase inhibitors": "hmg-coa reductase inhibitor",
    "anticoagulants": "anticoagulant",
    "antiplatelet agents": "platelet aggregation inhibitor",
    "antiplatelets": "platelet aggregation inhibitor",
    "platelet aggregation inhibitors": "platelet aggregation inhibitor",
    "proton pump inhibitors": "proton pump inhibitor",
    "ppis": "proton pump inhibitor",
    "corticosteroids": "corticosteroid",
    "fluoroquinolones": "fluoroquinolone antibacterial",
    "quinolones": "fluoroquinolone antibacterial",
    "azole antifungals": "azole antifungal",
    "ace inhibitors": "angiotensin converting enzyme inhibitor",
    "angiotensin-converting enzyme inhibitors": "angiotensin converting enzyme inhibitor",
}

_WS = re.compile(r"\s+")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def norm_class(name: str) -> str:
    n = _WS.sub(" ", name.lower().strip())
    n = re.sub(r"\s*\[epc\]$", "", n)
    n = re.sub(r"^(other|strong|moderate|weak)\s+", "", n)
    return CLASS_ALIASES.get(n, n)


def singular(name: str) -> str:
    """'anticoagulants' -> 'anticoagulant' for plain plurals; leaves 'diabetes' alone."""
    return name[:-1] if name.endswith("s") and not name.endswith(("ss", "us", "is", "es")) else name


class Loader:
    def __init__(self, driver, database: str):
        self.driver = driver
        self.database = database
        self.stats: Counter = Counter()

    def run(self, cypher: str, **params):
        with self.driver.session(database=self.database) as s:
            return s.run(cypher, **params).consume()

    def reset(self):
        with self.driver.session(database=self.database) as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("  database wiped")

    def constraints(self):
        for c in CONSTRAINTS:
            self.run(c)
        print(f"  {len(CONSTRAINTS)} constraints ensured")

    def drugs(self, records: list[dict], aliases: dict[str, list[str]]):
        rows = [{
            "id": r["id"], "name": r["name"], "label_id": r["label_id"], "set_id": r["set_id"],
            "effective_time": r["effective_time"], "product_type": r["product_type"],
            "brand_names": r["brand_names"], "generic_names": r["generic_names"],
            "rxcui": r["rxcui"], "aliases": aliases.get(r["id"], []),
        } for r in records]
        self.run("""
            UNWIND $rows AS row
            MERGE (d:Drug {id: row.id})
            SET d += row, d.in_corpus = true
        """, rows=rows)
        self.stats["Drug(in_corpus)"] += len(rows)

    def structured_edges(self, records: list[dict]):
        cls_rows, ing_rows = [], []
        for r in records:
            base = {"drug": r["id"], "source_label_id": r["label_id"], "set_id": r["set_id"], "extracted_by": "structured"}
            for c in r["classes"]:
                cls_rows.append({**base, "name": norm_class(c), "source_field": "openfda.pharm_class_epc",
                                 "evidence": f"openfda.pharm_class_epc: {c}"})
            for i in r["ingredients"]:
                ing_rows.append({**base, "name": i, "source_field": "openfda.substance_name",
                                 "evidence": f"openfda.substance_name: {i}"})
        self.run("""
            UNWIND $rows AS row
            MATCH (d:Drug {id: row.drug})
            MERGE (c:DrugClass {name: row.name})
            MERGE (d)-[e:BELONGS_TO_CLASS {source_label_id: row.source_label_id, evidence: row.evidence}]->(c)
            SET e.set_id = row.set_id, e.source_field = row.source_field, e.extracted_by = row.extracted_by, e.source_drug = row.drug
        """, rows=cls_rows)
        self.run("""
            UNWIND $rows AS row
            MATCH (d:Drug {id: row.drug})
            MERGE (i:Ingredient {name: row.name})
            MERGE (d)-[e:HAS_INGREDIENT {source_label_id: row.source_label_id, evidence: row.evidence}]->(i)
            SET e.set_id = row.set_id, e.source_field = row.source_field, e.extracted_by = row.extracted_by, e.source_drug = row.drug
        """, rows=ing_rows)
        self.stats["BELONGS_TO_CLASS"] += len(cls_rows)
        self.stats["HAS_INGREDIENT"] += len(ing_rows)

    def extracted_edges(self, triplets: list[dict], alias_to_id: dict[str, str]):
        by_kind: dict[tuple[str, str], list[dict]] = {}
        for t in triplets:
            if not t.get("evidence_verified"):
                self.stats["skipped_unverified"] += 1
                continue
            for k in PROVENANCE_KEYS:
                assert t.get(k), f"triplet missing provenance {k}: {t}"
            obj = t["object"]
            if t["object_type"] == "drug":
                target_id = alias_to_id.get(obj) or alias_to_id.get(singular(obj))
                if target_id is None:
                    target_id = slug(obj)
                    self.stats["Drug(stub)"] += 1
                key = (t["predicate"], "Drug")
                row = {**t, "target": target_id, "target_name": obj}
            elif t["object_type"] == "drug_class":
                key = (t["predicate"], "DrugClass")
                row = {**t, "target": norm_class(obj)}
            elif t["object_type"] == "condition":
                key = (t["predicate"], "Condition")
                row = {**t, "target": obj}
            else:
                key = (t["predicate"], "SideEffect")
                row = {**t, "target": obj}
            by_kind.setdefault(key, []).append(row)

        for (pred, label), rows in by_kind.items():
            if label == "Drug":
                cypher = f"""
                    UNWIND $rows AS row
                    MATCH (s:Drug {{id: row.subject}})
                    MERGE (o:Drug {{id: row.target}})
                      ON CREATE SET o.name = row.target_name, o.in_corpus = false
                    MERGE (s)-[e:{pred} {{source_label_id: row.source_label_id, evidence: row.evidence}}]->(o)
                    SET e.set_id = row.set_id, e.source_field = row.source_field, e.source_drug = row.subject,
                        e.extracted_by = row.extracted_by, e.evidence_match = row.evidence_match
                """
            else:
                cypher = f"""
                    UNWIND $rows AS row
                    MATCH (s:Drug {{id: row.subject}})
                    MERGE (o:{label} {{name: row.target}})
                    MERGE (s)-[e:{pred} {{source_label_id: row.source_label_id, evidence: row.evidence}}]->(o)
                    SET e.set_id = row.set_id, e.source_field = row.source_field, e.source_drug = row.subject,
                        e.extracted_by = row.extracted_by, e.evidence_match = row.evidence_match
                """
            self.run(cypher, rows=rows)
            self.stats[f"{pred}->{label}"] += len(rows)

    def build_info(self, n_labels: int, n_triplets: int, models: list[str]):
        self.run("""
            MERGE (b:BuildInfo {id: 'current'})
            SET b.built_at = $built_at, b.n_labels = $n_labels, b.n_triplets = $n_triplets,
                b.extraction_models = $models, b.source = 'openFDA drug/label'
        """, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             n_labels=n_labels, n_triplets=n_triplets, models=models)

    def counts(self) -> dict:
        with self.driver.session(database=self.database) as s:
            nodes = {r["label"]: r["c"] for r in s.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label")}
            rels = {r["type"]: r["c"] for r in s.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS c ORDER BY type")}
        return {"nodes": nodes, "relationships": rels}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="DETACH DELETE everything before loading")
    args = ap.parse_args()

    p = settings.PROCESSED_DIR
    records = [json.loads(l) for l in (p / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    aliases = json.loads((p / "aliases.json").read_text(encoding="utf-8"))
    triplets = [json.loads(l) for l in (p / "triplets.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    alias_to_id = {a: d for d, forms in aliases.items() for a in forms}

    print(f"neo4j: {settings.neo4j_uri().split('://')[0]}://… db={settings.NEO4J_DATABASE} | "
          f"{len(records)} labels, {len(triplets)} triplets")
    with GraphDatabase.driver(settings.neo4j_uri(), auth=(settings.neo4j_username(), settings.neo4j_password())) as driver:
        driver.verify_connectivity()
        loader = Loader(driver, settings.NEO4J_DATABASE)
        if args.reset:
            loader.reset()
        loader.constraints()
        loader.drugs(records, aliases)
        loader.structured_edges(records)
        loader.extracted_edges(triplets, alias_to_id)
        loader.build_info(len(records), len(triplets), sorted({t["extracted_by"] for t in triplets}))
        print("  load stats:", dict(loader.stats))
        counts = loader.counts()
        print("  graph nodes:", counts["nodes"])
        print("  graph rels: ", counts["relationships"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
