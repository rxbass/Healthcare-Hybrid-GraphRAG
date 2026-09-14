"""Neo4j schema for the drug-label graph (DESIGN.md section 4).

Nodes:  Drug, DrugClass, Condition, SideEffect, Ingredient
Edges:  TREATS, INTERACTS_WITH, CAUSES_SIDE_EFFECT, CONTRAINDICATED_FOR,
        BELONGS_TO_CLASS, HAS_INGREDIENT

Every edge carries provenance: `source_label_id`, `set_id`, `source_field`,
`evidence` (the sentence it came from) and `extracted_by` (either "structured"
for openfda fields or the extraction model name). Citations are built from
these properties, so nothing may write an edge without them.

One deliberate extension of the DESIGN.md table: INTERACTS_WITH may target a
DrugClass as well as a Drug, because labels often state interactions at class
level ("NSAIDs", "CYP3A4 inhibitors"). This is what makes the
drug -> class <- member multi-hop in the demo query possible.

Drug nodes have `in_corpus` = true when we hold their FDA label, false when
the node exists only because another label mentions the drug.

Vector side (not a fact node): (:Drug)-[:HAS_CHUNK]->(:Chunk {text, field,
embedding}) holds label prose for the dense-only vector index. Chunks carry
label_id/set_id for citation but never define a fact — see DESIGN.md section 5.
"""

from __future__ import annotations

NODE_LABELS = ("Drug", "DrugClass", "Condition", "SideEffect", "Ingredient")

# predicate -> (object node label(s), source field(s) it is extracted from)
EDGE_TYPES: dict[str, dict] = {
    "TREATS": {"object": ("Condition",), "fields": ("indications_and_usage",)},
    "INTERACTS_WITH": {"object": ("Drug", "DrugClass"), "fields": ("drug_interactions", "contraindications")},
    "CAUSES_SIDE_EFFECT": {"object": ("SideEffect",), "fields": ("adverse_reactions",)},
    "CONTRAINDICATED_FOR": {"object": ("Condition",), "fields": ("contraindications",)},
    "BELONGS_TO_CLASS": {"object": ("DrugClass",), "fields": ("openfda.pharm_class_epc",)},
    "HAS_INGREDIENT": {"object": ("Ingredient",), "fields": ("openfda.substance_name",)},
}

# Free-text label sections the LLM extractor reads, and the predicates it may
# emit from each. Keeping this explicit stops the extractor inventing an edge
# type from the wrong section (e.g. a TREATS from adverse_reactions).
EXTRACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "indications_and_usage": ("TREATS",),
    "drug_interactions": ("INTERACTS_WITH",),
    "contraindications": ("CONTRAINDICATED_FOR", "INTERACTS_WITH"),
    "adverse_reactions": ("CAUSES_SIDE_EFFECT",),
}

PROVENANCE_KEYS = ("source_label_id", "set_id", "source_field", "evidence", "extracted_by")

# Idempotent: CREATE ... IF NOT EXISTS.
CONSTRAINTS = [
    "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT drugclass_name IF NOT EXISTS FOR (c:DrugClass) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT sideeffect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT buildinfo_id IF NOT EXISTS FOR (b:BuildInfo) REQUIRE b.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
]

VECTOR_INDEX_NAME = "chunk_embeddings"
VECTOR_DIMENSIONS = 1536  # text-embedding-3-small
VECTOR_INDEX = f"""
CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}
"""
