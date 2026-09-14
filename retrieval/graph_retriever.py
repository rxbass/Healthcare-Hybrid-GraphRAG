"""Graph retriever: linked entities -> exact facts with provenance.

This is the source-of-truth side of the hybrid (DESIGN.md section 5). Given the
entity linker's output it runs, in order:

  * pair queries for every pair of linked drugs — and records explicitly when
    the graph holds NO interaction for a pair. "No documented interaction" is a
    graph fact the generator must state and the vector text must not override.
  * the multi-hop demo query for (drug, condition keyword) combinations.
  * per-drug facts (treats / contraindications / classes / interactions /
    side effects), capped per predicate so a single drug's 150 adverse
    reactions cannot crowd the context.

Every fact keeps source_label_id + evidence for citation.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import queries  # noqa: E402
from retrieval.entity_linker import LinkResult  # noqa: E402

FACT_CAPS = {
    "TREATS": 15,
    "CONTRAINDICATED_FOR": 15,
    "BELONGS_TO_CLASS": 5,
    "INTERACTS_WITH": 40,
    "CAUSES_SIDE_EFFECT": 30,
}

# Which predicates a question is asking about. This is fact *selection* on the
# graph side (both retrievers still always run — no router); it keeps a
# two-drug question from dragging 60 unrelated adverse reactions into context.
# A question matching nothing gets the descriptive set (what it is / treats).
FOCUS_PATTERNS = {
    "INTERACTS_WITH": r"interact|together|combin|with (my|other)|avoid|safe (to take )?with|mix|concomitant|affect",
    "TREATS": r"treat|(?<!contra)indicat|used for|prescribed for|use of|what is .* for|help with|good for|overview|kind of drug|what is",
    "CONTRAINDICATED_FOR": r"contraindicat|should not|shouldn't|who (can|should) not|not (be )?used|when .* not",
    "CAUSES_SIDE_EFFECT": r"side effect|adverse|reaction|cause|risk",
    "BELONGS_TO_CLASS": r"class|kind of drug|type of drug|is .* an? |what is",
}
DESCRIPTIVE_FOCUS = ("TREATS", "BELONGS_TO_CLASS")


def focus(question: str) -> set[str]:
    q = question.lower()
    hits = {p for p, pat in FOCUS_PATTERNS.items() if re.search(pat, q)}
    return hits or set(DESCRIPTIVE_FOCUS)


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    object_type: str
    source_label_id: str
    source_field: str
    evidence: str
    via: str = "direct"
    set_id: str | None = None
    source_drug: str | None = None  # drug id whose label the evidence sentence is from

    def key(self) -> tuple:
        return (self.subject, self.predicate, self.object, self.source_label_id, self.evidence)


@dataclass
class GraphResult:
    facts: list[Fact] = field(default_factory=list)
    focus: set[str] = field(default_factory=set)
    no_interaction_pairs: list[tuple[str, str]] = field(default_factory=list)
    demo_hits: list[dict] = field(default_factory=list)   # raw rows of the multi-hop query
    drugs: dict[str, dict] = field(default_factory=dict)  # drug id -> node props (label id etc.)
    error: str | None = None                              # set when the graph was unreachable

    @property
    def ok(self) -> bool:
        return self.error is None


def _add(result: GraphResult, seen: set, fact: Fact) -> None:
    if fact.key() not in seen:
        seen.add(fact.key())
        result.facts.append(fact)


def retrieve(link: LinkResult, question: str = "") -> GraphResult:
    result = GraphResult()
    seen: set = set()
    wanted = focus(question) if question else set(FACT_CAPS)
    result.focus = wanted
    pair_question = len(link.drug_ids) >= 2
    try:
        drug_ids = link.drug_ids
        for d in drug_ids:
            node = queries.drug(d)
            if node:
                result.drugs[d] = node

        # 1. pairwise interactions (both directions, direct or via class)
        for a, b in combinations(drug_ids, 2):
            row = queries.pair(a, b)
            if row and row["interactions"]:
                for i in row["interactions"]:
                    _add(result, seen, Fact(row["drug_a"], "INTERACTS_WITH", row["drug_b"], "Drug",
                                            i["source_label_id"], i["source_field"], i["evidence"], i["via"], i.get("set_id"), i.get("source_drug")))
            else:
                result.no_interaction_pairs.append((a, b))

        # 2. multi-hop: drug + condition -> drugs to avoid
        for d in link.corpus_drug_ids:
            for kw in link.condition_keywords:
                for row in queries.demo(d, kw):
                    result.demo_hits.append({"drug": d, "condition": kw, **row})
                    for t in row["treats"][:2]:
                        _add(result, seen, Fact(row["drug"], "TREATS", t["condition"], "Condition",
                                                t["source_label_id"], "indications_and_usage", t["evidence"], "direct", t.get("set_id"), t.get("source_drug")))
                    for i in row["interactions"][:3]:
                        _add(result, seen, Fact(result.drugs.get(d, {}).get("name", d), "INTERACTS_WITH", row["drug"], "Drug",
                                                i["source_label_id"], i["source_field"], i["evidence"], i["via"], i.get("set_id"), i.get("source_drug")))

        # 3. per-drug facts for the predicates the question asks about, capped.
        #    When the pair query or the multi-hop query already answered the
        #    interaction part, a drug's interactions with *third* drugs are noise.
        skip_fanout = pair_question or bool(result.demo_hits)
        for d in link.corpus_drug_ids:
            counts: dict[str, int] = {}
            for f in queries.facts(d, tuple(wanted)):
                p = f["predicate"]
                if p == "INTERACTS_WITH" and skip_fanout:
                    continue
                counts[p] = counts.get(p, 0) + 1
                if counts[p] > FACT_CAPS.get(p, 20):
                    continue
                _add(result, seen, Fact(f["subject"], p, f["object"], f["object_type"],
                                        f["source_label_id"], f["source_field"], f["evidence"], "direct", f.get("set_id"), f.get("source_drug")))
    except Exception as exc:  # graph down -> caller degrades to vectors-only
        result.error = f"{type(exc).__name__}: {exc}"
    return result
