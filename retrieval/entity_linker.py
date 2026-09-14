"""Entity linking: question text -> graph node ids (DESIGN.md section 5).

This is the precision layer. The graph retriever can only be exact if the
drug names in the question resolve to the right nodes, so this module is
deterministic and conservative: exact, case-insensitive, word-boundary phrase
matching against known surface forms — no fuzzy matching, no LLM.

Sources of surface forms, in priority order:
  1. data/processed/aliases.json — every generic/brand/substance/manual alias
     of a corpus drug (built by ingestion/parse_labels.py, also used by the
     graph loader, so both resolve names identically).
  2. Stub drugs from the graph (drugs another label mentions but we hold no
     label for). Linking these lets "does warfarin interact with disulfiram?"
     reach the edge; they are flagged in_corpus=False so the answer can say
     "mentioned in warfarin's label, but no label for disulfiram itself".
  3. DrugClass and Condition names from the graph, plus a small lay-term map
     ("blood thinner" -> anticoagulant, "kidney" -> renal) so the demo
     question "something for pain" yields a condition keyword.

Longest match wins and matched spans are consumed, so "naproxen sodium" does
not also link "sodium", and "sulfamethoxazole and trimethoprim" links once.

The graph is optional: without it (graph down, or offline tests) only corpus
aliases are used. That is the graceful-degradation path.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

MIN_ALIAS_LEN = 4  # "asa", "ppi" etc. are too short to match safely in free text

# Lay phrasing -> keyword(s) that appear in Condition node names. Matched as
# phrases in the question; the keyword is what the demo query CONTAINS-matches.
CONDITION_SYNONYMS: dict[str, str] = {
    "pain": "pain", "aches": "pain", "headache": "headache", "fever": "fever",
    "blood clot": "thrombosis", "clots": "thrombosis", "clot": "thrombosis",
    "high blood pressure": "hypertension", "blood pressure": "hypertension",
    "cholesterol": "cholesterol", "high cholesterol": "hypercholesterolemia",
    "diabetes": "diabetes", "sugar": "diabetes",
    "infection": "infection", "infections": "infection", "antibiotic": "infection",
    "fungal": "fungal", "yeast infection": "candidiasis", "thrush": "candidiasis",
    "heartburn": "gastroesophageal reflux", "acid reflux": "gastroesophageal reflux", "reflux": "reflux", "ulcer": "ulcer",
    "depression": "depressive", "anxiety": "anxiety",
    "seizure": "seizure", "seizures": "seizure", "epilepsy": "epilep",
    "irregular heartbeat": "fibrillation", "afib": "atrial fibrillation", "a-fib": "atrial fibrillation",
    "heart failure": "heart failure", "arrhythmia": "arrhythmia", "arrhythmias": "arrhythmia",
    "thyroid": "thyroid", "kidney": "renal", "kidneys": "renal", "liver": "hepatic",
    "inflammation": "inflammat", "arthritis": "arthritis", "stroke": "stroke", "pregnancy": "pregnan", "pregnant": "pregnan",
}

CLASS_SYNONYMS: dict[str, str] = {
    "nsaid": "nonsteroidal anti-inflammatory drug", "nsaids": "nonsteroidal anti-inflammatory drug",
    "blood thinner": "anticoagulant", "blood thinners": "anticoagulant", "anticoagulant": "anticoagulant", "anticoagulants": "anticoagulant",
    "statin": "hmg-coa reductase inhibitor", "statins": "hmg-coa reductase inhibitor",
    "antibiotic": "antibacterial", "antibiotics": "antibacterial",
    "antifungal": "antifungal", "antifungals": "antifungal",
    "ssri": "selective serotonin reuptake inhibitor", "ssris": "selective serotonin reuptake inhibitor",
    "ppi": "proton pump inhibitor", "ppis": "proton pump inhibitor", "proton pump inhibitor": "proton pump inhibitor",
    "ace inhibitor": "angiotensin converting enzyme inhibitor", "ace inhibitors": "angiotensin converting enzyme inhibitor",
    "painkiller": "analgesic", "painkillers": "analgesic", "pain reliever": "analgesic",
}


@dataclass
class Entity:
    id: str            # graph id (Drug.id / DrugClass.name / condition keyword)
    kind: str          # "drug" | "drug_class" | "condition"
    name: str          # canonical display name
    surface: str       # the text span in the question that matched
    start: int
    end: int
    in_corpus: bool = True


@dataclass
class LinkResult:
    drugs: list[Entity] = field(default_factory=list)
    classes: list[Entity] = field(default_factory=list)
    conditions: list[Entity] = field(default_factory=list)

    @property
    def drug_ids(self) -> list[str]:
        return [e.id for e in self.drugs]

    @property
    def corpus_drug_ids(self) -> list[str]:
        return [e.id for e in self.drugs if e.in_corpus]

    @property
    def condition_keywords(self) -> list[str]:
        return [e.id for e in self.conditions]

    def as_dict(self) -> dict:
        return {
            "drugs": [vars(e) for e in self.drugs],
            "classes": [vars(e) for e in self.classes],
            "conditions": [vars(e) for e in self.conditions],
        }


class EntityLinker:
    def __init__(self, aliases: dict[str, list[str]], stub_drugs: dict[str, str] | None = None,
                 class_names: list[str] | None = None, condition_names: list[str] | None = None):
        # surface form -> (id, name, in_corpus)
        self._drug_forms: dict[str, tuple[str, str, bool]] = {}
        for drug_id, forms in aliases.items():
            canonical = drug_id.replace("_", " ")
            for f in forms:
                f = f.lower().strip()
                if len(f) >= MIN_ALIAS_LEN:
                    self._drug_forms[f] = (drug_id, canonical, True)
        for stub_id, name in (stub_drugs or {}).items():
            n = name.lower().strip()
            if len(n) >= MIN_ALIAS_LEN and n not in self._drug_forms:
                self._drug_forms[n] = (stub_id, n, False)

        self._class_forms: dict[str, str] = {}
        for name in class_names or []:
            self._class_forms[name.lower()] = name.lower()
        self._class_forms.update(CLASS_SYNONYMS)

        self._condition_forms: dict[str, str] = {}
        for name in condition_names or []:
            if len(name) >= MIN_ALIAS_LEN:
                self._condition_forms[name.lower()] = name.lower()
        self._condition_forms.update(CONDITION_SYNONYMS)

        self._drug_re = self._compile(self._drug_forms)
        self._class_re = self._compile(self._class_forms)
        self._condition_re = self._compile(self._condition_forms)

    @staticmethod
    def _compile(forms: dict) -> re.Pattern | None:
        if not forms:
            return None
        # longest first so alternation prefers "naproxen sodium" over "naproxen"
        alts = sorted((re.escape(f) for f in forms), key=len, reverse=True)
        # allow optional trailing possessive / plural on the surface form
        return re.compile(r"(?<![a-z0-9])(" + "|".join(alts) + r")(?:'s|s)?(?![a-z0-9])", re.I)

    @staticmethod
    def _scan(pattern: re.Pattern | None, text: str, consumed: list[tuple[int, int]]) -> list[re.Match]:
        if pattern is None:
            return []
        out = []
        for m in pattern.finditer(text):
            if any(s <= m.start() < e or s < m.end() <= e for s, e in consumed):
                continue
            consumed.append((m.start(), m.end()))
            out.append(m)
        return out

    def link(self, question: str) -> LinkResult:
        text = question.lower().replace("’", "'")
        consumed: list[tuple[int, int]] = []
        result = LinkResult()
        seen: set[tuple[str, str]] = set()

        for m in self._scan(self._drug_re, text, consumed):
            drug_id, name, in_corpus = self._drug_forms[m.group(1).lower()]
            if ("drug", drug_id) not in seen:
                seen.add(("drug", drug_id))
                result.drugs.append(Entity(drug_id, "drug", name, m.group(0), m.start(), m.end(), in_corpus))

        for m in self._scan(self._class_re, text, consumed):
            cls = self._class_forms[m.group(1).lower()]
            if ("class", cls) not in seen:
                seen.add(("class", cls))
                result.classes.append(Entity(cls, "drug_class", cls, m.group(0), m.start(), m.end()))

        for m in self._scan(self._condition_re, text, consumed):
            kw = self._condition_forms[m.group(1).lower()]
            if ("condition", kw) not in seen:
                seen.add(("condition", kw))
                result.conditions.append(Entity(kw, "condition", kw, m.group(0), m.start(), m.end()))

        return result


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------
def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    p = path or settings.PROCESSED_DIR / "aliases.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _graph_vocab() -> tuple[dict[str, str], list[str], list[str]]:
    """Stub drugs, class names, condition names from Neo4j. Empty on any failure."""
    try:
        from graph import db
        stubs = {r["id"]: r["name"] for r in db.run("MATCH (d:Drug {in_corpus: false}) RETURN d.id AS id, d.name AS name")}
        classes = [r["n"] for r in db.run("MATCH (c:DrugClass) RETURN c.name AS n")]
        conditions = [r["n"] for r in db.run("MATCH (c:Condition) RETURN c.name AS n")]
        return stubs, classes, conditions
    except Exception:  # graph down / no credentials -> corpus aliases only
        return {}, [], []


@lru_cache(maxsize=1)
def get_linker(use_graph: bool = True) -> EntityLinker:
    aliases = load_aliases()
    stubs, classes, conditions = _graph_vocab() if use_graph else ({}, [], [])
    return EntityLinker(aliases, stubs, classes, conditions)


def link(question: str, use_graph: bool = True) -> LinkResult:
    return get_linker(use_graph).link(question)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "I take warfarin and need something for pain. What should I avoid?"
    r = link(q)
    print(q)
    print("  drugs:     ", [(e.id, e.surface, e.in_corpus) for e in r.drugs])
    print("  classes:   ", [(e.id, e.surface) for e in r.classes])
    print("  conditions:", [(e.id, e.surface) for e in r.conditions])
