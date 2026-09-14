"""Step 2 of the build phase: parse raw openFDA labels into clean, split records.

Reads data/raw/<slug>.json and writes:
  data/processed/labels.jsonl   one record per drug: identity, structured
                                openfda facts (classes, ingredients, brands)
                                and the cleaned free-text sections the
                                extractor and the vector index consume.
  data/processed/aliases.json   drug id -> every surface form we know
                                (generic, brand, substance, manual aliases).
                                Shared by the graph loader and the entity
                                linker so both resolve names identically.

Structured fields become nodes/edges without an LLM (cheap, exact). Free-text
fields hold the valuable edges and go to extract_triplets.py.

Usage:
    python ingestion/parse_labels.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from graph.schema import EXTRACTION_FIELDS  # noqa: E402

# Free-text sections kept for the vector index (superset of EXTRACTION_FIELDS).
TEXT_FIELDS = (
    "indications_and_usage",
    "drug_interactions",
    "contraindications",
    "adverse_reactions",
    "warnings_and_cautions",
    "warnings",
    "boxed_warning",
    "description",
    "clinical_pharmacology",
    "mechanism_of_action",
)
assert set(EXTRACTION_FIELDS) <= set(TEXT_FIELDS)

MANUAL_ALIASES_PATH = settings.DATA_DIR / "manual_aliases.json"

_SECTION_HEADER = re.compile(r"^\s*(\d+(\.\d+)?\s+)?[A-Z][A-Z &/,\-]{4,}\s*(?=[A-Z0-9(])")
_WS = re.compile(r"\s+")


def clean_text(chunks: list[str]) -> str:
    """Join a label field's paragraphs, strip the leading 'N SECTION TITLE', collapse whitespace."""
    text = " ".join(chunks)
    text = _WS.sub(" ", text).strip()
    m = _SECTION_HEADER.match(text)
    if m and len(m.group(0)) < 60:
        text = text[m.end():].strip()
    return text


def normalise_name(name: str) -> str:
    return _WS.sub(" ", name.strip().lower())


def strip_epc(name: str) -> str:
    """'Nonsteroidal Anti-inflammatory Drug [EPC]' -> 'nonsteroidal anti-inflammatory drug'."""
    return normalise_name(re.sub(r"\s*\[EPC\]\s*$", "", name))


def build_aliases(drug_id: str, seed_generic: str, openfda: dict, manual: dict[str, list[str]]) -> list[str]:
    forms = {seed_generic}
    forms.update(openfda.get("generic_name", []))
    forms.update(openfda.get("brand_name", []))
    forms.update(openfda.get("substance_name", []))
    forms.update(manual.get(drug_id, []))
    # Salt-stripped variants: "warfarin sodium" -> "warfarin". Only common salts.
    salts = r"\s+(sodium|hydrochloride|hcl|potassium|calcium|sulfate|acetate|maleate|tartrate|mesylate|citrate)$"
    for f in list(forms):
        stripped = re.sub(salts, "", normalise_name(f))
        if stripped:
            forms.add(stripped)
    return sorted({normalise_name(f) for f in forms if f and len(normalise_name(f)) >= 3})


def parse_one(label: dict) -> dict:
    meta = label["_meta"]
    openfda = label.get("openfda", {})
    seed_generic = meta["seed_generic"]
    drug_id = re.sub(r"[^a-z0-9]+", "_", seed_generic.lower()).strip("_")
    sections = {f: clean_text(label[f]) for f in TEXT_FIELDS if label.get(f)}
    return {
        "id": drug_id,
        "name": normalise_name(seed_generic),
        "label_id": label.get("id"),
        "set_id": label.get("set_id"),
        "effective_time": label.get("effective_time"),
        "version": label.get("version"),
        "brand_names": sorted({normalise_name(b) for b in openfda.get("brand_name", [])}),
        "generic_names": sorted({normalise_name(g) for g in openfda.get("generic_name", [])}),
        "product_type": (openfda.get("product_type") or [None])[0],
        "rxcui": openfda.get("rxcui", []),
        "classes": sorted({strip_epc(c) for c in openfda.get("pharm_class_epc", [])}),
        "ingredients": sorted({normalise_name(s) for s in openfda.get("substance_name", [])}),
        "sections": sections,
    }


def main() -> int:
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manual = {k: v for k, v in json.loads(MANUAL_ALIASES_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    raw_files = sorted(p for p in settings.RAW_DIR.glob("*.json") if p.name != "manifest.json")
    if not raw_files:
        print(f"no raw labels in {settings.RAW_DIR}; run ingestion/fetch_openfda.py first", file=sys.stderr)
        return 1

    records, aliases = [], {}
    for path in raw_files:
        label = json.loads(path.read_text(encoding="utf-8"))
        rec = parse_one(label)
        records.append(rec)
        aliases[rec["id"]] = build_aliases(rec["id"], label["_meta"]["seed_generic"], label.get("openfda", {}), manual)
        n_chars = sum(len(t) for t in rec["sections"].values())
        print(f"  {rec['id']:34s} sections={len(rec['sections']):2d} chars={n_chars:6d} classes={len(rec['classes'])} aliases={len(aliases[rec['id']])}")

    out = settings.PROCESSED_DIR / "labels.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (settings.PROCESSED_DIR / "aliases.json").write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8")

    # Aliases must be unambiguous: one surface form -> one drug.
    seen: dict[str, str] = {}
    for drug_id, forms in aliases.items():
        for f in forms:
            if f in seen and seen[f] != drug_id:
                print(f"  WARNING ambiguous alias '{f}': {seen[f]} vs {drug_id}", file=sys.stderr)
            seen[f] = drug_id
    print(f"\nwrote {len(records)} records -> {out}; {len(seen)} aliases -> aliases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
