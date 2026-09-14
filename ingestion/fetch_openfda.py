"""Step 1 of the build phase: pull FDA drug labels from openFDA and save locally.

For each generic name in data/seed_drugs.txt this script:
  1. queries openFDA for prescription labels of that generic (falls back to any
     label, e.g. OTC-only drugs like aspirin);
  2. scores each candidate label by how much usable content it carries
     (interactions, contraindications, indications, adverse reactions, EPC
     class) and keeps the best one;
  3. writes data/raw/<slug>.json — the label exactly as the API returned it,
     plus a small `_meta` block (query, score, fetched_at) — and updates
     data/raw/manifest.json.

Idempotent: re-running overwrites files in place, which is what the refresh job
wants. Keyless openFDA allows 40 req/min; we stay well under that.

Usage:
    python ingestion/fetch_openfda.py                 # all seed drugs
    python ingestion/fetch_openfda.py --only warfarin # one drug
    python ingestion/fetch_openfda.py --candidates 50 # widen the search
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

# Free-text sections we will later parse/extract from. Presence of each is a
# point in the richness score; `drug_interactions` is weighted highest because
# it is the field the whole project is about.
SECTION_WEIGHTS = {
    "drug_interactions": 4,
    "contraindications": 2,
    "indications_and_usage": 2,
    "adverse_reactions": 2,
    "warnings_and_cautions": 1,
    "warnings": 1,
    "boxed_warning": 1,
    "description": 1,
    "clinical_pharmacology": 1,
}
OPENFDA_WEIGHTS = {
    "pharm_class_epc": 2,
    "substance_name": 1,
    "brand_name": 1,
    "rxcui": 1,
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def read_seed_drugs(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def is_combination(label: dict, seed_generic: str) -> bool:
    """True when the label is a multi-ingredient product the seed did not ask for
    (e.g. 'lisinopril and hydrochlorothiazide' for seed 'lisinopril')."""
    openfda = label.get("openfda", {})
    n_seed = seed_generic.lower().count(" and ") + 1
    substances = openfda.get("substance_name", [])
    generics = " ".join(openfda.get("generic_name", [])).lower()
    return len(substances) > n_seed or generics.count(" and ") + generics.count(",") + 1 > n_seed


def score_label(label: dict, seed_generic: str = "") -> int:
    score = 0
    if seed_generic and is_combination(label, seed_generic):
        score -= 25  # any single-ingredient label beats a combination label (its facts would be misattributed)
    for field, weight in SECTION_WEIGHTS.items():
        if label.get(field):
            score += weight
    openfda = label.get("openfda", {})
    for field, weight in OPENFDA_WEIGHTS.items():
        if openfda.get(field):
            score += weight
    if "HUMAN PRESCRIPTION DRUG" in openfda.get("product_type", []):
        score += 3
    # Prefer the most recently revised label among equals.
    return score


def query_openfda(search: str, limit: int, session: requests.Session) -> list[dict]:
    params = {"search": search, "limit": limit}
    if settings.OPENFDA_API_KEY:
        params["api_key"] = settings.OPENFDA_API_KEY
    resp = session.get(settings.OPENFDA_BASE_URL, params=params, timeout=60)
    if resp.status_code == 404:  # openFDA returns 404 for "no matches"
        return []
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_best_label(generic: str, candidates: int, session: requests.Session) -> tuple[dict | None, str]:
    """Return (best_label, search_used).

    Pools prescription-only and any-product-type candidates, then picks the
    highest score. Pooling matters for drugs that are single-ingredient only
    as OTC (acetaminophen, aspirin): the Rx pool holds nothing but combination
    products, and a plain OTC label should beat those.
    """
    name_clause = f'openfda.generic_name:"{generic}"'
    searches = [
        f'{name_clause}+AND+openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
        name_clause,
    ]
    pool: dict[str, tuple[dict, str]] = {}
    for search in searches:
        for r in query_openfda(search, candidates, session):
            pool.setdefault(r.get("id"), (r, search))
        time.sleep(0.3)
    if not pool:
        return None, searches[-1]
    label, search = max(pool.values(), key=lambda t: (score_label(t[0], generic), t[0].get("effective_time", "")))
    return label, search


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="fetch a single generic name from the seed list")
    ap.add_argument("--candidates", type=int, default=100, help="labels to consider per drug per query (max 100)")
    args = ap.parse_args()

    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    drugs = read_seed_drugs(settings.SEED_DRUGS_PATH)
    if args.only:
        drugs = [d for d in drugs if d.lower() == args.only.lower()]
        if not drugs:
            print(f"'{args.only}' is not in {settings.SEED_DRUGS_PATH}", file=sys.stderr)
            return 2

    manifest_path = settings.RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    session = requests.Session()
    failures: list[str] = []

    print(f"openFDA key: {settings.redact(settings.OPENFDA_API_KEY)} | {len(drugs)} drugs -> {settings.RAW_DIR}")
    for generic in drugs:
        slug = slugify(generic)
        try:
            label, search = fetch_best_label(generic, min(args.candidates, 100), session)
        except requests.RequestException as exc:
            print(f"  FAIL {generic}: {exc}")
            failures.append(generic)
            continue
        if label is None:
            print(f"  MISS {generic}: no label found")
            failures.append(generic)
            continue

        openfda = label.get("openfda", {})
        label["_meta"] = {
            "seed_generic": generic,
            "search": search,
            "score": score_label(label, generic),
            "combination_product": is_combination(label, generic),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out = settings.RAW_DIR / f"{slug}.json"
        out.write_text(json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest[slug] = {
            "seed_generic": generic,
            "label_id": label.get("id"),
            "set_id": label.get("set_id"),
            "effective_time": label.get("effective_time"),
            "brand_name": openfda.get("brand_name", [None])[0],
            "generic_name": openfda.get("generic_name", [None])[0],
            "product_type": openfda.get("product_type", [None])[0],
            "pharm_class_epc": openfda.get("pharm_class_epc", []),
            "sections": sorted(f for f in SECTION_WEIGHTS if label.get(f)),
            "score": label["_meta"]["score"],
            "file": out.name,
        }
        flags = " ".join(
            f"{k[:4]}={'Y' if label.get(k) else '-'}"
            for k in ("drug_interactions", "contraindications", "indications_and_usage", "adverse_reactions")
        )
        epc = "epc=Y" if openfda.get("pharm_class_epc") else "epc=-"
        print(f"  ok   {generic:34s} {manifest[slug]['product_type'] or '?':26s} {flags} {epc} score={label['_meta']['score']}")
        time.sleep(0.5)  # polite to the keyless rate limit

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(manifest)} labels; manifest -> {manifest_path}")
    if failures:
        print(f"failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
