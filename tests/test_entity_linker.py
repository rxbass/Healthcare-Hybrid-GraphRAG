"""Entity linking must resolve exactly the drugs the golden set expects — no more, no less.

Offline: uses data/processed/aliases.json only (no graph), so it runs in CI
without credentials. Precision matters as much as recall: a spurious drug
would pull wrong facts into the context.
"""

import json

import pytest

from config import settings
from retrieval.entity_linker import EntityLinker, load_aliases

GOLDEN = [json.loads(l) for l in settings.GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def linker():
    return EntityLinker(load_aliases())


@pytest.mark.parametrize("row", GOLDEN, ids=[r["id"] for r in GOLDEN])
def test_golden_entities_exact(linker, row):
    got = sorted(linker.link(row["question"]).drug_ids)
    expected = sorted(e.replace(" ", "_") for e in row["expected_entities"])
    assert got == expected, f"{row['id']}: linked {got}, expected {expected}"


@pytest.mark.parametrize("question,expected", [
    ("Does Coumadin interact with Advil?", ["warfarin", "ibuprofen"]),
    ("Is Bactrim safe with warfarin?", ["sulfamethoxazole_and_trimethoprim", "warfarin"]),
    ("warfarin's interactions", ["warfarin"]),
    ("WARFARIN SODIUM tablets", ["warfarin"]),
    ("naproxen sodium and ibuprofens", ["naproxen", "ibuprofen"]),
    ("Tell me about Tylenol.", ["acetaminophen"]),
    ("Does aspirinase exist?", []),          # no partial-word match
    ("What is the capital of France?", []),
])
def test_aliases_and_boundaries(linker, question, expected):
    assert sorted(linker.link(question).drug_ids) == sorted(expected)


def test_demo_question_yields_condition_keyword(linker):
    r = linker.link("I take warfarin and need something for pain. What should I avoid?")
    assert r.drug_ids == ["warfarin"]
    assert "pain" in r.condition_keywords


def test_class_mentions(linker):
    r = linker.link("Is ibuprofen an NSAID?")
    assert r.drug_ids == ["ibuprofen"]
    assert [c.id for c in r.classes] == ["nonsteroidal anti-inflammatory drug"]
