"""Custom graph-grounding gate — the domain truth check (DESIGN.md section 6).

Deterministic checks on the structured Answer against the merged context.
No off-the-shelf tool knows the Neo4j graph is the source of truth, so this is
hand-written. It fails CLOSED: anything it cannot verify is removed, and an
answer with nothing verifiable left becomes an honest "not found".

Checks, per claim:
  1. Every cited id must exist in the context. A made-up [F9] is a hallucinated
     citation -> claim dropped.
  2. A factual claim must cite at least one [F#]/[N#]. A claim with only [C#]
     is background prose: allowed only if it does not assert an interaction,
     contraindication, indication or adverse reaction (those are graph-only
     facts). Otherwise dropped — this is the "vector text cannot override the
     graph" rule, enforced.
  3. A cited fact must actually be about what the claim says: the claim must
     name the fact's object, or share at least two content words with the
     fact's evidence sentence (beyond the drug name). Mentioning only the drug
     is not enough, so citing a random [F#] to launder an unsupported sentence
     fails.
  4. A claim that asserts an interaction between a pair the graph says has none
     ([N#] pair) is dropped, whatever it cites.
  5. Dosing leakage: a claim containing a numeric dose (e.g. "5 mg", "twice
     daily") is dropped — informational only, never dosing.
  6. status="answer" with no surviving factual claim -> status becomes
     not_found. status="refuse"/"not_found" must carry no claims.

Recoverable misses: the caller may retry generation once with the problems
listed; on the second pass whatever still fails is dropped for good.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generation.generate import Answer, Claim  # noqa: E402
from retrieval.merge import MergedContext  # noqa: E402

FACT_WORDS = re.compile(
    r"interact|contraindicat|indicated|treats?|used (for|to treat)|adverse|side effect|causes?|"
    r"avoid|should not|together|combin|concomitant|co-?administ|risk|bleed|hemorrhag|toxic|prolong|"
    r"increase|decrease|reduce|raise|lower|potentiat|inhibit|induce|belongs to|class of|not documented", re.I)
# A dose is a quantity of drug; a lab threshold ("eGFR below 30 mL/min", "creatinine 1.5 mg/dL") is not.
DOSE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|µg|g|ml|mL|units?|iu)\b(?!\s*/)|\b(once|twice|three times|[1-9]x)\s+(a|per)\s+day\b|\bevery \d+ hours\b|\bq\d+h\b", re.I)
_ID = re.compile(r"^[FNC]\d+$")
# Trailing speculation the model sometimes appends to relate a fact to the question
# ("..., which may be relevant to kidney issues"). The fact stays; the hedge goes.
SPECULATION = re.compile(
    r",?\s+(which|this|that)\s+(may|might|could|can)\s+(also\s+)?(be\s+)?(relevant|related|relate|apply|occur|matter|connected|linked|pertain)\b[^.]*",
    re.I)
_STOP = {"the", "and", "with", "for", "of", "in", "to", "a", "an", "is", "are", "or", "drug", "drugs", "tablets", "sodium"}


@dataclass
class GateReport:
    passed: bool
    status: str                                  # final status after gating
    kept: list[Claim] = field(default_factory=list)
    dropped: list[tuple[Claim, str]] = field(default_factory=list)
    background: list[Claim] = field(default_factory=list)  # [C#]-only descriptive claims, kept but labelled

    @property
    def problems(self) -> str:
        return "; ".join(f"'{c.statement[:60]}…': {why}" for c, why in self.dropped)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 2}


def _mentions(statement: str, *names: str | None) -> bool:
    st = _tokens(statement)
    for n in names:
        if n and (_tokens(n) & st):
            return True
    return False


def _supports(statement: str, cite: dict) -> bool:
    """A cited fact supports a claim if the claim names the fact's object, or shares
    >= 2 content words (len > 4, not the drug name) with the evidence sentence.
    Mentioning only the drug is not enough — that is how a vector-only statement
    gets laundered through a real [F#]."""
    if _mentions(statement, cite.get("object")):
        return True
    drug_tokens = _tokens(cite.get("drug") or "") | _tokens(cite.get("label_drug") or "")
    st = {t for t in _tokens(statement) if len(t) > 4} - drug_tokens
    ev = {t for t in _tokens(cite.get("quote") or "") if len(t) > 4} - drug_tokens
    return len(st & ev) >= 2


def _asserts_interaction_between(statement: str, a: str, b: str) -> bool:
    s = statement.lower()
    return ("interact" in s or "combin" in s or "together" in s) and _mentions(s, a) and _mentions(s, b) \
        and not re.search(r"\bno\b|\bnot\b|none|without", s)


def check(answer: Answer, ctx: MergedContext) -> GateReport:
    cites = ctx.citations()
    none_pairs = [(cites[k]["drug"], cites[k]["object"]) for k in cites if k.startswith("N")]

    if answer.status in ("refuse", "not_found"):
        # never leak facts through a refusal / not-found
        return GateReport(passed=True, status=answer.status, kept=[], dropped=[(c, f"claims not allowed with status={answer.status}") for c in answer.claims])

    report = GateReport(passed=True, status="answer")
    for claim in answer.claims:
        cleaned = SPECULATION.sub("", claim.statement).strip()
        if cleaned and cleaned != claim.statement:
            claim.statement = cleaned.rstrip(",;") + ("" if cleaned.endswith(".") else ".")
        ids = [i.strip("[] ").upper() for i in claim.fact_ids + claim.context_ids]
        bad = [i for i in ids if not _ID.match(i) or i not in cites]
        if bad:
            report.dropped.append((claim, f"unknown citation {bad}"))
            continue
        if DOSE_PATTERN.search(claim.statement):
            report.dropped.append((claim, "contains dosing"))
            continue
        fact_ids = [i for i in ids if i[0] in "FN"]
        if any(_asserts_interaction_between(claim.statement, a, b) for a, b in none_pairs):
            report.dropped.append((claim, "asserts an interaction the graph says is not documented"))
            continue
        if not fact_ids:
            linked = [e.name for e in ctx.link.drugs]
            names_two = sum(1 for n in linked if _mentions(claim.statement, n)) >= 2
            if FACT_WORDS.search(claim.statement) or names_two:
                report.dropped.append((claim, "factual claim supported only by background text"))
            else:
                report.background.append(claim)
            continue
        supported = any(_supports(claim.statement, cites[i]) for i in fact_ids)
        if not supported:
            report.dropped.append((claim, f"cited facts {fact_ids} are not about this statement"))
            continue
        report.kept.append(claim)

    if not report.kept:
        report.passed = False
        report.status = "not_found"
    return report
