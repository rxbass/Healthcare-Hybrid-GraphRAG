"""Render the gated answer with citations back to FDA labels.

Every kept claim shows its [F#]/[N#] ids inline; a Sources section maps each id
to the label (DailyMed link by set id when known), the section, and the exact
evidence sentence. Background [C#] claims are shown separately and labelled as
label text, not graph facts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generation.generate import Claim  # noqa: E402
from generation.grounding_gate import GateReport  # noqa: E402
from retrieval.merge import MergedContext  # noqa: E402

DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"
FOOTER = ("Informational only — documented FDA label information, not medical advice. "
          "Talk to a pharmacist or clinician about your own medications.")

REFUSAL_TEXT = ("I can only report documented information from FDA drug labels — I can't give dosing, diagnose "
                "symptoms, or advise on starting, stopping or choosing a medication. Please ask a pharmacist or clinician.")
NOT_FOUND_TEXT = "I couldn't find that in the FDA labels I hold, so I can't answer it."
# A not-found must never be worded as an absence-of-interaction finding.
NO_INTERACTION_WORDING = re.compile(r"no (documented |known )?interaction|does not interact|do(es)? not document (any )?interaction|not documented to interact", re.I)


def _ids(claim: Claim) -> list[str]:
    return [i.strip("[] ").upper() for i in claim.fact_ids + claim.context_ids]


def _source_line(cid: str, c: dict) -> str:
    if cid.startswith("N"):
        return f"[{cid}] Graph query result — {c['quote']}."
    link = DAILYMED.format(set_id=c["set_id"]) if c.get("set_id") else f"FDA label {c['label_id']}"
    return f"[{cid}] {c.get('label_drug', c['drug'])} label, section {c['field']} — {link}\n      \"{c['quote']}\""


def render(summary: str, report: GateReport, ctx: MergedContext, built_on: str | None = None) -> str:
    cites = ctx.citations()
    parts: list[str] = []

    if report.status == "refuse":
        parts.append(summary.strip() or REFUSAL_TEXT)
    elif report.status == "not_found":
        # Deterministic wording: "not in my data" must never read as "no interaction".
        parts.append(NOT_FOUND_TEXT)
        known = [e.name for e in ctx.link.drugs if e.in_corpus]
        if known:
            parts.append(f"Drugs I recognised in your question: {', '.join(known)}. Anything else named is not among "
                         f"the FDA labels held, so I can't say whether it interacts with them.")
        else:
            parts.append("None of the drugs or products named are among the FDA labels held.")
        if summary.strip() and not NO_INTERACTION_WORDING.search(summary) and not report.dropped:
            parts.append(summary.strip())
        if report.dropped:
            parts.append("(Some statements were removed because the graph facts did not support them.)")
    else:
        if summary.strip():
            parts.append(summary.strip())
        parts.append("")
        for claim in report.kept:
            parts.append(f"• {claim.statement.strip()} " + " ".join(f"[{i}]" for i in _ids(claim)))
        if report.background:
            parts.append("\nBackground from the label text (descriptive, not a graph fact):")
            for claim in report.background:
                parts.append(f"  – {claim.statement.strip()} " + " ".join(f"[{i}]" for i in _ids(claim)))
        used = []
        for claim in report.kept + report.background:
            for i in _ids(claim):
                if i in cites and i not in used:
                    used.append(i)
        if used:
            parts.append("\nSources:")
            for i in sorted(used, key=lambda x: (x[0], int(x[1:]))):
                parts.append(_source_line(i, cites[i]))

    if ctx.degraded:
        parts.append("\nNote: " + "; ".join(ctx.degraded) + " — answer may be incomplete.")
    stamp = f" Graph built on {built_on[:10]}." if built_on else ""
    parts.append(f"\n{FOOTER}{stamp}")
    return "\n".join(parts)


def render_facts_only(ctx: MergedContext, built_on: str | None = None) -> str:
    """LLM unavailable: return the raw graph facts with citations, no prose."""
    lines = ["The language model is unavailable; here are the documented facts retrieved for your question:", ""]
    for i, f in enumerate(ctx.facts, 1):
        lines.append(f"• {f.subject} — {f.predicate.replace('_', ' ').lower()} — {f.object} [F{i}]")
    for i, (a, b) in enumerate(ctx.nones, 1):
        lines.append(f"• no documented interaction between {a} and {b} [N{i}]")
    if len(lines) == 2:
        lines.append(NOT_FOUND_TEXT)
    cites = ctx.citations()
    lines.append("\nSources:")
    for cid in [k for k in cites if k[0] in "FN"]:
        lines.append(_source_line(cid, cites[cid]))
    stamp = f" Graph built on {built_on[:10]}." if built_on else ""
    lines.append(f"\n{FOOTER}{stamp}")
    return "\n".join(lines)
