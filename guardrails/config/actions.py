"""Custom NeMo action: deterministic scope check (DESIGN.md section 6).

"No dosing, diagnosis, or personal medical advice" is enforced here as a rail,
not as a prompt instruction, so the refusal is deterministic and testable
offline. The generation prompt and the grounding gate remain as second and
third lines of defence.

Returned reason drives which refusal message the Colang flow emits:
  dosing | diagnosis | treatment | off_topic
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from nemoguardrails.actions import action

DOSING = re.compile(
    r"\b(how (much|many|often)|what dose|dosage|dosing|dose (of|for|should)|max(imum)? dose|"
    r"(double|halve|increase|reduce|lower|raise) (my|the) .{0,30}(dose|pill|tablet)|milligrams?|\d+\s*mg\b|mcg\b|micrograms?|"
    r"(take|taking) .{0,40}(per|a|each) day|how many .{0,30}(tablets?|pills?)|times a day|can i take (more|less|extra)|"
    r"dose .{0,30}survive|lethal (dose|amount)|overdose (amount|threshold))",
    re.I,
)
DIAGNOSIS = re.compile(
    r"\b(what (condition|disease|illness) (do|might|could) i have|do i have\b|diagnos|"
    r"what('s| is) wrong with me|what could (this|it) be|is (this|it) (serious|cancer|a heart attack)|"
    r"(i have|i've got|i am having|i'm having|experiencing) .{0,80}(what|which) (condition|disease|is it|do i))",
    re.I,
)
TREATMENT = re.compile(
    r"\b(should i (stop|start|skip|take|switch|continue|keep taking|double|reduce|increase)|"
    r"can i (stop|skip|switch|combine|take .{0,30}(with|and) my)|"
    r"(which|what) (is|one is|drug is|medication is) (better|best|safer|right|safest) for me|"
    r"(is it (safe|ok|okay) (for me )?to (take|stop|skip|combine))|"
    r"(what|which) should i (take|use)|recommend .{0,30}for me|prescribe me|for my (condition|symptoms))",
    re.I,
)
# In-scope vocabulary: anything about medications is in scope even when no
# drug name is recognised (the pipeline then answers "not found").
MEDICAL = re.compile(
    r"\b(drug|drugs|medication|medicine|medicines|pill|tablet|capsule|prescri\w*|pharmac\w*|label|interact\w*|"
    r"side effects?|adverse|contraindicat\w*|indicat\w*|treat\w*|dose|dosing|ingredient|generic|brand|"
    r"anticoagulant|antibiotic|antifungal|statin|nsaid|inhibitor|blocker|"
    r"[a-z]{3,}(mab|nib|pril|sartan|olol|statin|azole|mycin|cillin|cycline|floxacin|prazole|tidine|dipine|"
    r"parin|gliptin|glutide|formin|tinib|zepam|zolam|oxetine|pramine|triptan|dronate|lukast|profen|"
    r"aban|grel|arone|oxin|nidazole|toin|mazepine|thyroxine|isone|olone|coxib|semide|terol))\b",
    re.I,
)

_LINKER = None


def _names_a_known_drug(text: str) -> bool:
    """Any corpus drug alias in the text puts the question in scope (offline alias table)."""
    global _LINKER
    try:
        if _LINKER is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from retrieval.entity_linker import EntityLinker, load_aliases
            _LINKER = EntityLinker(load_aliases())
        return bool(_LINKER.link(text).drugs)
    except Exception:
        return False


def scope_of(text: str) -> dict:
    t = text or ""
    if DOSING.search(t):
        return {"blocked": True, "reason": "dosing"}
    if DIAGNOSIS.search(t):
        return {"blocked": True, "reason": "diagnosis"}
    if TREATMENT.search(t):
        return {"blocked": True, "reason": "treatment"}
    if not MEDICAL.search(t) and not _names_a_known_drug(t):
        return {"blocked": True, "reason": "off_topic"}
    return {"blocked": False, "reason": ""}


@action(name="check_scope")
async def check_scope(text: str = "") -> dict:
    return scope_of(text)
