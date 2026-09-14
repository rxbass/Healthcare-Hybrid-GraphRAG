"""LCEL generation chain: merged context + question -> structured Answer.

Model-agnostic in the LCEL sense: swap `ChatOpenAI` for any chat model that
supports structured output. Returns the parsed Answer *and* the raw message so
token usage can be logged (with_structured_output(include_raw=True)).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from generation.prompts import RETRY_SUFFIX, SYSTEM_PROMPT, USER_PROMPT  # noqa: E402


class Claim(BaseModel):
    statement: str = Field(description="One short sentence stating one fact.")
    fact_ids: list[str] = Field(default_factory=list, description="[F#]/[N#] ids that support the statement, e.g. ['F1','N1'].")
    context_ids: list[str] = Field(default_factory=list, description="[C#] ids used for wording, e.g. ['C2'].")


class Answer(BaseModel):
    status: Literal["answer", "not_found", "refuse"]
    summary: str
    claims: list[Claim] = Field(default_factory=list)


@lru_cache(maxsize=4)
def build_chain(model_name: str | None = None):
    llm = ChatOpenAI(model=model_name or settings.LLM_MODEL, temperature=0, api_key=settings.openai_api_key())
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT + "{retry}")])
    return prompt | llm.with_structured_output(Answer, include_raw=True)


def generate(context: str, question: str, problems: str = "", model_name: str | None = None) -> tuple[Answer, dict]:
    """Return (Answer, usage) where usage = {input_tokens, output_tokens, total_tokens, model}."""
    retry = RETRY_SUFFIX.format(problems=problems) if problems else ""
    out = build_chain(model_name).invoke({"context": context, "question": question, "retry": retry})
    if out.get("parsing_error") or out.get("parsed") is None:
        raise ValueError(f"structured output failed: {out.get('parsing_error')}")
    raw = out["raw"]
    usage = dict(getattr(raw, "usage_metadata", None) or {})
    usage["model"] = model_name or settings.LLM_MODEL
    return out["parsed"], usage
