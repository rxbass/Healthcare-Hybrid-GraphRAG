"""Query logging and the ops report — offline."""

import json

from eval.report import build, to_markdown
from observability import logger
from pipeline import Response


def test_record_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(logger.settings, "LOG_DIR", tmp_path)
    resp = Response("My name is <PERSON>. Does warfarin interact with aspirin?", "answer", "…", stats={
        "linked_drugs": ["warfarin", "aspirin"], "graph_facts": 1, "vector_chunks": 6, "pii_masked": True,
        "gate_status": "answer", "gate_passed": True, "claims_kept": 1, "claims_dropped": 0, "gate_retries": 0,
        "total_ms": 3000, "generate_ms": 2000, "usage": [{"model": "gpt-4o-mini", "input_tokens": 3000, "output_tokens": 100}],
    })
    rec = logger.record(resp)
    assert rec["pii_masked"] and rec["question"].startswith("My name is <PERSON>")
    assert rec["cost_usd"] > 0 and rec["tokens"]["input"] == 3000
    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["status"] == "answer"

    refused = Response("How many mg?", "refuse", "…", stats={"blocked_by": "check scope", "total_ms": 20, "input_rails_ms": 20})
    logger.record(refused)
    rep = build(logger.read_log(tmp_path / "queries.jsonl"))
    assert rep["queries"] == 2 and rep["status_mix"] == {"answer": 1, "refuse": 1}
    assert rep["blocked_by"] == {"check scope": 1} and rep["cost_usd"]["per_answered_query"] > 0
    md = to_markdown(rep, tmp_path / "queries.jsonl")
    assert "p50" in md and "Grounding gate" in md


def test_logging_never_breaks_answering(monkeypatch):
    from guardrails import rails
    monkeypatch.setattr(rails, "_guarded_answer", lambda q: Response(q, "answer", "ok", stats={}))
    monkeypatch.setattr(rails.logger, "record", lambda r: (_ for _ in ()).throw(OSError("disk full")))
    r = rails.guarded_answer("x")
    assert r.text == "ok" and "disk full" in r.stats["log_error"]


def test_price_override(monkeypatch):
    monkeypatch.setenv("PRICE_TABLE_JSON", json.dumps({"gpt-4o-mini": [1000.0, 0.0]}))
    assert logger.estimate_cost([{"model": "gpt-4o-mini", "input_tokens": 1_000_000}]) >= 1000.0
