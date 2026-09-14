"""Single place that reads configuration from the environment.

Every other module imports from here — nothing else touches os.environ, and
nothing here has a literal secret. Values come from the process environment or
a git-ignored `.env` file in the project root (loaded via python-dotenv).

Required settings are validated lazily via `require()` so that offline scripts
(e.g. the openFDA fetch) can run without an OpenAI key or Neo4j credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ---- paths (not secrets, but centralised so scripts agree) ----
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.jsonl"
SEED_DRUGS_PATH = DATA_DIR / "seed_drugs.txt"
LOG_DIR = PROJECT_ROOT / os.environ.get("LOG_DIR", "logs")


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or default


def require(name: str) -> str:
    """Return an env var or raise a clear error naming the missing key."""
    value = _get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


# ---- OpenAI ----
def openai_api_key() -> str:
    return require("OPENAI_API_KEY")


EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = _get("LLM_MODEL", "gpt-4o-mini")
JUDGE_MODEL = _get("JUDGE_MODEL", "gpt-4o")


# ---- Neo4j ----
def neo4j_uri() -> str:
    return require("NEO4J_URI")


def neo4j_username() -> str:
    return require("NEO4J_USERNAME")


def neo4j_password() -> str:
    return require("NEO4J_PASSWORD")


NEO4J_DATABASE = _get("NEO4J_DATABASE", "neo4j")


# ---- openFDA ----
OPENFDA_BASE_URL = _get("OPENFDA_BASE_URL", "https://api.fda.gov/drug/label.json")
OPENFDA_API_KEY = _get("OPENFDA_API_KEY")  # optional


def redact(value: str | None) -> str:
    """For logging: never print a secret, only whether it is set."""
    return "<set>" if value else "<missing>"
