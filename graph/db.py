"""Shared Neo4j driver (credentials from config.settings only)."""

from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase

from config import settings


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    driver = GraphDatabase.driver(
        settings.neo4j_uri(),
        auth=(settings.neo4j_username(), settings.neo4j_password()),
        notifications_min_severity="OFF",  # server deprecation notices are not actionable at runtime
    )
    driver.verify_connectivity()
    return driver


def run(cypher: str, **params) -> list[dict]:
    """Run a read query and return records as plain dicts."""
    with get_driver().session(database=settings.NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(cypher, **params)]
