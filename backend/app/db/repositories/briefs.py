"""Saved briefs, in Postgres.

Two things are worth saying about this module.

The first is that a brief is optional. Nothing else in the application needs a
database, so an unconfigured checkout has to run exactly as it did before this
existed; ``BriefsUnavailable`` is how the routes tell "not set up" apart from
"broken" without either one becoming a crash.

The second is that the rows are written by the server. The caller sends questions
and programs, and the figures stored beside them are what the engine and the check
return when those programs are run again here. Storing what the browser had on
screen would make the saved page a photograph of an answer rather than an answer,
and the two drift apart the first time anything about the engine changes.
"""

from __future__ import annotations

import secrets
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import get_settings

# A brief id is a path segment, so it is short, URL-safe, and long enough not to be
# guessable: 9 random bytes is 72 bits.
ID_BYTES = 9

CONNECT_TIMEOUT_SECONDS = 10


class BriefsUnavailable(RuntimeError):
    """No database, or one that did not answer. The answer is a 503, not a 500."""


def _dsn() -> str:
    url = get_settings().database_url.strip()
    if not url:
        raise BriefsUnavailable("no database is configured for saved briefs")
    # The setting is written with SQLAlchemy's driver suffix because that is what
    # the field was created for. psycopg wants the plain URL.
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _connect() -> psycopg.Connection:
    try:
        return psycopg.connect(
            _dsn(),
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            row_factory=dict_row,
        )
    except psycopg.Error as exc:
        # The reason is in the chain for whoever reads the server log. It is not in
        # the response: a connection string is not a thing to hand to a caller.
        raise BriefsUnavailable("the brief store is not reachable") from exc


def new_id() -> str:
    return secrets.token_urlsafe(ID_BYTES)


def save(document_id: str, filename: str, rows: list[dict[str, Any]]) -> str:
    """Write one brief and return its id."""
    brief_id = new_id()
    with _connect() as connection:
        try:
            connection.execute(
                """
                insert into briefs (id, document_id, filename, rows)
                values (%s, %s, %s, %s)
                """,
                (brief_id, document_id, filename, Jsonb(rows)),
            )
        except psycopg.Error as exc:
            raise BriefsUnavailable("the brief store did not accept the write") from exc
    return brief_id


def load(brief_id: str) -> dict[str, Any] | None:
    """One brief, or None when there is no such id."""
    with _connect() as connection:
        try:
            row = connection.execute(
                """
                select id, document_id, filename, rows, created_at
                from briefs
                where id = %s
                """,
                (brief_id,),
            ).fetchone()
        except psycopg.Error as exc:
            raise BriefsUnavailable("the brief store did not answer the read") from exc
    return row
