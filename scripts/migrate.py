"""Apply the SQL migrations, in order, once each.

    python scripts/migrate.py            # against DATABASE_URL
    python scripts/migrate.py --status   # what has been applied, without applying

A migration is a file in backend/migrations numbered by the order it should run in.
Each one is applied inside a transaction together with the row recording it, so a run
that fails halfway leaves no half-applied file behind. The table that records them is
created first, which means this script is safe to run against an empty database and
safe to run twice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "backend" / "migrations"

# The application reads its settings from backend/, which is not on the path when a
# script runs from the repository root.
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings

LEDGER = """
create table if not exists schema_migrations (
    name       text primary key,
    applied_at timestamptz not null default now()
)
"""


def dsn() -> str:
    url = get_settings().database_url.strip()
    if not url:
        raise SystemExit("DATABASE_URL is empty; set it in .env or the environment")
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def files() -> list[Path]:
    # Numbered names sort in the order they should run. Refuse anything else rather
    # than depend on the filesystem's idea of alphabetical.
    found = sorted(path for path in MIGRATIONS.glob("*.sql") if path.is_file())
    for path in found:
        if not path.name[:3].isdigit():
            raise SystemExit(f"{path.name}: migration names start with a number")
    return found


def applied(connection: psycopg.Connection) -> set[str]:
    connection.execute(LEDGER)
    rows = connection.execute("select name from schema_migrations").fetchall()
    return {row[0] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="list files and what has run")
    args = parser.parse_args()

    with psycopg.connect(dsn(), connect_timeout=15) as connection:
        done = applied(connection)
        for path in files():
            if path.name in done:
                print(f"applied  {path.name}")
                continue
            if args.status:
                print(f"pending  {path.name}")
                continue
            with connection.transaction():
                connection.execute(path.read_text(encoding="utf-8"))
                connection.execute("insert into schema_migrations (name) values (%s)", (path.name,))
            print(f"applied  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
