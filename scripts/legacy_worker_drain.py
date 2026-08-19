#!/usr/bin/env python3
"""Drain helper for pausing legacy SQLite worker claims.

Default mode is read-only. Use --apply to install the claim-blocking trigger and
--clear to remove it after migration.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

TRIGGER_NAME = "legacy_worker_drain_block_claims"
EXPECTED_TRIGGER_SQL = f"""CREATE TRIGGER {TRIGGER_NAME}
BEFORE UPDATE OF status ON tasks
FOR EACH ROW
WHEN OLD.status = 'pending' AND NEW.status = 'processing'
BEGIN
    SELECT RAISE(IGNORE);
END"""


class DrainError(RuntimeError):
    pass


@dataclass(frozen=True)
class TriggerState:
    status: str
    sql: str | None = None


def default_database_path(env: dict[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    instance_id = values.get("INSTANCE_ID")
    if not instance_id:
        raise DrainError("INSTANCE_ID is required when --database is not provided")
    return Path("/share/wangjiong/databases/mineru_database") / instance_id / "mineru_tianshu.db"


def normalize_sql(sql: str | None) -> str:
    return " ".join((sql or "").strip().rstrip(";").split())


def trigger_matches(sql: str | None) -> bool:
    return normalize_sql(sql) == normalize_sql(EXPECTED_TRIGGER_SQL)


def get_sqlite_write_lock():
    try:
        from task_db import _SQLiteWriteLock  # type: ignore
    except Exception as exc:  # pragma: no cover - exact import failure varies
        raise DrainError(f"cannot import backend.task_db._SQLiteWriteLock: {exc}") from exc
    if not callable(_SQLiteWriteLock) or not hasattr(_SQLiteWriteLock, "__enter__"):
        raise DrainError("backend.task_db._SQLiteWriteLock is not a usable context manager")
    return _SQLiteWriteLock


def open_readonly_conn(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def open_write_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def fetch_trigger_state(conn: sqlite3.Connection) -> TriggerState:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (TRIGGER_NAME,),
    ).fetchone()
    if row is None:
        return TriggerState("absent")
    sql = row["sql"]
    return TriggerState("active" if trigger_matches(sql) else "unexpected", sql)


def assert_expected_or_absent(state: TriggerState) -> None:
    if state.status == "unexpected":
        raise DrainError(f"refusing unexpected SQL for existing trigger {TRIGGER_NAME}")


def pending_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM tasks WHERE status = 'pending'").fetchone()
    return int(row["count"])


def active_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT status, COALESCE(worker_id, '(none)') AS worker_id, COUNT(*) AS count
        FROM tasks
        WHERE status IN ('processing', 'merging')
        GROUP BY status, COALESCE(worker_id, '(none)')
        ORDER BY status, worker_id
        """
    ).fetchall()


@contextmanager
def schema_mutation(db_path: Path) -> Iterator[sqlite3.Connection]:
    if not db_path.exists():
        raise DrainError(f"database does not exist: {db_path}")
    lock_cls = get_sqlite_write_lock()
    with lock_cls(str(db_path)):
        conn = open_write_conn(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def apply_trigger(db_path: Path) -> str:
    with schema_mutation(db_path) as conn:
        state = fetch_trigger_state(conn)
        assert_expected_or_absent(state)
        if state.status == "active":
            return "already-active"
        conn.execute(EXPECTED_TRIGGER_SQL)
        return "applied"


def clear_trigger(db_path: Path) -> str:
    with schema_mutation(db_path) as conn:
        state = fetch_trigger_state(conn)
        assert_expected_or_absent(state)
        if state.status == "absent":
            return "already-absent"
        conn.execute(f"DROP TRIGGER {TRIGGER_NAME}")
        return "cleared"


def collect_status(db_path: Path) -> dict:
    with open_readonly_conn(db_path) as conn:
        state = fetch_trigger_state(conn)
        return {
            "database": str(db_path),
            "pending": pending_count(conn),
            "active": [dict(row) for row in active_counts(conn)],
            "trigger": state.status,
        }


def print_status(report: dict) -> None:
    print(f"database: {report['database']}")
    print(f"pending: {report['pending']}")
    print("active:")
    if report["active"]:
        for row in report["active"]:
            print(f"  {row['status']} worker_id={row['worker_id']} count={row['count']}")
    else:
        print("  none")
    print(f"trigger {TRIGGER_NAME}: {report['trigger']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely drain legacy SQLite worker claims")
    parser.add_argument("--database", type=Path, help="Path to mineru_tianshu.db")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="Install the drain trigger")
    action.add_argument("--clear", action="store_true", help="Remove the drain trigger")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        db_path = args.database or default_database_path()
        if args.apply:
            result = apply_trigger(db_path)
            print(f"action: {result}")
        elif args.clear:
            result = clear_trigger(db_path)
            print(f"action: {result}")
        else:
            print("action: dry-run")
        print_status(collect_status(db_path))
        return 0
    except (DrainError, sqlite3.Error, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
