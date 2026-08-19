import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task_db import TaskDB
import legacy_worker_drain as drain


def trigger_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (drain.TRIGGER_NAME,),
        ).fetchone()
    return row[0]


def trigger_sql(db_path: Path) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (drain.TRIGGER_NAME,),
        ).fetchone()
    return row[0] if row else None


def test_dry_run_status_does_not_mutate_database(tmp_path):
    db_path = tmp_path / "tasks.db"
    db = TaskDB(db_path)
    db.create_task("pending.pdf", "/tmp/pending.pdf")

    report = drain.collect_status(db_path)

    assert report["pending"] == 1
    assert report["trigger"] == "absent"
    assert trigger_count(db_path) == 0


def test_apply_and_clear_are_idempotent(tmp_path):
    db_path = tmp_path / "tasks.db"
    TaskDB(db_path)

    assert drain.apply_trigger(db_path) == "applied"
    assert drain.apply_trigger(db_path) == "already-active"
    assert drain.trigger_matches(trigger_sql(db_path))
    assert trigger_count(db_path) == 1

    assert drain.clear_trigger(db_path) == "cleared"
    assert drain.clear_trigger(db_path) == "already-absent"
    assert trigger_count(db_path) == 0


def test_apply_and_clear_refuse_unexpected_existing_trigger(tmp_path):
    db_path = tmp_path / "tasks.db"
    TaskDB(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TRIGGER {drain.TRIGGER_NAME}
            BEFORE UPDATE OF status ON tasks
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
        )

    with pytest.raises(drain.DrainError, match="unexpected SQL"):
        drain.apply_trigger(db_path)
    with pytest.raises(drain.DrainError, match="unexpected SQL"):
        drain.clear_trigger(db_path)
    assert trigger_count(db_path) == 1


def test_drain_trigger_blocks_claim_but_allows_terminal_transition(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    db_path = tmp_path / "tasks.db"
    db = TaskDB(db_path)
    task_id = db.create_task("claim.pdf", "/tmp/claim.pdf")["task_id"]
    assert drain.apply_trigger(db_path) == "applied"

    assert db.get_next_task("worker-a") is None
    pending = db.get_task(task_id)
    assert pending["status"] == "pending"
    assert pending["worker_id"] is None

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'processing', worker_id = ? WHERE task_id = ?",
            ("worker-a", task_id),
        )
        row = conn.execute("SELECT status, worker_id FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row == ("pending", None)

    assert drain.clear_trigger(db_path) == "cleared"
    claimed = db.get_next_task("worker-a")
    assert claimed["task_id"] == task_id
    assert db.update_task_status(task_id, "completed", worker_id="worker-a", result_path="/tmp/out") is True
    assert db.get_task(task_id)["status"] == "completed"


def test_terminal_transition_succeeds_while_drain_trigger_is_active(tmp_path):
    db_path = tmp_path / "tasks.db"
    db = TaskDB(db_path)
    task_id = db.create_task("done.pdf", "/tmp/done.pdf")["task_id"]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'processing', worker_id = ? WHERE task_id = ?",
            ("worker-a", task_id),
        )
    assert db.get_task(task_id)["status"] == "processing"
    assert drain.apply_trigger(db_path) == "applied"

    assert db.update_task_status(task_id, "completed", worker_id="worker-a", result_path="/tmp/out") is True
    assert db.get_task(task_id)["status"] == "completed"
