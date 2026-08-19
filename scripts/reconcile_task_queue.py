#!/usr/bin/env python3
"""Reconcile SQLite task state with the optional Redis queue.

Default mode is dry-run. Use --apply during a maintenance window.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from task_db import TaskDB  # noqa: E402
from redis_queue import get_redis_queue  # noqa: E402

MAX_BATCH_SIZE = 500
MAX_ACTION_DETAILS = 100
DEFAULT_ACTIVE_STALE_SECONDS = 3600


@dataclass
class ReconcileReport:
    dry_run: bool
    active_only: bool = False
    scanned_pending: int = 0
    scanned_queued: int = 0
    scanned_processing: int = 0
    scanned_sqlite_active: int = 0
    enqueued_missing: int = 0
    stale_queued_removed: int = 0
    stale_processing_removed: int = 0
    stale_sqlite_reset: int = 0
    fresh_processing_protected: int = 0
    parent_waiting_protected: int = 0
    checkpoints: dict = field(default_factory=dict)
    omitted_actions: int = 0
    actions: list[dict] = field(default_factory=list)

    def add_action(self, action: str, task_id: str, applied: bool, reason: str) -> None:
        if len(self.actions) >= MAX_ACTION_DETAILS:
            self.omitted_actions += 1
            return
        self.actions.append({"action": action, "task_id": task_id, "applied": applied, "reason": reason})


def clamp_batch_size(batch_size: int) -> int:
    return max(1, min(batch_size, MAX_BATCH_SIZE))


def load_checkpoint(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def save_checkpoint(path: Optional[Path], checkpoints: dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoints, indent=2, sort_keys=True))


def decode(value):
    return value.decode() if isinstance(value, bytes) else value


def open_readonly_conn(db_path: str) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_processing(data: str | bytes | None) -> dict:
    if data is None:
        return {}
    data = decode(data)
    try:
        return json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return {}


def is_fresh_processing(raw_data, now: float, fresh_heartbeat_seconds: int) -> bool:
    heartbeat = parse_processing(raw_data)
    claimed_at = float(heartbeat.get("claimed_at") or 0)
    return bool(claimed_at and now - claimed_at <= fresh_heartbeat_seconds)


def iter_pending(conn: sqlite3.Connection, after_task_id: str, batch_size: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT task_id, priority, file_name, backend
        FROM tasks
        WHERE status = 'pending' AND task_id > ?
        ORDER BY task_id
        LIMIT ?
        """,
        (after_task_id, batch_size),
    ).fetchall()


def iter_zscan_batches(client, key: str, batch_size: int) -> Iterable[list[str]]:
    cursor = 0
    while True:
        cursor, rows = client.zscan(key, cursor=cursor, count=batch_size)
        batch = [decode(member) for member, _score in rows]
        if batch:
            yield batch
        if int(cursor) == 0:
            break


def iter_hscan_batches(client, key: str, batch_size: int) -> Iterable[list[tuple[str, str]]]:
    cursor = 0
    while True:
        cursor, rows = client.hscan(key, cursor=cursor, count=batch_size)
        batch = [(decode(task_id), decode(raw_data)) for task_id, raw_data in rows.items()]
        if batch:
            yield batch
        if int(cursor) == 0:
            break


def select_statuses(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not task_ids:
        return {}
    placeholders = ",".join("?" for _ in task_ids)
    return {
        row["task_id"]: row
        for row in conn.execute(
            f"SELECT task_id, status, priority, file_name, backend FROM tasks WHERE task_id IN ({placeholders})",
            tuple(task_ids),
        ).fetchall()
    }


def redis_tasks_present(redis_queue, task_ids: list[str]) -> dict[str, bool]:
    if not task_ids:
        return {}
    client = redis_queue.client
    queue_key = redis_queue.config.queue_key
    processing_key = redis_queue.config.processing_key

    if hasattr(client, "zmscore") and hasattr(client, "hmget"):
        queued_scores = client.zmscore(queue_key, task_ids)
        processing_rows = client.hmget(processing_key, task_ids)
    else:
        pipe = client.pipeline()
        for task_id in task_ids:
            pipe.zscore(queue_key, task_id)
            pipe.hget(processing_key, task_id)
        results = pipe.execute()
        queued_scores = results[0::2]
        processing_rows = results[1::2]

    return {
        task_id: queued_scores[index] is not None or processing_rows[index] is not None
        for index, task_id in enumerate(task_ids)
    }


def fetch_active_candidates(conn: sqlite3.Connection, batch_size: int, active_stale_seconds: int) -> tuple[list[sqlite3.Row], int]:
    parent_waiting = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM tasks
        WHERE status = 'processing'
          AND is_parent = 1
          AND child_count > 0
          AND child_completed < child_count
          AND started_at < datetime('now', '-' || ? || ' seconds')
        """,
        (active_stale_seconds,),
    ).fetchone()["count"]
    rows = conn.execute(
        """
        SELECT task_id
        FROM tasks
        WHERE status IN ('processing', 'merging')
          AND started_at < datetime('now', '-' || ? || ' seconds')
          AND NOT (
            is_parent = 1
            AND status = 'processing'
            AND child_count > 0
            AND child_completed < child_count
          )
        ORDER BY started_at ASC, task_id
        LIMIT ?
        """,
        (active_stale_seconds, batch_size),
    ).fetchall()
    return rows, parent_waiting


def pause_claims_for_apply(redis_queue) -> tuple[bool, bool]:
    if not hasattr(redis_queue, "is_claim_maintenance") or not hasattr(redis_queue, "set_claim_maintenance"):
        return False, False
    if redis_queue.is_claim_maintenance():
        return True, False
    acquired = bool(redis_queue.set_claim_maintenance("queue reconciliation"))
    return acquired, acquired


def reset_sqlite_active_and_enqueue(db: TaskDB, redis_queue, task_id: str) -> tuple[bool, bool]:
    task_data = None
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            SELECT task_id, priority, file_name, backend
            FROM tasks
            WHERE task_id = ? AND status IN ('processing', 'merging')
            """,
            (task_id,),
        )
        row = cursor.fetchone()
        if not row:
            return False, False
        task_data = {
            "task_id": row["task_id"],
            "priority": row["priority"] or 0,
            "file_name": row["file_name"],
            "backend": row["backend"],
        }
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'pending', worker_id = NULL, started_at = NULL
            WHERE task_id = ? AND status IN ('processing', 'merging')
            """,
            (task_id,),
        )
        if cursor.rowcount == 0:
            return False, False

    try:
        redis_queue.client.hdel(redis_queue.config.processing_key, task_id)
        redis_ok = redis_queue.enqueue(
            task_id,
            priority=task_data["priority"],
            task_data={"file_name": task_data["file_name"], "backend": task_data["backend"]},
        )
    except Exception:
        redis_ok = False
    return True, bool(redis_ok)


def reconcile(
    db_path: str,
    redis_queue,
    *,
    apply: bool = False,
    checkpoint: Optional[Path] = None,
    batch_size: int = MAX_BATCH_SIZE,
    fresh_heartbeat_seconds: int = 300,
    active_stale_seconds: int = DEFAULT_ACTIVE_STALE_SECONDS,
    ensure_indexes: bool = False,
    active_only: bool = False,
) -> ReconcileReport:
    if redis_queue is None:
        raise RuntimeError("Redis queue is unavailable; reconciliation needs Redis access")
    if ensure_indexes and not apply:
        raise RuntimeError("--ensure-indexes requires --apply")

    batch_size = clamp_batch_size(batch_size)
    db_path = str(Path(db_path).resolve())
    db = None
    report = ReconcileReport(dry_run=not apply, active_only=active_only)
    loaded_checkpoint = load_checkpoint(checkpoint) if apply else {}
    next_checkpoints = dict(loaded_checkpoint)
    after_task_id = str(loaded_checkpoint.get("pending_after_task_id", ""))
    now = time.time()
    paused_by_reconcile = False
    apply_success = False

    if apply or ensure_indexes:
        pause_ok, paused_by_reconcile = pause_claims_for_apply(redis_queue)
        if not pause_ok:
            raise RuntimeError("Refusing to apply reconciliation because Redis claim pause could not be enabled")
        db = TaskDB(db_path)

    if ensure_indexes:
        db.ensure_queue_indexes()

    conn = open_readonly_conn(db_path)
    try:
        if not active_only:
            while True:
                pending_rows = iter_pending(conn, after_task_id, batch_size)
                pending_presence = redis_tasks_present(redis_queue, [row["task_id"] for row in pending_rows])
                for row in pending_rows:
                    task_id = row["task_id"]
                    report.scanned_pending += 1
                    if not pending_presence.get(task_id, False):
                        if apply:
                            redis_ok = redis_queue.enqueue(
                                task_id,
                                priority=row["priority"] or 0,
                                task_data={"file_name": row["file_name"], "backend": row["backend"]},
                            )
                            if not redis_ok:
                                raise RuntimeError(f"Redis enqueue failed while repairing missing pending task {task_id}")
                        report.enqueued_missing += 1
                        report.add_action("enqueue_missing_pending", task_id, apply, "pending in SQLite but absent from Redis")
                    after_task_id = task_id
                    if apply:
                        next_checkpoints["pending_after_task_id"] = task_id
                if apply:
                    save_checkpoint(checkpoint, next_checkpoints)
                if len(pending_rows) < batch_size:
                    if apply:
                        next_checkpoints.pop("pending_after_task_id", None)
                        save_checkpoint(checkpoint, next_checkpoints)
                    break

            for queued_batch in iter_zscan_batches(redis_queue.client, redis_queue.config.queue_key, batch_size):
                report.scanned_queued += len(queued_batch)
                statuses = select_statuses(conn, queued_batch)
                for task_id in queued_batch:
                    row = statuses.get(task_id)
                    if row and row["status"] == "pending":
                        continue
                    if apply:
                        redis_queue.client.zrem(redis_queue.config.queue_key, task_id)
                    report.stale_queued_removed += 1
                    report.add_action("remove_stale_queued", task_id, apply, "Redis queued task is not SQLite pending")

        for processing_batch in iter_hscan_batches(redis_queue.client, redis_queue.config.processing_key, batch_size):
            report.scanned_processing += len(processing_batch)
            task_ids = [task_id for task_id, _raw in processing_batch]
            statuses = select_statuses(conn, task_ids)
            for task_id, raw_data in processing_batch:
                if is_fresh_processing(raw_data, now, fresh_heartbeat_seconds):
                    report.fresh_processing_protected += 1
                    continue
                row = statuses.get(task_id)
                if row and row["status"] in ("processing", "merging"):
                    continue
                if row and row["status"] == "pending":
                    if apply:
                        redis_queue.client.hdel(redis_queue.config.processing_key, task_id)
                        redis_ok = redis_queue.enqueue(
                            task_id=task_id,
                            priority=row["priority"] or 0,
                            task_data={"file_name": row["file_name"], "backend": row["backend"]},
                        )
                        if not redis_ok:
                            raise RuntimeError(f"Redis enqueue failed while repairing pending processing task {task_id}")
                    report.stale_processing_removed += 1
                    report.enqueued_missing += 1
                    report.add_action(
                        "requeue_pending_processing",
                        task_id,
                        apply,
                        "Redis processing task is SQLite pending; remove processing entry and enqueue",
                    )
                    continue
                if apply:
                    redis_queue.client.hdel(redis_queue.config.processing_key, task_id)
                report.stale_processing_removed += 1
                report.add_action("remove_stale_processing", task_id, apply, "Redis processing task is missing or not active in SQLite")

        stale_active, parent_waiting = fetch_active_candidates(conn, batch_size, active_stale_seconds)
        report.parent_waiting_protected = parent_waiting
        report.scanned_sqlite_active = len(stale_active)
        for row in stale_active:
            task_id = row["task_id"]
            raw_data = redis_queue.client.hget(redis_queue.config.processing_key, task_id)
            if is_fresh_processing(raw_data, now, fresh_heartbeat_seconds):
                report.fresh_processing_protected += 1
                continue
            redis_ok = True
            if apply:
                if db is None:
                    continue
                sqlite_reset, redis_ok = reset_sqlite_active_and_enqueue(db, redis_queue, task_id)
                if not sqlite_reset:
                    continue
            report.stale_sqlite_reset += 1
            reason = "SQLite active task exceeded active stale age and has no fresh Redis heartbeat"
            if apply and not redis_ok:
                reason += "; Redis requeue failed after SQLite reset, next pass can recover pending task"
            report.add_action(
                "reset_stale_sqlite_active",
                task_id,
                apply,
                reason,
            )
            if apply and not redis_ok:
                raise RuntimeError(f"Redis enqueue failed after SQLite reset for stale active task {task_id}")

        if apply:
            redis_queue.client.hset(
                redis_queue.config.claim_maintenance_key,
                mapping={"last_reconcile_at": str(now), "last_reconcile_actions": str(len(report.actions) + report.omitted_actions)},
            )
            report.checkpoints = next_checkpoints
            apply_success = True
    finally:
        conn.close()
        if paused_by_reconcile and apply_success and hasattr(redis_queue, "clear_claim_maintenance"):
            if not redis_queue.clear_claim_maintenance():
                raise RuntimeError("Redis claim pause was not cleared after successful reconciliation")

    return report


def default_db_path() -> str:
    if os.getenv("DATABASE_PATH"):
        return os.environ["DATABASE_PATH"]
    instance_id = os.getenv("INSTANCE_ID") or socket.gethostname()
    return f"/share/wangjiong/databases/mineru_database/{instance_id}/mineru_tianshu.db"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=default_db_path())
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint JSON path for resumable pending scans. Only advances with --apply.")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--fresh-heartbeat-seconds", type=int, default=300)
    parser.add_argument("--active-stale-seconds", type=int, default=DEFAULT_ACTIVE_STALE_SECONDS)
    parser.add_argument("--ensure-indexes", action="store_true", help="Create queue indexes first; run only during maintenance.")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Repair Redis processing and stale SQLite active tasks without scanning pending/queued tasks.",
    )
    args = parser.parse_args()

    redis_queue = get_redis_queue()
    report = reconcile(
        args.db,
        redis_queue,
        apply=args.apply,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        fresh_heartbeat_seconds=args.fresh_heartbeat_seconds,
        active_stale_seconds=args.active_stale_seconds,
        ensure_indexes=args.ensure_indexes,
        active_only=args.active_only,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
