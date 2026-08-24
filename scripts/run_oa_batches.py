#!/usr/bin/env python3
"""Run OA inventory batches serially with health gates and checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import evaluate_local_efficiency as efficiency
import manage_oa_batches as batches

DEFAULT_LOCK_NAME = "run_oa_batches.lock"
DEFAULT_CHECKPOINT_NAME = "run_oa_batches_checkpoint.json"
DEFAULT_HEALTH_BACKOFF_INITIAL_SECONDS = 60.0
DEFAULT_HEALTH_BACKOFF_MAX_SECONDS = 300.0
DEFAULT_HEALTH_RECOVERY_SUCCESSES = 3
ACTIVE_BATCH_STATES = ("running", "partial")
TERMINAL_BATCH_STATES = batches.TERMINAL_BATCH_STATES

_STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_is_stale(value: object, age_seconds: float) -> bool:
    if not value:
        return False
    try:
        observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return observed <= datetime.now(timezone.utc) - timedelta(seconds=age_seconds)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def runner_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another OA batch runner holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def install_signal_handlers() -> None:
    def request_stop(_signum: int, _frame: object) -> None:
        global _STOP
        _STOP = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"time": utc_now(), "event": event, **fields}, ensure_ascii=False, sort_keys=True), flush=True)



def connect_inventory_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def open_inventory_for_run(args: argparse.Namespace) -> sqlite3.Connection:
    if args.apply:
        return batches.connect_inventory(args.inventory)
    return connect_inventory_read_only(args.inventory)

def get_batch(conn: sqlite3.Connection, batch_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT b.*,
               (SELECT COUNT(*) FROM batch_items bi
                WHERE bi.batch_id=b.batch_id AND bi.item_status='deferred') AS deferred_count
        FROM batches b WHERE b.batch_key=?
        """,
        (batch_key,),
    ).fetchone()
    return dict(row) if row else None


def first_batch(conn: sqlite3.Connection, statuses: tuple[str, ...]) -> dict[str, Any] | None:
    placeholders = ",".join("?" for _ in statuses)
    row = conn.execute(
        f"SELECT * FROM batches WHERE status IN ({placeholders}) ORDER BY generation,ordinal LIMIT 1",
        statuses,
    ).fetchone()
    return get_batch(conn, row["batch_key"]) if row else None


def checkpoint_payload(args: argparse.Namespace, *, action: str, stop_reason: str | None, batch: dict[str, Any] | None, errors: int) -> dict[str, Any]:
    return {
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "dry_run": not args.apply,
        "action": action,
        "stop_reason": stop_reason,
        "current_batch": batch,
        "last_refresh_at": getattr(args, "_last_refresh_at", None),
        "last_submit_at": getattr(args, "_last_submit_at", None),
        "consecutive_health_failures": errors,
        "consecutive_health_successes": getattr(args, "_consecutive_health_successes", 0),
        "health_first_failure_at": getattr(args, "_health_first_failure_at", None),
        "health_last_failure_at": getattr(args, "_health_last_failure_at", None),
        "health_last_failure_reason": getattr(args, "_health_last_failure_reason", None),
        "next_health_check_at": getattr(args, "_next_health_check_at", None),
        "deferred_backlog": getattr(args, "_deferred_backlog", 0),
    }


def write_checkpoint(args: argparse.Namespace, *, action: str, stop_reason: str | None = None, batch: dict[str, Any] | None = None, errors: int = 0) -> None:
    atomic_write_json(args.checkpoint, checkpoint_payload(args, action=action, stop_reason=stop_reason, batch=batch, errors=errors))



def stop_requested(args: argparse.Namespace, *, batch: dict[str, Any] | None = None, errors: int = 0) -> bool:
    if not _STOP:
        return False
    write_checkpoint(args, action="stopped", stop_reason="signal", batch=batch, errors=errors)
    emit("stop", reason="signal")
    return True


def health_failure_reason(health: dict[str, Any]) -> str:
    checks = health.get("checks")
    if not isinstance(checks, dict):
        return "health_gate"
    failed = [
        name
        for name, detail in checks.items()
        if not isinstance(detail, dict) or detail.get("passed") is not True
    ]
    return ",".join(sorted(failed)) or "health_gate"


def health_retry_delay(args: argparse.Namespace, failures: int) -> float:
    pause_attempt = max(0, failures - max(1, args.max_health_failures))
    if pause_attempt >= 2:
        return args.health_backoff_max_seconds
    delay = args.health_backoff_initial_seconds * (2**pause_attempt)
    return min(delay, args.health_backoff_max_seconds)


def next_health_check_at(delay_seconds: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=max(0.0, delay_seconds))
    ).isoformat(timespec="seconds")


def wait_for_health_retry(
    args: argparse.Namespace,
    delay_seconds: float,
    *,
    batch: dict[str, Any] | None,
    errors: int,
) -> bool:
    remaining = max(0.0, delay_seconds)
    while remaining > 0:
        if stop_requested(args, batch=batch, errors=errors):
            return False
        interval = min(1.0, remaining)
        time.sleep(interval)
        remaining -= interval
    return not stop_requested(args, batch=batch, errors=errors)


def health_gate(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    report = efficiency.run_evaluation(
        SimpleNamespace(
            duration=0.0,
            interval=5.0,
            baseline=None,
            stale_parent_threshold_seconds=args.stale_parent_threshold_seconds,
            record_baseline=None,
            evaluate_report=None,
            sample=True,
            host=args.host,
            database=args.task_db,
            log_path=args.log_path,
            worker_ports=args.worker_ports,
            vllm_ports=args.vllm_ports,
        )
    )
    checks = report.get("evaluation", {}).get("checks", {})
    required = (
        "services_healthy",
        "stale_parent_merges_zero",
        "api_500_delta_zero",
        "db_lock_terminal_failures_zero",
        "no_oom_or_preemption",
        "hbm_p99_lte_95_5_percent",
        "hbm_max_lt_98_percent",
    )
    ok = all(checks.get(name, {}).get("passed") is True for name in required)
    return ok, {"checks": {name: checks.get(name) for name in required}, "fleet": report.get("fleet")}


def full_refresh(args: argparse.Namespace) -> dict[str, Any]:
    conn = batches.connect_inventory(args.inventory)
    try:
        report = batches.refresh_inventory(conn, args.inventory, args.source_root, args.parsed_root, args.task_db, args.legacy_progress)
    finally:
        conn.close()
    args._last_refresh_at = utc_now()
    return report


def fast_refresh(conn: sqlite3.Connection, args: argparse.Namespace, batch_key: str) -> dict[str, Any]:
    report = batches.refresh_batch_from_task_db(
        conn,
        args.inventory,
        args.legacy_progress,
        args.task_db,
        batch_key,
        write_reports=args.write_progress_reports,
    )
    args._last_refresh_at = utc_now()
    return report


def submit(conn: sqlite3.Connection, args: argparse.Namespace, batch_key: str) -> dict[str, Any]:
    submit_args = SimpleNamespace(
        batch_key=batch_key,
        limit=None,
        priority=args.priority,
        high_watermark=args.high_watermark + args.max_deferred_backlog,
        allow_live_claims=args.allow_live_claims,
        task_db=args.task_db,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_password=args.redis_password,
        redis_queue_key=args.redis_queue_key,
        redis_processing_key=args.redis_processing_key,
        redis_maintenance_key=args.redis_maintenance_key,
        redis_pause_key=args.redis_pause_key,
    )
    report = batches.submit_batch(conn, args.inventory, args.legacy_progress, args.source_root, submit_args)
    args._last_submit_at = utc_now()
    return report


def deferred_backlog_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM batch_items WHERE item_status='deferred'"
    ).fetchone()
    return int(row["count"] if row else 0)


def stale_tail_task_ids(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    batch: dict[str, Any],
) -> list[str]:
    max_per_batch = max(0, int(args.max_deferred_items_per_batch))
    if (
        args.defer_stale_after_seconds <= 0
        or max_per_batch == 0
        or args.max_deferred_backlog == 0
        or batch["submitted_count"] <= 0
        or batch["submitted_count"] > max_per_batch
        or batch["item_count"] <= 0
    ):
        return []

    terminal_fraction = (batch["item_count"] - batch["submitted_count"]) / batch["item_count"]
    if terminal_fraction < args.min_terminal_fraction:
        return []

    backlog = deferred_backlog_count(conn)
    budget = min(
        batch["submitted_count"],
        max_per_batch,
        max(0, args.max_deferred_backlog - backlog),
    )
    if budget <= 0 or not args.task_db.exists():
        return []

    rows = conn.execute(
        """
        SELECT bi.task_id,d.last_submit_at
        FROM batch_items bi JOIN documents d USING(sha256)
        WHERE bi.batch_id=? AND bi.item_status='submitted'
          AND bi.task_id IS NOT NULL
        ORDER BY bi.position
        """,
        (batch["batch_id"],),
    ).fetchall()
    task_ids = [row["task_id"] for row in rows]
    if not task_ids:
        return []

    inventory_stale_ids = {
        row["task_id"]
        for row in rows
        if timestamp_is_stale(row["last_submit_at"], args.defer_stale_after_seconds)
    }
    placeholders = ",".join("?" for _ in task_ids)
    task_uri = f"file:{args.task_db}?mode=ro"
    task_conn = sqlite3.connect(task_uri, uri=True, timeout=10.0)
    task_conn.row_factory = sqlite3.Row
    try:
        task_rows = task_conn.execute(
            f"""
            SELECT task_id,status,
                   CASE
                       WHEN COALESCE(started_at,created_at) <= datetime('now', ?)
                       THEN 1 ELSE 0
                   END AS stale
            FROM tasks
            WHERE task_id IN ({placeholders})
            """,
            (f"-{int(args.defer_stale_after_seconds)} seconds", *task_ids),
        ).fetchall()
    finally:
        task_conn.close()

    known_ids = {row["task_id"] for row in task_rows}
    eligible_ids = {
        row["task_id"]
        for row in task_rows
        if row["status"] in ("pending", "processing", "merging") and row["stale"] == 1
    }
    eligible_ids.update(task_id for task_id in task_ids if task_id not in known_ids)
    eligible_ids.update(inventory_stale_ids)
    return [task_id for task_id in task_ids if task_id in eligible_ids][:budget]


def defer_stale_batch_tail(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    batch: dict[str, Any],
) -> dict[str, Any]:
    task_ids = stale_tail_task_ids(conn, args, batch)
    if not task_ids:
        return {"deferred": 0, "task_ids": []}
    reason = (
        f"batch tail <= {args.max_deferred_items_per_batch}; "
        f"task age >= {int(args.defer_stale_after_seconds)}s; "
        f"terminal fraction >= {args.min_terminal_fraction:.4f}"
    )
    report = batches.defer_batch_items(
        conn,
        batch["batch_key"],
        task_ids,
        reason=reason,
    )
    args._deferred_backlog = deferred_backlog_count(conn)
    report["deferred_backlog"] = args._deferred_backlog
    if args.write_progress_reports:
        batches.write_progress_reports(conn, args.inventory, args.legacy_progress)
    return report


def retry_exhausted_task_ids(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    batch: dict[str, Any],
) -> list[str]:
    max_per_batch = max(0, int(args.max_deferred_items_per_batch))
    if (
        args.max_item_submit_attempts <= 0
        or max_per_batch == 0
        or args.max_deferred_backlog == 0
        or batch["submitted_count"] > 0
        or batch["item_count"] <= 0
    ):
        return []

    rows = conn.execute(
        """
        SELECT bi.task_id
        FROM batch_items bi JOIN documents d USING(sha256)
        WHERE bi.batch_id=? AND bi.item_status='error'
          AND bi.task_id IS NOT NULL
          AND d.state='failed_retryable'
          AND d.submit_attempts>=?
        ORDER BY bi.position
        """,
        (batch["batch_id"], args.max_item_submit_attempts),
    ).fetchall()
    if not rows or len(rows) > max_per_batch:
        return []
    terminal_fraction = (batch["item_count"] - len(rows)) / batch["item_count"]
    if terminal_fraction < args.min_terminal_fraction:
        return []

    backlog = deferred_backlog_count(conn)
    budget = min(len(rows), max(0, args.max_deferred_backlog - backlog))
    return [row["task_id"] for row in rows[:budget]]


def defer_retry_exhausted_tail(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    batch: dict[str, Any],
) -> dict[str, Any]:
    task_ids = retry_exhausted_task_ids(conn, args, batch)
    if not task_ids:
        return {"deferred": 0, "task_ids": []}
    reason = (
        f"retryable batch tail exhausted {args.max_item_submit_attempts} submit attempts; "
        f"terminal fraction >= {args.min_terminal_fraction:.4f}"
    )
    report = batches.defer_batch_items(
        conn,
        batch["batch_key"],
        task_ids,
        reason=reason,
    )
    args._deferred_backlog = deferred_backlog_count(conn)
    report["deferred_backlog"] = args._deferred_backlog
    if args.write_progress_reports:
        batches.write_progress_reports(conn, args.inventory, args.legacy_progress)
    return report


def run_loop(args: argparse.Namespace, health_check: Callable[[argparse.Namespace], tuple[bool, dict[str, Any]]] = health_gate) -> int:
    install_signal_handlers()
    health_failures = 0
    health_successes = 0
    health_paused = False
    with runner_lock(args.lock_file):
        write_checkpoint(args, action="started")
        if args.initial_refresh and args.apply:
            emit("full_refresh_start")
            full_refresh(args)
            if stop_requested(args, errors=health_failures):
                return 130
            emit("full_refresh_done")
        conn = open_inventory_for_run(args)
        args._deferred_backlog = deferred_backlog_count(conn)
        try:
            while True:
                if stop_requested(args, errors=health_failures):
                    return 130

                batch = first_batch(conn, ACTIVE_BATCH_STATES)
                if batch:
                    if args.apply:
                        fast_refresh(conn, args, batch["batch_key"])
                        batch = get_batch(conn, batch["batch_key"])
                        if stop_requested(args, batch=batch, errors=health_failures):
                            return 130
                    if batch and batch["status"] in TERMINAL_BATCH_STATES:
                        emit("batch_terminal", batch_key=batch["batch_key"], status=batch["status"], failed_count=batch["failed_count"])
                        write_checkpoint(args, action="batch_terminal", batch=batch, errors=health_failures)
                        if args.once:
                            return 0
                        continue
                    if batch and args.apply:
                        exhausted = defer_retry_exhausted_tail(conn, args, batch)
                        if exhausted["deferred"]:
                            batch = get_batch(conn, batch["batch_key"])
                            emit("batch_retry_tail_deferred", **exhausted)
                            write_checkpoint(args, action="batch_retry_tail_deferred", batch=batch, errors=health_failures)
                            if args.once:
                                return 0
                            continue
                    if batch and batch["failed_count"] > args.max_batch_errors:
                        write_checkpoint(args, action="stopped", stop_reason="batch_error_threshold", batch=batch, errors=health_failures)
                        emit("stop", reason="batch_error_threshold", batch_key=batch["batch_key"], failed_count=batch["failed_count"])
                        return 2
                    if batch and batch["submitted_count"] > 0 and args.apply:
                        deferred = defer_stale_batch_tail(conn, args, batch)
                        if deferred["deferred"]:
                            batch = get_batch(conn, batch["batch_key"])
                            emit("batch_tail_deferred", **deferred)
                            write_checkpoint(args, action="batch_tail_deferred", batch=batch, errors=health_failures)
                            if args.once:
                                return 0
                            continue
                    if batch and batch["submitted_count"] > 0:
                        emit("wait_batch", batch_key=batch["batch_key"], status=batch["status"], submitted_count=batch["submitted_count"], completed_count=batch["completed_count"], failed_count=batch["failed_count"], deferred_count=batch.get("deferred_count", 0))
                        write_checkpoint(args, action="wait_batch", batch=batch, errors=health_failures)
                        if args.once:
                            return 0
                        time.sleep(args.poll_seconds)
                        continue
                    target = batch
                else:
                    target = first_batch(conn, ("prepared",))
                    if not target:
                        backlog = deferred_backlog_count(conn)
                        args._deferred_backlog = backlog
                        if backlog:
                            write_checkpoint(args, action="primary_complete_with_deferred", stop_reason="deferred_backfill_required", errors=health_failures)
                            emit("primary_complete_with_deferred", deferred_backlog=backlog)
                        else:
                            write_checkpoint(args, action="complete", stop_reason="no_remaining_batches", errors=health_failures)
                            emit("complete", reason="no_remaining_batches")
                        return 0

                if stop_requested(args, batch=target, errors=health_failures):
                    return 130

                if target["failed_count"] > args.max_batch_errors:
                    write_checkpoint(args, action="stopped", stop_reason="batch_error_threshold", batch=target, errors=health_failures)
                    emit("stop", reason="batch_error_threshold", batch_key=target["batch_key"], failed_count=target["failed_count"])
                    return 2

                ok, health = health_check(args)
                if not ok:
                    observed_at = utc_now()
                    if health_failures == 0:
                        args._health_first_failure_at = observed_at
                    health_failures += 1
                    health_successes = 0
                    args._consecutive_health_successes = 0
                    args._health_last_failure_at = observed_at
                    args._health_last_failure_reason = health_failure_reason(health)
                    if args.once or (
                        args.exit_on_health_failure
                        and health_failures >= args.max_health_failures
                    ):
                        write_checkpoint(args, action="health_failed", batch=target, errors=health_failures)
                        emit("health_failed", batch_key=target["batch_key"], consecutive=health_failures, health=health)
                        write_checkpoint(args, action="stopped", stop_reason="health_gate", batch=target, errors=health_failures)
                        return 3
                    if health_failures >= args.max_health_failures:
                        health_paused = True
                        delay = health_retry_delay(args, health_failures)
                        args._next_health_check_at = next_health_check_at(delay)
                        write_checkpoint(args, action="health_paused", batch=target, errors=health_failures)
                        emit(
                            "health_paused",
                            batch_key=target["batch_key"],
                            consecutive=health_failures,
                            next_check_at=args._next_health_check_at,
                            health=health,
                        )
                        if not wait_for_health_retry(args, delay, batch=target, errors=health_failures):
                            return 130
                    else:
                        args._next_health_check_at = next_health_check_at(args.poll_seconds)
                        write_checkpoint(args, action="health_failed", batch=target, errors=health_failures)
                        emit("health_failed", batch_key=target["batch_key"], consecutive=health_failures, health=health)
                        if not wait_for_health_retry(args, args.poll_seconds, batch=target, errors=health_failures):
                            return 130
                    continue
                if stop_requested(args, batch=target, errors=health_failures):
                    return 130
                if health_paused:
                    health_successes += 1
                    args._consecutive_health_successes = health_successes
                    if health_successes < args.health_recovery_successes:
                        args._next_health_check_at = next_health_check_at(args.poll_seconds)
                        write_checkpoint(args, action="health_recovering", batch=target, errors=health_failures)
                        emit(
                            "health_recovering",
                            batch_key=target["batch_key"],
                            consecutive_successes=health_successes,
                            required_successes=args.health_recovery_successes,
                            next_check_at=args._next_health_check_at,
                        )
                        if not wait_for_health_retry(args, args.poll_seconds, batch=target, errors=health_failures):
                            return 130
                        continue
                    emit(
                        "health_resumed",
                        batch_key=target["batch_key"],
                        consecutive_successes=health_successes,
                    )
                    args._next_health_check_at = None
                    write_checkpoint(args, action="health_resumed", batch=target, errors=health_failures)
                    health_paused = False
                health_failures = 0
                health_successes = 0
                args._consecutive_health_successes = 0
                args._health_first_failure_at = None
                args._health_last_failure_at = None
                args._health_last_failure_reason = None
                args._next_health_check_at = None

                if not args.apply:
                    write_checkpoint(args, action="dry_run_submit", stop_reason="dry_run", batch=target, errors=health_failures)
                    emit("dry_run_submit", batch_key=target["batch_key"], status=target["status"], item_count=target["item_count"])
                    return 0

                if stop_requested(args, batch=target, errors=health_failures):
                    return 130
                emit("submit_start", batch_key=target["batch_key"], status=target["status"])
                try:
                    submit(conn, args, target["batch_key"])
                    fast_refresh(conn, args, target["batch_key"])
                    refreshed = get_batch(conn, target["batch_key"])
                except Exception as exc:
                    write_checkpoint(args, action="stopped", stop_reason=f"submit_failed: {type(exc).__name__}: {exc}", batch=target, errors=health_failures)
                    emit("stop", reason="submit_failed", batch_key=target["batch_key"], error=f"{type(exc).__name__}: {exc}")
                    return 4
                if refreshed and refreshed["failed_count"] > args.max_batch_errors:
                    write_checkpoint(args, action="stopped", stop_reason="batch_error_threshold", batch=refreshed, errors=health_failures)
                    emit("stop", reason="batch_error_threshold", batch_key=refreshed["batch_key"], failed_count=refreshed["failed_count"])
                    return 2
                write_checkpoint(args, action="submitted", batch=refreshed, errors=health_failures)
                emit("submit_done", batch_key=target["batch_key"], batch=refreshed)
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
        finally:
            conn.close()


def parse_ports(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="submit work; default only reports the next action")
    parser.add_argument("--once", action="store_true", help="run one decision loop and exit")
    parser.add_argument("--initial-refresh", action="store_true", help="run one full inventory refresh before the loop when --apply is set")
    parser.add_argument("--inventory", type=Path, default=batches.DEFAULT_INVENTORY)
    parser.add_argument("--source-root", type=Path, default=batches.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--parsed-root", type=Path, default=batches.DEFAULT_PARSED_ROOT)
    parser.add_argument("--task-db", type=Path, default=batches.DEFAULT_TASK_DB)
    parser.add_argument("--legacy-progress", type=Path, default=batches.DEFAULT_LEGACY_PROGRESS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-health-failures", type=int, default=3)
    parser.add_argument("--exit-on-health-failure", action="store_true", help="exit after the health failure threshold instead of waiting for recovery")
    parser.add_argument("--health-backoff-initial-seconds", type=float, default=DEFAULT_HEALTH_BACKOFF_INITIAL_SECONDS)
    parser.add_argument("--health-backoff-max-seconds", type=float, default=DEFAULT_HEALTH_BACKOFF_MAX_SECONDS)
    parser.add_argument("--health-recovery-successes", type=int, default=DEFAULT_HEALTH_RECOVERY_SUCCESSES)
    parser.add_argument("--max-batch-errors", type=int, default=100)
    parser.add_argument("--write-progress-reports", action="store_true", help="rewrite full progress.json/batches.csv on every batch poll")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--high-watermark", type=int, default=batches.DEFAULT_HIGH_WATERMARK)
    parser.add_argument("--allow-live-claims", action="store_true")
    parser.add_argument("--defer-stale-after-seconds", type=float, default=3600.0, help="defer a small stale batch tail after this age; 0 disables")
    parser.add_argument("--max-deferred-items-per-batch", type=int, default=20, help="maximum stale tail items released from one batch")
    parser.add_argument("--max-deferred-backlog", type=int, default=100, help="global deferred backlog cap and queue headroom")
    parser.add_argument("--min-terminal-fraction", type=float, default=0.99, help="minimum terminal fraction before stale tail deferral")
    parser.add_argument("--max-item-submit-attempts", type=int, default=2, help="defer a retryable tail after this many submissions; 0 disables")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--worker-ports", type=parse_ports, default=parse_ports("8101,8102,8103,8104,8105,8106,8107,8108"))
    parser.add_argument("--vllm-ports", type=parse_ports, default=parse_ports("30025,30026,30027,30028,30029,30030,30031,30032"))
    parser.add_argument("--log-path", type=Path, action="append")
    parser.add_argument("--stale-parent-threshold-seconds", type=float, default=efficiency.DEFAULT_STALE_PARENT_MERGE_THRESHOLD_SECONDS)
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "127.0.0.1"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--redis-db", type=int, default=int(os.getenv("REDIS_DB", "0")))
    parser.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD", "redis123"))
    parser.add_argument("--redis-queue-key", default=os.getenv("REDIS_QUEUE_KEY", "tianshu:task_queue:mineru-runner-worker-0"))
    parser.add_argument("--redis-processing-key", default=os.getenv("REDIS_PROCESSING_KEY", "tianshu:processing:mineru-runner-worker-0"))
    parser.add_argument("--redis-maintenance-key", default=os.getenv("REDIS_CLAIM_MAINTENANCE_KEY", "tianshu:claim_maintenance:mineru-runner-worker-0"))
    parser.add_argument("--redis-pause-key", default=os.getenv("REDIS_CLAIM_PAUSE_KEY", "tianshu:claim_pause:mineru-runner-worker-0"))
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.checkpoint is None:
        args.checkpoint = args.inventory.parent / DEFAULT_CHECKPOINT_NAME
    if args.lock_file is None:
        args.lock_file = args.inventory.parent / DEFAULT_LOCK_NAME
    if args.log_path is None:
        args.log_path = [Path("/share/wangjiong/databases/mineru_database/mineru-runner-worker-0/mineru_logs")]
    args._last_refresh_at = None
    args._last_submit_at = None
    args._consecutive_health_successes = 0
    args._health_first_failure_at = None
    args._health_last_failure_at = None
    args._health_last_failure_reason = None
    args._next_health_check_at = None
    if not 0.0 <= args.min_terminal_fraction <= 1.0:
        raise ValueError("min-terminal-fraction must be between 0 and 1")
    if args.max_health_failures < 1 or args.health_recovery_successes < 1:
        raise ValueError("health failure and recovery thresholds must be at least 1")
    if args.health_backoff_initial_seconds < 0 or args.health_backoff_max_seconds < args.health_backoff_initial_seconds:
        raise ValueError("health backoff must be non-negative and max must be >= initial")
    if args.defer_stale_after_seconds < 0 or args.max_deferred_items_per_batch < 0 or args.max_deferred_backlog < 0 or args.max_item_submit_attempts < 0:
        raise ValueError("deferred-tail limits must be non-negative")
    args._deferred_backlog = 0
    return args


def main() -> int:
    return run_loop(normalize_args(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
