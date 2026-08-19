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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import evaluate_local_efficiency as efficiency
import manage_oa_batches as batches

DEFAULT_LOCK_NAME = "run_oa_batches.lock"
DEFAULT_CHECKPOINT_NAME = "run_oa_batches_checkpoint.json"
ACTIVE_BATCH_STATES = ("running", "partial")
TERMINAL_BATCH_STATES = batches.TERMINAL_BATCH_STATES

_STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    row = conn.execute("SELECT * FROM batches WHERE batch_key=?", (batch_key,)).fetchone()
    return dict(row) if row else None


def first_batch(conn: sqlite3.Connection, statuses: tuple[str, ...]) -> dict[str, Any] | None:
    placeholders = ",".join("?" for _ in statuses)
    row = conn.execute(
        f"SELECT * FROM batches WHERE status IN ({placeholders}) ORDER BY generation,ordinal LIMIT 1",
        statuses,
    ).fetchone()
    return dict(row) if row else None


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
    }


def write_checkpoint(args: argparse.Namespace, *, action: str, stop_reason: str | None = None, batch: dict[str, Any] | None = None, errors: int = 0) -> None:
    atomic_write_json(args.checkpoint, checkpoint_payload(args, action=action, stop_reason=stop_reason, batch=batch, errors=errors))



def stop_requested(args: argparse.Namespace, *, batch: dict[str, Any] | None = None, errors: int = 0) -> bool:
    if not _STOP:
        return False
    write_checkpoint(args, action="stopped", stop_reason="signal", batch=batch, errors=errors)
    emit("stop", reason="signal")
    return True

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
        high_watermark=args.high_watermark,
        allow_live_claims=args.allow_live_claims,
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


def run_loop(args: argparse.Namespace, health_check: Callable[[argparse.Namespace], tuple[bool, dict[str, Any]]] = health_gate) -> int:
    install_signal_handlers()
    health_failures = 0
    with runner_lock(args.lock_file):
        write_checkpoint(args, action="started")
        if args.initial_refresh and args.apply:
            emit("full_refresh_start")
            full_refresh(args)
            if stop_requested(args, errors=health_failures):
                return 130
            emit("full_refresh_done")
        conn = open_inventory_for_run(args)
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
                    if batch and batch["failed_count"] > args.max_batch_errors:
                        write_checkpoint(args, action="stopped", stop_reason="batch_error_threshold", batch=batch, errors=health_failures)
                        emit("stop", reason="batch_error_threshold", batch_key=batch["batch_key"], failed_count=batch["failed_count"])
                        return 2
                    if batch and batch["submitted_count"] > 0:
                        emit("wait_batch", batch_key=batch["batch_key"], status=batch["status"], submitted_count=batch["submitted_count"], completed_count=batch["completed_count"], failed_count=batch["failed_count"])
                        write_checkpoint(args, action="wait_batch", batch=batch, errors=health_failures)
                        if args.once:
                            return 0
                        time.sleep(args.poll_seconds)
                        continue
                    target = batch
                else:
                    target = first_batch(conn, ("prepared",))
                    if not target:
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
                    health_failures += 1
                    write_checkpoint(args, action="health_failed", batch=target, errors=health_failures)
                    emit("health_failed", batch_key=target["batch_key"], consecutive=health_failures, health=health)
                    if health_failures >= args.max_health_failures:
                        write_checkpoint(args, action="stopped", stop_reason="health_gate", batch=target, errors=health_failures)
                        return 3
                    if args.once:
                        return 3
                    time.sleep(args.poll_seconds)
                    continue
                if stop_requested(args, batch=target, errors=health_failures):
                    return 130
                health_failures = 0

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
    parser.add_argument("--max-batch-errors", type=int, default=100)
    parser.add_argument("--write-progress-reports", action="store_true", help="rewrite full progress.json/batches.csv on every batch poll")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--high-watermark", type=int, default=batches.DEFAULT_HIGH_WATERMARK)
    parser.add_argument("--allow-live-claims", action="store_true")
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
    return args


def main() -> int:
    return run_loop(normalize_args(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
