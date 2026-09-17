#!/usr/bin/env python3
"""Reconcile split-parent merges from SQLite evidence.

Dry-run is the default. Use --apply to claim and merge eligible parents.
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from parent_merge import rebuild_completed_parent_artifacts, merge_parent_task_results, validate_parent_merge_inputs
from task_db import TaskDB

INSTANCE_ID = os.environ.get("INSTANCE_ID") or socket.gethostname()
DATA_DIR = Path(f"/share/wangjiong/databases/mineru_database/{INSTANCE_ID}")
DEFAULT_DB_PATH = Path(os.environ.get("DATABASE_PATH", str(DATA_DIR / "mineru_tianshu.db")))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", str(DATA_DIR / "mineru_outputs")))


def build_parser():
    parser = argparse.ArgumentParser(description="Recover and reconcile parent task merges")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--apply", action="store_true", help="claim and merge eligible parents")
    parser.add_argument("--watch", action="store_true", help="repeat until interrupted")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--task-id")
    parser.add_argument("--force", action="store_true", help="allow claiming currently merging parents")
    parser.add_argument(
        "--rebuild-completed",
        action="store_true",
        help="rebuild artifacts for completed split parents without changing task status",
    )
    parser.add_argument("--report-json", action="store_true")
    return parser


def summarize(candidates):
    summary = {"finalizable": 0, "remergeable": 0, "blocked": 0}
    for item in candidates:
        summary[item["classification"]] = summary.get(item["classification"], 0) + 1
    return summary


def run_completed_rebuild(args, db):
    candidates = db.list_completed_parent_tasks(limit=args.limit, task_id=args.task_id)
    report = {
        "summary": {"rebuildable": 0, "blocked": 0},
        "tasks": [],
        "applied": args.apply,
        "rebuild_completed": True,
    }
    for item in candidates:
        parent = db.get_task_with_children(item["task_id"])
        _children, blocked_reasons = validate_parent_merge_inputs(parent or {})
        classification = "blocked" if blocked_reasons else "rebuildable"
        entry = {
            "task_id": item["task_id"],
            "status": item["status"],
            "classification": classification,
            "child_count": item["child_count"],
            "real_child_completed": item.get("real_child_completed") or 0,
        }
        if blocked_reasons:
            entry["blocked_reasons"] = blocked_reasons
        elif args.apply:
            try:
                out = rebuild_completed_parent_artifacts(
                    task_db=db,
                    parent_task_id=item["task_id"],
                    output_dir=args.output_dir,
                )
                entry["rebuilt"] = True
                entry["result_path"] = str(out)
            except Exception as exc:
                entry["classification"] = "blocked"
                entry["rebuilt"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
        report["tasks"].append(entry)
        report["summary"][entry["classification"]] += 1
    return report


def run_once(args, owner):
    db = TaskDB(args.db_path, initialize=args.apply, read_only=not args.apply)
    if args.rebuild_completed:
        return run_completed_rebuild(args, db)
    candidates = db.list_parent_merge_candidates(
        stale_seconds=args.stale_seconds,
        max_attempts=args.max_attempts,
        limit=args.limit,
        task_id=args.task_id,
        force=args.force,
    )
    report = {"summary": {"finalizable": 0, "remergeable": 0, "blocked": 0}, "tasks": [], "applied": args.apply}

    for item in candidates:
        classification = item["classification"]
        blocked_reasons = []
        if classification in {"finalizable", "remergeable"}:
            parent = db.get_task_with_children(item["task_id"])
            _completed_children, blocked_reasons = validate_parent_merge_inputs(parent or {})
            if blocked_reasons:
                classification = "blocked"
        entry = {
            "task_id": item["task_id"],
            "status": item["status"],
            "classification": classification,
            "child_count": item["child_count"],
            "real_child_completed": item["real_child_completed"],
            "merge_attempts": item.get("merge_attempts") or 0,
            "merge_error": item.get("merge_error"),
        }
        if blocked_reasons:
            entry["blocked_reasons"] = blocked_reasons
            if args.apply:
                claimed = db.claim_parent_merge(
                    item["task_id"],
                    owner,
                    stale_seconds=args.stale_seconds,
                    max_attempts=args.max_attempts,
                    force=args.force,
                )
                entry["claimed"] = claimed
                if claimed:
                    message = "Blocked parent merge prevalidation: " + "; ".join(blocked_reasons)
                    db.block_parent_merge(item["task_id"], owner, message, max_attempts=args.max_attempts)
                    entry["blocked"] = True
                    entry["merge_error"] = message
        if not blocked_reasons and args.apply and classification in {"finalizable", "remergeable"}:
            claimed = db.claim_parent_merge(
                item["task_id"],
                owner,
                stale_seconds=args.stale_seconds,
                max_attempts=args.max_attempts,
                force=args.force,
            )
            entry["claimed"] = claimed
            if claimed:
                try:
                    out = merge_parent_task_results(
                        task_db=db,
                        parent_task_id=item["task_id"],
                        output_dir=args.output_dir,
                        merge_owner=owner,
                        cleanup_child_files=True,
                    )
                    entry["merged"] = True
                    entry["result_path"] = str(out)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    entry["merged"] = False
                    entry["error"] = message
                    refreshed = db.get_task(item["task_id"]) or {}
                    attempts = int(refreshed.get("merge_attempts") or 0)
                    if attempts >= args.max_attempts:
                        db.block_parent_merge(item["task_id"], owner, message, max_attempts=args.max_attempts)
                        entry["blocked"] = True
                    else:
                        db.release_parent_merge(item["task_id"], owner, message)
        report["tasks"].append(entry)
    report["summary"] = summarize(report["tasks"])
    return report


def emit(report, as_json):
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    summary = report["summary"]
    if report.get("rebuild_completed"):
        mode = "apply" if report["applied"] else "dry-run"
        print(f"mode={mode} rebuildable={summary.get('rebuildable', 0)} blocked={summary.get('blocked', 0)}")
        for item in report["tasks"]:
            suffix = f" rebuilt result_path={item.get('result_path')}" if item.get("rebuilt") else ""
            if item.get("error"):
                suffix = f" error={item['error']}"
            print(
                f"{item['classification']} {item['task_id']} status={item['status']} "
                f"children={item['real_child_completed']}/{item['child_count']}{suffix}"
            )
        return
    mode = "apply" if report["applied"] else "dry-run"
    print(f"mode={mode} finalizable={summary.get('finalizable', 0)} remergeable={summary.get('remergeable', 0)} blocked={summary.get('blocked', 0)}")
    for item in report["tasks"]:
        suffix = ""
        if item.get("merged"):
            suffix = f" merged result_path={item.get('result_path')}"
        elif item.get("error"):
            suffix = f" error={item['error']}"
        print(
            f"{item['classification']} {item['task_id']} status={item['status']} "
            f"children={item['real_child_completed']}/{item['child_count']} attempts={item['merge_attempts']}{suffix}"
        )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rebuild_completed and args.watch:
        parser.error("--rebuild-completed cannot be combined with --watch")
    owner = f"reconcile-parent-merges:{os.getpid()}"
    while True:
        report = run_once(args, owner)
        emit(report, args.report_json)
        if not args.watch:
            return 0
        time.sleep(max(args.interval_seconds, 0.1))


if __name__ == "__main__":
    raise SystemExit(main())
