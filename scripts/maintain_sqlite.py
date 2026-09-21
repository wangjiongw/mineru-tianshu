#!/usr/bin/env python3
"""Inspect SQLite storage or run an explicitly requested WAL checkpoint."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from task_db import TaskDB


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("checkpoint", nargs="?", choices=["checkpoint"])
    parser.add_argument("--mode", choices=["PASSIVE", "FULL", "RESTART", "TRUNCATE"], default="PASSIVE")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expect-services-stopped", action="store_true")
    args = parser.parse_args(argv)
    db = TaskDB(args.db_path, initialize=False, read_only=not bool(args.checkpoint and args.apply))
    report = {"storage": db.storage_health(), "applied": False}
    if args.checkpoint:
        if not args.apply:
            report["planned_checkpoint"] = args.mode
        else:
            if args.mode == "TRUNCATE" and not args.expect_services_stopped:
                parser.error("TRUNCATE requires --expect-services-stopped")
            report["checkpoint"] = db.checkpoint_wal(args.mode)
            report["applied"] = True
            report["storage_after"] = db.storage_health()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
