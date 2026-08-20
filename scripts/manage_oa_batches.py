#!/usr/bin/env python3
"""Build and operate a durable OA corpus inventory and bounded task batches.

The inventory is a sidecar database. Source PDFs and parsed outputs are read-only.
Only ``submit`` writes MinerU task state, and it requires the Redis claim pause by
default so a complete batch can be staged before workers start claiming it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_SOURCE_ROOT = Path("/share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256")
DEFAULT_PARSED_ROOT = Path("/share/wangjiong/databases/escorpus-assets/pdfs_parsed")
DEFAULT_TASK_DB = Path(
    "/share/wangjiong/databases/mineru_database/mineru-runner-worker-0/mineru_tianshu.db"
)
DEFAULT_INVENTORY = DEFAULT_PARSED_ROOT / "_inventory" / "oa_queue.sqlite3"
DEFAULT_LEGACY_PROGRESS = DEFAULT_PARSED_ROOT / "_submit_progress.json"
DEFAULT_BATCH_SIZE = 5000
DEFAULT_HIGH_WATERMARK = 5000
DB_CHUNK = 500
SOURCE_FLUSH = 5000
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SHA_PDF_RE = re.compile(r"^([0-9a-f]{64})\.pdf$")
ELIGIBLE_STATES = ("unsubmitted", "failed_retryable")
READY_STATES = ("complete_valid", "legacy_result_only")
TERMINAL_BATCH_STATES = ("completed", "completed_with_errors", "completed_with_deferred")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%F %T')}] {message}", flush=True)


def chunks(values: Sequence[str], size: int = DB_CHUNK) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def is_sha256(value: str | None) -> bool:
    return bool(value and SHA_RE.fullmatch(value.lower()))


def validate_source_path(source_root: Path, sha256: str, source_path: Path) -> Path:
    """Return a trusted corpus path or raise if it escapes the SHA layout."""
    sha256 = sha256.lower()
    if not is_sha256(sha256):
        raise ValueError(f"invalid sha256: {sha256!r}")
    root = source_root.resolve(strict=True)
    resolved = source_path.resolve(strict=True)
    expected = (root / sha256[:2] / f"{sha256}.pdf").resolve(strict=True)
    if resolved != expected or resolved.parent.parent != root:
        raise ValueError(f"source path is outside the trusted SHA layout: {source_path}")
    if not resolved.is_file():
        raise ValueError(f"source path is not a file: {resolved}")
    return resolved


def strict_result_complete(result_dir: Path) -> tuple[bool, str | None]:
    """Require a non-empty Markdown result and parseable MinerU model JSON."""
    if not result_dir.is_dir():
        return False, "result directory missing"
    markdown = result_dir / "result.md"
    model = result_dir / "mineru_model.json"
    try:
        if not markdown.is_file() or markdown.stat().st_size <= 0:
            return False, "result.md missing or empty"
        if not model.is_file() or model.stat().st_size <= 0:
            return False, "mineru_model.json missing or empty"
        with model.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"invalid result: {type(exc).__name__}: {exc}"
    return True, None


def connect_inventory(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    initialize_schema(conn)
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            sha256 TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_size INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            seen_generation INTEGER NOT NULL,
            parsed_path TEXT,
            parsed_complete INTEGER NOT NULL DEFAULT 0,
            parsed_error TEXT,
            db_task_id TEXT,
            db_status TEXT,
            db_result_path TEXT,
            db_error TEXT,
            state TEXT NOT NULL DEFAULT 'unsubmitted',
            current_batch_id INTEGER,
            batch_position INTEGER,
            submit_attempts INTEGER NOT NULL DEFAULT 0,
            last_submit_at TEXT,
            last_submit_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_documents_state ON documents(state, sha256);
        CREATE INDEX IF NOT EXISTS idx_documents_batch ON documents(current_batch_id, batch_position);
        CREATE INDEX IF NOT EXISTS idx_documents_db_task ON documents(db_task_id);

        CREATE TABLE IF NOT EXISTS batches (
            batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_key TEXT NOT NULL UNIQUE,
            generation INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'prepared',
            item_count INTEGER NOT NULL,
            submitted_count INTEGER NOT NULL DEFAULT 0,
            completed_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS batch_items (
            batch_id INTEGER NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            sha256 TEXT NOT NULL REFERENCES documents(sha256),
            item_status TEXT NOT NULL DEFAULT 'prepared',
            task_id TEXT,
            submit_error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (batch_id, position),
            UNIQUE (batch_id, sha256)
        );

        CREATE INDEX IF NOT EXISTS idx_batch_items_sha ON batch_items(sha256);
        CREATE INDEX IF NOT EXISTS idx_batch_items_status ON batch_items(batch_id, item_status);

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL,
            batch_id INTEGER,
            sha256 TEXT,
            detail_json TEXT NOT NULL
        );
        """
    )
    conn.commit()


def get_metadata(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_metadata(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO metadata(key,value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def iter_source_pdfs(source_root: Path) -> Iterator[tuple[str, Path, os.stat_result]]:
    for shard in sorted(source_root.iterdir(), key=lambda item: item.name):
        if not shard.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", shard.name):
            continue
        with os.scandir(shard) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                match = SHA_PDF_RE.fullmatch(entry.name)
                if not match or not entry.is_file(follow_symlinks=False):
                    continue
                sha256 = match.group(1)
                if sha256[:2] != shard.name:
                    continue
                yield sha256, Path(entry.path), entry.stat(follow_symlinks=False)


def scan_sources(
    conn: sqlite3.Connection,
    source_root: Path,
    generation: int,
) -> set[str]:
    now = utc_now()
    source_shas: set[str] = set()
    pending: list[tuple[object, ...]] = []
    sql = """
        INSERT INTO documents(
            sha256,source_path,source_size,source_mtime_ns,seen_generation,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(sha256) DO UPDATE SET
            source_path=excluded.source_path,
            source_size=excluded.source_size,
            source_mtime_ns=excluded.source_mtime_ns,
            seen_generation=excluded.seen_generation,
            updated_at=excluded.updated_at
    """
    for sha256, path, stat_result in iter_source_pdfs(source_root):
        source_shas.add(sha256)
        pending.append(
            (
                sha256,
                str(path),
                stat_result.st_size,
                stat_result.st_mtime_ns,
                generation,
                now,
                now,
            )
        )
        if len(pending) >= SOURCE_FLUSH:
            conn.executemany(sql, pending)
            conn.commit()
            pending.clear()
            if len(source_shas) % 50000 == 0:
                log(f"源文件扫描: {len(source_shas):,}")
    if pending:
        conn.executemany(sql, pending)
        conn.commit()
    conn.execute(
        """
        UPDATE documents
        SET state='missing_source', updated_at=?
        WHERE seen_generation<>?
        """,
        (now, generation),
    )
    conn.commit()
    return source_shas


def open_task_db_readonly(task_db: Path) -> sqlite3.Connection:
    uri = task_db.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def load_task_state(conn: sqlite3.Connection, task_db: Path, source_shas: set[str]) -> Counter:
    conn.execute(
        """
        UPDATE documents
        SET db_task_id=NULL,db_status=NULL,db_result_path=NULL,db_error=NULL
        WHERE seen_generation=(SELECT CAST(value AS INTEGER) FROM metadata WHERE key='scan_generation')
        """
    )
    conn.commit()
    counts: Counter = Counter()
    task_conn = open_task_db_readonly(task_db)
    try:
        # Later statuses overwrite earlier ones so completed remains authoritative.
        statuses = ("failed", "timeout", "pending", "processing", "merging", "completed")
        update_sql = """
            UPDATE documents
            SET db_task_id=?,db_status=?,db_result_path=?,db_error=?,updated_at=?
            WHERE sha256=?
        """
        for status in statuses:
            rows: list[tuple[object, ...]] = []
            query = """
                SELECT task_id,file_hash,status,result_path,error_message
                FROM tasks INDEXED BY idx_status
                WHERE status=? AND parent_task_id IS NULL
                  AND backend='hybrid-auto-engine' AND lang='auto' AND method='auto'
                  AND length(file_hash)=64
            """
            scanned = 0
            for row in task_conn.execute(query, (status,)):
                scanned += 1
                sha256 = (row["file_hash"] or "").lower()
                if sha256 not in source_shas:
                    continue
                rows.append(
                    (
                        row["task_id"],
                        row["status"],
                        row["result_path"],
                        row["error_message"],
                        utc_now(),
                        sha256,
                    )
                )
                if len(rows) >= SOURCE_FLUSH:
                    conn.executemany(update_sql, rows)
                    conn.commit()
                    counts[status] += len(rows)
                    rows.clear()
            if rows:
                conn.executemany(update_sql, rows)
                conn.commit()
                counts[status] += len(rows)
            log(f"任务库扫描 {status}: scanned={scanned:,} matched={counts[status]:,}")
    finally:
        task_conn.close()
    return counts


def iter_result_dirs(parsed_root: Path) -> Iterator[tuple[str, Path]]:
    seen: set[str] = set()
    for shard_name in (f"{value:02x}" for value in range(256)):
        shard = parsed_root / shard_name
        if not shard.is_dir():
            continue
        with os.scandir(shard) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if entry.is_dir(follow_symlinks=False) and is_sha256(entry.name):
                    sha256 = entry.name.lower()
                    if sha256[:2] == shard_name:
                        seen.add(sha256)
                        yield sha256, Path(entry.path)
    with os.scandir(parsed_root) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if not entry.is_dir(follow_symlinks=False) or not is_sha256(entry.name):
                continue
            sha256 = entry.name.lower()
            if sha256 not in seen:
                yield sha256, Path(entry.path)


def scan_results(conn: sqlite3.Connection, parsed_root: Path, source_shas: set[str]) -> Counter:
    conn.execute(
        "UPDATE documents SET parsed_path=NULL,parsed_complete=0,parsed_error=NULL "
        "WHERE seen_generation=(SELECT CAST(value AS INTEGER) FROM metadata WHERE key='scan_generation')"
    )
    conn.commit()
    counts: Counter = Counter()
    updates: list[tuple[object, ...]] = []
    for sha256, result_dir in iter_result_dirs(parsed_root):
        if sha256 not in source_shas:
            counts["result_not_in_source"] += 1
            continue
        complete, error = strict_result_complete(result_dir)
        counts["complete" if complete else "incomplete"] += 1
        updates.append((str(result_dir), int(complete), error, utc_now(), sha256))
        if len(updates) >= SOURCE_FLUSH:
            conn.executemany(
                """
                UPDATE documents
                SET parsed_path=?,parsed_complete=?,parsed_error=?,updated_at=?
                WHERE sha256=?
                """,
                updates,
            )
            conn.commit()
            updates.clear()
            seen = counts["complete"] + counts["incomplete"]
            if seen % 50000 == 0:
                log(f"结果目录扫描: {seen:,}")
    if updates:
        conn.executemany(
            """
            UPDATE documents
            SET parsed_path=?,parsed_complete=?,parsed_error=?,updated_at=?
            WHERE sha256=?
            """,
            updates,
        )
        conn.commit()
    return counts


def classify_documents(conn: sqlite3.Connection, generation: int) -> Counter:
    now = utc_now()
    conn.execute(
        """
        UPDATE documents
        SET state = CASE
            WHEN seen_generation<>? THEN 'missing_source'
            WHEN parsed_complete=1 AND db_status='completed' THEN 'complete_valid'
            WHEN parsed_complete=1 THEN 'legacy_result_only'
            WHEN db_status='completed' THEN 'completed_db_unsynced'
            WHEN db_status='pending' THEN 'queued'
            WHEN db_status IN ('processing','merging') THEN 'active'
            WHEN db_status IN ('failed','timeout') AND (
                lower(COALESCE(db_error,'')) LIKE '%invalid pdf%'
                OR lower(COALESCE(db_error,'')) LIKE '%pdfium%'
                OR lower(COALESCE(db_error,'')) LIKE '%password%'
                OR lower(COALESCE(db_error,'')) LIKE '%encrypted%'
                OR lower(COALESCE(db_error,'')) LIKE '%cannot open%'
            ) THEN 'failed_permanent'
            WHEN db_status IN ('failed','timeout') THEN 'failed_retryable'
            ELSE 'unsubmitted'
        END,
        updated_at=?
        """,
        (generation, now),
    )
    conn.commit()
    refresh_batch_progress(conn)
    return Counter(
        {row["state"]: row["count"] for row in conn.execute(
            "SELECT state,COUNT(*) AS count FROM documents GROUP BY state ORDER BY state"
        )}
    )


def classify_batch_documents(conn: sqlite3.Connection, batch_key: str) -> Counter:
    now = utc_now()
    conn.execute(
        """
        UPDATE documents
        SET state = CASE
            WHEN parsed_complete=1 AND db_status='completed' THEN 'complete_valid'
            WHEN parsed_complete=1 THEN 'legacy_result_only'
            WHEN db_status='completed' THEN 'completed_db_unsynced'
            WHEN db_status='pending' THEN 'queued'
            WHEN db_status IN ('processing','merging') THEN 'active'
            WHEN db_status IN ('failed','timeout') AND (
                lower(COALESCE(db_error,'')) LIKE '%invalid pdf%'
                OR lower(COALESCE(db_error,'')) LIKE '%pdfium%'
                OR lower(COALESCE(db_error,'')) LIKE '%password%'
                OR lower(COALESCE(db_error,'')) LIKE '%encrypted%'
                OR lower(COALESCE(db_error,'')) LIKE '%cannot open%'
            ) THEN 'failed_permanent'
            WHEN db_status IN ('failed','timeout') THEN 'failed_retryable'
            ELSE state
        END,
        updated_at=?
        WHERE current_batch_id=(SELECT batch_id FROM batches WHERE batch_key=?)
        """,
        (now, batch_key),
    )
    return Counter(
        {
            row["state"]: row["count"]
            for row in conn.execute(
                """
                SELECT d.state,COUNT(*) AS count
                FROM documents d JOIN batches b ON d.current_batch_id=b.batch_id
                WHERE b.batch_key=?
                GROUP BY d.state
                """,
                (batch_key,),
            )
        }
    )




def refresh_one_batch_progress(conn: sqlite3.Connection, batch_key: str) -> None:
    now = utc_now()
    batch = conn.execute("SELECT batch_id,item_count FROM batches WHERE batch_key=?", (batch_key,)).fetchone()
    if not batch:
        raise RuntimeError(f"unknown batch: {batch_key}")
    conn.execute(
        """
        UPDATE batch_items
        SET item_status = CASE
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 IN ('complete_valid','legacy_result_only','completed_db_unsynced') THEN 'completed'
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 = 'failed_permanent' THEN 'error'
            WHEN item_status='deferred' THEN 'deferred'
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 IN ('queued','active') THEN 'submitted'
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 IN ('failed_retryable','failed_permanent') THEN 'error'
            ELSE item_status
        END,
        task_id=COALESCE(
            (SELECT db_task_id FROM documents d WHERE d.sha256=batch_items.sha256),
            task_id
        ),
        updated_at=?
        WHERE batch_id=?
        """,
        (now, batch["batch_id"]),
    )
    item_counts = Counter(
        {
            item["item_status"]: item["count"]
            for item in conn.execute(
                "SELECT item_status,COUNT(*) AS count FROM batch_items WHERE batch_id=? GROUP BY item_status",
                (batch["batch_id"],),
            )
        }
    )
    state_counts = Counter(
        {
            item["state"]: item["count"]
            for item in conn.execute(
                """
                SELECT d.state,COUNT(*) AS count
                FROM batch_items bi JOIN documents d USING(sha256)
                WHERE bi.batch_id=?
                GROUP BY d.state
                """,
                (batch["batch_id"],),
            )
        }
    )
    completed = item_counts["completed"]
    submitted = item_counts["submitted"]
    deferred = item_counts["deferred"]
    permanent_failed = state_counts["failed_permanent"]
    failed = int(
        conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM batch_items bi JOIN documents d USING(sha256)
            WHERE bi.batch_id=?
              AND (
                  bi.item_status='error'
                  OR (bi.item_status!='deferred' AND d.state IN ('failed_retryable','failed_permanent'))
              )
            """,
            (batch["batch_id"],),
        ).fetchone()["count"]
    )
    if completed == batch["item_count"]:
        status = "completed"
    elif completed + permanent_failed == batch["item_count"] and permanent_failed:
        status = "completed_with_errors"
    elif completed + permanent_failed + deferred == batch["item_count"] and deferred:
        status = "completed_with_deferred"
    elif submitted or completed:
        status = "partial" if failed else "running"
    elif failed or deferred:
        status = "partial"
    else:
        status = "prepared"
    conn.execute(
        """
        UPDATE batches
        SET status=?,submitted_count=?,completed_count=?,failed_count=?,updated_at=?
        WHERE batch_id=?
        """,
        (status, submitted, completed, failed, now, batch["batch_id"]),
    )
    conn.commit()


def defer_batch_items(
    conn: sqlite3.Connection,
    batch_key: str,
    task_ids: Sequence[str],
    *,
    reason: str,
) -> dict[str, object]:
    """Release stale batch tails without cancelling their underlying tasks."""
    unique_task_ids = tuple(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    if not unique_task_ids:
        return {"batch_key": batch_key, "deferred": 0, "task_ids": []}
    batch = conn.execute(
        "SELECT batch_id FROM batches WHERE batch_key=?",
        (batch_key,),
    ).fetchone()
    if not batch:
        raise RuntimeError(f"unknown batch: {batch_key}")
    placeholders = ",".join("?" for _ in unique_task_ids)
    rows = conn.execute(
        f"""
        SELECT task_id,sha256
        FROM batch_items
        WHERE batch_id=? AND item_status IN ('submitted','error')
          AND task_id IN ({placeholders})
        ORDER BY position
        """,
        (batch["batch_id"], *unique_task_ids),
    ).fetchall()
    selected_task_ids = [row["task_id"] for row in rows]
    if not selected_task_ids:
        return {"batch_key": batch_key, "deferred": 0, "task_ids": []}
    selected_placeholders = ",".join("?" for _ in selected_task_ids)
    now = utc_now()
    conn.execute(
        f"""
        UPDATE batch_items
        SET item_status='deferred',submit_error=NULL,updated_at=?
        WHERE batch_id=? AND item_status IN ('submitted','error')
          AND task_id IN ({selected_placeholders})
        """,
        (now, batch["batch_id"], *selected_task_ids),
    )
    conn.execute(
        "INSERT INTO events(event_time,event_type,batch_id,detail_json) VALUES (?,?,?,?)",
        (
            now,
            "batch_tail_deferred",
            batch["batch_id"],
            json.dumps(
                {"count": len(selected_task_ids), "reason": reason, "task_ids": selected_task_ids},
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    )
    conn.commit()
    refresh_one_batch_progress(conn, batch_key)
    return {
        "batch_key": batch_key,
        "deferred": len(selected_task_ids),
        "task_ids": selected_task_ids,
        "reason": reason,
    }


def refresh_batch_progress(conn: sqlite3.Connection) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE batch_items
        SET item_status = CASE
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 IN ('complete_valid','legacy_result_only','completed_db_unsynced') THEN 'completed'
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 = 'failed_permanent' THEN 'error'
            WHEN item_status='deferred' THEN 'deferred'
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 IN ('queued','active') THEN 'submitted'
            WHEN (SELECT state FROM documents d WHERE d.sha256=batch_items.sha256)
                 IN ('failed_retryable','failed_permanent') THEN 'error'
            ELSE item_status
        END,
        task_id=COALESCE(
            (SELECT db_task_id FROM documents d WHERE d.sha256=batch_items.sha256),
            task_id
        ),
        updated_at=?
        """,
        (now,),
    )
    for row in conn.execute("SELECT batch_id,item_count FROM batches ORDER BY batch_id").fetchall():
        item_counts = Counter(
            {
                item["item_status"]: item["count"]
                for item in conn.execute(
                    "SELECT item_status,COUNT(*) AS count FROM batch_items WHERE batch_id=? GROUP BY item_status",
                    (row["batch_id"],),
                )
            }
        )
        state_counts = Counter(
            {
                item["state"]: item["count"]
                for item in conn.execute(
                    """
                    SELECT d.state,COUNT(*) AS count
                    FROM batch_items bi JOIN documents d USING(sha256)
                    WHERE bi.batch_id=?
                    GROUP BY d.state
                    """,
                    (row["batch_id"],),
                )
            }
        )
        completed = item_counts["completed"]
        submitted = item_counts["submitted"]
        deferred = item_counts["deferred"]
        permanent_failed = state_counts["failed_permanent"]
        failed = int(
            conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM batch_items bi JOIN documents d USING(sha256)
                WHERE bi.batch_id=?
                  AND (
                      bi.item_status='error'
                      OR (bi.item_status!='deferred' AND d.state IN ('failed_retryable','failed_permanent'))
                  )
                """,
                (row["batch_id"],),
            ).fetchone()["count"]
        )
        if completed == row["item_count"]:
            status = "completed"
        elif completed + permanent_failed == row["item_count"] and permanent_failed:
            status = "completed_with_errors"
        elif completed + permanent_failed + deferred == row["item_count"] and deferred:
            status = "completed_with_deferred"
        elif submitted or completed:
            status = "partial" if failed else "running"
        elif failed or deferred:
            status = "partial"
        else:
            status = "prepared"
        conn.execute(
            """
            UPDATE batches
            SET status=?,submitted_count=?,completed_count=?,failed_count=?,updated_at=?
            WHERE batch_id=?
            """,
            (status, submitted, completed, failed, now, row["batch_id"]),
        )
    conn.commit()


def refresh_batch_from_task_db(
    conn: sqlite3.Connection,
    inventory_path: Path,
    legacy_progress_path: Path,
    task_db_path: Path,
    batch_key: str,
    write_reports: bool = True,
) -> dict[str, object]:
    batch = conn.execute("SELECT batch_id FROM batches WHERE batch_key=?", (batch_key,)).fetchone()
    if not batch:
        raise RuntimeError(f"unknown batch: {batch_key}")
    rows = conn.execute(
        """
        SELECT bi.sha256,COALESCE(bi.task_id,d.db_task_id) AS task_id
        FROM batch_items bi JOIN documents d USING(sha256)
        WHERE bi.batch_id=? AND COALESCE(bi.task_id,d.db_task_id) IS NOT NULL
        """,
        (batch["batch_id"],),
    ).fetchall()
    task_ids = [row["task_id"] for row in rows if row["task_id"]]
    task_to_sha = {row["task_id"]: row["sha256"] for row in rows if row["task_id"]}
    updated = 0
    if task_ids and task_db_path.exists():
        task_uri = f"file:{task_db_path}?mode=ro"
        task_conn = sqlite3.connect(task_uri, uri=True, timeout=10.0)
        task_conn.row_factory = sqlite3.Row
        try:
            columns = {row["name"] for row in task_conn.execute("PRAGMA table_info(tasks)").fetchall()}
            select_columns = ["task_id", "status"]
            for optional in ("result_path", "error_message", "file_hash"):
                if optional in columns:
                    select_columns.append(optional)
            for task_chunk in chunks(task_ids):
                placeholders = ",".join("?" for _ in task_chunk)
                task_rows = task_conn.execute(
                    f"SELECT {','.join(select_columns)} FROM tasks WHERE task_id IN ({placeholders})",
                    tuple(task_chunk),
                ).fetchall()
                now = utc_now()
                updates = []
                for task in task_rows:
                    keys = task.keys()
                    sha = task["file_hash"] if "file_hash" in keys and task["file_hash"] else task_to_sha.get(task["task_id"])
                    if not sha:
                        continue
                    updates.append(
                        (
                            task["task_id"],
                            task["status"],
                            task["result_path"] if "result_path" in keys else None,
                            task["error_message"] if "error_message" in keys else None,
                            now,
                            sha,
                        )
                    )
                if updates:
                    conn.executemany(
                        """
                        UPDATE documents
                        SET db_task_id=?,db_status=?,db_result_path=?,db_error=?,updated_at=?
                        WHERE sha256=?
                        """,
                        updates,
                    )
                    updated += len(updates)
        finally:
            task_conn.close()
    states = classify_batch_documents(conn, batch_key)
    refresh_one_batch_progress(conn, batch_key)
    batch_row = conn.execute("SELECT * FROM batches WHERE batch_key=?", (batch_key,)).fetchone()
    batch_payload = dict(batch_row) if batch_row else None
    refresh_payload = {"batch_key": batch_key, "task_ids": len(task_ids), "updated": updated, "states": dict(states)}
    if write_reports:
        report = write_progress_reports(conn, inventory_path, legacy_progress_path)
        report["batch_refresh"] = refresh_payload
        return report
    return {
        "inventory_path": str(inventory_path),
        "updated_at": utc_now(),
        "batch_refresh": refresh_payload,
        "batch": batch_payload,
    }


def legacy_progress_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        submitted = payload.get("submitted")
        return len(submitted) if isinstance(submitted, list) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def read_live_queue_stats() -> dict[str, object]:
    if os.getenv("REDIS_QUEUE_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
        return {"available": False}
    try:
        import redis

        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=5,
        )
        return {
            "available": bool(client.ping()),
            "queued": client.zcard(os.getenv("REDIS_QUEUE_KEY", "tianshu:task_queue")),
            "processing": client.hlen(os.getenv("REDIS_PROCESSING_KEY", "tianshu:processing")),
            "claim_pause": client.get(os.getenv("REDIS_CLAIM_PAUSE_KEY", "tianshu:claim_pause")),
        }
    except Exception as exc:  # status reporting must remain available without Redis
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def build_progress_report(
    conn: sqlite3.Connection,
    inventory_path: Path,
    legacy_progress_path: Path,
) -> dict[str, object]:
    states = {
        row["state"]: row["count"]
        for row in conn.execute(
            "SELECT state,COUNT(*) AS count FROM documents GROUP BY state ORDER BY state"
        )
    }
    total = sum(states.values())
    ready = sum(states.get(state, 0) for state in READY_STATES)
    db_completed = ready + states.get("completed_db_unsynced", 0)
    batches = [dict(row) for row in conn.execute(
        """
        SELECT b.batch_id,b.batch_key,b.generation,b.ordinal,b.status,b.item_count,
               b.submitted_count,b.completed_count,b.failed_count,
               (SELECT COUNT(*) FROM batch_items bi
                WHERE bi.batch_id=b.batch_id AND bi.item_status='deferred') AS deferred_count,
               b.created_at,b.updated_at
        FROM batches b ORDER BY b.batch_id
        """
    )]
    return {
        "schema_version": 2,
        "updated_at": utc_now(),
        "inventory_path": str(inventory_path),
        "total_source": total,
        "states": states,
        "strict_results_ready": ready,
        "strict_results_ready_ratio": round(ready / total, 8) if total else 0.0,
        "db_completed_or_ready": db_completed,
        "db_completed_or_ready_ratio": round(db_completed / total, 8) if total else 0.0,
        "remaining_without_strict_result": total - ready,
        "legacy_submit_progress_count": legacy_progress_count(legacy_progress_path),
        "legacy_submit_progress_is_authoritative": False,
        "queue": read_live_queue_stats(),
        "batch_size": int(get_metadata(conn, "batch_size", str(DEFAULT_BATCH_SIZE))),
        "batch_count": len(batches),
        "batches": batches,
    }


def write_progress_reports(
    conn: sqlite3.Connection,
    inventory_path: Path,
    legacy_progress_path: Path,
) -> dict[str, object]:
    report = build_progress_report(conn, inventory_path, legacy_progress_path)
    report_dir = inventory_path.parent
    atomic_write_text(
        report_dir / "progress.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    csv_path = report_dir / "batches.csv"
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "batch_id",
                "batch_key",
                "generation",
                "ordinal",
                "status",
                "item_count",
                "submitted_count",
                "completed_count",
                "failed_count",
                "deferred_count",
                "created_at",
                "updated_at",
            ),
        )
        writer.writeheader()
        writer.writerows(report["batches"])
    tmp.replace(csv_path)
    return report


def refresh_inventory(
    conn: sqlite3.Connection,
    inventory_path: Path,
    source_root: Path,
    parsed_root: Path,
    task_db: Path,
    legacy_progress_path: Path,
) -> dict[str, object]:
    generation = int(get_metadata(conn, "scan_generation", "0")) + 1
    set_metadata(conn, "scan_generation", generation)
    set_metadata(conn, "source_root", source_root)
    set_metadata(conn, "parsed_root", parsed_root)
    set_metadata(conn, "task_db", task_db)
    set_metadata(conn, "last_refresh_started_at", utc_now())
    conn.commit()

    log(f"开始全局刷新 generation={generation}")
    source_shas = scan_sources(conn, source_root, generation)
    log(f"源文件总数: {len(source_shas):,}")
    task_counts = load_task_state(conn, task_db, source_shas)
    result_counts = scan_results(conn, parsed_root, source_shas)
    state_counts = classify_documents(conn, generation)
    if sum(state_counts.values()) != len(source_shas):
        raise RuntimeError(
            f"inventory invariant failed: states={sum(state_counts.values())} sources={len(source_shas)}"
        )
    set_metadata(conn, "last_refresh_completed_at", utc_now())
    set_metadata(conn, "source_count", len(source_shas))
    conn.commit()
    report = write_progress_reports(conn, inventory_path, legacy_progress_path)
    log(f"任务库匹配: {dict(task_counts)}")
    log(f"结果扫描: {dict(result_counts)}")
    log(f"全局分类: {dict(state_counts)}")
    return report


def remove_prepared_batches(conn: sqlite3.Connection) -> int:
    ids = [
        row["batch_id"]
        for row in conn.execute("SELECT batch_id FROM batches WHERE status='prepared'")
    ]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE documents SET current_batch_id=NULL,batch_position=NULL "
        f"WHERE current_batch_id IN ({placeholders})",
        ids,
    )
    conn.execute(f"DELETE FROM batches WHERE batch_id IN ({placeholders})", ids)
    conn.commit()
    return len(ids)


def write_batch_manifests(conn: sqlite3.Connection, inventory_path: Path) -> None:
    manifest_dir = inventory_path.parent / "batches"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for batch in conn.execute("SELECT batch_id,batch_key FROM batches ORDER BY batch_id"):
        path = manifest_dir / f"{batch['batch_key']}.jsonl"
        tmp = path.with_suffix(path.suffix + ".tmp")
        digest = hashlib.sha256()
        with tmp.open("w", encoding="utf-8") as handle:
            for row in conn.execute(
                """
                SELECT bi.position,bi.sha256,bi.item_status,d.source_path,d.state
                FROM batch_items bi JOIN documents d USING(sha256)
                WHERE bi.batch_id=? ORDER BY bi.position
                """,
                (batch["batch_id"],),
            ):
                line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
        tmp.replace(path)
        atomic_write_text(path.with_suffix(".sha256"), digest.hexdigest() + "\n")


def prepare_batches(
    conn: sqlite3.Connection,
    inventory_path: Path,
    legacy_progress_path: Path,
    batch_size: int,
    replace_prepared: bool,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if replace_prepared:
        removed = remove_prepared_batches(conn)
        log(f"替换 prepared 批次: removed={removed}")
    elif conn.execute("SELECT 1 FROM batches WHERE status='prepared' LIMIT 1").fetchone():
        raise RuntimeError("prepared batches already exist; use --replace-prepared to rebuild them")

    eligible = [
        row["sha256"]
        for row in conn.execute(
            """
            SELECT sha256 FROM documents
            WHERE state IN ('unsubmitted','failed_retryable')
              AND current_batch_id IS NULL
            ORDER BY sha256
            """
        )
    ]
    generation = int(get_metadata(conn, "batch_generation", "0")) + 1
    set_metadata(conn, "batch_generation", generation)
    set_metadata(conn, "batch_size", batch_size)
    now = utc_now()
    created = 0
    for ordinal, group in enumerate(chunks(eligible, batch_size), 1):
        batch_key = f"oa-g{generation:03d}-b{ordinal:05d}"
        cursor = conn.execute(
            """
            INSERT INTO batches(batch_key,generation,ordinal,status,item_count,created_at,updated_at)
            VALUES (?,?,?,'prepared',?,?,?)
            """,
            (batch_key, generation, ordinal, len(group), now, now),
        )
        batch_id = cursor.lastrowid
        conn.executemany(
            """
            INSERT INTO batch_items(batch_id,position,sha256,item_status,updated_at)
            VALUES (?,?,?,'prepared',?)
            """,
            ((batch_id, position, sha256, now) for position, sha256 in enumerate(group, 1)),
        )
        conn.executemany(
            "UPDATE documents SET current_batch_id=?,batch_position=?,updated_at=? WHERE sha256=?",
            ((batch_id, position, now, sha256) for position, sha256 in enumerate(group, 1)),
        )
        conn.execute(
            "INSERT INTO events(event_time,event_type,batch_id,detail_json) VALUES (?,?,?,?)",
            (now, "batch_prepared", batch_id, json.dumps({"batch_key": batch_key, "items": len(group)})),
        )
        conn.commit()
        created += 1
        if created % 10 == 0:
            log(f"批次生成: {created:,}")
    write_batch_manifests(conn, inventory_path)
    report = write_progress_reports(conn, inventory_path, legacy_progress_path)
    log(f"批次准备完成: documents={len(eligible):,} batches={created:,} batch_size={batch_size:,}")
    return report


def default_task_options() -> dict[str, object]:
    return {
        "lang": "auto",
        "method": "auto",
        "effort": "high",
        "formula_enable": True,
        "table_enable": True,
        "dump_markdown": True,
        "dump_middle_json": True,
        "dump_model_output": True,
        "dump_content_list": True,
        "dump_orig_pdf": True,
        "draw_layout_bbox": True,
        "draw_span_bbox": True,
        "useLayoutDetection": True,
        "useSealRecognition": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "ocr",
        "repetitionPenalty": 1.0,
        "temperature": 0.0,
        "topP": 1.0,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "upload_images": os.getenv("RUSTFS_ENABLED", "true").lower() == "true",
    }


def configure_submit_environment(args: argparse.Namespace) -> None:
    values = {
        "DATABASE_PATH": str(args.task_db),
        "REDIS_QUEUE_ENABLED": "true",
        "REDIS_HOST": args.redis_host,
        "REDIS_PORT": str(args.redis_port),
        "REDIS_DB": str(args.redis_db),
        "REDIS_PASSWORD": args.redis_password or "",
        "REDIS_QUEUE_KEY": args.redis_queue_key,
        "REDIS_PROCESSING_KEY": args.redis_processing_key,
        "REDIS_CLAIM_MAINTENANCE_KEY": args.redis_maintenance_key,
        "REDIS_CLAIM_PAUSE_KEY": args.redis_pause_key,
        "SQLITE_QUEUE_FALLBACK": "false",
    }
    os.environ.update(values)


def load_task_db_api(args: argparse.Namespace):
    configure_submit_environment(args)
    root = Path(__file__).resolve().parents[1]
    backend = root / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from redis_queue import get_redis_queue
    from task_db import TaskDB

    queue = get_redis_queue()
    if queue is None or not queue.is_available():
        raise RuntimeError("Redis queue is unavailable")
    return TaskDB(str(args.task_db)), queue


SPLIT_PROCESSING_RETRY_MODES = {
    "split_children_requeued",
    "split_ready_to_merge",
    "split_awaiting_children",
}


def task_present_in_live_queue(queue: object, task_id: str) -> bool:
    return queue.client.zscore(queue.config.queue_key, task_id) is not None or queue.client.hget(
        queue.config.processing_key,
        task_id,
    ) is not None


def batch_item_status_from_create_result(result: dict[str, object], queue: object) -> str:
    retry_info = result.get("retry_info") if isinstance(result, dict) else None
    if isinstance(retry_info, dict) and retry_info.get("enqueue_failed_task_ids"):
        raise RuntimeError(f"task enqueue failed: {retry_info['enqueue_failed_task_ids']}")
    task_id = str(result["task_id"])
    status = result["status"]
    if status == "completed":
        return "completed"
    if status == "pending":
        if not task_present_in_live_queue(queue, task_id):
            raise RuntimeError(f"task {task_id} is pending in SQLite but absent from Redis")
        return "submitted"
    if status == "processing":
        if not isinstance(retry_info, dict):
            raise RuntimeError(f"task {task_id} returned processing without split retry_info")
        mode = retry_info.get("mode")
        if mode not in SPLIT_PROCESSING_RETRY_MODES:
            raise RuntimeError(f"task {task_id} returned unsupported processing retry mode: {mode}")
        requeued = retry_info.get("requeued_task_ids") or []
        if not isinstance(requeued, list):
            raise RuntimeError(f"task {task_id} returned non-list requeued split children")
        if mode == "split_children_requeued" and not requeued:
            raise RuntimeError(f"task {task_id} returned split_children_requeued without requeued split children")
        missing = [str(child_id) for child_id in requeued if not task_present_in_live_queue(queue, str(child_id))]
        if missing:
            raise RuntimeError(f"split children absent from Redis: {missing}")
        return "submitted"
    raise RuntimeError(f"task {task_id} returned non-queueable status: {status}")


def submit_batch(
    conn: sqlite3.Connection,
    inventory_path: Path,
    legacy_progress_path: Path,
    source_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    task_db, queue = load_task_db_api(args)
    pause_reason = queue.client.get(queue.config.claim_pause_key)
    if not pause_reason and not args.allow_live_claims:
        raise RuntimeError("claim pause is required; pass --allow-live-claims only for an intentional live refill")
    batch = conn.execute(
        "SELECT * FROM batches WHERE batch_key=?",
        (args.batch_key,),
    ).fetchone()
    if not batch:
        raise RuntimeError(f"unknown batch: {args.batch_key}")
    rows = conn.execute(
        """
        SELECT bi.position,bi.sha256,bi.item_status,d.source_path,d.state
        FROM batch_items bi JOIN documents d USING(sha256)
        WHERE bi.batch_id=? AND (bi.item_status='prepared' OR (bi.item_status='error' AND d.state='failed_retryable'))
        ORDER BY bi.position
        """,
        (batch["batch_id"],),
    ).fetchall()
    if args.limit is not None:
        rows = rows[: args.limit]
    queued = queue.client.zcard(queue.config.queue_key)
    capacity = max(0, args.high_watermark - queued)
    if len(rows) > capacity:
        raise RuntimeError(
            f"batch would exceed high watermark: queued={queued} selected={len(rows)} high={args.high_watermark}"
        )

    now = utc_now()
    successes = 0
    errors = 0
    for index, row in enumerate(rows, 1):
        sha256 = row["sha256"]
        source_path = validate_source_path(source_root, sha256, Path(row["source_path"]))
        try:
            result = task_db.create_task(
                file_name=f"{sha256}.pdf",
                file_path=str(source_path),
                backend="hybrid-auto-engine",
                options=default_task_options(),
                priority=args.priority,
                user_id=None,
                file_hash=sha256,
                lang="auto",
                method="auto",
            )
            task_id = result["task_id"]
            status = result["status"]
            item_status = batch_item_status_from_create_result(result, queue)
            conn.execute(
                """
                UPDATE batch_items
                SET item_status=?,task_id=?,submit_error=NULL,updated_at=?
                WHERE batch_id=? AND sha256=?
                """,
                (item_status, task_id, now, batch["batch_id"], sha256),
            )
            conn.execute(
                """
                UPDATE documents
                SET db_task_id=?,db_status=?,state=?,submit_attempts=submit_attempts+1,
                    last_submit_at=?,last_submit_error=NULL,updated_at=?
                WHERE sha256=?
                """,
                (
                    task_id,
                    status,
                    "complete_valid" if status == "completed" else "queued",
                    now,
                    now,
                    sha256,
                ),
            )
            successes += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            conn.execute(
                """
                UPDATE batch_items SET item_status='error',submit_error=?,updated_at=?
                WHERE batch_id=? AND sha256=?
                """,
                (error, now, batch["batch_id"], sha256),
            )
            conn.execute(
                """
                UPDATE documents SET submit_attempts=submit_attempts+1,last_submit_at=?,
                    last_submit_error=?,updated_at=? WHERE sha256=?
                """,
                (now, error, now, sha256),
            )
            errors += 1
        if index % 100 == 0 or index == len(rows):
            conn.commit()
            log(f"批次提交 {args.batch_key}: {index}/{len(rows)} success={successes} error={errors}")
    conn.execute(
        "INSERT INTO events(event_time,event_type,batch_id,detail_json) VALUES (?,?,?,?)",
        (
            utc_now(),
            "batch_submit",
            batch["batch_id"],
            json.dumps({"selected": len(rows), "success": successes, "errors": errors}),
        ),
    )
    conn.commit()
    refresh_batch_progress(conn)
    return write_progress_reports(conn, inventory_path, legacy_progress_path)


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    parser.add_argument("--task-db", type=Path, default=DEFAULT_TASK_DB)
    parser.add_argument("--legacy-progress", type=Path, default=DEFAULT_LEGACY_PROGRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh = subparsers.add_parser("refresh", help="Rescan sources, strict results, and MinerU DB state")
    add_common_paths(refresh)

    prepare = subparsers.add_parser("prepare", help="Create deterministic bounded batches")
    add_common_paths(prepare)
    prepare.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    prepare.add_argument("--replace-prepared", action="store_true")

    status = subparsers.add_parser("status", help="Render progress from the current inventory")
    add_common_paths(status)

    refresh_batch = subparsers.add_parser("refresh-batch", help="Refresh one batch from task DB without scanning history")
    add_common_paths(refresh_batch)
    refresh_batch.add_argument("--batch-key", required=True)

    submit = subparsers.add_parser("submit", help="Stage one prepared batch through trusted local paths")
    add_common_paths(submit)
    submit.add_argument("--batch-key", required=True)
    submit.add_argument("--limit", type=int)
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--high-watermark", type=int, default=DEFAULT_HIGH_WATERMARK)
    submit.add_argument("--allow-live-claims", action="store_true")
    submit.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "127.0.0.1"))
    submit.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    submit.add_argument("--redis-db", type=int, default=int(os.getenv("REDIS_DB", "0")))
    submit.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD", "redis123"))
    submit.add_argument(
        "--redis-queue-key",
        default=os.getenv("REDIS_QUEUE_KEY", "tianshu:task_queue:mineru-runner-worker-0"),
    )
    submit.add_argument(
        "--redis-processing-key",
        default=os.getenv("REDIS_PROCESSING_KEY", "tianshu:processing:mineru-runner-worker-0"),
    )
    submit.add_argument(
        "--redis-maintenance-key",
        default=os.getenv(
            "REDIS_CLAIM_MAINTENANCE_KEY",
            "tianshu:claim_maintenance:mineru-runner-worker-0",
        ),
    )
    submit.add_argument(
        "--redis-pause-key",
        default=os.getenv("REDIS_CLAIM_PAUSE_KEY", "tianshu:claim_pause:mineru-runner-worker-0"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    conn = connect_inventory(args.inventory)
    try:
        if args.command == "refresh":
            report = refresh_inventory(
                conn,
                args.inventory,
                args.source_root,
                args.parsed_root,
                args.task_db,
                args.legacy_progress,
            )
        elif args.command == "prepare":
            report = prepare_batches(
                conn,
                args.inventory,
                args.legacy_progress,
                args.batch_size,
                args.replace_prepared,
            )
        elif args.command == "refresh-batch":
            report = refresh_batch_from_task_db(
                conn,
                args.inventory,
                args.legacy_progress,
                args.task_db,
                args.batch_key,
            )
        elif args.command == "submit":
            report = submit_batch(
                conn,
                args.inventory,
                args.legacy_progress,
                args.source_root,
                args,
            )
        else:
            refresh_batch_progress(conn)
            report = write_progress_reports(conn, args.inventory, args.legacy_progress)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
