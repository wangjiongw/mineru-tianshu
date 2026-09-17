import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parent_merge import ParentMergeInputError, merge_parent_task_results
import task_db as task_db_module
from task_db import TaskDB
import reconcile_parent_merges


def _make_parent_with_children(tmp_path, child_count=2, db=None, stem="parent"):
    db = db or TaskDB(tmp_path / "tasks.db")
    parent_pdf = tmp_path / f"{stem}.pdf"
    parent_pdf.write_text("pdf", encoding="utf-8")
    parent_id = db.create_task(f"{stem}.pdf", str(parent_pdf))["task_id"]
    db.convert_to_parent_task(parent_id, child_count=0)
    child_ids = []
    for idx in range(child_count):
        child_file = tmp_path / f"{stem}-child-{idx}.pdf"
        child_file.write_text(f"child {idx}", encoding="utf-8")
        child_id = db.create_child_task(
            parent_task_id=parent_id,
            file_name=child_file.name,
            file_path=str(child_file),
            options={"chunk_info": {"start_page": idx + 1, "page_count": 1}},
        )
        result_dir = tmp_path / f"{stem}-result-{idx}"
        result_dir.mkdir()
        (result_dir / "result.md").write_text(f"markdown {idx}", encoding="utf-8")
        (result_dir / "result.json").write_text(json.dumps([{"page_idx": 0, "text": str(idx)}]), encoding="utf-8")
        (result_dir / "mineru_model.json").write_text(json.dumps([[{"type": "text", "chunk": idx}]]), encoding="utf-8")
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks
                SET status = 'completed', result_path = ?, worker_id = 'worker-child', completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (str(result_dir), child_id),
            )
        child_ids.append(child_id)
    db.convert_to_parent_task(parent_id, child_count=child_count)
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status = 'processing', worker_id = 'parent-worker' WHERE task_id = ?", (parent_id,))
    return db, parent_id, child_ids


def test_callback_replay_uses_real_completed_children_and_single_merge_claim(tmp_path):
    db, parent_id, child_ids = _make_parent_with_children(tmp_path)

    assert db.on_child_task_completed(child_ids[0], merge_owner="worker-a") == parent_id
    assert db.on_child_task_completed(child_ids[0], merge_owner="worker-b") is None
    assert db.on_child_task_completed(child_ids[1], merge_owner="worker-c") is None

    parent = db.get_task(parent_id)
    assert parent["status"] == "merging"
    assert parent["merge_owner"] == "worker-a"
    assert parent["child_completed"] == 2
    assert parent["merge_attempts"] == 1


def test_concurrent_parent_merge_claim_allows_only_one_owner(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)

    def claim(owner):
        return owner, db.claim_parent_merge(parent_id, owner)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, [f"worker-{i}" for i in range(8)]))

    winners = [owner for owner, won in results if won]
    assert len(winners) == 1
    parent = db.get_task(parent_id)
    assert parent["merge_owner"] == winners[0]
    assert parent["merge_attempts"] == 1


def test_merge_does_not_cleanup_child_files_until_db_complete_succeeds(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    assert db.claim_parent_merge(parent_id, "worker-a")
    child_files = [Path(child["file_path"]) for child in db.get_child_tasks(parent_id)]

    class RejectingCompleteDB:
        def get_task_with_children(self, task_id):
            return db.get_task_with_children(task_id)

        def complete_parent_merge(self, task_id, result_path, merge_owner, data=None):
            return False

    try:
        merge_parent_task_results(
            task_db=RejectingCompleteDB(),
            parent_task_id=parent_id,
            output_dir=str(tmp_path / "out"),
            merge_owner="worker-a",
            cleanup_child_files=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("merge should fail when DB completion CAS is rejected")

    assert (tmp_path / "out" / "parent" / "result.md").exists()
    assert all(path.exists() for path in child_files)
    assert db.get_task(parent_id)["status"] == "merging"


def test_reconciler_apply_merges_once_and_then_has_no_candidates(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status = 'pending', worker_id = NULL WHERE task_id = ?", (parent_id,))

    args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(tmp_path / "tasks.db"),
        "--output-dir", str(tmp_path / "out"),
        "--apply",
    ])
    first = reconcile_parent_merges.run_once(args, "reconciler-test")
    second = reconcile_parent_merges.run_once(args, "reconciler-test")

    assert first["summary"]["finalizable"] == 1
    assert first["tasks"][0]["merged"] is True
    assert db.get_task(parent_id)["status"] == "completed"
    assert (tmp_path / "out" / "parent" / "result.md").read_text(encoding="utf-8") == "markdown 0\n\n\n\nmarkdown 1"
    assert second["summary"] == {"finalizable": 0, "remergeable": 0, "blocked": 0}



def test_release_and_block_keep_parent_merging_and_clear_only_merge_lease(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'merging', worker_id = 'original-worker', merge_owner = 'owner-a',
                merge_claimed_at = CURRENT_TIMESTAMP, merge_attempts = 2
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    assert db.release_parent_merge(parent_id, "owner-a", "boom")
    parent = db.get_task(parent_id)
    assert parent["status"] == "merging"
    assert parent["worker_id"] == "original-worker"
    assert parent["merge_owner"] is None
    assert parent["merge_claimed_at"] is None
    assert parent["merge_error"] == "boom"

    with db.get_cursor() as cursor:
        cursor.execute(
            "UPDATE tasks SET merge_owner = 'owner-b', merge_claimed_at = CURRENT_TIMESTAMP, merge_attempts = 2 WHERE task_id = ?",
            (parent_id,),
        )
    assert db.block_parent_merge(parent_id, "owner-b", "still bad", max_attempts=5)
    parent = db.get_task(parent_id)
    assert parent["status"] == "merging"
    assert parent["worker_id"] == "original-worker"
    assert parent["merge_owner"] is None
    assert parent["merge_claimed_at"] is None
    assert parent["merge_error"] == "still bad"
    assert parent["merge_attempts"] == 5


def test_claim_unowned_merging_parent_immediately_preserves_worker_id(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'merging', worker_id = 'worker-original', merge_owner = NULL,
                merge_claimed_at = CURRENT_TIMESTAMP, merge_attempts = 0
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    assert db.claim_parent_merge(parent_id, "reconciler", stale_seconds=3600)
    parent = db.get_task(parent_id)
    assert parent["status"] == "merging"
    assert parent["worker_id"] == "worker-original"
    assert parent["merge_owner"] == "reconciler"
    assert parent["merge_attempts"] == 1


def test_convert_to_parent_task_resets_merge_lease_and_attempts(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'merging', merge_owner = 'old-owner', merge_claimed_at = CURRENT_TIMESTAMP,
                merge_attempts = 4, merge_error = 'old error'
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    db.convert_to_parent_task(parent_id, child_count=0)
    parent = db.get_task(parent_id)
    assert parent["status"] == "processing"
    assert parent["child_count"] == 0
    assert parent["child_completed"] == 0
    assert parent["merge_owner"] is None
    assert parent["merge_claimed_at"] is None
    assert parent["merge_attempts"] == 0
    assert parent["merge_error"] is None


def test_recover_orphans_does_not_reset_merging_parent(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'merging', worker_id = 'merge-worker', started_at = datetime('now', '-2 hours'),
                merge_owner = NULL, merge_claimed_at = NULL, retry_count = 0
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    result = db.recover_orphans(stale_minutes=30)
    parent = db.get_task(parent_id)
    assert result["sqlite_reset"] == 0
    assert result["sqlite_failed"] == 0
    assert parent["status"] == "merging"
    assert parent["worker_id"] == "merge-worker"


def _create_legacy_parent_merge_db(db_path):
    conn = sqlite3.connect(db_path)
    result_dirs = []
    for idx in range(2):
        result_dir = db_path.parent / f"legacy-result-{idx}"
        result_dir.mkdir()
        (result_dir / "result.md").write_text(f"legacy markdown {idx}", encoding="utf-8")
        (result_dir / "result.json").write_text(json.dumps([{"page_idx": 0, "text": str(idx)}]), encoding="utf-8")
        (result_dir / "mineru_model.json").write_text(json.dumps([[{"type": "text", "chunk": idx}]]), encoding="utf-8")
        result_dirs.append(result_dir)
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 0,
            backend TEXT DEFAULT 'pipeline',
            options TEXT,
            result_path TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            worker_id TEXT,
            shared_read INTEGER DEFAULT 0,
            file_hash TEXT,
            lang TEXT,
            method TEXT,
            retry_count INTEGER DEFAULT 0,
            parent_task_id TEXT,
            is_parent INTEGER DEFAULT 0,
            child_count INTEGER DEFAULT 0,
            child_completed INTEGER DEFAULT 0,
            data TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO tasks(task_id, file_name, file_path, status, is_parent, child_count, child_completed)
        VALUES ('parent-legacy', 'parent.pdf', '/tmp/parent.pdf', 'pending', 1, 2, 0)
        """
    )
    for idx in range(2):
        conn.execute(
            """
            INSERT INTO tasks(task_id, parent_task_id, file_name, file_path, status, result_path, options)
            VALUES (?, 'parent-legacy', ?, ?, 'completed', ?, ?)
            """,
            (
                f"child-{idx}",
                f"child-{idx}.pdf",
                f"/tmp/child-{idx}.pdf",
                str(result_dirs[idx]),
                json.dumps({"chunk_info": {"start_page": idx + 1}}),
            ),
        )
    conn.commit()
    before_schema = conn.execute("PRAGMA table_info(tasks)").fetchall()
    before_rows = conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
    conn.close()
    return before_schema, before_rows


def test_reconciler_dry_run_is_read_only_and_supports_legacy_schema_without_merge_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    before_schema, before_rows = _create_legacy_parent_merge_db(db_path)

    args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(db_path),
        "--output-dir", str(tmp_path / "out"),
        "--report-json",
    ])
    report = reconcile_parent_merges.run_once(args, "dry-run-owner")

    conn = sqlite3.connect(db_path)
    after_schema = conn.execute("PRAGMA table_info(tasks)").fetchall()
    after_rows = conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
    conn.close()

    assert report["applied"] is False
    assert report["summary"]["finalizable"] == 1
    assert report["tasks"][0]["merge_attempts"] == 0
    assert before_schema == after_schema
    assert before_rows == after_rows
    assert "merge_owner" not in {row[1] for row in after_schema}



def test_reconciler_reports_missing_child_result_as_blocked_without_claiming(tmp_path):
    db, parent_id, child_ids = _make_parent_with_children(tmp_path)
    missing_child = child_ids[0]
    with db.get_cursor() as cursor:
        cursor.execute("SELECT result_path FROM tasks WHERE task_id = ?", (missing_child,))
        missing_result = Path(cursor.fetchone()["result_path"])
        cursor.execute("UPDATE tasks SET status = 'pending', worker_id = NULL WHERE task_id = ?", (parent_id,))
    for path in missing_result.iterdir():
        path.unlink()
    missing_result.rmdir()

    args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(tmp_path / "tasks.db"),
        "--output-dir", str(tmp_path / "out"),
        "--apply",
    ])
    report = reconcile_parent_merges.run_once(args, "reconciler-test")

    assert report["summary"] == {"finalizable": 0, "remergeable": 0, "blocked": 1}
    assert report["tasks"][0]["classification"] == "blocked"
    assert "result_path missing" in report["tasks"][0]["blocked_reasons"][0]
    parent = db.get_task(parent_id)
    assert parent["status"] == "merging"
    assert parent["merge_owner"] is None
    assert parent["merge_attempts"] == 3
    assert "result_path missing" in parent["merge_error"]
    assert not (tmp_path / "out" / "parent" / "result.md").exists()


def test_merge_retry_after_files_written_but_db_completion_rejected_is_idempotent(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    assert db.claim_parent_merge(parent_id, "worker-a")
    child_files = [Path(child["file_path"]) for child in db.get_child_tasks(parent_id)]

    class RejectingCompleteDB:
        def get_task_with_children(self, task_id):
            return db.get_task_with_children(task_id)

        def complete_parent_merge(self, task_id, result_path, merge_owner, data=None):
            return False

    try:
        merge_parent_task_results(
            task_db=RejectingCompleteDB(),
            parent_task_id=parent_id,
            output_dir=str(tmp_path / "out"),
            merge_owner="worker-a",
            cleanup_child_files=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("merge should fail when DB completion is rejected")

    parent_out = tmp_path / "out" / "parent"
    assert (parent_out / "result.md").read_text(encoding="utf-8") == "markdown 0\n\n\n\nmarkdown 1"
    assert all(path.exists() for path in child_files)
    assert db.get_task(parent_id)["status"] == "merging"

    out = merge_parent_task_results(
        task_db=db,
        parent_task_id=parent_id,
        output_dir=str(tmp_path / "out"),
        merge_owner="worker-a",
        cleanup_child_files=True,
    )

    assert out == parent_out
    assert db.get_task(parent_id)["status"] == "completed"
    assert (parent_out / "result.md").read_text(encoding="utf-8") == "markdown 0\n\n\n\nmarkdown 1"
    assert all(not path.exists() for path in child_files)
    assert not list((tmp_path / "out").glob(".parent.merge-*"))



def test_reconciler_remergeable_race_does_not_steal_new_active_lease(tmp_path, monkeypatch):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'merging', merge_owner = 'stale-owner', merge_claimed_at = datetime('now', '-2 hours'),
                merge_attempts = 0
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    class RacingTaskDB(TaskDB):
        raced = False

        def claim_parent_merge(self, parent_task_id, merge_owner, stale_seconds=None, max_attempts=None, force=False):
            assert force is False
            if not RacingTaskDB.raced:
                RacingTaskDB.raced = True
                with self.get_cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET merge_owner = 'fresh-owner', merge_claimed_at = CURRENT_TIMESTAMP
                        WHERE task_id = ?
                        """,
                        (parent_task_id,),
                    )
            return super().claim_parent_merge(parent_task_id, merge_owner, stale_seconds, max_attempts, force)

    monkeypatch.setattr(reconcile_parent_merges, "TaskDB", RacingTaskDB)
    args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(tmp_path / "tasks.db"),
        "--output-dir", str(tmp_path / "out"),
        "--apply",
        "--stale-seconds", "60",
    ])
    report = reconcile_parent_merges.run_once(args, "reconciler-test")

    assert report["tasks"][0]["classification"] == "remergeable"
    assert report["tasks"][0]["claimed"] is False
    parent = db.get_task(parent_id)
    assert parent["merge_owner"] == "fresh-owner"
    assert parent["status"] == "merging"
    assert not (tmp_path / "out" / "parent" / "result.md").exists()


def test_apply_pins_missing_artifact_blocked_and_processes_later_valid_candidate(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    _blocked_db, blocked_parent, blocked_children = _make_parent_with_children(tmp_path, db=db, stem="blocked")
    _valid_db, valid_parent, _valid_children = _make_parent_with_children(tmp_path, db=db, stem="valid")
    missing_child = blocked_children[0]
    with db.get_cursor() as cursor:
        cursor.execute("SELECT result_path FROM tasks WHERE task_id = ?", (missing_child,))
        missing_result = Path(cursor.fetchone()["result_path"])
        cursor.execute("UPDATE tasks SET status = 'pending', worker_id = NULL WHERE task_id IN (?, ?)", (blocked_parent, valid_parent))
    for path in missing_result.iterdir():
        path.unlink()
    missing_result.rmdir()

    args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(tmp_path / "tasks.db"),
        "--output-dir", str(tmp_path / "out"),
        "--apply",
        "--limit", "2",
        "--max-attempts", "3",
    ])
    report = reconcile_parent_merges.run_once(args, "reconciler-test")

    by_id = {item["task_id"]: item for item in report["tasks"]}
    assert by_id[blocked_parent]["classification"] == "blocked"
    assert by_id[blocked_parent]["blocked"] is True
    assert "result_path missing" in by_id[blocked_parent]["merge_error"]
    assert by_id[valid_parent]["merged"] is True
    blocked = db.get_task(blocked_parent)
    valid = db.get_task(valid_parent)
    assert blocked["status"] == "merging"
    assert blocked["merge_owner"] is None
    assert blocked["merge_attempts"] == 3
    assert "result_path missing" in blocked["merge_error"]
    assert valid["status"] == "completed"


def test_parent_merge_rejects_extra_child_rows(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    extra_file = tmp_path / "extra-child.pdf"
    extra_file.write_text("extra", encoding="utf-8")
    extra_id = "extra-child-row"
    extra_result = tmp_path / "extra-result"
    extra_result.mkdir()
    (extra_result / "result.md").write_text("extra", encoding="utf-8")
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO tasks(task_id, parent_task_id, file_name, file_path, status, result_path, options)
            VALUES (?, ?, ?, ?, 'completed', ?, ?)
            """,
            (extra_id, parent_id, extra_file.name, str(extra_file), str(extra_result), json.dumps({"chunk_info": {"start_page": 99}})),
        )
    assert db.claim_parent_merge(parent_id, "worker-a", force=True)

    try:
        merge_parent_task_results(
            task_db=db,
            parent_task_id=parent_id,
            output_dir=str(tmp_path / "out"),
            merge_owner="worker-a",
        )
    except Exception as exc:
        assert "child row count mismatch" in str(exc)
    else:
        raise AssertionError("merge should reject extra child rows")
    assert db.get_task(parent_id)["status"] == "merging"
    assert not (tmp_path / "out" / "parent" / "result.md").exists()


def test_retry_without_required_json_keeps_existing_parent_output(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    assert db.claim_parent_merge(parent_id, "worker-a")
    parent_out = tmp_path / "out" / "parent"
    parent_out.mkdir(parents=True)
    (parent_out / "result.json").write_text('{"stale": true}', encoding="utf-8")
    for child in db.get_child_tasks(parent_id):
        (Path(child["result_path"]) / "result.json").unlink()

    try:
        merge_parent_task_results(
            task_db=db,
            parent_task_id=parent_id,
            output_dir=str(tmp_path / "out"),
            merge_owner="worker-a",
        )
    except ParentMergeInputError as exc:
        assert "content-list artifact" in str(exc)
    else:
        raise AssertionError("merge should reject missing MinerU content lists")

    assert db.get_task(parent_id)["status"] == "merging"
    assert json.loads((parent_out / "result.json").read_text(encoding="utf-8")) == {"stale": True}


def test_list_parent_merge_candidates_excludes_terminal_failed_and_cancelled(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    _db2, failed_parent, _failed_children = _make_parent_with_children(tmp_path, db=db, stem="failed")
    _db3, cancelled_parent, _cancelled_children = _make_parent_with_children(tmp_path, db=db, stem="cancelled")
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status = 'pending' WHERE task_id = ?", (parent_id,))
        cursor.execute("UPDATE tasks SET status = 'failed' WHERE task_id = ?", (failed_parent,))
        cursor.execute("UPDATE tasks SET status = 'cancelled' WHERE task_id = ?", (cancelled_parent,))

    candidates = db.list_parent_merge_candidates(limit=10)
    candidate_ids = {item["task_id"] for item in candidates}
    assert parent_id in candidate_ids
    assert failed_parent not in candidate_ids
    assert cancelled_parent not in candidate_ids



def test_sqlite_normal_claim_skips_pending_split_parent_and_preserves_children(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", False)
    db, parent_id, child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status = 'pending', worker_id = NULL, priority = 100 WHERE task_id = ?", (parent_id,))
    ordinary_id = db.create_task("ordinary.pdf", str(tmp_path / "ordinary.pdf"), priority=1)["task_id"]

    claimed = db.get_next_task("worker-a")

    assert claimed["task_id"] == ordinary_id
    parent = db.get_task(parent_id)
    assert parent["status"] == "pending"
    assert parent["worker_id"] is None
    assert {child["task_id"] for child in db.get_child_tasks(parent_id)} == set(child_ids)


def test_redis_normal_claim_removes_pending_split_parent_without_requeue_and_preserves_children(tmp_path, monkeypatch):
    db, parent_id, child_ids = _make_parent_with_children(tmp_path)
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status = 'pending', worker_id = NULL WHERE task_id = ?", (parent_id,))

    class ClaimedRedis:
        def __init__(self, task_id):
            self.task_id = task_id
            self.failed = []

        def enqueue(self, *args, **kwargs):
            return True

        def claim(self, worker_id, timeout=1.0):
            return type("Claim", (), {"status": "claimed", "task_id": self.task_id})()

        def fail(self, task_id, worker_id, requeue=False):
            self.failed.append((task_id, worker_id, requeue))
            return True

    redis_queue = ClaimedRedis(parent_id)
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: redis_queue)
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")

    assert db.get_next_task("worker-a") is None
    assert redis_queue.failed == [(parent_id, "worker-a", False)]
    parent = db.get_task(parent_id)
    assert parent["status"] == "pending"
    assert parent["worker_id"] is None
    assert {child["task_id"] for child in db.get_child_tasks(parent_id)} == set(child_ids)


def test_retry_failed_split_parent_requeues_only_failed_children_and_can_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", False)
    db, parent_id, child_ids = _make_parent_with_children(tmp_path)
    completed_child, failed_child = child_ids
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'ReadTimeout', completed_at = CURRENT_TIMESTAMP,
                worker_id = 'worker-failed', retry_count = 1, result_path = NULL
            WHERE task_id = ?
            """,
            (failed_child,),
        )
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'Subtask failed', completed_at = CURRENT_TIMESTAMP,
                worker_id = 'parent-worker', child_completed = 0, merge_owner = 'stale',
                merge_claimed_at = CURRENT_TIMESTAMP, merge_error = 'old merge error'
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    enqueued = []
    monkeypatch.setattr(db, "_enqueue_to_redis", lambda *args: enqueued.append(args) or True)

    first = db.retry_failed_logical_task(parent_id)
    second = db.retry_failed_logical_task(parent_id)

    assert first["mode"] == "split_children_requeued"
    assert first["requeued_task_ids"] == [failed_child]
    assert first["requeued_child_count"] == 1
    assert first["enqueue_failed_task_ids"] == []
    assert second["mode"] == "split_awaiting_children"
    assert second["requeued_task_ids"] == []
    assert second["enqueue_failed_task_ids"] == []
    assert [item[0] for item in enqueued] == [failed_child]

    parent = db.get_task(parent_id)
    assert parent["status"] == "processing"
    assert parent["child_completed"] == 1
    assert parent["error_message"] is None
    assert parent["completed_at"] is None
    assert parent["worker_id"] is None
    assert parent["merge_owner"] is None
    assert parent["merge_claimed_at"] is None
    assert parent["merge_error"] is None

    completed = db.get_task(completed_child)
    assert completed["status"] == "completed"
    assert completed["retry_count"] == 0
    failed = db.get_task(failed_child)
    assert failed["status"] == "pending"
    assert failed["retry_count"] == 2
    assert failed["error_message"] is None
    assert failed["worker_id"] is None

    claimed = db.get_next_task("worker-retry")
    assert claimed["task_id"] == failed_child
    retry_result = tmp_path / "retried-child-result"
    retry_result.mkdir()
    (retry_result / "result.md").write_text("retried", encoding="utf-8")
    (retry_result / "result.json").write_text(json.dumps([{"page_idx": 1, "text": "retried"}]), encoding="utf-8")
    assert db.update_task_status(failed_child, "completed", worker_id="worker-retry", result_path=str(retry_result))
    assert db.on_child_task_completed(failed_child, merge_owner="worker-merge") == parent_id
    assert db.get_task(parent_id)["status"] == "merging"


def test_create_task_dedup_retries_failed_split_parent_without_enqueueing_parent(tmp_path, monkeypatch):
    db = TaskDB(tmp_path / "tasks.db")
    file_hash = "sha256-parent"
    parent_pdf = tmp_path / "dedup-parent.pdf"
    parent_pdf.write_text("pdf", encoding="utf-8")
    created = db.create_task("dedup-parent.pdf", str(parent_pdf), file_hash=file_hash, lang="en", method="auto")
    parent_id = created["task_id"]
    db.convert_to_parent_task(parent_id, child_count=0)
    child_id = db.create_child_task(parent_id, "dedup-parent-0001.pdf", str(tmp_path / "dedup-parent-0001.pdf"))
    db.convert_to_parent_task(parent_id, child_count=1)
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'ReadTimeout', completed_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (child_id,),
        )
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'Subtask failed', completed_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    enqueued = []
    monkeypatch.setattr(db, "_enqueue_to_redis", lambda *args: enqueued.append(args) or True)

    deduped = db.create_task("dedup-parent.pdf", str(parent_pdf), file_hash=file_hash, lang="en", method="auto")

    assert deduped["task_id"] == parent_id
    assert deduped["deduped"] is True
    assert deduped["status"] == "processing"
    assert deduped["requeued"] is True
    assert deduped["retry_info"]["mode"] == "split_children_requeued"
    assert [item[0] for item in enqueued] == [child_id]
    assert db.get_task(parent_id)["status"] == "processing"
    assert db.get_task(child_id)["status"] == "pending"


def test_retry_failed_logical_task_keeps_leaf_dedup_retry_contract(tmp_path, monkeypatch):
    db = TaskDB(tmp_path / "tasks.db")
    file_hash = "sha256-leaf"
    pdf = tmp_path / "leaf.pdf"
    pdf.write_text("pdf", encoding="utf-8")
    task_id = db.create_task("leaf.pdf", str(pdf), file_hash=file_hash, lang="en", method="auto")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'boom', completed_at = CURRENT_TIMESTAMP,
                worker_id = 'worker-leaf'
            WHERE task_id = ?
            """,
            (task_id,),
        )

    enqueued = []
    monkeypatch.setattr(db, "_enqueue_to_redis", lambda *args: enqueued.append(args) or True)

    deduped = db.create_task("leaf.pdf", str(pdf), file_hash=file_hash, lang="en", method="auto")

    assert deduped == {
        "task_id": task_id,
        "status": "pending",
        "deduped": True,
        "requeued": True,
        "retry_info": {
            "task_id": task_id,
            "mode": "leaf",
            "status": "pending",
            "requeued_task_ids": [task_id],
            "requeued_child_count": 0,
            "enqueue_failed_task_ids": [],
        },
    }
    task = db.get_task(task_id)
    assert task["status"] == "pending"
    assert task["retry_count"] == 1
    assert task["error_message"] is None
    assert task["worker_id"] is None
    assert [item[0] for item in enqueued] == [task_id]


def test_retry_failed_split_parent_compensates_enqueue_failure_and_later_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    db, parent_id, child_ids = _make_parent_with_children(tmp_path)
    completed_child, failed_child = child_ids
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'ReadTimeout', completed_at = CURRENT_TIMESTAMP,
                worker_id = 'worker-failed', retry_count = 4, result_path = NULL
            WHERE task_id = ?
            """,
            (failed_child,),
        )
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'Subtask failed', completed_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    calls = []

    def enqueue_once_failed(task_id, priority, payload):
        calls.append(task_id)
        return len(calls) > 1

    monkeypatch.setattr(db, "_enqueue_to_redis", enqueue_once_failed)

    first = db.retry_failed_logical_task(parent_id)
    assert first["status"] == "failed"
    assert first["requeued_task_ids"] == []
    assert first["enqueue_failed_task_ids"] == [failed_child]
    assert db.get_task(parent_id)["status"] == "failed"
    retry_after_first = db.get_task(failed_child)["retry_count"]
    assert retry_after_first == 5
    assert db.get_task(failed_child)["status"] == "failed"
    assert db.get_task(completed_child)["status"] == "completed"

    second = db.retry_failed_logical_task(parent_id)
    assert second["status"] == "processing"
    assert second["requeued_task_ids"] == [failed_child]
    assert second["enqueue_failed_task_ids"] == []
    assert db.get_task(failed_child)["status"] == "pending"
    assert db.get_task(failed_child)["retry_count"] == retry_after_first
    assert calls == [failed_child, failed_child]


def test_retry_failed_leaf_compensates_enqueue_failure_and_later_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    db = TaskDB(tmp_path / "tasks.db")
    pdf = tmp_path / "leaf-enqueue.pdf"
    pdf.write_text("pdf", encoding="utf-8")
    task_id = db.create_task("leaf-enqueue.pdf", str(pdf))["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'failed', error_message = 'boom', completed_at = CURRENT_TIMESTAMP,
                worker_id = 'worker-leaf', retry_count = 2
            WHERE task_id = ?
            """,
            (task_id,),
        )

    calls = []

    def enqueue_once_failed(task_id, priority, payload):
        calls.append(task_id)
        return len(calls) > 1

    monkeypatch.setattr(db, "_enqueue_to_redis", enqueue_once_failed)

    first = db.retry_failed_logical_task(task_id)
    assert first["status"] == "failed"
    assert first["requeued_task_ids"] == []
    assert first["enqueue_failed_task_ids"] == [task_id]
    retry_after_first = db.get_task(task_id)["retry_count"]
    assert retry_after_first == 3
    assert db.get_task(task_id)["status"] == "failed"

    second = db.retry_failed_logical_task(task_id)
    assert second["status"] == "pending"
    assert second["requeued_task_ids"] == [task_id]
    assert second["enqueue_failed_task_ids"] == []
    assert db.get_task(task_id)["status"] == "pending"
    assert db.get_task(task_id)["retry_count"] == retry_after_first
    assert calls == [task_id, task_id]


def test_api_retry_uses_logical_retry_and_reports_enqueue_failure(tmp_path, monkeypatch):
    import api_server

    class FakeUser:
        user_id = "owner"

        def has_permission(self, permission):
            return True

    class RejectOldRetryDB:
        def __init__(self):
            self.logical_calls = []

        def get_task(self, task_id):
            return {"task_id": task_id, "user_id": "owner"}

        def retry_task(self, task_id):
            raise AssertionError("public API must not call blind retry_task")

        def retry_failed_logical_task(self, task_id):
            self.logical_calls.append(task_id)
            return {
                "task_id": task_id,
                "mode": "split_children_requeued",
                "status": "failed",
                "requeued_task_ids": [],
                "requeued_child_count": 0,
                "enqueue_failed_task_ids": ["child-1"],
            }

    fake_db = RejectOldRetryDB()
    monkeypatch.setattr(api_server, "db", fake_db)
    monkeypatch.setattr(api_server, "OUTPUT_DIR", tmp_path)

    response = api_server.retry_task("parent-1", current_user=FakeUser())

    assert fake_db.logical_calls == ["parent-1"]
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["success"] is False
    assert body["retry_info"]["enqueue_failed_task_ids"] == ["child-1"]


def test_api_retry_returns_conflict_when_leaf_has_no_retry_work(tmp_path, monkeypatch):
    import api_server

    class FakeUser:
        user_id = "owner"

        def has_permission(self, permission):
            return True

    class NoWorkDB:
        def get_task(self, task_id):
            return {"task_id": task_id, "user_id": "owner"}

        def retry_failed_logical_task(self, task_id):
            return {
                "task_id": task_id,
                "mode": "leaf",
                "status": "completed",
                "requeued_task_ids": [],
                "requeued_child_count": 0,
                "enqueue_failed_task_ids": [],
            }

    monkeypatch.setattr(api_server, "db", NoWorkDB())
    monkeypatch.setattr(api_server, "OUTPUT_DIR", tmp_path)

    response = api_server.retry_task("done-leaf", current_user=FakeUser())

    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["success"] is False
    assert body["retry_info"]["mode"] == "leaf"


def test_api_retry_treats_split_ready_or_awaiting_as_recoverable(tmp_path, monkeypatch):
    import api_server

    class FakeUser:
        user_id = "owner"

        def has_permission(self, permission):
            return True

    class SplitReadyDB:
        def __init__(self):
            self.mode = "split_ready_to_merge"

        def get_task(self, task_id):
            return {"task_id": task_id, "user_id": "owner"}

        def retry_failed_logical_task(self, task_id):
            return {
                "task_id": task_id,
                "mode": self.mode,
                "status": "processing",
                "requeued_task_ids": [],
                "requeued_child_count": 0,
                "enqueue_failed_task_ids": [],
            }

    fake_db = SplitReadyDB()
    monkeypatch.setattr(api_server, "db", fake_db)
    monkeypatch.setattr(api_server, "OUTPUT_DIR", tmp_path)

    ready = api_server.retry_task("parent-ready", current_user=FakeUser())
    fake_db.mode = "split_awaiting_children"
    awaiting = api_server.retry_task("parent-awaiting", current_user=FakeUser())

    assert ready["success"] is True
    assert ready["retry_info"]["mode"] == "split_ready_to_merge"
    assert awaiting["success"] is True
    assert awaiting["retry_info"]["mode"] == "split_awaiting_children"


def _call_submit_task_for_api_test(api_server, file_obj, user, **overrides):
    kwargs = {
        "file": file_obj,
        "backend": "pipeline",
        "lang": "en",
        "method": "auto",
        "formula_enable": True,
        "table_enable": True,
        "priority": 1,
        "start_page": None,
        "end_page": None,
        "force_ocr": False,
        "server_url": None,
        "draw_layout_bbox": True,
        "draw_span_bbox": True,
        "dump_markdown": True,
        "dump_middle_json": True,
        "dump_model_output": True,
        "dump_content_list": True,
        "dump_orig_pdf": True,
        "draw_layout": True,
        "draw_span": True,
        "keep_audio": False,
        "enable_keyframe_ocr": False,
        "ocr_backend": "paddleocr-vl",
        "keep_keyframes": False,
        "enable_speaker_diarization": False,
        "remove_watermark": False,
        "watermark_conf_threshold": 0.35,
        "watermark_dilation": 10,
        "convert_office_to_pdf": False,
        "effort": "high",
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": True,
        "useChartRecognition": False,
        "useSealRecognition": True,
        "useOcrForImageBlock": False,
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
        "markdownIgnoreLabels": "header,footer",
        "current_user": user,
    }
    kwargs.update(overrides)
    return api_server.submit_task(**kwargs)


def test_api_submit_returns_503_for_initial_enqueue_failure(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    from fastapi import HTTPException
    import api_server

    class FakeUser:
        user_id = "owner"

    class InitialEnqueueFailedDB:
        def create_task(self, **kwargs):
            return {
                "task_id": "task-new",
                "status": "failed",
                "deduped": False,
                "requeued": False,
                "enqueue_failed": True,
                "enqueue_failed_task_ids": ["task-new"],
            }

    monkeypatch.setattr(api_server, "db", InitialEnqueueFailedDB())
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path)
    upload = SimpleNamespace(filename="new.pdf", file=io.BytesIO(b"%PDF-1.4"))

    try:
        _call_submit_task_for_api_test(api_server, upload, FakeUser())
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["task_id"] == "task-new"
        assert exc.detail["status"] == "failed"
        assert exc.detail["enqueue_failed_task_ids"] == ["task-new"]
    else:
        raise AssertionError("submit should fail with 503 when Redis enqueue failed")


def test_api_submit_returns_503_for_dedup_retry_enqueue_failure(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    from fastapi import HTTPException
    import api_server

    class FakeUser:
        user_id = "owner"

    class DedupRetryEnqueueFailedDB:
        def create_task(self, **kwargs):
            return {
                "task_id": "parent-1",
                "status": "failed",
                "deduped": True,
                "requeued": False,
                "retry_info": {
                    "task_id": "parent-1",
                    "mode": "split_children_requeued",
                    "status": "failed",
                    "requeued_task_ids": [],
                    "requeued_child_count": 0,
                    "enqueue_failed_task_ids": ["child-1"],
                },
            }

    monkeypatch.setattr(api_server, "db", DedupRetryEnqueueFailedDB())
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path)
    upload = SimpleNamespace(filename="dedup.pdf", file=io.BytesIO(b"%PDF-1.4"))

    try:
        _call_submit_task_for_api_test(api_server, upload, FakeUser())
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["task_id"] == "parent-1"
        assert exc.detail["status"] == "failed"
        assert exc.detail["deduped"] is True
        assert exc.detail["enqueue_failed_task_ids"] == ["child-1"]
        assert exc.detail["retry_info"]["mode"] == "split_children_requeued"
    else:
        raise AssertionError("dedup retry enqueue failure should surface as 503")


def _prepare_artifact_merge(db, parent_id, preserve_all):
    children = db.get_child_tasks(parent_id)
    with db.get_cursor() as cursor:
        cursor.execute(
            "UPDATE tasks SET options = ? WHERE task_id = ?",
            (json.dumps({"preserve_all_artifacts": preserve_all}), parent_id),
        )
    for idx, child in enumerate(children):
        result_dir = Path(child["result_path"])
        image_dir = result_dir / "images"
        image_dir.mkdir()
        (image_dir / "same.jpg").write_bytes(f"image-{idx}".encode())
        (result_dir / "full.md").write_text(
            f"chunk {idx} ![figure](images/same.jpg)", encoding="utf-8"
        )
        (result_dir / "result.md").write_text(
            f"wrong child URL /api/v1/files/output/child-{idx}/images/same.jpg", encoding="utf-8"
        )
        (result_dir / "result.json").write_text(
            json.dumps([{"page_idx": 0, "img_path": "images/same.jpg", "text": str(idx)}]),
            encoding="utf-8",
        )
        (result_dir / "mineru_model.json").write_text(
            json.dumps([[{"type": "image", "img_path": "images/same.jpg", "chunk": idx}]]),
            encoding="utf-8",
        )
        diagnostic = result_dir / "raw" / "layout.pdf"
        diagnostic.parent.mkdir()
        diagnostic.write_text(f"layout-{idx}", encoding="utf-8")
    return children


def test_parent_merge_combines_core_artifacts_images_and_full_chunks(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    _prepare_artifact_merge(db, parent_id, preserve_all=True)
    assert db.claim_parent_merge(parent_id, "worker-a")

    out = merge_parent_task_results(
        task_db=db,
        parent_task_id=parent_id,
        output_dir=str(tmp_path / "out"),
        merge_owner="worker-a",
        cleanup_child_files=False,
    )

    full_markdown = (out / "full.md").read_text(encoding="utf-8")
    result_markdown = (out / "result.md").read_text(encoding="utf-8")
    assert "images/same.jpg" in full_markdown
    assert "images/p2_same.jpg" in full_markdown
    assert "/api/v1/files/output/parent/images/same.jpg" in result_markdown
    assert "/api/v1/files/output/parent/images/p2_same.jpg" in result_markdown
    assert (out / "images" / "same.jpg").read_bytes() == b"image-0"
    assert (out / "images" / "p2_same.jpg").read_bytes() == b"image-1"

    content = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert content == json.loads((out / "content_list.json").read_text(encoding="utf-8"))
    assert [item["page_idx"] for item in content] == [0, 1]
    assert [item["img_path"] for item in content] == ["images/same.jpg", "images/p2_same.jpg"]

    model = json.loads((out / "mineru_model.json").read_text(encoding="utf-8"))
    assert len(model) == 2
    assert model[1][0]["img_path"] == "images/p2_same.jpg"
    assert (out / "chunks" / "0001_pages_1-1" / "raw" / "layout.pdf").exists()
    assert (out / "chunks" / "0002_pages_2-2" / "raw" / "layout.pdf").exists()


def test_compact_parent_merge_omits_full_chunk_trees(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    _prepare_artifact_merge(db, parent_id, preserve_all=False)
    assert db.claim_parent_merge(parent_id, "worker-a")

    out = merge_parent_task_results(
        task_db=db,
        parent_task_id=parent_id,
        output_dir=str(tmp_path / "out"),
        merge_owner="worker-a",
        cleanup_child_files=False,
    )

    assert (out / "mineru_model.json").exists()
    assert (out / "content_list.json").exists()
    assert (out / "images" / "same.jpg").exists()
    assert not (out / "chunks").exists()


def test_completed_parent_rebuild_is_dry_run_then_explicit_apply(tmp_path):
    db, parent_id, _child_ids = _make_parent_with_children(tmp_path)
    _prepare_artifact_merge(db, parent_id, preserve_all=False)
    assert db.claim_parent_merge(parent_id, "worker-a")
    out = merge_parent_task_results(
        task_db=db,
        parent_task_id=parent_id,
        output_dir=str(tmp_path / "out"),
        merge_owner="worker-a",
        cleanup_child_files=False,
    )
    (out / "mineru_model.json").unlink()
    (out / "content_list.json").unlink()

    dry_args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(tmp_path / "tasks.db"),
        "--output-dir", str(tmp_path / "out"),
        "--rebuild-completed",
        "--task-id", parent_id,
    ])
    dry_report = reconcile_parent_merges.run_once(dry_args, "rebuild-test")
    assert dry_report["summary"] == {"rebuildable": 1, "blocked": 0}
    assert not (out / "mineru_model.json").exists()

    apply_args = reconcile_parent_merges.build_parser().parse_args([
        "--db-path", str(tmp_path / "tasks.db"),
        "--output-dir", str(tmp_path / "out"),
        "--rebuild-completed",
        "--task-id", parent_id,
        "--apply",
    ])
    apply_report = reconcile_parent_merges.run_once(apply_args, "rebuild-test")
    assert apply_report["tasks"][0]["rebuilt"] is True
    assert db.get_task(parent_id)["status"] == "completed"
    assert (out / "mineru_model.json").exists()
    assert (out / "content_list.json").exists()


def test_parent_status_prefers_merged_root_files_over_full_mode_chunks(tmp_path, monkeypatch):
    import api_server

    result_dir = tmp_path / "parent-result"
    result_dir.mkdir()
    (result_dir / "result.md").write_text("PARENT MARKDOWN", encoding="utf-8")
    (result_dir / "result.json").write_text('[{"scope":"parent"}]', encoding="utf-8")
    (result_dir / "content_list.json").write_text('[{"scope":"parent"}]', encoding="utf-8")
    (result_dir / "mineru_model.json").write_text('[{"scope":"parent-model"}]', encoding="utf-8")
    (result_dir / "parent.pdf").write_text("parent pdf", encoding="utf-8")
    chunk = result_dir / "chunks" / "0001_pages_1-1"
    chunk.mkdir(parents=True)
    (chunk / "result.md").write_text("CHILD MARKDOWN", encoding="utf-8")
    (chunk / "result.json").write_text('[{"scope":"child"}]', encoding="utf-8")
    (chunk / "mineru_model.json").write_text('[{"scope":"child-model"}]', encoding="utf-8")
    (chunk / "child_layout.pdf").write_text("child layout", encoding="utf-8")

    class FakeDB:
        def get_task(self, task_id):
            return {
                "task_id": task_id,
                "status": "completed",
                "file_name": "parent.pdf",
                "file_path": str(tmp_path / "parent.pdf"),
                "backend": "hybrid-auto-engine",
                "priority": 0,
                "error_message": None,
                "created_at": "now",
                "started_at": "now",
                "completed_at": "now",
                "user_id": "user-1",
                "shared_read": 0,
                "is_parent": 1,
                "child_count": 1,
                "child_completed": 1,
                "result_path": str(result_dir),
            }

        def get_child_tasks(self, task_id):
            return []

    class FakeUser:
        user_id = "user-1"

        def has_permission(self, permission):
            return True

    monkeypatch.setattr(api_server, "db", FakeDB())
    monkeypatch.setattr(api_server, "OUTPUT_DIR", tmp_path)
    response = api_server.get_task_status("parent-task", format="both", current_user=FakeUser())

    assert response["data"]["content"] == "PARENT MARKDOWN"
    assert response["data"]["json_content"] == [{"scope": "parent"}]
    assert response["data"]["mineru_model_content"] == [{"scope": "parent-model"}]
    assert response["data"]["pdf_path"] == "parent-result/parent.pdf"
