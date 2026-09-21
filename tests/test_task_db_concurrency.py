import fcntl
import multiprocessing
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import task_db as task_db_module
from redis_queue import QueueClaim
from task_db import TaskDB, _SQLiteWriteLock


def _create_task_process(db_path: str, idx: int, barrier, queue):
    try:
        barrier.wait(timeout=20)
        db = TaskDB(db_path)
        result = db.create_task(f"proc-{idx}.pdf", f"/tmp/proc-{idx}.pdf", priority=idx % 5)
        queue.put((True, result["task_id"]))
    except Exception as exc:
        queue.put((False, repr(exc)))


def test_sqlite_writes_are_serialized_across_32_processes(tmp_path):
    db_path = str(tmp_path / "tasks.db")
    TaskDB(db_path)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(32)
    queue = ctx.Queue()
    processes = [ctx.Process(target=_create_task_process, args=(db_path, idx, barrier, queue)) for idx in range(32)]

    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)

    assert all(ok for ok, _value in results), results
    created = [value for ok, value in results if ok]
    assert len(created) == len(set(created)) == 32
    assert TaskDB(db_path).get_queue_stats()["pending"] == 32


def test_sqlite_claims_are_unique_with_32_simultaneous_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    db = TaskDB(tmp_path / "tasks.db")
    task_ids = [db.create_task(f"f-{i}.pdf", f"/tmp/f-{i}.pdf", priority=i % 3)["task_id"] for i in range(32)]

    def claim(worker_idx):
        task = db.get_next_task(f"worker-{worker_idx}")
        return task["task_id"] if task else None

    with ThreadPoolExecutor(max_workers=32) as pool:
        claimed = [task_id for task_id in pool.map(claim, range(32)) if task_id]

    assert len(claimed) == len(task_ids)
    assert len(set(claimed)) == len(task_ids)
    assert set(claimed) == set(task_ids)


def test_readonly_context_does_not_take_fcntl_write_lock(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("read.pdf", "/tmp/read.pdf")["task_id"]
    lock = _SQLiteWriteLock(db.db_path)
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    with lock.path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        assert db.get_task(task_id)["task_id"] == task_id
        assert time.monotonic() - started < 1.0
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_completion_metric_columns_are_migrated(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    with db.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(tasks)")
        columns = {row["name"] for row in cursor.fetchall()}

    assert {
        "page_count",
        "processing_seconds",
        "worker_group_index",
        "worker_child_index",
    }.issubset(columns)


def test_completed_status_persists_typed_metrics(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("metrics.pdf", "/tmp/metrics.pdf")["task_id"]
    task = db.get_next_task("worker-a")
    assert task["task_id"] == task_id

    assert db.update_task_status(
        task_id,
        "completed",
        worker_id="worker-a",
        result_path="/tmp/out",
        page_count=12,
        processing_seconds=3.5,
        worker_group_index=4,
        worker_child_index=2,
    ) is True

    completed = db.get_task(task_id)
    assert completed["status"] == "completed"
    assert completed["page_count"] == 12
    assert completed["processing_seconds"] == 3.5
    assert completed["worker_group_index"] == 4
    assert completed["worker_child_index"] == 2


def test_completed_status_old_call_leaves_metrics_null(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("old.pdf", "/tmp/old.pdf")["task_id"]
    task = db.get_next_task("worker-a")
    assert task["task_id"] == task_id

    assert db.update_task_status(task_id, "completed", worker_id="worker-a", result_path="/tmp/out") is True

    completed = db.get_task(task_id)
    assert completed["status"] == "completed"
    assert completed["page_count"] is None
    assert completed["processing_seconds"] is None
    assert completed["worker_group_index"] is None
    assert completed["worker_child_index"] is None


def test_completed_status_survives_transient_sqlite_busy(tmp_path, monkeypatch):
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("done.pdf", "/tmp/done.pdf")["task_id"]
    task = db.get_next_task("worker-a")
    assert task["task_id"] == task_id

    calls = {"n": 0}
    real_connect = sqlite3.connect

    class BusyOnceConnection:
        def __init__(self, conn):
            object.__setattr__(self, "_conn", conn)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            setattr(self._conn, name, value)

        def commit(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return self._conn.commit()

    def connect_busy_once(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        if calls["n"] == 0:
            return BusyOnceConnection(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect_busy_once)
    assert db.update_task_status(task_id, "completed", worker_id="worker-a", result_path="/tmp/out") is True
    assert db.get_task(task_id)["status"] == "completed"


def test_redis_unavailable_with_sqlite_fallback_false_fails_closed(tmp_path, monkeypatch):
    class UnavailableRedis:
        def enqueue(self, *args, **kwargs):
            return True

        def claim(self, worker_id, timeout=1.0):
            return QueueClaim(status="unavailable", error="down")

    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "false")
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: UnavailableRedis())
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("pending.pdf", "/tmp/pending.pdf")["task_id"]

    assert db.get_next_task("worker-a") is None
    assert db.get_task(task_id)["status"] == "pending"



def test_redis_maintenance_does_not_fall_back_to_sqlite(tmp_path, monkeypatch):
    class MaintenanceRedis:
        def enqueue(self, *args, **kwargs):
            return True

        def claim(self, worker_id, timeout=1.0):
            return QueueClaim(status="maintenance")

    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: MaintenanceRedis())
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("paused.pdf", "/tmp/paused.pdf")["task_id"]

    assert db.get_next_task("worker-a") is None
    assert db.get_task(task_id)["status"] == "pending"



def test_redis_claim_sqlite_failure_requeues_and_does_not_fallback(tmp_path, monkeypatch):
    class ClaimedRedis:
        def __init__(self, task_id):
            self.task_id = task_id
            self.failed = []

        def enqueue(self, *args, **kwargs):
            return True

        def claim(self, worker_id, timeout=1.0):
            return QueueClaim(status="claimed", task_id=self.task_id)

        def fail(self, task_id, worker_id, requeue=False):
            self.failed.append((task_id, worker_id, requeue))
            return True

    db = TaskDB(tmp_path / "tasks.db")
    claimed_id = db.create_task("claimed.pdf", "/tmp/claimed.pdf", priority=10)["task_id"]
    other_id = db.create_task("other.pdf", "/tmp/other.pdf", priority=1)["task_id"]
    redis_queue = ClaimedRedis(claimed_id)

    real_get_cursor = db.get_cursor

    def failing_get_cursor():
        ctx = real_get_cursor()
        cursor = ctx.__enter__()

        class FailingContext:
            def __enter__(self):
                return cursor

            def __exit__(self, exc_type, exc, tb):
                ctx.__exit__(sqlite3.OperationalError, sqlite3.OperationalError("database is locked"), None)
                raise sqlite3.OperationalError("database is locked")

        return FailingContext()

    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: redis_queue)
    monkeypatch.setattr(db, "get_cursor", failing_get_cursor)

    assert db.get_next_task("worker-a") is None
    assert redis_queue.failed == [(claimed_id, "worker-a", True)]

    monkeypatch.setattr(db, "get_cursor", real_get_cursor)
    assert db.get_task(claimed_id)["status"] == "pending"
    assert db.get_task(other_id)["status"] == "pending"



def test_redis_claim_requeues_when_pending_update_is_ignored_by_legacy_trigger(tmp_path, monkeypatch):
    class ClaimedRedis:
        def __init__(self, task_id):
            self.task_id = task_id
            self.failed = []

        def enqueue(self, *args, **kwargs):
            return True

        def claim(self, worker_id, timeout=1.0):
            return QueueClaim(status="claimed", task_id=self.task_id)

        def fail(self, task_id, worker_id, requeue=False):
            self.failed.append((task_id, worker_id, requeue))
            return True

    db = TaskDB(tmp_path / "tasks.db")
    claimed_id = db.create_task("claimed.pdf", "/tmp/claimed.pdf", priority=10)["task_id"]
    other_id = db.create_task("other.pdf", "/tmp/other.pdf", priority=1)["task_id"]
    redis_queue = ClaimedRedis(claimed_id)

    with db.get_cursor() as cursor:
        cursor.execute(
            """
            CREATE TRIGGER ignore_processing_claim
            BEFORE UPDATE OF status ON tasks
            WHEN OLD.task_id = '%s' AND OLD.status = 'pending' AND NEW.status = 'processing'
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
            % claimed_id
        )

    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: redis_queue)

    assert db.get_next_task("worker-a") is None
    assert redis_queue.failed == [(claimed_id, "worker-a", True)]
    assert db.get_task(claimed_id)["status"] == "pending"
    assert db.get_task(other_id)["status"] == "pending"


def test_create_task_redis_only_enqueue_failure_marks_failed_and_dedup_recovers(tmp_path, monkeypatch):
    class FlakyRedis:
        def __init__(self):
            self.calls = []

        def enqueue(self, task_id, priority=0, task_data=None):
            self.calls.append(task_id)
            return len(self.calls) > 1

    redis_queue = FlakyRedis()
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: redis_queue)
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "false")
    db = TaskDB(tmp_path / "tasks.db")

    first = db.create_task("redis-only.pdf", "/tmp/redis-only.pdf", file_hash="redis-only", lang="en", method="auto")
    assert first["status"] == "failed"
    assert first["enqueue_failed"] is True
    assert first["enqueue_failed_task_ids"] == [first["task_id"]]
    task = db.get_task(first["task_id"])
    assert task["status"] == "failed"
    assert "Initial enqueue failed" in task["error_message"]

    second = db.create_task("redis-only.pdf", "/tmp/redis-only.pdf", file_hash="redis-only", lang="en", method="auto")
    assert second["task_id"] == first["task_id"]
    assert second["deduped"] is True
    assert second["status"] == "pending"
    assert second["requeued"] is True
    assert second["retry_info"]["enqueue_failed_task_ids"] == []
    assert db.get_task(first["task_id"])["retry_count"] == 1
    assert redis_queue.calls == [first["task_id"], first["task_id"]]


def test_create_task_enqueue_failure_uses_sqlite_fallback_when_enabled(tmp_path, monkeypatch):
    class FailingRedis:
        def enqueue(self, task_id, priority=0, task_data=None):
            return False

    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", True)
    monkeypatch.setattr(task_db_module, "get_redis_queue", lambda: FailingRedis())
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "true")
    db = TaskDB(tmp_path / "tasks.db")

    result = db.create_task("fallback.pdf", "/tmp/fallback.pdf")

    assert result["status"] == "pending"
    assert result["enqueue_failed"] is False
    assert result["enqueue_failed_task_ids"] == []
    assert db.get_task(result["task_id"])["status"] == "pending"


def test_create_task_no_redis_module_keeps_sqlite_fallback_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(task_db_module, "REDIS_QUEUE_AVAILABLE", False)
    monkeypatch.setenv("SQLITE_QUEUE_FALLBACK", "false")
    db = TaskDB(tmp_path / "tasks.db")

    result = db.create_task("sqlite-only.pdf", "/tmp/sqlite-only.pdf")

    assert result["status"] == "pending"
    assert result["enqueue_failed"] is False
    assert result["enqueue_failed_task_ids"] == []
    assert db.get_task(result["task_id"])["status"] == "pending"


def test_storage_health_and_manual_passive_checkpoint(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    db.create_task("health.pdf", "/tmp/health.pdf")
    health = db.storage_health()
    assert health["database_bytes"] > 0
    assert health["level"] in {"ok", "warning", "critical"}
    checkpoint = db.checkpoint_wal("PASSIVE")
    assert checkpoint["mode"] == "PASSIVE"
    assert set(checkpoint) == {"mode", "busy", "log_pages", "checkpointed_pages"}


def test_force_claim_can_bypass_attempt_limit_only_when_explicit(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    parent = db.create_task("parent.pdf", "/tmp/parent.pdf")["task_id"]
    db.convert_to_parent_task(parent, child_count=0)
    child = db.create_child_task(parent, "child.pdf", "/tmp/child.pdf")
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status='completed' WHERE task_id=?", (child,))
        cursor.execute("UPDATE tasks SET status='merging', merge_attempts=3 WHERE task_id=?", (parent,))
    assert not db.claim_parent_merge(parent, "normal", max_attempts=3)
    assert db.claim_parent_merge(parent, "forced", max_attempts=3, force=True)
