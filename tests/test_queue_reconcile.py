import json
import sys
import time

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from redis_queue import QueueClaim, RedisConfig, RedisTaskQueue
from task_db import TaskDB
import reconcile_task_queue as reconcile_module
from reconcile_task_queue import reconcile


class FakeRedisClient:
    def __init__(self):
        self.zsets = {}
        self.hashes = {}
        self.strings = {}
        self.hscan_calls = 0
        self.zscan_calls = 0
        self.hgetall_calls = 0
        self.zscore_calls = 0
        self.hget_calls = 0
        self.zmscore_calls = 0
        self.hmget_calls = 0

    def pipeline(self):
        return self

    def execute(self):
        return []

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def zscan(self, key, cursor=0, count=500):
        self.zscan_calls += 1
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[0])
        start = int(cursor)
        end = min(start + count, len(items))
        next_cursor = 0 if end >= len(items) else end
        return next_cursor, items[start:end]

    def zscore(self, key, member):
        self.zscore_calls += 1
        return self.zsets.get(key, {}).get(member)

    def zmscore(self, key, members):
        self.zmscore_calls += 1
        return [self.zsets.get(key, {}).get(member) for member in members]

    def zrem(self, key, member):
        return 1 if self.zsets.get(key, {}).pop(member, None) is not None else 0

    def zcard(self, key):
        return len(self.zsets.get(key, {}))

    def hset(self, key, name=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(key, {})
        if mapping:
            bucket.update(mapping)
            return len(mapping)
        bucket[name] = value
        return 1

    def hget(self, key, field):
        self.hget_calls += 1
        return self.hashes.get(key, {}).get(field)

    def hmget(self, key, fields):
        self.hmget_calls += 1
        return [self.hashes.get(key, {}).get(field) for field in fields]

    def hscan(self, key, cursor=0, count=500):
        self.hscan_calls += 1
        items = sorted(self.hashes.get(key, {}).items(), key=lambda kv: kv[0])
        start = int(cursor)
        end = min(start + count, len(items))
        next_cursor = 0 if end >= len(items) else end
        return next_cursor, dict(items[start:end])

    def hgetall(self, key):
        self.hgetall_calls += 1
        raise AssertionError("reconcile must use HSCAN, not HGETALL")

    def hdel(self, key, field):
        return 1 if self.hashes.get(key, {}).pop(field, None) is not None else 0

    def hlen(self, key):
        return len(self.hashes.get(key, {}))

    def set(self, key, value):
        self.strings[key] = value
        return True

    def get(self, key):
        return self.strings.get(key)

    def delete(self, *keys):
        for key in keys:
            self.zsets.pop(key, None)
            self.hashes.pop(key, None)
            self.strings.pop(key, None)
        return len(keys)

    def expire(self, key, seconds):
        return True

    def script_load(self, script):
        return "sha1"

    def evalsha(self, sha, numkeys, queue_key, processing_key, maintenance_key, pause_key, processing_data, now, worker_id):
        if self.get(pause_key):
            return ["maintenance", None]
        items = sorted(self.zsets.get(queue_key, {}).items(), key=lambda kv: kv[1])
        if not items:
            return ["empty", None]
        task_id, _score = items[0]
        self.zrem(queue_key, task_id)
        self.hset(processing_key, task_id, processing_data)
        self.hset(maintenance_key, mapping={"last_claimed_at": now, "last_worker_id": worker_id})
        return ["claimed", task_id]


class FakeRedisQueue:
    def __init__(self):
        self.config = RedisConfig(queue_key="q", processing_key="p", claim_maintenance_key="m", claim_pause_key="pause", task_data_prefix="t:")
        self.client = FakeRedisClient()

    def enqueue(self, task_id, priority=0, task_data=None):
        score = -priority * 1e10 + time.time()
        self.client.zadd(self.config.queue_key, {task_id: score})
        return True

    def set_claim_maintenance(self, reason="maintenance"):
        self.client.set(self.config.claim_pause_key, reason)
        return True

    def clear_claim_maintenance(self):
        self.client.delete(self.config.claim_pause_key)
        return True

    def is_claim_maintenance(self):
        return bool(self.client.get(self.config.claim_pause_key))


def make_queue_with_fake_client():
    queue = object.__new__(RedisTaskQueue)
    queue.config = RedisConfig(queue_key="q", processing_key="p", claim_maintenance_key="m", claim_pause_key="pause", task_data_prefix="t:")
    queue._client = FakeRedisClient()
    queue._connected = True
    queue._claim_script_sha = None
    return queue


def test_redis_claim_is_atomic_and_distinguishes_empty_and_maintenance():
    queue = make_queue_with_fake_client()
    queue.enqueue("low", priority=1)
    queue.enqueue("high", priority=5)

    claim = queue.claim("worker-a", timeout=0)
    assert claim == QueueClaim(status="claimed", task_id="high")
    assert "high" not in queue.client.zsets[queue.config.queue_key]
    assert "high" in queue.client.hashes[queue.config.processing_key]
    assert queue.claim("worker-a", timeout=0).task_id == "low"
    assert queue.claim("worker-a", timeout=0).status == "empty"

    queue.enqueue("paused", priority=9)
    assert queue.set_claim_maintenance("test") is True
    assert queue.is_claim_maintenance() is True
    assert queue.claim("worker-a", timeout=0).status == "maintenance"
    assert "paused" in queue.client.zsets[queue.config.queue_key]
    assert queue.clear_claim_maintenance() is True
    assert queue.claim("worker-a", timeout=0).task_id == "paused"


def test_reconcile_dry_run_does_not_advance_apply_checkpoint(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    for idx in range(3):
        db.create_task(f"f-{idx}.pdf", f"/tmp/f-{idx}.pdf")
    checkpoint = tmp_path / "checkpoint.json"
    queue = FakeRedisQueue()

    dry = reconcile(db.db_path, queue, apply=False, checkpoint=checkpoint, batch_size=2)
    assert dry.scanned_pending == 3
    assert not checkpoint.exists()

    applied = reconcile(db.db_path, queue, apply=True, checkpoint=checkpoint, batch_size=2)
    assert applied.scanned_pending == 3
    assert json.loads(checkpoint.read_text()) == {}


def test_reconcile_dry_run_then_apply_is_idempotent_and_uses_hscan(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    missing = db.create_task("missing.pdf", "/tmp/missing.pdf", priority=7)["task_id"]
    queued_ok = db.create_task("queued.pdf", "/tmp/queued.pdf")["task_id"]
    stale = "stale-redis-task"
    queue = FakeRedisQueue()
    queue.enqueue(queued_ok)
    queue.enqueue(stale)

    dry = reconcile(db.db_path, queue, apply=False, batch_size=500)
    assert dry.dry_run is True
    assert dry.enqueued_missing == 1
    assert dry.stale_queued_removed == 1
    assert missing not in queue.client.zsets[queue.config.queue_key]
    assert stale in queue.client.zsets[queue.config.queue_key]

    applied = reconcile(db.db_path, queue, apply=True, batch_size=500)
    assert applied.enqueued_missing == 1
    assert applied.stale_queued_removed == 1
    assert missing in queue.client.zsets[queue.config.queue_key]
    assert stale not in queue.client.zsets[queue.config.queue_key]
    assert queue.client.hscan_calls > 0
    assert queue.client.hgetall_calls == 0
    assert not queue.is_claim_maintenance()

    again = reconcile(db.db_path, queue, apply=True, batch_size=500)
    assert again.enqueued_missing == 0
    assert again.stale_queued_removed == 0


def test_reconcile_pending_presence_uses_batch_redis_checks_and_preserves_results(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    missing = db.create_task("missing.pdf", "/tmp/missing.pdf", priority=7)["task_id"]
    queued = db.create_task("queued.pdf", "/tmp/queued.pdf", priority=4)["task_id"]
    processing = db.create_task("processing.pdf", "/tmp/processing.pdf", priority=2)["task_id"]
    queue = FakeRedisQueue()
    queue.enqueue(queued)
    queue.client.hset(queue.config.processing_key, processing, json.dumps({"worker_id": "w", "claimed_at": time.time()}))

    dry = reconcile(db.db_path, queue, apply=False, batch_size=500)
    assert dry.enqueued_missing == 1
    assert dry.actions[0]["task_id"] == missing
    assert queue.client.zmscore_calls == 1
    assert queue.client.hmget_calls == 1
    assert queue.client.zscore_calls == 0

    queue = FakeRedisQueue()
    queue.enqueue(queued)
    queue.client.hset(queue.config.processing_key, processing, json.dumps({"worker_id": "w", "claimed_at": time.time()}))
    applied = reconcile(db.db_path, queue, apply=True, batch_size=500)
    assert applied.enqueued_missing == 1
    assert missing in queue.client.zsets[queue.config.queue_key]
    assert queued in queue.client.zsets[queue.config.queue_key]
    assert processing in queue.client.hashes[queue.config.processing_key]
    assert queue.client.zmscore_calls == 1
    assert queue.client.hmget_calls == 1
    assert queue.client.zscore_calls == 0


def test_reconcile_active_only_skips_pending_and_queued_scans(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    missing_pending = db.create_task("missing.pdf", "/tmp/missing.pdf")["task_id"]
    stale_active = db.create_task("stale-active.pdf", "/tmp/stale-active.pdf")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            "UPDATE tasks SET status='processing', started_at=datetime('now','-2 hours'), worker_id='dead' WHERE task_id=?",
            (stale_active,),
        )

    queue = FakeRedisQueue()
    queue.enqueue("stale-queued")
    queue.client.hset(
        queue.config.processing_key,
        "stale-processing",
        json.dumps({"worker_id": "dead", "claimed_at": time.time() - 9999}),
    )

    report = reconcile(
        db.db_path,
        queue,
        apply=True,
        active_only=True,
        fresh_heartbeat_seconds=300,
        active_stale_seconds=3600,
    )

    assert report.active_only is True
    assert report.scanned_pending == 0
    assert report.scanned_queued == 0
    assert queue.client.zscan_calls == 0
    assert queue.client.zmscore_calls == 0
    assert queue.client.hmget_calls == 0
    assert report.scanned_processing == 1
    assert report.stale_processing_removed == 1
    assert report.stale_sqlite_reset == 1
    assert "stale-queued" in queue.client.zsets[queue.config.queue_key]
    assert missing_pending not in queue.client.zsets[queue.config.queue_key]
    assert stale_active in queue.client.zsets[queue.config.queue_key]
    assert db.get_task(stale_active)["status"] == "pending"
    assert not queue.is_claim_maintenance()


def test_reconcile_mixed_fresh_and_stale_processing_batch(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    fresh = "fresh-ghost"
    stale = "stale-ghost"
    queue = FakeRedisQueue()
    queue.client.hset(queue.config.processing_key, fresh, json.dumps({"worker_id": "w", "claimed_at": time.time()}))
    queue.client.hset(queue.config.processing_key, stale, json.dumps({"worker_id": "w", "claimed_at": time.time() - 9999}))

    report = reconcile(db.db_path, queue, apply=True, fresh_heartbeat_seconds=300)
    assert report.fresh_processing_protected == 1
    assert report.stale_processing_removed == 1
    assert fresh in queue.client.hashes[queue.config.processing_key]
    assert stale not in queue.client.hashes[queue.config.processing_key]


def test_reconcile_requeues_sqlite_pending_stuck_in_redis_processing(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("pending-processing.pdf", "/tmp/pending-processing.pdf", priority=9)["task_id"]
    queue = FakeRedisQueue()
    queue.client.hset(queue.config.processing_key, task_id, json.dumps({"worker_id": "w", "claimed_at": time.time() - 9999}))

    dry = reconcile(db.db_path, queue, apply=False, fresh_heartbeat_seconds=300)
    assert dry.dry_run is True
    assert dry.stale_processing_removed == 1
    assert dry.enqueued_missing == 1
    assert dry.actions[0]["action"] == "requeue_pending_processing"
    assert task_id in queue.client.hashes[queue.config.processing_key]
    assert task_id not in queue.client.zsets.get(queue.config.queue_key, {})

    applied = reconcile(db.db_path, queue, apply=True, fresh_heartbeat_seconds=300)
    assert applied.stale_processing_removed == 1
    assert applied.enqueued_missing == 1
    assert applied.actions[0]["action"] == "requeue_pending_processing"
    assert task_id not in queue.client.hashes.get(queue.config.processing_key, {})
    assert task_id in queue.client.zsets[queue.config.queue_key]
    assert db.get_task(task_id)["status"] == "pending"


def test_reconcile_stale_active_reset_protects_fresh_heartbeat_and_parent_waiting(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    stale_active = db.create_task("stale.pdf", "/tmp/stale.pdf")["task_id"]
    fresh_active = db.create_task("fresh.pdf", "/tmp/fresh.pdf")["task_id"]
    parent = db.create_parent_task("parent.pdf", "/tmp/parent.pdf")
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE tasks SET status='processing', started_at=datetime('now','-2 hours'), worker_id='w' WHERE task_id IN (?, ?)", (stale_active, fresh_active))
        cursor.execute("UPDATE tasks SET started_at=datetime('now','-2 hours'), child_count=2, child_completed=1 WHERE task_id=?", (parent,))

    queue = FakeRedisQueue()
    queue.client.hset(queue.config.processing_key, fresh_active, json.dumps({"worker_id": "w", "claimed_at": time.time()}))
    report = reconcile(db.db_path, queue, apply=True, fresh_heartbeat_seconds=300, batch_size=500)

    assert report.stale_sqlite_reset == 1
    assert report.fresh_processing_protected >= 1
    assert report.parent_waiting_protected == 1
    assert db.get_task(stale_active)["status"] == "pending"
    assert stale_active in queue.client.zsets[queue.config.queue_key]
    assert stale_active not in queue.client.hashes.get(queue.config.processing_key, {})
    assert db.get_task(fresh_active)["status"] == "processing"
    assert db.get_task(parent)["status"] == "processing"


def test_reconcile_never_uses_sqlite_bind_batches_over_500(tmp_path, monkeypatch):
    db = TaskDB(tmp_path / "tasks.db")
    queue = FakeRedisQueue()
    for idx in range(1001):
        queue.enqueue(f"stale-{idx:04d}")

    real_connect = reconcile_module.sqlite3.connect

    class CheckedConnection:
        def __init__(self, conn):
            object.__setattr__(self, "_conn", conn)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            setattr(self._conn, name, value)

        def execute(self, sql, params=()):
            assert len(tuple(params)) <= 500
            return self._conn.execute(sql, params)

    def checked_connect(*args, **kwargs):
        return CheckedConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(reconcile_module.sqlite3, "connect", checked_connect)
    report = reconcile(db.db_path, queue, apply=False, batch_size=9999)
    assert report.scanned_queued == 1001



def test_reconcile_active_stale_age_is_separate_from_heartbeat_freshness(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    long_running = db.create_task("long.pdf", "/tmp/long.pdf")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            "UPDATE tasks SET status='processing', started_at=datetime('now','-6 minutes'), worker_id='w' WHERE task_id=?",
            (long_running,),
        )
    queue = FakeRedisQueue()

    report = reconcile(db.db_path, queue, apply=True, fresh_heartbeat_seconds=90, active_stale_seconds=3600)
    assert report.stale_sqlite_reset == 0
    assert db.get_task(long_running)["status"] == "processing"
    assert long_running not in queue.client.zsets.get(queue.config.queue_key, {})


def test_reconcile_apply_fails_closed_when_claim_pause_fails(tmp_path):
    class NoPauseQueue(FakeRedisQueue):
        def set_claim_maintenance(self, reason="maintenance"):
            return False

    db = TaskDB(tmp_path / "tasks.db")
    db.create_task("pending.pdf", "/tmp/pending.pdf")
    queue = NoPauseQueue()

    try:
        reconcile(db.db_path, queue, apply=True)
    except RuntimeError as exc:
        assert "claim pause" in str(exc)
    else:
        raise AssertionError("apply must fail closed when claims cannot be paused")


def test_reconcile_ensure_indexes_requires_apply_before_pause(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    queue = FakeRedisQueue()

    with pytest.raises(RuntimeError, match="ensure-indexes requires --apply"):
        reconcile(db.db_path, queue, apply=False, ensure_indexes=True)

    assert not queue.is_claim_maintenance()


def test_reconcile_preserves_preexisting_claim_pause(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    queue = FakeRedisQueue()
    queue.set_claim_maintenance("operator")

    reconcile(db.db_path, queue, apply=True)
    assert queue.is_claim_maintenance() is True


def test_reconcile_caps_action_details(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    queue = FakeRedisQueue()
    for idx in range(150):
        queue.enqueue(f"ghost-{idx:03d}")

    report = reconcile(db.db_path, queue, apply=False, batch_size=500)
    assert report.stale_queued_removed == 150
    assert len(report.actions) == 100
    assert report.omitted_actions == 50


def test_reconcile_loops_all_pending_batches_before_processing_scan(tmp_path):
    db = TaskDB(tmp_path / "tasks.db")
    for idx in range(1001):
        db.create_task(f"pending-{idx}.pdf", f"/tmp/pending-{idx}.pdf")
    queue = FakeRedisQueue()
    queue.client.hset(queue.config.processing_key, "ghost", json.dumps({"claimed_at": time.time() - 9999}))

    report = reconcile(db.db_path, queue, apply=False, batch_size=500)
    assert report.scanned_pending == 1001
    assert queue.client.zmscore_calls == 3
    assert queue.client.hmget_calls == 3
    assert queue.client.zscore_calls == 0
    assert queue.client.hget_calls == 0
    assert queue.client.hscan_calls == 1



def test_reconcile_dry_run_does_not_instantiate_taskdb_or_touch_mtime(tmp_path, monkeypatch):
    db = TaskDB(tmp_path / "tasks.db")
    db.create_task("pending.pdf", "/tmp/pending.pdf")
    db_path = Path(db.db_path)
    before = db_path.stat().st_mtime_ns

    class ExplodingTaskDB:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run must not instantiate TaskDB")

    monkeypatch.setattr(reconcile_module, "TaskDB", ExplodingTaskDB)
    report = reconcile(db.db_path, FakeRedisQueue(), apply=False)

    assert report.scanned_pending == 1
    assert db_path.stat().st_mtime_ns == before


def test_default_db_path_matches_live_instance_layout(monkeypatch):
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.setenv("INSTANCE_ID", "abc123")
    assert reconcile_module.default_db_path() == "/share/wangjiong/databases/mineru_database/abc123/mineru_tianshu.db"

    monkeypatch.setenv("DATABASE_PATH", "/tmp/custom.db")
    assert reconcile_module.default_db_path() == "/tmp/custom.db"



def test_reconcile_sqlite_reset_failure_raises_and_leaves_claim_pause(tmp_path):
    class FailingEnqueueQueue(FakeRedisQueue):
        def enqueue(self, task_id, priority=0, task_data=None):
            return False

    db = TaskDB(tmp_path / "tasks.db")
    task_id = db.create_task("stale.pdf", "/tmp/stale.pdf")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            "UPDATE tasks SET status='processing', started_at=datetime('now','-2 hours'), worker_id='w' WHERE task_id=?",
            (task_id,),
        )
    queue = FailingEnqueueQueue()
    queue.client.hset(queue.config.processing_key, task_id, json.dumps({"worker_id": "w", "claimed_at": time.time() - 9999}))

    with pytest.raises(RuntimeError, match="Redis enqueue failed after SQLite reset"):
        reconcile(db.db_path, queue, apply=True, active_stale_seconds=3600)

    assert db.get_task(task_id)["status"] == "pending"
    assert task_id not in queue.client.hashes.get(queue.config.processing_key, {})
    assert task_id not in queue.client.zsets.get(queue.config.queue_key, {})
    assert queue.is_claim_maintenance()
