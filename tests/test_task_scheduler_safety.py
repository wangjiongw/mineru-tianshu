import json
import sys
import time
import types
from pathlib import Path


class _NoopLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("loguru", types.SimpleNamespace(logger=_NoopLogger()))
sys.modules.setdefault("redis", types.SimpleNamespace(Redis=object))
sys.modules.setdefault(
    "aiohttp",
    types.SimpleNamespace(ClientSession=object, ClientTimeout=lambda *args, **kwargs: None),
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import redis_queue
from task_scheduler import TaskScheduler


class FakeRedisConfig:
    processing_key = "processing"
    queue_key = "queue"


class FakeRedisClient:
    def __init__(self):
        self.hashes = {"processing": {}}
        self.hgetall_calls = 0

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hlen(self, key):
        return len(self.hashes.get(key, {}))

    def hgetall(self, key):
        self.hgetall_calls += 1
        return dict(self.hashes.get(key, {}))

    def hdel(self, key, field):
        return 1 if self.hashes.get(key, {}).pop(field, None) is not None else 0


class FakeRedisQueue:
    def __init__(self):
        self.config = FakeRedisConfig()
        self.client = FakeRedisClient()
        self.requeued = []

    def fail(self, task_id, worker_id, requeue=False):
        if requeue:
            self.requeued.append(task_id)
        self.client.hdel(self.config.processing_key, task_id)
        return True


class LargeHashNoScanClient(FakeRedisClient):
    def hlen(self, key):
        return 80_000

    def hgetall(self, key):
        raise AssertionError("scheduler must not HGETALL a large processing hash")


def make_scheduler(tmp_path, monkeypatch, *, apply=False, batch_size=500, redis=None, stale_minutes=10):
    db_path = tmp_path / "tasks.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setattr(redis_queue, "get_redis_queue", lambda: redis)
    scheduler = TaskScheduler(
        stale_task_timeout=stale_minutes,
        orphan_recovery_apply=apply,
        orphan_recovery_batch_size=batch_size,
    )
    return scheduler, scheduler.db


def create_processing_task(db, task_id_suffix, *, minutes_old=70, retry_count=0):
    task_id = db.create_task(f"{task_id_suffix}.pdf", f"/tmp/{task_id_suffix}.pdf")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'processing', worker_id = 'worker-a', retry_count = ?,
                started_at = datetime('now', '-' || ? || ' minutes')
            WHERE task_id = ?
            """,
            (retry_count, minutes_old, task_id),
        )
    return task_id


def task_status(db, task_id):
    return db.get_task(task_id)["status"]


def test_stale_threshold_uses_conservative_maximum():
    assert (
        TaskScheduler._effective_stale_timeout_minutes(10, task_p99_seconds=None, heartbeat_interval_seconds=30)
        == 60
    )
    assert (
        TaskScheduler._effective_stale_timeout_minutes(20, task_p99_seconds=7_500, heartbeat_interval_seconds=30)
        == 188
    )
    assert (
        TaskScheduler._effective_stale_timeout_minutes(20, task_p99_seconds=None, heartbeat_interval_seconds=1_500)
        == 75
    )


def test_orphan_recovery_is_dry_run_by_default(tmp_path, monkeypatch):
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=False, redis=None)
    task_id = create_processing_task(db, "stale")

    stats = scheduler._recover_orphans_safely()

    assert stats["dry_run"] is True
    assert stats["stale_minutes"] == 60
    assert stats["planned_sqlite_reset"] == 1
    assert stats["applied_sqlite_reset"] == 0
    assert task_status(db, task_id) == "processing"


def test_dead_heartbeat_is_not_treated_as_fresh(tmp_path, monkeypatch):
    redis = FakeRedisQueue()
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, redis=redis)
    task_id = create_processing_task(db, "dead-heartbeat")
    redis.client.hashes[redis.config.processing_key][task_id] = json.dumps({
        "worker_id": "worker-a",
        "claimed_at": time.time() - (50 * 60),
    })

    stats = scheduler._recover_orphans_safely()

    assert stats["heartbeat_freshness_seconds"] == 90
    assert stats["fresh_heartbeat_protected"] == 0
    assert stats["applied_sqlite_reset"] == 1
    assert task_status(db, task_id) == "pending"


def test_apply_resets_only_stale_unprotected_tasks(tmp_path, monkeypatch):
    redis = FakeRedisQueue()
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, redis=redis)
    stale_task_id = create_processing_task(db, "stale-unprotected")
    fresh_task_id = create_processing_task(db, "fresh-heartbeat")
    redis.client.hashes[redis.config.processing_key][fresh_task_id] = json.dumps({
        "worker_id": "worker-a",
        "claimed_at": time.time(),
    })

    parent_id = db.create_task("parent.pdf", "/tmp/parent.pdf")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'processing', worker_id = 'worker-a', is_parent = 1,
                child_count = 3, child_completed = 1,
                started_at = datetime('now', '-70 minutes')
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    stats = scheduler._recover_orphans_safely()

    assert stats["dry_run"] is False
    assert stats["planned_sqlite_reset"] == 1
    assert stats["applied_sqlite_reset"] == 1
    assert stats["applied_redis_requeued"] == 1
    assert stats["fresh_heartbeat_protected"] >= 1
    assert stats["parent_waiting_protected"] == 1
    assert task_status(db, stale_task_id) == "pending"
    assert task_status(db, fresh_task_id) == "processing"
    assert task_status(db, parent_id) == "processing"
    assert redis.requeued == [stale_task_id]


def test_merging_parent_stays_parked_across_recovery_cycles(tmp_path, monkeypatch):
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=False, redis=None)
    parent_id = db.create_task("merge-parent.pdf", "/tmp/merge-parent.pdf")["task_id"]
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = 'merging', worker_id = 'worker-a', is_parent = 1,
                child_count = 3, child_completed = 3, retry_count = 2,
                started_at = datetime('now', '-70 minutes')
            WHERE task_id = ?
            """,
            (parent_id,),
        )

    first_dry = scheduler._recover_orphans_safely()
    second_dry = scheduler._recover_orphans_safely()

    assert first_dry["planned_parent_merging"] == 0
    assert first_dry["planned_parent_already_merging"] == 1
    assert first_dry["planned_sqlite_reset"] == 0
    assert first_dry["planned_sqlite_failed"] == 0
    assert second_dry["planned_parent_merging"] == 0
    assert second_dry["planned_parent_already_merging"] == 1
    assert second_dry["planned_sqlite_reset"] == 0
    assert second_dry["planned_sqlite_failed"] == 0
    assert task_status(db, parent_id) == "merging"

    scheduler.orphan_recovery_apply = True
    first_apply = scheduler._recover_orphans_safely()
    second_apply = scheduler._recover_orphans_safely()

    assert first_apply["planned_parent_merging"] == 0
    assert first_apply["planned_parent_already_merging"] == 1
    assert first_apply["applied_parent_merging"] == 0
    assert first_apply["applied_sqlite_reset"] == 0
    assert first_apply["applied_sqlite_failed"] == 0
    assert second_apply["planned_parent_merging"] == 0
    assert second_apply["planned_parent_already_merging"] == 1
    assert second_apply["applied_parent_merging"] == 0
    assert second_apply["applied_sqlite_reset"] == 0
    assert second_apply["applied_sqlite_failed"] == 0
    assert task_status(db, parent_id) == "merging"


def test_processing_scan_does_not_hgetall_large_hash_without_scan_api(tmp_path, monkeypatch):
    redis = FakeRedisQueue()
    redis.client = LargeHashNoScanClient()
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, batch_size=500, redis=redis)
    create_processing_task(db, "stale")

    stats = scheduler._recover_orphans_safely()

    assert stats["applied_sqlite_reset"] == 1
    assert stats["planned_ghosts_purged"] == 0


def create_parent_with_children(db, name, *, status, child_statuses, minutes_old=70, merge_owner=None, merge_attempts=0):
    parent_id = db.create_task(f"{name}.pdf", f"/tmp/{name}.pdf")["task_id"]
    child_ids = [
        db.create_task(f"{name}-{index}.pdf", f"/tmp/{name}-{index}.pdf")["task_id"]
        for index, _child_status in enumerate(child_statuses)
    ]
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET status = ?, worker_id = 'worker-a', is_parent = 1,
                child_count = ?, child_completed = ?,
                merge_owner = ?, merge_claimed_at = CASE WHEN ? IS NULL THEN NULL ELSE datetime('now', '-' || ? || ' minutes') END,
                merge_attempts = ?, started_at = datetime('now', '-' || ? || ' minutes')
            WHERE task_id = ?
            """,
            (
                status,
                len(child_statuses),
                sum(1 for child_status in child_statuses if child_status == "completed"),
                merge_owner,
                merge_owner,
                minutes_old,
                merge_attempts,
                minutes_old,
                parent_id,
            ),
        )
        for child_id, child_status in zip(child_ids, child_statuses):
            cursor.execute(
                """
                UPDATE tasks
                SET parent_task_id = ?, status = ?, completed_at = CASE WHEN ? = 'completed' THEN datetime('now') ELSE NULL END
                WHERE task_id = ?
                """,
                (parent_id, child_status, child_status, child_id),
            )
    return parent_id


def test_processing_parent_transitions_to_merging_once(tmp_path, monkeypatch):
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, redis=None)
    parent_id = create_parent_with_children(
        db, "ready-processing-parent", status="processing", child_statuses=["completed", "completed"]
    )

    first = scheduler._recover_orphans_safely()
    second = scheduler._recover_orphans_safely()

    assert first["planned_parent_merging"] == 1
    assert first["planned_parent_already_merging"] == 0
    assert first["applied_parent_merging"] == 1
    assert task_status(db, parent_id) == "merging"
    assert second["planned_parent_merging"] == 0
    assert second["planned_parent_already_merging"] == 1
    assert second["applied_parent_merging"] == 0


def test_parent_merge_backlog_stats_are_read_only_and_classified(tmp_path, monkeypatch):
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, redis=None, batch_size=50)
    create_parent_with_children(db, "ready", status="processing", child_statuses=["completed", "completed"])
    create_parent_with_children(db, "recoverable", status="merging", child_statuses=["completed"], merge_owner=None)
    create_parent_with_children(
        db, "leased", status="merging", child_statuses=["completed"], merge_owner="worker-a", minutes_old=5
    )
    create_parent_with_children(
        db, "stale-leased", status="merging", child_statuses=["completed"], merge_owner="worker-b", minutes_old=70
    )
    create_parent_with_children(
        db, "blocked", status="merging", child_statuses=["completed"], merge_owner=None, merge_attempts=3
    )

    before = db.get_queue_stats()
    stats = scheduler._get_parent_merge_backlog_stats(stale_seconds=30 * 60, max_attempts=3)
    after = db.get_queue_stats()

    assert before == after
    assert stats["total"] == 5
    assert stats["ready"] == 1
    assert stats["recoverable"] == 2
    assert stats["active_leased"] == 1
    assert stats["stale_leased"] == 1
    assert stats["blocked"] == 1
    assert stats["oldest_merging_age_seconds"] >= 60 * 60


def test_parent_merge_backlog_total_is_not_batch_limited(tmp_path, monkeypatch):
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, redis=None, batch_size=2)
    for index in range(5):
        create_parent_with_children(db, f"ready-{index}", status="processing", child_statuses=["completed"])

    stats = scheduler._get_parent_merge_backlog_stats(stale_seconds=30 * 60, max_attempts=3)

    assert stats["total"] == 5
    assert stats["ready"] == 5


def test_parent_merge_age_uses_fresh_claim_before_old_started_at(tmp_path, monkeypatch):
    scheduler, db = make_scheduler(tmp_path, monkeypatch, apply=True, redis=None, batch_size=50)
    create_parent_with_children(
        db, "fresh-lease-old-parent", status="merging", child_statuses=["completed"],
        merge_owner="worker-a", minutes_old=5
    )
    with db.get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE tasks
            SET started_at = datetime('now', '-70 minutes'),
                created_at = datetime('now', '-80 minutes')
            WHERE file_name = 'fresh-lease-old-parent.pdf'
            """
        )

    stats = scheduler._get_parent_merge_backlog_stats(stale_seconds=30 * 60, max_attempts=3)

    assert stats["active_leased"] == 1
    assert stats["stale_leased"] == 0
    assert 0 <= stats["oldest_merging_age_seconds"] < 10 * 60
