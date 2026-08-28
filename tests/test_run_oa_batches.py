import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

RUNNER_PATH = SCRIPTS / "run_oa_batches.py"
SPEC = importlib.util.spec_from_file_location("run_oa_batches", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

import manage_oa_batches as batches


@pytest.fixture(autouse=True)
def reset_stop_flag():
    runner._STOP = False
    yield
    runner._STOP = False


def make_args(tmp_path: Path, *, apply: bool = False, once: bool = True) -> SimpleNamespace:
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    return SimpleNamespace(
        apply=apply,
        once=once,
        initial_refresh=False,
        inventory=inventory,
        source_root=tmp_path / "source",
        parsed_root=tmp_path / "parsed",
        task_db=tmp_path / "tasks.sqlite3",
        legacy_progress=tmp_path / "legacy.json",
        checkpoint=tmp_path / "inventory" / "checkpoint.json",
        lock_file=tmp_path / "inventory" / "runner.lock",
        poll_seconds=0.01,
        max_health_failures=1,
        exit_on_health_failure=False,
        health_backoff_initial_seconds=0.01,
        health_backoff_max_seconds=0.03,
        health_recovery_successes=3,
        max_batch_errors=10,
        write_progress_reports=False,
        priority=0,
        high_watermark=5000,
        defer_stale_after_seconds=3600.0,
        max_deferred_items_per_batch=20,
        max_deferred_backlog=100,
        min_terminal_fraction=0.99,
        max_item_submit_attempts=2,
        _deferred_backlog=0,
        allow_live_claims=True,
        host="localhost",
        worker_ports=[8101],
        vllm_ports=[30025],
        log_path=[tmp_path / "logs"],
        stale_parent_threshold_seconds=600.0,
        redis_host="127.0.0.1",
        redis_port=6379,
        redis_db=0,
        redis_password="redis123",
        redis_queue_key="queue",
        redis_processing_key="processing",
        redis_maintenance_key="maintenance",
        redis_pause_key="pause",
        _last_refresh_at=None,
        _last_submit_at=None,
        _consecutive_health_successes=0,
        _health_first_failure_at=None,
        _health_last_failure_at=None,
        _health_last_failure_reason=None,
        _next_health_check_at=None,
    )


def add_doc(conn: sqlite3.Connection, sha: str, *, state: str = "unsubmitted", task_id: str | None = None) -> None:
    now = batches.utc_now()
    conn.execute(
        """
        INSERT INTO documents(
            sha256,source_path,source_size,source_mtime_ns,seen_generation,state,
            db_task_id,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (sha, f"/source/{sha}.pdf", 1, 1, 1, state, task_id, now, now),
    )


def add_batch(conn: sqlite3.Connection, batch_key: str, shas: list[str], *, status: str = "prepared", item_status: str = "prepared", submitted: int = 0, completed: int = 0, failed: int = 0) -> None:
    now = batches.utc_now()
    conn.execute(
        """
        INSERT INTO batches(batch_key,generation,ordinal,status,item_count,submitted_count,completed_count,failed_count,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (batch_key, 1, int(batch_key[-5:]), status, len(shas), submitted, completed, failed, now, now),
    )
    batch_id = conn.execute("SELECT batch_id FROM batches WHERE batch_key=?", (batch_key,)).fetchone()["batch_id"]
    conn.executemany(
        """
        INSERT INTO batch_items(batch_id,position,sha256,item_status,task_id,updated_at)
        VALUES (?,?,?,?,?,?)
        """,
        [(batch_id, index, sha, item_status, f"task-{sha[-8:]}", now) for index, sha in enumerate(shas, 1)],
    )
    conn.executemany(
        "UPDATE documents SET current_batch_id=?,batch_position=? WHERE sha256=?",
        [(batch_id, index, sha) for index, sha in enumerate(shas, 1)],
    )
    conn.commit()


def setup_inventory(args: SimpleNamespace, specs: list[tuple[str, str, str, int, int, int]]) -> None:
    conn = batches.connect_inventory(args.inventory)
    try:
        for index, (sha, batch_key, status, submitted, completed, failed) in enumerate(specs, 1):
            add_doc(conn, sha, state="queued" if submitted else "unsubmitted", task_id=f"task-{sha[-8:]}")
            add_batch(
                conn,
                batch_key,
                [sha],
                status=status,
                item_status="submitted" if submitted else "prepared",
                submitted=submitted,
                completed=completed,
                failed=failed,
            )
    finally:
        conn.close()


def healthy(_args):
    names = (
        "services_healthy",
        "stale_parent_merges_zero",
        "api_500_delta_zero",
        "db_lock_terminal_failures_zero",
        "no_oom_or_preemption",
        "hbm_p99_lte_95_5_percent",
        "hbm_max_lt_98_percent",
    )
    return True, {"checks": {name: {"passed": True} for name in names}}


def unhealthy(_args):
    return False, {"checks": {"services_healthy": {"passed": False}}}


def test_dry_run_reports_next_prepared_batch_without_submit(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=False, once=True)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called")))

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = __import__("json").loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 0
    assert checkpoint["action"] == "dry_run_submit"
    assert checkpoint["current_batch"]["batch_key"] == "oa-g001-b00001"


def test_waits_for_running_batch_before_next_prepared(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    setup_inventory(
        args,
        [
            ("a" * 64, "oa-g001-b00001", "running", 1, 0, 0),
            ("b" * 64, "oa-g001-b00002", "prepared", 0, 0, 0),
        ],
    )
    monkeypatch.setattr(runner.batches, "refresh_inventory", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full refresh used")))
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("next batch submitted")))

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = __import__("json").loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 0
    assert checkpoint["action"] == "wait_batch"
    assert checkpoint["current_batch"]["batch_key"] == "oa-g001-b00001"


def test_health_failure_stops_before_submit(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called")))

    rc = runner.run_loop(args, health_check=unhealthy)
    checkpoint = __import__("json").loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 3
    assert checkpoint["stop_reason"] == "health_gate"


def test_health_retry_delay_uses_60_120_300_schedule(tmp_path):
    args = make_args(tmp_path)
    args.max_health_failures = 3
    args.health_backoff_initial_seconds = 60.0
    args.health_backoff_max_seconds = 300.0

    assert [runner.health_retry_delay(args, failures) for failures in (3, 4, 5, 20)] == [60.0, 120.0, 300.0, 300.0]


def test_health_pause_recovers_after_three_successes_and_submits_once(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=False)
    args.max_health_failures = 3
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    health_results = iter([False, False, False, True, True, True])
    events = []
    submissions = []

    def sequenced_health(_args):
        ok = next(health_results)
        return healthy(_args) if ok else unhealthy(_args)

    def fake_submit(_conn, _args, batch_key):
        submissions.append(batch_key)
        args.once = True
        return {"submitted": 1}

    monkeypatch.setattr(runner, "submit", fake_submit)
    monkeypatch.setattr(runner, "fast_refresh", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "emit", lambda event, **fields: events.append((event, fields)))

    rc = runner.run_loop(args, health_check=sequenced_health)

    assert rc == 0
    assert submissions == ["oa-g001-b00001"]
    assert "health_paused" in [event for event, _fields in events]
    assert [event for event, _fields in events].count("health_recovering") == 2
    assert "health_resumed" in [event for event, _fields in events]


def test_signal_during_health_pause_exits_and_checkpoints(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=False)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called")))

    def request_stop(_seconds):
        runner._STOP = True

    monkeypatch.setattr(runner.time, "sleep", request_stop)

    rc = runner.run_loop(args, health_check=unhealthy)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 130
    assert checkpoint["action"] == "stopped"
    assert checkpoint["stop_reason"] == "signal"
    assert checkpoint["health_last_failure_reason"] == "services_healthy"


def test_exit_on_health_failure_keeps_one_shot_job_behavior(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=False)
    args.exit_on_health_failure = True
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called")))

    rc = runner.run_loop(args, health_check=unhealthy)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 3
    assert checkpoint["stop_reason"] == "health_gate"


def test_submit_exception_remains_a_hard_failure(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=False)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])

    def fail_submit(*_args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runner, "submit", fail_submit)

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 4
    assert checkpoint["action"] == "stopped"
    assert checkpoint["stop_reason"].startswith("submit_failed: OperationalError")


def test_lock_prevents_duplicate_runner(tmp_path):
    args = make_args(tmp_path)
    args.lock_file.parent.mkdir(parents=True)
    with args.lock_file.open("a+", encoding="utf-8") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another OA batch runner"):
            with runner.runner_lock(args.lock_file):
                pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_dry_run_does_not_modify_source_pdf(tmp_path):
    args = make_args(tmp_path, apply=False, once=True)
    source = tmp_path / "source" / "aa" / f"{'a' * 64}.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])

    rc = runner.run_loop(args, health_check=healthy)

    assert rc == 0
    assert source.read_bytes() == b"original"


def test_dry_run_active_batch_does_not_refresh_or_write_inventory(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=False, once=True)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "running", 1, 0, 0)])
    monkeypatch.setattr(runner, "fast_refresh", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run refreshed")))
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called")))

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = __import__("json").loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 0
    assert checkpoint["action"] == "wait_batch"
    assert checkpoint["current_batch"]["batch_key"] == "oa-g001-b00001"


def test_health_gate_uses_sample_safe_checks_not_overall_passed(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    required = healthy(args)[1]["checks"]
    report = {"evaluation": {"passed": False, "checks": required}, "fleet": {"hbm_p99_percent": 60.0}}
    monkeypatch.setattr(runner.efficiency, "run_evaluation", lambda _args: report)

    ok, details = runner.health_gate(args)

    assert ok is True
    assert details["checks"]["hbm_max_lt_98_percent"]["passed"] is True



def test_dry_run_opens_inventory_read_only_without_schema_init(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=False, once=True)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    monkeypatch.setattr(runner.batches, "connect_inventory", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("write connection used")))
    real_connect = runner.sqlite3.connect
    calls = []

    def spy_connect(database, *args_, **kwargs):
        calls.append((database, kwargs.copy()))
        return real_connect(database, *args_, **kwargs)

    monkeypatch.setattr(runner.sqlite3, "connect", spy_connect)

    rc = runner.run_loop(args, health_check=healthy)

    assert rc == 0
    assert any(str(database).startswith(f"file:{args.inventory}") and "mode=ro" in str(database) and kwargs.get("uri") is True for database, kwargs in calls)


def test_initial_refresh_runs_under_runner_lock(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    args.initial_refresh = True
    conn = batches.connect_inventory(args.inventory)
    conn.close()
    lock_checked = []

    def locked_full_refresh(_args):
        import fcntl

        with args.lock_file.open("a+", encoding="utf-8") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_checked.append(True)
        return {}

    monkeypatch.setattr(runner, "full_refresh", locked_full_refresh)

    rc = runner.run_loop(args, health_check=healthy)

    assert rc == 0
    assert lock_checked == [True]


def test_stop_after_health_gate_prevents_submit(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "prepared", 0, 0, 0)])
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called after stop")))

    def stop_after_health(_args):
        runner._STOP = True
        return True, healthy(_args)[1]

    rc = runner.run_loop(args, health_check=stop_after_health)
    checkpoint = __import__("json").loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 130
    assert checkpoint["stop_reason"] == "signal"


def test_stop_after_fast_refresh_prevents_retry_submit(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    setup_inventory(args, [("a" * 64, "oa-g001-b00001", "partial", 0, 0, 1)])
    monkeypatch.setattr(runner, "submit", lambda *_args: (_ for _ in ()).throw(AssertionError("submit called after stop")))

    def stop_after_refresh(_conn, _args, _batch_key):
        runner._STOP = True
        return {}

    monkeypatch.setattr(runner, "fast_refresh", stop_after_refresh)

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = __import__("json").loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 130
    assert checkpoint["current_batch"]["batch_key"] == "oa-g001-b00001"



def test_submit_bridges_task_db_and_redis_fields_to_manage_submit(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    seen = {}

    def fake_submit_batch(conn, inventory, legacy, source_root, submit_args):
        seen["conn"] = conn
        seen["inventory"] = inventory
        seen["legacy"] = legacy
        seen["source_root"] = source_root
        required = (
            "batch_key",
            "limit",
            "priority",
            "high_watermark",
            "allow_live_claims",
            "task_db",
            "redis_host",
            "redis_port",
            "redis_db",
            "redis_password",
            "redis_queue_key",
            "redis_processing_key",
            "redis_maintenance_key",
            "redis_pause_key",
        )
        seen["missing"] = [name for name in required if not hasattr(submit_args, name)]
        seen["task_db"] = submit_args.task_db
        seen["redis_queue_key"] = submit_args.redis_queue_key
        return {"ok": True}

    monkeypatch.setattr(runner.batches, "submit_batch", fake_submit_batch)
    conn = object()

    report = runner.submit(conn, args, "oa-g001-b00001")

    assert report == {"ok": True}
    assert seen["conn"] is conn
    assert seen["inventory"] == args.inventory
    assert seen["legacy"] == args.legacy_progress
    assert seen["source_root"] == args.source_root
    assert seen["missing"] == []
    assert seen["task_db"] == args.task_db
    assert seen["redis_queue_key"] == args.redis_queue_key
    assert args._last_submit_at is not None


def test_stale_tail_is_deferred_and_batch_becomes_terminal(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    args.min_terminal_fraction = 0.5
    completed_sha = "a" * 64
    stale_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, completed_sha, state="complete_valid", task_id=f"task-{completed_sha[-8:]}")
    add_doc(conn, stale_sha, state="active", task_id=f"task-{stale_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [completed_sha, stale_sha],
        status="partial",
        item_status="submitted",
        submitted=1,
        completed=1,
    )
    conn.execute(
        "UPDATE batch_items SET item_status='completed' WHERE sha256=?",
        (completed_sha,),
    )
    conn.commit()

    task_conn = sqlite3.connect(args.task_db)
    task_conn.executescript(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT,
            created_at TIMESTAMP,
            started_at TIMESTAMP
        );
        """
    )
    task_conn.execute(
        """
        INSERT INTO tasks(task_id,status,created_at,started_at)
        VALUES (?,'processing',datetime('now','-2 hours'),datetime('now','-2 hours'))
        """,
        (f"task-{stale_sha[-8:]}",),
    )
    task_conn.commit()
    task_conn.close()

    batch = runner.get_batch(conn, "oa-g001-b00001")
    report = runner.defer_stale_batch_tail(conn, args, batch)
    refreshed = runner.get_batch(conn, "oa-g001-b00001")
    event = conn.execute(
        "SELECT event_type,detail_json FROM events WHERE event_type='batch_tail_deferred'"
    ).fetchone()
    conn.close()

    assert report["deferred"] == 1
    assert report["deferred_backlog"] == 1
    assert refreshed["status"] == "completed_with_deferred"
    assert refreshed["submitted_count"] == 0
    assert refreshed["deferred_count"] == 1
    assert event["event_type"] == "batch_tail_deferred"
    assert json.loads(event["detail_json"])["task_ids"] == [f"task-{stale_sha[-8:]}"]


def test_recovered_tail_uses_original_submit_time_not_reset_start_time(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    args.min_terminal_fraction = 0.5
    completed_sha = "a" * 64
    recovered_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, completed_sha, state="complete_valid", task_id=f"task-{completed_sha[-8:]}")
    add_doc(conn, recovered_sha, state="active", task_id=f"task-{recovered_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [completed_sha, recovered_sha],
        status="partial",
        item_status="submitted",
        submitted=1,
        completed=1,
    )
    conn.execute(
        "UPDATE batch_items SET item_status='completed' WHERE sha256=?",
        (completed_sha,),
    )
    conn.execute(
        "UPDATE documents SET last_submit_at=datetime('now','-2 hours') WHERE sha256=?",
        (recovered_sha,),
    )
    conn.commit()

    task_conn = sqlite3.connect(args.task_db)
    task_conn.executescript(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT,
            created_at TIMESTAMP,
            started_at TIMESTAMP
        );
        """
    )
    task_conn.execute(
        """
        INSERT INTO tasks(task_id,status,created_at,started_at)
        VALUES (?,'processing',datetime('now'),datetime('now'))
        """,
        (f"task-{recovered_sha[-8:]}",),
    )
    task_conn.commit()
    task_conn.close()

    batch = runner.get_batch(conn, "oa-g001-b00001")
    report = runner.defer_stale_batch_tail(conn, args, batch)
    refreshed = runner.get_batch(conn, "oa-g001-b00001")
    conn.close()

    assert report["deferred"] == 1
    assert report["task_ids"] == [f"task-{recovered_sha[-8:]}"]
    assert refreshed["status"] == "completed_with_deferred"


def test_retry_exhausted_tail_is_deferred_after_one_retry(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    args.min_terminal_fraction = 0.5
    completed_sha = "a" * 64
    failed_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, completed_sha, state="complete_valid", task_id=f"task-{completed_sha[-8:]}")
    add_doc(conn, failed_sha, state="failed_retryable", task_id=f"task-{failed_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [completed_sha, failed_sha],
        status="partial",
        item_status="error",
        failed=1,
        completed=1,
    )
    conn.execute(
        "UPDATE batch_items SET item_status='completed' WHERE sha256=?",
        (completed_sha,),
    )
    conn.execute(
        "UPDATE documents SET submit_attempts=2 WHERE sha256=?",
        (failed_sha,),
    )
    conn.commit()

    batch = runner.get_batch(conn, "oa-g001-b00001")
    report = runner.defer_retry_exhausted_tail(conn, args, batch)
    refreshed = runner.get_batch(conn, "oa-g001-b00001")
    conn.close()

    assert report["deferred"] == 1
    assert refreshed["status"] == "completed_with_deferred"
    assert refreshed["deferred_count"] == 1


def test_stale_tail_ignores_global_deferred_backlog_cap(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    args.min_terminal_fraction = 0.0
    args.max_deferred_backlog = 1
    old_sha = "a" * 64
    stale_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, old_sha, state="active", task_id=f"task-{old_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [old_sha],
        status="completed_with_deferred",
        item_status="deferred",
    )
    add_doc(conn, stale_sha, state="active", task_id=f"task-{stale_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00002",
        [stale_sha],
        status="partial",
        item_status="submitted",
        submitted=1,
    )
    batch = runner.get_batch(conn, "oa-g001-b00002")
    conn.close()

    task_conn = sqlite3.connect(args.task_db)
    task_conn.execute(
        "CREATE TABLE tasks (task_id TEXT PRIMARY KEY,status TEXT,created_at TIMESTAMP,started_at TIMESTAMP)"
    )
    task_conn.commit()
    task_conn.close()

    check_conn = batches.connect_inventory(args.inventory)
    try:
        assert runner.stale_tail_task_ids(check_conn, args, batch) == [f"task-{stale_sha[-8:]}"]
    finally:
        check_conn.close()


def test_retry_exhausted_tail_ignores_global_deferred_backlog_cap(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    args.max_deferred_backlog = 1
    old_sha = "a" * 64
    failed_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, old_sha, state="active", task_id=f"task-{old_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [old_sha],
        status="completed_with_deferred",
        item_status="deferred",
    )
    add_doc(conn, failed_sha, state="failed_retryable", task_id=f"task-{failed_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00002",
        [failed_sha],
        status="partial",
        item_status="error",
        failed=1,
    )
    conn.execute(
        "UPDATE documents SET submit_attempts=? WHERE sha256=?",
        (args.max_item_submit_attempts, failed_sha),
    )
    conn.commit()

    batch = runner.get_batch(conn, "oa-g001-b00002")
    report = runner.defer_retry_exhausted_tail(conn, args, batch)
    refreshed = runner.get_batch(conn, "oa-g001-b00002")
    conn.close()

    assert report["deferred"] == 1
    assert report["deferred_backlog"] == 2
    assert refreshed["status"] == "completed_with_deferred"


def test_run_loop_never_resubmits_item_at_attempt_cap(tmp_path, monkeypatch):
    args = make_args(tmp_path, apply=True, once=True)
    failed_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, failed_sha, state="failed_retryable", task_id=f"task-{failed_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [failed_sha],
        status="partial",
        item_status="error",
        failed=1,
    )
    conn.execute(
        "UPDATE documents SET submit_attempts=? WHERE sha256=?",
        (args.max_item_submit_attempts, failed_sha),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(runner, "fast_refresh", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "submit",
        lambda *_args: (_ for _ in ()).throw(AssertionError("exhausted item was resubmitted")),
    )

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 0
    assert checkpoint["action"] == "batch_retry_tail_deferred"


def test_retryable_item_below_attempt_cap_is_not_deferred(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    failed_sha = "b" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, failed_sha, state="failed_retryable", task_id=f"task-{failed_sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [failed_sha],
        status="partial",
        item_status="error",
        failed=1,
    )
    conn.execute(
        "UPDATE documents SET submit_attempts=? WHERE sha256=?",
        (args.max_item_submit_attempts - 1, failed_sha),
    )
    conn.commit()
    batch = runner.get_batch(conn, "oa-g001-b00001")

    assert runner.retry_exhausted_task_ids(conn, args, batch) == []
    conn.close()


def test_primary_batches_finish_with_explicit_deferred_backfill_checkpoint(tmp_path):
    args = make_args(tmp_path, apply=True, once=True)
    sha = "a" * 64
    conn = batches.connect_inventory(args.inventory)
    add_doc(conn, sha, state="active", task_id=f"task-{sha[-8:]}")
    add_batch(
        conn,
        "oa-g001-b00001",
        [sha],
        status="completed_with_deferred",
        item_status="deferred",
    )
    conn.close()

    rc = runner.run_loop(args, health_check=healthy)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))

    assert rc == 0
    assert checkpoint["action"] == "primary_complete_with_deferred"
    assert checkpoint["stop_reason"] == "deferred_backfill_required"
    assert checkpoint["deferred_backlog"] == 1
