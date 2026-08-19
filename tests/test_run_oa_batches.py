import importlib.util
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
        max_batch_errors=10,
        write_progress_reports=False,
        priority=0,
        high_watermark=5000,
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
