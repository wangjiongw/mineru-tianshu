import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "manage_oa_batches.py"
SPEC = importlib.util.spec_from_file_location("manage_oa_batches", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def make_source(root: Path, sha: str) -> Path:
    path = root / sha[:2] / f"{sha}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")
    return path


def make_result(root: Path, sha: str, *, valid: bool = True) -> Path:
    path = root / sha[:2] / sha
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.md").write_text("content" if valid else "", encoding="utf-8")
    (path / "mineru_model.json").write_text("[]", encoding="utf-8")
    return path


def make_task_db(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            file_name TEXT,
            file_path TEXT,
            status TEXT,
            backend TEXT,
            lang TEXT,
            method TEXT,
            file_hash TEXT,
            result_path TEXT,
            error_message TEXT,
            parent_task_id TEXT
        );
        CREATE INDEX idx_status ON tasks(status);
        """
    )
    conn.executemany(
        """
        INSERT INTO tasks(
            task_id,file_name,file_path,status,backend,lang,method,file_hash,
            result_path,error_message,parent_task_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def task_row(sha: str, status: str, error: str | None = None) -> tuple:
    return (
        f"task-{sha[-8:]}",
        f"{sha}.pdf",
        f"/source/{sha}.pdf",
        status,
        "hybrid-auto-engine",
        "auto",
        "auto",
        sha,
        f"/output/{sha}",
        error,
        None,
    )


def test_strict_result_complete_rejects_empty_and_invalid_json(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    (result / "result.md").write_text("", encoding="utf-8")
    (result / "mineru_model.json").write_text("[]", encoding="utf-8")
    assert module.strict_result_complete(result)[0] is False

    (result / "result.md").write_text("ok", encoding="utf-8")
    (result / "mineru_model.json").write_text("not-json", encoding="utf-8")
    assert module.strict_result_complete(result)[0] is False

    (result / "mineru_model.json").write_text("{}", encoding="utf-8")
    assert module.strict_result_complete(result) == (True, None)


def test_validate_source_path_enforces_sharded_sha_layout(tmp_path):
    root = tmp_path / "source"
    sha = "a" * 64
    path = make_source(root, sha)
    assert module.validate_source_path(root, sha, path) == path.resolve()

    wrong = root / "bb" / f"{sha}.pdf"
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(b"pdf")
    with pytest.raises(ValueError, match="trusted SHA layout"):
        module.validate_source_path(root, sha, wrong)


def test_refresh_classifies_global_states_and_preserves_total(tmp_path):
    source = tmp_path / "source"
    parsed = tmp_path / "parsed"
    inventory = parsed / "_inventory" / "oa.sqlite3"
    task_db = tmp_path / "tasks.sqlite3"
    shas = [f"{index:064x}" for index in range(1, 8)]
    for sha in shas:
        make_source(source, sha)
    make_result(parsed, shas[0])
    make_result(parsed, shas[1])
    rows = [
        task_row(shas[0], "completed"),
        task_row(shas[2], "completed"),
        task_row(shas[3], "pending"),
        task_row(shas[4], "failed", "database is locked"),
        task_row(shas[5], "failed", "Pdfium: invalid PDF"),
    ]
    make_task_db(task_db, rows)

    conn = module.connect_inventory(inventory)
    try:
        report = module.refresh_inventory(
            conn,
            inventory,
            source,
            parsed,
            task_db,
            parsed / "_submit_progress.json",
        )
    finally:
        conn.close()

    assert report["total_source"] == 7
    assert report["states"] == {
        "complete_valid": 1,
        "completed_db_unsynced": 1,
        "failed_permanent": 1,
        "failed_retryable": 1,
        "legacy_result_only": 1,
        "queued": 1,
        "unsubmitted": 1,
    }
    assert sum(report["states"].values()) == report["total_source"]


def test_prepare_batches_is_deterministic_and_skips_noneligible(tmp_path):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    conn = module.connect_inventory(inventory)
    now = module.utc_now()
    rows = []
    for index in range(12):
        sha = f"{index:064x}"
        state = "unsubmitted" if index < 11 else "complete_valid"
        rows.append((sha, f"/source/{sha}.pdf", 1, 1, 1, state, now, now))
    conn.executemany(
        """
        INSERT INTO documents(
            sha256,source_path,source_size,source_mtime_ns,seen_generation,state,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    try:
        report = module.prepare_batches(
            conn,
            inventory,
            tmp_path / "legacy.json",
            batch_size=5,
            replace_prepared=False,
        )
        batches = conn.execute(
            "SELECT batch_key,item_count FROM batches ORDER BY ordinal"
        ).fetchall()
        first = [
            row[0]
            for row in conn.execute(
                "SELECT sha256 FROM batch_items WHERE batch_id=1 ORDER BY position"
            )
        ]
    finally:
        conn.close()

    assert [row["item_count"] for row in batches] == [5, 5, 1]
    assert first == [f"{index:064x}" for index in range(5)]
    assert report["batch_size"] == 5
    assert report["batch_count"] == 3
    manifest = inventory.parent / "batches" / f"{batches[0]['batch_key']}.jsonl"
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 5
    assert json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])["position"] == 1



def insert_document(conn, sha: str, source_path: Path, state: str, *, task_id: str | None = None, db_status: str | None = None, db_error: str | None = None) -> None:
    now = module.utc_now()
    conn.execute(
        """
        INSERT INTO documents(
            sha256,source_path,source_size,source_mtime_ns,seen_generation,state,
            db_task_id,db_status,db_error,created_at,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (sha, str(source_path), 1, 1, 1, state, task_id, db_status, db_error, now, now),
    )


def insert_batch(conn, batch_key: str, shas: list[str], *, item_status: str = "submitted") -> int:
    now = module.utc_now()
    conn.execute(
        """
        INSERT INTO batches(batch_key,generation,ordinal,status,item_count,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (batch_key, 1, int(batch_key[-5:]), "running", len(shas), now, now),
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
    return batch_id


def test_batch_progress_treats_db_completed_as_terminal_with_permanent_errors(tmp_path):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    conn = module.connect_inventory(inventory)
    shas = [f"{index:064x}" for index in range(1, 4)]
    source = tmp_path / "source"
    for sha in shas:
        insert_document(conn, sha, make_source(source, sha), "completed_db_unsynced", db_status="completed")
    conn.execute("UPDATE documents SET state='failed_permanent',db_status='failed',db_error='Pdfium invalid PDF' WHERE sha256=?", (shas[-1],))
    batch_id = insert_batch(conn, "oa-g001-b00001", shas, item_status="submitted")

    module.refresh_batch_progress(conn)
    batch = conn.execute("SELECT status,completed_count,failed_count FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    statuses = {row["item_status"]: row["count"] for row in conn.execute("SELECT item_status,COUNT(*) AS count FROM batch_items WHERE batch_id=? GROUP BY item_status", (batch_id,))}
    conn.close()

    assert dict(batch) == {"status": "completed_with_errors", "completed_count": 2, "failed_count": 1}
    assert statuses == {"completed": 2, "error": 1}


def test_submit_batch_retries_retryable_errors_but_skips_permanent_errors(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    source = tmp_path / "source"
    legacy = tmp_path / "legacy.json"
    conn = module.connect_inventory(inventory)
    retryable = "a" * 64
    permanent = "b" * 64
    insert_document(conn, retryable, make_source(source, retryable), "failed_retryable")
    insert_document(conn, permanent, make_source(source, permanent), "failed_permanent")
    insert_batch(conn, "oa-g001-b00001", [retryable, permanent], item_status="error")

    created = []

    class FakeTaskDB:
        def create_task(self, **kwargs):
            created.append(kwargs["file_hash"])
            return {"task_id": "new-task", "status": "pending"}

    class FakeRedis:
        def get(self, _key):
            return b"paused"

        def zcard(self, _key):
            return 0

        def zscore(self, _key, _task_id):
            return 1.0

        def hget(self, _key, _task_id):
            return None

    fake_queue = SimpleNamespace(
        client=FakeRedis(),
        config=SimpleNamespace(claim_pause_key="pause", queue_key="queue", processing_key="processing"),
    )
    monkeypatch.setattr(module, "load_task_db_api", lambda _args: (FakeTaskDB(), fake_queue))
    args = SimpleNamespace(
        batch_key="oa-g001-b00001",
        allow_live_claims=False,
        limit=None,
        high_watermark=5000,
        priority=0,
    )

    module.submit_batch(conn, inventory, legacy, source, args)
    permanent_row = conn.execute("SELECT item_status,task_id FROM batch_items WHERE sha256=?", (permanent,)).fetchone()
    conn.close()

    assert created == [retryable]
    assert permanent_row["item_status"] == "error"
    assert permanent_row["task_id"] == f"task-{permanent[-8:]}"


def test_refresh_batch_from_task_db_only_updates_selected_batch(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    legacy = tmp_path / "legacy.json"
    task_db = tmp_path / "tasks.sqlite3"
    source = tmp_path / "source"
    conn = module.connect_inventory(inventory)
    shas = ["c" * 64, "d" * 64]
    for sha in shas:
        insert_document(conn, sha, make_source(source, sha), "queued", task_id=f"task-{sha[-8:]}", db_status="pending")
    insert_batch(conn, "oa-g001-b00001", [shas[0]], item_status="submitted")
    insert_batch(conn, "oa-g001-b00002", [shas[1]], item_status="submitted")
    make_task_db(
        task_db,
        [
            task_row(shas[0], "completed"),
            task_row(shas[1], "failed", "Pdfium invalid PDF"),
        ],
    )
    monkeypatch.setattr(module, "load_task_state", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full refresh used")))

    report = module.refresh_batch_from_task_db(conn, inventory, legacy, task_db, "oa-g001-b00001")
    first = conn.execute("SELECT state FROM documents WHERE sha256=?", (shas[0],)).fetchone()["state"]
    second = conn.execute("SELECT state FROM documents WHERE sha256=?", (shas[1],)).fetchone()["state"]
    conn.close()

    assert report["batch_refresh"]["updated"] == 1
    assert first == "completed_db_unsynced"
    assert second == "queued"



def test_submit_batch_keeps_item_error_when_create_task_reports_enqueue_failure(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    source = tmp_path / "source"
    legacy = tmp_path / "legacy.json"
    conn = module.connect_inventory(inventory)
    sha = "e" * 64
    insert_document(conn, sha, make_source(source, sha), "failed_retryable")
    insert_batch(conn, "oa-g001-b00001", [sha], item_status="error")

    class FakeTaskDB:
        def create_task(self, **_kwargs):
            return {
                "task_id": "failed-task",
                "status": "failed",
                "retry_info": {"enqueue_failed_task_ids": ["failed-task"]},
            }

    class FakeRedis:
        def get(self, _key):
            return b"paused"

        def zcard(self, _key):
            return 0

    fake_queue = SimpleNamespace(
        client=FakeRedis(),
        config=SimpleNamespace(claim_pause_key="pause", queue_key="queue", processing_key="processing"),
    )
    monkeypatch.setattr(module, "load_task_db_api", lambda _args: (FakeTaskDB(), fake_queue))
    args = SimpleNamespace(
        batch_key="oa-g001-b00001",
        allow_live_claims=False,
        limit=None,
        high_watermark=5000,
        priority=0,
    )

    module.submit_batch(conn, inventory, legacy, source, args)
    item = conn.execute("SELECT item_status,task_id,submit_error FROM batch_items WHERE sha256=?", (sha,)).fetchone()
    doc = conn.execute("SELECT state,db_status,last_submit_error FROM documents WHERE sha256=?", (sha,)).fetchone()
    conn.close()

    assert item["item_status"] == "error"
    assert item["task_id"] == f"task-{sha[-8:]}"
    assert "enqueue failed" in item["submit_error"]
    assert doc["state"] == "failed_retryable"
    assert doc["db_status"] is None
    assert "enqueue failed" in doc["last_submit_error"]



def test_submit_batch_accepts_split_processing_when_requeued_children_are_live(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    source = tmp_path / "source"
    legacy = tmp_path / "legacy.json"
    conn = module.connect_inventory(inventory)
    sha = "f" * 64
    insert_document(conn, sha, make_source(source, sha), "failed_retryable")
    insert_batch(conn, "oa-g001-b00001", [sha], item_status="error")

    class FakeTaskDB:
        def create_task(self, **_kwargs):
            return {
                "task_id": "parent-task",
                "status": "processing",
                "retry_info": {
                    "mode": "split_children_requeued",
                    "requeued_task_ids": ["child-1", "child-2"],
                    "enqueue_failed_task_ids": [],
                },
            }

    class FakeRedis:
        def get(self, _key):
            return b"paused"

        def zcard(self, _key):
            return 0

        def zscore(self, _key, task_id):
            return 1.0 if task_id in {"child-1", "child-2"} else None

        def hget(self, _key, _task_id):
            return None

    fake_queue = SimpleNamespace(
        client=FakeRedis(),
        config=SimpleNamespace(claim_pause_key="pause", queue_key="queue", processing_key="processing"),
    )
    monkeypatch.setattr(module, "load_task_db_api", lambda _args: (FakeTaskDB(), fake_queue))
    args = SimpleNamespace(batch_key="oa-g001-b00001", allow_live_claims=False, limit=None, high_watermark=5000, priority=0)

    module.submit_batch(conn, inventory, legacy, source, args)
    item = conn.execute("SELECT item_status,task_id,submit_error FROM batch_items WHERE sha256=?", (sha,)).fetchone()
    doc = conn.execute("SELECT state,db_status,last_submit_error FROM documents WHERE sha256=?", (sha,)).fetchone()
    conn.close()

    assert item["item_status"] == "submitted"
    assert item["task_id"] == "parent-task"
    assert item["submit_error"] is None
    assert doc["state"] == "queued"
    assert doc["db_status"] == "processing"
    assert doc["last_submit_error"] is None


def test_submit_batch_rejects_split_processing_when_requeued_child_is_missing(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    source = tmp_path / "source"
    legacy = tmp_path / "legacy.json"
    conn = module.connect_inventory(inventory)
    sha = "1" * 64
    insert_document(conn, sha, make_source(source, sha), "failed_retryable")
    insert_batch(conn, "oa-g001-b00001", [sha], item_status="error")

    class FakeTaskDB:
        def create_task(self, **_kwargs):
            return {
                "task_id": "parent-task",
                "status": "processing",
                "retry_info": {
                    "mode": "split_ready_to_merge",
                    "requeued_task_ids": ["child-1", "child-missing"],
                    "enqueue_failed_task_ids": [],
                },
            }

    class FakeRedis:
        def get(self, _key):
            return b"paused"

        def zcard(self, _key):
            return 0

        def zscore(self, _key, task_id):
            return 1.0 if task_id == "child-1" else None

        def hget(self, _key, _task_id):
            return None

    fake_queue = SimpleNamespace(
        client=FakeRedis(),
        config=SimpleNamespace(claim_pause_key="pause", queue_key="queue", processing_key="processing"),
    )
    monkeypatch.setattr(module, "load_task_db_api", lambda _args: (FakeTaskDB(), fake_queue))
    args = SimpleNamespace(batch_key="oa-g001-b00001", allow_live_claims=False, limit=None, high_watermark=5000, priority=0)

    module.submit_batch(conn, inventory, legacy, source, args)
    item = conn.execute("SELECT item_status,task_id,submit_error FROM batch_items WHERE sha256=?", (sha,)).fetchone()
    doc = conn.execute("SELECT state,db_status,last_submit_error FROM documents WHERE sha256=?", (sha,)).fetchone()
    conn.close()

    assert item["item_status"] == "error"
    assert item["task_id"] == f"task-{sha[-8:]}"
    assert "child-missing" in item["submit_error"]
    assert doc["state"] == "failed_retryable"
    assert doc["db_status"] is None
    assert "child-missing" in doc["last_submit_error"]



@pytest.mark.parametrize("mode", ["split_ready_to_merge", "split_awaiting_children"])
def test_submit_batch_accepts_split_processing_modes_with_empty_requeued_children(tmp_path, monkeypatch, mode):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    source = tmp_path / "source"
    legacy = tmp_path / "legacy.json"
    conn = module.connect_inventory(inventory)
    sha = "2" * 64 if mode == "split_ready_to_merge" else "3" * 64
    insert_document(conn, sha, make_source(source, sha), "failed_retryable")
    insert_batch(conn, "oa-g001-b00001", [sha], item_status="error")

    class FakeTaskDB:
        def create_task(self, **_kwargs):
            return {
                "task_id": "parent-task",
                "status": "processing",
                "retry_info": {
                    "mode": mode,
                    "requeued_task_ids": [],
                    "enqueue_failed_task_ids": [],
                },
            }

    class FakeRedis:
        def get(self, _key):
            return b"paused"

        def zcard(self, _key):
            return 0

        def zscore(self, _key, _task_id):
            return None

        def hget(self, _key, _task_id):
            return None

    fake_queue = SimpleNamespace(
        client=FakeRedis(),
        config=SimpleNamespace(claim_pause_key="pause", queue_key="queue", processing_key="processing"),
    )
    monkeypatch.setattr(module, "load_task_db_api", lambda _args: (FakeTaskDB(), fake_queue))
    args = SimpleNamespace(batch_key="oa-g001-b00001", allow_live_claims=False, limit=None, high_watermark=5000, priority=0)

    module.submit_batch(conn, inventory, legacy, source, args)
    item = conn.execute("SELECT item_status,task_id,submit_error FROM batch_items WHERE sha256=?", (sha,)).fetchone()
    doc = conn.execute("SELECT state,db_status,last_submit_error FROM documents WHERE sha256=?", (sha,)).fetchone()
    conn.close()

    assert item["item_status"] == "submitted"
    assert item["task_id"] == "parent-task"
    assert item["submit_error"] is None
    assert doc["state"] == "queued"
    assert doc["db_status"] == "processing"
    assert doc["last_submit_error"] is None


def test_deferred_items_are_tracked_and_reconcile_when_results_arrive(tmp_path):
    inventory = tmp_path / "inventory" / "oa.sqlite3"
    conn = module.connect_inventory(inventory)
    completed_sha = "a" * 64
    deferred_sha = "b" * 64
    source = tmp_path / "source"
    insert_document(
        conn,
        completed_sha,
        make_source(source, completed_sha),
        "completed_db_unsynced",
        task_id=f"task-{completed_sha[-8:]}",
        db_status="completed",
    )
    insert_document(
        conn,
        deferred_sha,
        make_source(source, deferred_sha),
        "active",
        task_id=f"task-{deferred_sha[-8:]}",
        db_status="processing",
    )
    batch_id = insert_batch(
        conn,
        "oa-g001-b00001",
        [completed_sha, deferred_sha],
        item_status="submitted",
    )

    result = module.defer_batch_items(
        conn,
        "oa-g001-b00001",
        [f"task-{deferred_sha[-8:]}"],
        reason="stale test tail",
    )
    batch = conn.execute(
        "SELECT status,submitted_count,completed_count,failed_count FROM batches WHERE batch_id=?",
        (batch_id,),
    ).fetchone()
    report = module.build_progress_report(conn, inventory, tmp_path / "legacy.json")

    assert result["deferred"] == 1
    assert dict(batch) == {
        "status": "completed_with_deferred",
        "submitted_count": 0,
        "completed_count": 1,
        "failed_count": 0,
    }
    assert report["schema_version"] == 2
    assert report["batches"][0]["deferred_count"] == 1

    conn.execute(
        """
        UPDATE documents
        SET state='completed_db_unsynced',db_status='completed'
        WHERE sha256=?
        """,
        (deferred_sha,),
    )
    conn.commit()
    module.refresh_one_batch_progress(conn, "oa-g001-b00001")
    resolved = conn.execute(
        """
        SELECT b.status,
               (SELECT COUNT(*) FROM batch_items bi
                WHERE bi.batch_id=b.batch_id AND bi.item_status='deferred') AS deferred_count
        FROM batches b WHERE b.batch_id=?
        """,
        (batch_id,),
    ).fetchone()
    conn.close()

    assert dict(resolved) == {"status": "completed", "deferred_count": 0}
