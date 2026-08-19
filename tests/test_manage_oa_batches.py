import importlib.util
import json
import sqlite3
from pathlib import Path

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
