#!/usr/bin/env python3
"""Re-process files previously done with 'auto'(pipeline) backend using hybrid-auto-engine.

Steps:
1. Query DB for completed tasks with backend='auto'
2. Delete their output dirs, DB records, Redis cache
3. Submit each file fresh with hybrid-auto-engine + effort=high
"""
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import requests

INSTANCE_ID = os.environ.get("INSTANCE_ID", os.popen("hostname").read().strip())
DATA_DIR = Path(f"/share/wangjiong/databases/mineru_database/{INSTANCE_ID}")
DB_PATH = DATA_DIR / "mineru_tianshu.db"
OUTPUTS_DIR = DATA_DIR / "mineru_outputs"
SOURCE_DIR = Path("/share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256")
PARSED_OUT = Path("/share/wangjiong/databases/escorpus-assets/pdfs_parsed")

API_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin123"


def login():
    r = requests.post(
        f"{API_URL}/api/v1/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def get_pipeline_tasks():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT task_id, file_name, result_path FROM tasks WHERE backend='auto' AND status='completed'"
    ).fetchall()
    conn.close()
    return rows


def delete_task_artifacts(task_id, file_name, result_path):
    """Delete output dir, DB record, and pdfs_parsed marker."""
    # Delete output directory
    if result_path and Path(result_path).exists():
        shutil.rmtree(result_path, ignore_errors=True)

    # Delete DB record
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
    conn.commit()
    conn.close()

    # Delete pdfs_parsed marker if exists
    sha = file_name.replace(".pdf", "")
    parsed_marker = PARSED_OUT / sha
    if parsed_marker.exists():
        shutil.rmtree(parsed_marker, ignore_errors=True)


def find_source_pdf(sha):
    """Find the source PDF in by_sha256 directory."""
    src = SOURCE_DIR / sha[:2] / f"{sha}.pdf"
    if src.exists():
        return str(src)
    return None


def submit_pdf(token, pdf_path):
    """Submit a single PDF with hybrid-auto-engine."""
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{API_URL}/api/v1/tasks/submit",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (Path(pdf_path).name, f, "application/pdf")},
            data={"backend": "hybrid-auto-engine", "effort": "high"},
            timeout=30,
        )
    return resp.json()


def main():
    dry_run = "--dry-run" in sys.argv

    print("=== Step 1: Query pipeline tasks ===")
    tasks = get_pipeline_tasks()
    print(f"Found {len(tasks)} pipeline tasks to reprocess")

    if dry_run:
        for t in tasks[:5]:
            print(f"  {t['file_name']} -> {t['result_path']}")
        print(f"  ... ({len(tasks)} total)")
        return

    # Step 2: Delete artifacts
    print("\n=== Step 2: Delete old pipeline artifacts ===")
    sha_list = []
    deleted_count = 0
    for t in tasks:
        delete_task_artifacts(t["task_id"], t["file_name"], t["result_path"])
        sha_list.append(t["file_name"].replace(".pdf", ""))
        deleted_count += 1
        if deleted_count % 200 == 0:
            print(f"  Deleted {deleted_count}/{len(tasks)}...")
    print(f"  Deleted {deleted_count} output dirs + DB records")

    # Step 3: Submit with hybrid-auto-engine
    print("\n=== Step 3: Submit with hybrid-auto-engine ===")
    token = login()
    submitted = 0
    failed = 0
    skipped = 0

    for i, sha in enumerate(sha_list):
        pdf_path = find_source_pdf(sha)
        if not pdf_path:
            print(f"  SKIP (source not found): {sha}")
            skipped += 1
            continue

        try:
            result = submit_pdf(token, pdf_path)
            if "task_id" in result:
                submitted += 1
            else:
                print(f"  ERROR: {result}")
                failed += 1
        except Exception as e:
            print(f"  FAILED: {sha} - {e}")
            failed += 1
            # Re-login on auth failure
            if "401" in str(e) or "Unauthorized" in str(e):
                token = login()

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(sha_list)} (submitted={submitted}, failed={failed})")
            time.sleep(0.5)  # Small breather

    print(f"\n=== Done ===")
    print(f"  Submitted: {submitted}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (no source): {skipped}")


if __name__ == "__main__":
    main()
