#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

import requests


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def iter_pdfs(root: Path, recursive: bool) -> Iterator[Path]:
    """Yield PDF files under the given root."""
    if recursive:
        yield from root.rglob("*.pdf")
    else:
        yield from root.glob("*.pdf")


def login(base_url: str, username: str, password: str) -> str:
    """Login and return access token."""
    url = f"{base_url}/api/v1/auth/login"
    resp = requests.post(url, json={"username": username, "password": password}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def submit_task(
    base_url: str,
    token: str,
    pdf_path: Path,
    backend: str,
    lang: str,
    method: str,
) -> str:
    """Submit a single PDF task and return task_id."""
    url = f"{base_url}/api/v1/tasks/submit"
    headers = {"Authorization": f"Bearer {token}"}
    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {"backend": backend, "lang": lang, "method": method}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=600)
    resp.raise_for_status()
    result = resp.json()
    return result["task_id"]


def retry_task(base_url: str, token: str, task_id: str) -> None:
    """Retry a failed task via API."""
    url = f"{base_url}/api/v1/tasks/{task_id}/retry"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(url, headers=headers, timeout=30)
    resp.raise_for_status()


def poll_task(
    base_url: str,
    token: str,
    task_id: str,
    poll_interval: int,
    timeout_sec: int,
) -> dict:
    """Poll task until completed/failed or timeout."""
    url = f"{base_url}/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {token}"}
    start = time.time()
    last_status = None
    while True:
        resp = requests.get(url, headers=headers, params={"format": "both"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status != last_status:
            log(f"Task {task_id} status: {status}")
            last_status = status
        if status in ("completed", "failed"):
            return data
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Task {task_id} timed out after {timeout_sec} seconds")
        time.sleep(poll_interval)


def safe_mkdir(path: Path) -> None:
    """Create directory if not exists."""
    path.mkdir(parents=True, exist_ok=True)


def extract_image_urls(md_content: str) -> set[str]:
    """Extract image URLs from markdown and HTML tags."""
    urls = set()
    md_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    html_pattern = r"<img[^>]*src=[\"']([^\"']+)[\"'][^>]*>"
    for pattern in (md_pattern, html_pattern):
        for match in re.findall(pattern, md_content):
            urls.add(match)
    return urls


def normalize_url(base_url: str, url: str) -> Optional[str]:
    """Convert relative API URLs to absolute URLs if possible."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/api/") or url.startswith("/v1/"):
        return f"{base_url}{url}"
    return None


def download_images(
    base_url: str,
    token: str,
    md_content: str,
    image_dir: Path,
) -> None:
    """Download images referenced in markdown content."""
    urls = extract_image_urls(md_content)
    if not urls:
        return

    headers = {"Authorization": f"Bearer {token}"}
    safe_mkdir(image_dir)

    for url in urls:
        full_url = normalize_url(base_url, url)
        if not full_url:
            continue
        try:
            resp = requests.get(full_url, headers=headers, timeout=60)
            resp.raise_for_status()
            filename = Path(url).name
            if not filename:
                continue
            (image_dir / filename).write_bytes(resp.content)
        except Exception:
            continue


def save_result(
    out_dir: Path,
    task_result: dict,
    base_url: str,
    token: str,
    download_imgs: bool,
) -> None:
    """Save task result files into output directory."""
    safe_mkdir(out_dir)

    (out_dir / "task_meta.json").write_text(json.dumps(task_result, ensure_ascii=False, indent=2), encoding="utf-8")

    data = task_result.get("data") or {}
    md_content = data.get("content")
    if md_content:
        (out_dir / "result.md").write_text(md_content, encoding="utf-8")
        if download_imgs:
            download_images(base_url, token, md_content, out_dir / "images")

    json_content = data.get("json_content")
    if json_content is not None:
        (out_dir / "result.json").write_text(
            json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    mineru_model_content = data.get("mineru_model_content")
    if mineru_model_content is not None:
        (out_dir / "mineru_model.json").write_text(
            json.dumps(mineru_model_content, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def build_output_path(input_root: Path, output_root: Path, pdf_path: Path) -> Path:
    """Mirror input directory structure under output root."""
    rel = pdf_path.relative_to(input_root)
    return output_root / rel.parent / rel.stem


def is_output_complete(output_dir: Path, require_images: bool) -> bool:
    if not output_dir.exists():
        return False
    if not (output_dir / "result.md").exists():
        return False
    if not (output_dir / "mineru_model.json").exists():
        return False
    if require_images and not (output_dir / "images").exists():
        return False
    return True


def load_tasks(tasks_file: Path) -> Dict[str, dict]:
    if not tasks_file.exists():
        return {}
    try:
        data = json.loads(tasks_file.read_text(encoding="utf-8"))
        return data.get("tasks", {})
    except Exception:
        return {}


def save_tasks(tasks_file: Path, tasks: Dict[str, dict]) -> None:
    safe_mkdir(tasks_file.parent)
    payload = {
        "updated_at": datetime.now().isoformat(),
        "tasks": tasks,
    }
    tmp = tasks_file.with_suffix(tasks_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(tasks_file)


def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def update_summary(summary_json: Path, summary_csv: Path, tasks: Dict[str, dict]) -> None:
    entries = list(tasks.values())
    summary = {
        "updated_at": datetime.now().isoformat(),
        "total": len(entries),
        "completed": sum(1 for e in entries if e.get("status") == "completed"),
        "failed": sum(1 for e in entries if e.get("status") in ("failed", "timeout", "submit_failed")),
        "pending": sum(1 for e in entries if e.get("status") in ("pending", "processing")),
        "entries": entries,
    }
    safe_mkdir(summary_json.parent)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf_path",
                "relative_path",
                "task_id",
                "status",
                "submitted_at",
                "completed_at",
                "duration_sec",
                "retry_count",
                "output_dir",
                "error",
            ],
        )
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k) for k in writer.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch parse PDFs via MinerU Tianshu API")
    parser.add_argument("--base-url", required=True, help="API base URL, e.g. http://localhost:8000")
    parser.add_argument("--input-root", required=True, help="Root folder containing PDFs")
    parser.add_argument("--output-root", required=True, help="Root folder to store results")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan PDFs")
    parser.add_argument("--username", help="Login username")
    parser.add_argument("--password", help="Login password")
    parser.add_argument("--token", help="Bearer token (skip login if provided)")
    parser.add_argument("--backend", default="hybrid-auto-engine", help="Backend name")
    parser.add_argument("--lang", default="auto", help="Language")
    parser.add_argument("--method", default="auto", help="Method: auto/txt/ocr")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval (sec)")
    parser.add_argument("--timeout", type=int, default=3600, help="Task timeout (sec)")
    parser.add_argument("--download-images", action="store_true", help="Download images to output folder")
    parser.add_argument("--batch-size", type=int, default=100, help="Max PDFs per submit batch")
    parser.add_argument("--max-retries", type=int, default=0, help="Max retries per task (failed/timeout)")
    parser.add_argument("--tasks-file", help="Tasks state JSON file")
    parser.add_argument("--summary-json", help="Summary JSON output path")
    parser.add_argument("--summary-csv", help="Summary CSV output path")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume (ignore tasks file)")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    tasks_file = Path(args.tasks_file).resolve() if args.tasks_file else output_root / "batch_tasks.json"
    summary_json = Path(args.summary_json).resolve() if args.summary_json else output_root / "summary.json"
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else output_root / "summary.csv"

    log("Stage 1/6: validate input")
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    if args.token:
        log("Stage 2/6: using provided token")
        token = args.token
    else:
        if not args.username or not args.password:
            raise SystemExit("Provide --token or --username/--password")
        log("Stage 2/6: login")
        token = login(base_url, args.username, args.password)

    log("Stage 3/6: scan PDFs")
    pdfs = list(iter_pdfs(input_root, args.recursive))
    if not pdfs:
        raise SystemExit("No PDFs found")
    log(f"Found {len(pdfs)} PDFs")

    resume = not args.no_resume
    require_images = args.download_images

    tasks: Dict[str, dict] = {} if not resume else load_tasks(tasks_file)

    for pdf in pdfs:
        rel = str(pdf.relative_to(input_root))
        out_dir = build_output_path(input_root, output_root, pdf)
        if rel not in tasks:
            tasks[rel] = {
                "pdf_path": str(pdf),
                "relative_path": rel,
                "task_id": None,
                "status": "pending",
                "submitted_at": None,
                "completed_at": None,
                "duration_sec": None,
                "retry_count": 0,
                "output_dir": str(out_dir),
                "error": None,
                "last_update": datetime.now().isoformat(),
            }

    for rel, item in tasks.items():
        out_dir = Path(item["output_dir"])
        if is_output_complete(out_dir, require_images):
            item["status"] = "completed"
            item["completed_at"] = item.get("completed_at") or datetime.now().isoformat()
            item["duration_sec"] = item.get("duration_sec") or 0
            item["last_update"] = datetime.now().isoformat()

    save_tasks(tasks_file, tasks)

    def calc_avg_duration() -> Optional[float]:
        durations = [t.get("duration_sec") for t in tasks.values() if t.get("status") == "completed"]
        durations = [d for d in durations if isinstance(d, (int, float)) and d > 0]
        if not durations:
            return None
        return sum(durations) / len(durations)

    pending_rels = [r for r, t in tasks.items() if t.get("status") != "completed"]
    total_pending = len(pending_rels)
    log(f"Pending tasks: {total_pending}")

    for batch_start in range(0, total_pending, args.batch_size):
        batch_rels = pending_rels[batch_start: batch_start + args.batch_size]
        log(f"Stage 4/6: submit batch {batch_start // args.batch_size + 1} ({len(batch_rels)} tasks)")

        for rel in batch_rels:
            item = tasks[rel]
            pdf_path = Path(item["pdf_path"])

            if item.get("task_id") and item.get("status") in ("pending", "processing"):
                continue

            out_dir = Path(item["output_dir"])
            if is_output_complete(out_dir, require_images):
                item["status"] = "completed"
                item["completed_at"] = item.get("completed_at") or datetime.now().isoformat()
                item["duration_sec"] = item.get("duration_sec") or 0
                item["last_update"] = datetime.now().isoformat()
                continue

            try:
                task_id = submit_task(base_url, token, pdf_path, args.backend, args.lang, args.method)
                item["task_id"] = task_id
                item["status"] = "pending"
                item["submitted_at"] = datetime.now().isoformat()
                item["last_update"] = datetime.now().isoformat()
                log(f"Submitted: {rel} -> {task_id}")
            except Exception as e:
                item["status"] = "submit_failed"
                item["error"] = str(e)
                item["last_update"] = datetime.now().isoformat()
                log(f"Submit failed: {rel} ({e})")

        save_tasks(tasks_file, tasks)

        log("Stage 5/6: poll tasks")
        for rel in batch_rels:
            item = tasks[rel]
            task_id = item.get("task_id")
            if not task_id:
                continue
            if item.get("status") == "completed":
                continue

            attempts = 0
            while True:
                try:
                    start_time = time.time()
                    result = poll_task(base_url, token, task_id, args.poll_interval, args.timeout)
                    status = result.get("status")
                    item["status"] = status
                    if status == "completed":
                        save_result(Path(item["output_dir"]), result, base_url, token, args.download_images)
                        item["completed_at"] = datetime.now().isoformat()
                        item["duration_sec"] = time.time() - start_time
                        item["last_update"] = datetime.now().isoformat()
                        break
                    item["error"] = result.get("error_message")
                    item["last_update"] = datetime.now().isoformat()

                    if attempts < args.max_retries:
                        attempts += 1
                        item["retry_count"] = item.get("retry_count", 0) + 1
                        log(f"Retrying task {task_id} (attempt {attempts}/{args.max_retries})")
                        retry_task(base_url, token, task_id)
                        continue
                    break
                except TimeoutError as e:
                    item["status"] = "timeout"
                    item["error"] = str(e)
                    item["last_update"] = datetime.now().isoformat()
                    if attempts < args.max_retries:
                        attempts += 1
                        item["retry_count"] = item.get("retry_count", 0) + 1
                        log(f"Retrying task {task_id} after timeout (attempt {attempts}/{args.max_retries})")
                        retry_task(base_url, token, task_id)
                        continue
                    log(f"Timeout: {rel} ({e})")
                    break
                except Exception as e:
                    item["status"] = "failed"
                    item["error"] = str(e)
                    item["last_update"] = datetime.now().isoformat()
                    if attempts < args.max_retries:
                        attempts += 1
                        item["retry_count"] = item.get("retry_count", 0) + 1
                        log(f"Retrying task {task_id} after error (attempt {attempts}/{args.max_retries})")
                        retry_task(base_url, token, task_id)
                        continue
                    log(f"Failed: {rel} ({e})")
                    break

            avg = calc_avg_duration()
            remaining = sum(1 for t in tasks.values() if t.get("status") != "completed")
            eta = avg * remaining if avg else None
            log(
                f"Progress: completed={sum(1 for t in tasks.values() if t.get('status') == 'completed')}, "
                f"remaining={remaining}, avg={format_seconds(avg)}, ETA={format_seconds(eta)}"
            )

        save_tasks(tasks_file, tasks)
        update_summary(summary_json, summary_csv, tasks)

    log("Stage 6/6: finalize summary")
    update_summary(summary_json, summary_csv, tasks)
    log(f"Summary written: {summary_json} , {summary_csv}")


if __name__ == "__main__":
    main()
