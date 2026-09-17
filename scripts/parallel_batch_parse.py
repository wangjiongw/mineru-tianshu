#!/usr/bin/env python3
"""
parallel_batch_parse.py — 并行批量解析 PDF(复用 batch_parse_pdfs.py 的轮询/下载函数)。

特点:
  - ThreadPoolExecutor 并发提交 + 并发轮询 + 完成后下载(利用 worker 池并行)
  - 文件名 sanitize:上传的 multipart filename 与输出目录名都把空格/特殊字符/非ASCII → _
    (不改原 PDF;worker 收到的是安全文件名)
  - --input-root 支持逗号分隔多个根(一次跑多个文件夹,rel key 含 root 名保证唯一)

复用(直接 import batch_parse_pdfs as bp):
  login / poll_task / save_result / iter_pdfs / is_output_complete /
  load_tasks / save_tasks / update_summary / log

用法:
  python scripts/parallel_batch_parse.py \\
    --base-url http://localhost:8000 \\
    --input-root /data/projects/research_background/AI_Scientist,/data/projects/research_background/LLM-benchmark \\
    --output-root /data/projects/research_background/parsed \\
    --backend auto --recursive --concurrency 8 \\
    --username admin --password admin123
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests

# 复用同目录 batch_parse_pdfs.py 的函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_parse_pdfs as bp  # noqa: E402


def safe_name(name: str) -> str:
    """sanitize 文件/目录名:非 [a-zA-Z0-9._-] → _(空格/特殊字符/非ASCII 统一转 _)。"""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def submit_one(base_url, token, rel_key, pdf_path, backend, lang, method, preserve_all_artifacts=False):
    """提交单个 PDF(用 safe filename 上传),返回 (rel_key, task_id|None, error|None)。"""
    try:
        url = f"{base_url}/api/v1/tasks/submit"
        headers = {"Authorization": f"Bearer {token}"}
        safe_fname = safe_name(pdf_path.name)
        with pdf_path.open("rb") as f:
            files = {"file": (safe_fname, f, "application/pdf")}
            data = {
                "backend": backend,
                "lang": lang,
                "method": method,
                "preserve_all_artifacts": str(preserve_all_artifacts).lower(),
            }
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=600)
        resp.raise_for_status()
        return rel_key, resp.json()["task_id"], None
    except Exception as e:
        return rel_key, None, str(e)


def poll_and_save(base_url, token, rel_key, item, poll_interval, timeout, download_imgs):
    """轮询单个任务到完成并下载结果,更新 item。"""
    tid = item.get("task_id")
    if not tid:
        return rel_key, item["status"]
    start = time.time()
    try:
        result = bp.poll_task(base_url, token, tid, poll_interval, timeout)
        status = result.get("status")
        item["status"] = status
        if status == "completed":
            bp.save_result(Path(item["output_dir"]), result, base_url, token, download_imgs)
            item["completed_at"] = datetime.now().isoformat()
            item["duration_sec"] = round(time.time() - start, 1)
        else:
            item["error"] = (result.get("error_message") or result.get("message") or "")[:200]
        item["last_update"] = datetime.now().isoformat()
    except Exception as e:
        item["status"] = "failed"
        item["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        item["last_update"] = datetime.now().isoformat()
    return rel_key, item["status"]


def main():
    ap = argparse.ArgumentParser(description="并行批量解析 PDF(sanitize 文件名 + 多 input-root)")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--input-root", required=True, help="逗号分隔多个根目录")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--username", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--method", default="auto")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--poll-interval", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--download-images", action="store_true", help="Deprecated; images are downloaded by default")
    ap.add_argument("--download-all-artifacts", action="store_true", help="Use full server artifact policy")
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    input_roots = [Path(x).resolve() for x in args.input_root.split(",") if x.strip()]
    output_root = Path(args.output_root).resolve()
    tasks_file = Path(args.tasks_file).resolve() if args.tasks_file else output_root / "batch_tasks.json"
    summary_json = Path(args.summary_json).resolve() if args.summary_json else output_root / "summary.json"
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else output_root / "summary.csv"

    bp.log("Stage 1/5: validate")
    for r in input_roots:
        if not r.exists():
            raise SystemExit(f"input-root 不存在: {r}")

    bp.log("Stage 2/5: login")
    if args.token:
        token = args.token
    else:
        if not args.username or not args.password:
            raise SystemExit("需要 --token 或 --username/--password")
        token = bp.login(base_url, args.username, args.password)

    bp.log("Stage 3/5: scan PDFs(多 root)")
    pdf_list = []  # (pdf_path, root_name, rel)
    for root in input_roots:
        for p in bp.iter_pdfs(root, args.recursive):
            pdf_list.append((p, root.name, p.relative_to(root)))
    if not pdf_list:
        raise SystemExit("未找到 PDF")
    bp.log(f"  找到 {len(pdf_list)} 个 PDF, concurrency={args.concurrency}")

    resume = not args.no_resume
    preserve_all_artifacts = args.download_all_artifacts
    download_results = True
    tasks: Dict[str, dict] = {} if not resume else bp.load_tasks(tasks_file)

    # 初始化任务记录(rel_key = root_name/rel,唯一)
    for pdf, root_name, rel in pdf_list:
        rel_key = f"{root_name}/{rel.as_posix()}"
        out_dir = output_root / root_name / rel.parent / safe_name(rel.stem)
        if rel_key not in tasks:
            tasks[rel_key] = {
                "pdf_path": str(pdf),
                "relative_path": rel_key,
                "task_id": None,
                "status": "pending",
                "submitted_at": None,
                "completed_at": None,
                "duration_sec": None,
                "retry_count": 0,
                "output_dir": str(out_dir),
                "error": None,
                "last_update": datetime.now().isoformat(),
                "preserve_all_artifacts": preserve_all_artifacts,
            }
        previous_policy = bool(tasks[rel_key].get("preserve_all_artifacts", False))
        policy_changed = previous_policy != preserve_all_artifacts
        if policy_changed:
            tasks[rel_key]["task_id"] = None
            tasks[rel_key]["status"] = "pending"
            tasks[rel_key]["completed_at"] = None
            tasks[rel_key]["duration_sec"] = None
            tasks[rel_key]["preserve_all_artifacts"] = preserve_all_artifacts
        if not policy_changed and bp.is_output_complete(Path(tasks[rel_key]["output_dir"]), download_results):
            tasks[rel_key]["status"] = "completed"
        elif download_results and tasks[rel_key].get("status") == "completed" and tasks[rel_key].get("task_id"):
            tasks[rel_key]["status"] = "processing"

    # ---- 并发提交全部 ----
    to_submit = [k for k, t in tasks.items() if t["status"] != "completed" and not t.get("task_id")]
    bp.log(f"Stage 4/5: 并发提交 {len(to_submit)} 个")
    with ThreadPoolExecutor(max_workers=min(args.concurrency, max(1, len(to_submit))) if to_submit else 1) as ex:
        futs = {
            ex.submit(
                submit_one, base_url, token, k, Path(tasks[k]["pdf_path"]), args.backend, args.lang, args.method, preserve_all_artifacts
            ): k
            for k in to_submit
        }
        for fut in as_completed(futs):
            rel_key, tid, err = fut.result()
            if tid:
                tasks[rel_key]["task_id"] = tid
                tasks[rel_key]["status"] = "pending"
                tasks[rel_key]["submitted_at"] = datetime.now().isoformat()
                bp.log(f"  提交: {rel_key} -> {tid}")
            else:
                tasks[rel_key]["status"] = "submit_failed"
                tasks[rel_key]["error"] = err
                bp.log(f"  提交失败: {rel_key} ({err})")
    bp.save_tasks(tasks_file, tasks)

    # ---- 并发轮询 + 下载 ----
    to_poll = [k for k, t in tasks.items() if t.get("task_id") and t["status"] not in ("completed", "submit_failed")]
    bp.log(f"Stage 5/5: 并发轮询+下载 {len(to_poll)} 个(每个到 completed/failed)")
    done = 0
    total = len(to_poll)
    workers = min(args.concurrency, max(1, total)) if total else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(poll_and_save, base_url, token, k, tasks[k], args.poll_interval, args.timeout, download_results): k
            for k in to_poll
        }
        for fut in as_completed(futs):
            rel_key, status = fut.result()
            done += 1
            bp.save_tasks(tasks_file, tasks)
            bp.update_summary(summary_json, summary_csv, tasks)
            bp.log(f"  [{done}/{total}] {rel_key} -> {status}")

    bp.save_tasks(tasks_file, tasks)
    bp.update_summary(summary_json, summary_csv, tasks)

    ok = sum(1 for t in tasks.values() if t["status"] == "completed")
    fail = sum(1 for t in tasks.values() if t["status"] == "failed")
    other = len(tasks) - ok - fail
    bp.log(f"\n=== 完成: {ok} completed / {fail} failed / {other} other(共 {len(tasks)}) ===")
    if fail:
        bp.log("失败项:")
        for k, t in tasks.items():
            if t["status"] == "failed":
                bp.log(f"  {k}: {t.get('error')}")


if __name__ == "__main__":
    main()
