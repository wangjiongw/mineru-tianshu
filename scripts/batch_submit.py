#!/usr/bin/env python3
"""
batch_submit.py — Fire-and-forget 批量提交 PDF 到 MinerU 队列

只提交、不轮询、不下载。Worker 从 Redis 队列自动消费, 结果写入 mineru_outputs/。
自动跳过已处理的文件 (检查 pdfs_parsed/ 和 mineru_outputs/)。

用法:
  # 提交指定目录
  python scripts/batch_submit.py --dirs 0e 0f

  # 提交所有未完成目录
  python scripts/batch_submit.py --dirs all

  # 先看会提交多少, 不实际提交
  python scripts/batch_submit.py --dirs 0e --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
BY_SHA256 = Path("/share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256")
PARSED_OUT = Path("/share/wangjiong/databases/escorpus-assets/pdfs_parsed")
MINERU_OUT = Path("/share/wangjiong/databases/mineru_database/mineru-runner-worker-0/mineru_outputs")
PROGRESS_FILE = Path("/share/wangjiong/databases/escorpus-assets/pdfs_parsed/_submit_progress.json")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_processed_set():
    """构建已处理 SHA256 集合 (从 pdfs_parsed/{xx}/{sha}/ 分片布局、顶层平铺遗留、mineru_outputs/)"""
    processed = set()

    if PARSED_OUT.exists():
        for name in os.listdir(PARSED_OUT):
            d = PARSED_OUT / name
            if not d.is_dir():
                continue
            if len(name) == 64 and re.match(r"^[0-9a-f]{64}$", name):
                # 顶层平铺遗留布局
                if (d / "result.md").exists():
                    processed.add(name)
            elif len(name) == 2 and re.match(r"^[0-9a-f]{2}$", name):
                # 分片布局 {xx}/{sha256}/
                for sub in os.listdir(d):
                    if len(sub) == 64 and re.match(r"^[0-9a-f]{64}$", sub):
                        if (d / sub / "result.md").exists():
                            processed.add(sub)

    # mineru_outputs/: {md5}_{sha256}
    if MINERU_OUT.exists():
        for name in os.listdir(MINERU_OUT):
            parts = name.split("_", 1)
            if len(parts) == 2 and len(parts[1]) == 64:
                if re.match(r"^[0-9a-f]{64}$", parts[1]):
                    processed.add(parts[1])

    return processed


def scan_pdfs(dirs):
    """扫描指定目录下所有未处理的 PDF, 返回 [(sha256, path), ...]"""
    processed = build_processed_set()
    log(f"已处理 (跳过): {len(processed)}")

    # 加载提交进度 (断点续传)
    submitted = set()
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text())
            submitted = set(data.get("submitted", []))
            log(f"已提交 (续传跳过): {len(submitted)}")
        except Exception:
            pass

    pending = []
    for d in sorted(dirs):
        dir_path = BY_SHA256 / d
        if not dir_path.exists():
            log(f"⚠️  目录不存在: {d}")
            continue
        for pdf in sorted(dir_path.glob("*.pdf")):
            sha = pdf.stem
            if sha in processed or sha in submitted:
                continue
            pending.append((sha, str(pdf)))

    return pending, submitted


def login(username, password):
    """登录获取 JWT token"""
    resp = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def submit_one(token, sha, pdf_path, backend="hybrid-auto-engine", effort="high"):
    """提交单个 PDF, 返回 (sha, task_id|None, error|None)"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        with open(pdf_path, "rb") as f:
            files = {"file": (f"{sha}.pdf", f, "application/pdf")}
            data = {"backend": backend, "lang": "auto", "method": "auto", "effort": effort}
            resp = requests.post(
                f"{BASE_URL}/api/v1/tasks/submit",
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        return sha, task_id, None
    except Exception as e:
        return sha, None, str(e)[:200]


def save_progress(submitted):
    """保存提交进度"""
    PROGRESS_FILE.write_text(json.dumps({
        "updated_at": datetime.now().isoformat(),
        "submitted": list(submitted),
    }, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="Fire-and-forget 批量提交 PDF")
    ap.add_argument("--dirs", nargs="+", required=True,
                     help="要处理的目录 (如: 0e 0f 10) 或 'all'")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--backend", default="hybrid-auto-engine")
    ap.add_argument("--effort", default="high", choices=["medium", "high"],
                    help="Hybrid 解析强度 (默认: high)")
    ap.add_argument("--dry-run", action="store_true", help="只统计,不提交")
    args = ap.parse_args()

    # 确定目录列表
    if args.dirs == ["all"]:
        all_dirs = sorted(d for d in os.listdir(BY_SHA256)
                          if (BY_SHA256 / d).is_dir()
                          and any((BY_SHA256 / d).glob("*.pdf")))
        # 过滤掉 00-0d (已完成)
        done_prefixes = [f"0{x}" for x in "0123456789abcd"]
        dirs = [d for d in all_dirs if d not in done_prefixes]
    else:
        dirs = args.dirs

    log(f"扫描目录: {dirs}")
    pending, submitted = scan_pdfs(dirs)

    if not pending:
        log("✅ 无待处理文件")
        return

    log(f"待提交: {len(pending)} 个 PDF")

    if args.dry_run:
        # 按目录统计
        dir_counts = {}
        for sha, path in pending:
            d = Path(path).parent.name
            dir_counts[d] = dir_counts.get(d, 0) + 1
        for d, c in sorted(dir_counts.items()):
            log(f"  {d}/: {c}")
        return

    # 登录
    log("登录中...")
    token = login(args.username, args.password)
    log("✅ 登录成功")

    # 并发提交
    total = len(pending)
    done = 0
    ok = 0
    fail = 0
    start_time = time.time()
    last_save = time.time()

    workers = min(args.concurrency, total)
    log(f"开始提交 (并发={workers})")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(submit_one, token, sha, path, args.backend, args.effort): sha
            for sha, path in pending
        }
        for fut in as_completed(futs):
            sha, task_id, err = fut.result()
            done += 1
            if task_id:
                ok += 1
                submitted.add(sha)
            else:
                fail += 1
                log(f"❌ [{done}/{total}] {sha[:16]}... 失败: {err}")

            if done % 100 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                log(f"📊 [{done}/{total}] ok={ok} fail={fail} "
                    f"rate={rate:.1f}/s ETA={eta/60:.0f}min")

            # 定期保存进度
            if time.time() - last_save > 30:
                save_progress(submitted)
                last_save = time.time()

    save_progress(submitted)

    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"提交完成: {ok} 成功 / {fail} 失败 / {total} 总计")
    log(f"耗时: {elapsed/60:.1f} 分钟 ({total/elapsed:.1f}/s)")
    log(f"进度文件: {PROGRESS_FILE}")
    log(f"Worker 将从队列自动消费处理, 结果写入 mineru_outputs/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
