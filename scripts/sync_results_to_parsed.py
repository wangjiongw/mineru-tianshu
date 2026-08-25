#!/usr/bin/env python3
"""
sync_results_to_parsed.py — 将 mineru_outputs 结果归档到 pdfs_parsed/{xx}/{sha256}/ 分片布局

与源语料 /share/wangjiong/databases/escorpus-assets/pdfs/oa/by_sha256/{xx}/{sha256}.pdf
保持相同的 sha256 前两位分组。

三种模式:
  --reshard-flat   把 pdfs_parsed/ 顶层平铺的 {sha256}/ 目录移入 {xx}/{sha256}/
  --migrate        一次性迁移: DB 中已完成任务的结果目录 mineru_outputs/{md5}_{sha}/
                   → pdfs_parsed/{xx}/{sha}/, 并同步更新 tasks.result_path
  --daemon         常驻模式, 每 --interval 秒同步新完成的任务 (供后续任务持续落地)

安全设计:
  - 只处理 status='completed' 的顶层任务 (parent_task_id IS NULL)
  - 只处理 sha256 命名的文件 (escorpus 语料), 旧语料命名不动
  - 目标已存在时 no-clobber 合并 (旧结果优先), 完全覆盖后才删源目录
  - mv 为同卷 rename, 元数据操作不复制数据
  - DB 更新分批小事务, 锁冲突自动重试
"""

import argparse
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

INSTANCE_ID = os.environ.get("INSTANCE_ID") or os.popen("hostname").read().strip()
DATA_DIR = Path(f"/share/wangjiong/databases/mineru_database/{INSTANCE_ID}")
DB_PATH = Path(os.environ.get("DATABASE_PATH", str(DATA_DIR / "mineru_tianshu.db")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", str(DATA_DIR / "mineru_outputs")))
PARSED_ROOT = Path("/share/wangjiong/databases/escorpus-assets/pdfs_parsed")

SHA_RE = re.compile(r"^([0-9a-f]{64})$")
SHA_FILE_RE = re.compile(r"^([0-9a-f]{64})\.pdf$")
DB_BATCH = 200
LOCK_RETRIES = 5


def log(msg):
    ts = datetime.now().strftime("%F %T")
    print(f"[{ts}] {msg}", flush=True)


def db_connect():
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def db_exec_batch(conn, sql, params_list):
    """分批小事务执行, 锁冲突自动重试"""
    for i in range(0, len(params_list), DB_BATCH):
        chunk = params_list[i:i + DB_BATCH]
        for attempt in range(LOCK_RETRIES):
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(sql, chunk)
                conn.execute("COMMIT")
                break
            except sqlite3.OperationalError as e:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                if "locked" in str(e).lower() and attempt < LOCK_RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise


def move_result_dir(src: Path, dst: Path):
    """
    src → dst (同卷 rename)。
    dst 已存在时 no-clobber 合并: 只移动 dst 缺失的条目, 全部覆盖后删 src。
    返回 (状态, 详情): ('moved'|'merged'|'missing'|'conflict', str)
    """
    if not src.exists():
        return ("missing", str(src))
    if dst.exists():
        # 合并: 目标里已有的文件保留 (旧结果优先), 只补缺失的
        uncovered = []
        for item in src.iterdir():
            target_item = dst / item.name
            if target_item.exists():
                continue
            try:
                os.rename(item, target_item)
            except OSError:
                # rename 失败(如目录非空冲突), 回退为复制
                if item.is_dir():
                    shutil.copytree(item, target_item, dirs_exist_ok=True)
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    shutil.copy2(item, target_item)
                    item.unlink()
        # 判断是否全部条目已在目标中
        remaining = list(src.iterdir())
        for r in remaining:
            if not (dst / r.name).exists():
                uncovered.append(r.name)
        if uncovered:
            return ("conflict", f"{len(uncovered)} items uncovered, e.g. {uncovered[:3]}")
        shutil.rmtree(src, ignore_errors=True)
        return ("merged", str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
        return ("moved", str(dst))
    except OSError:
        # 跨设备兜底 (不应该发生, 都在 /share)
        shutil.copytree(src, dst)
        shutil.rmtree(src, ignore_errors=True)
        return ("moved", str(dst))


def ensure_shard_dirs():
    """预创建 256 个分片目录"""
    for i in range(256):
        (PARSED_ROOT / f"{i:02x}").mkdir(parents=True, exist_ok=True)


def cmd_reshard_flat(conn, dry_run=False):
    """把 pdfs_parsed/ 顶层平铺的 {sha256}/ 移入 {xx}/{sha256}/"""
    if not PARSED_ROOT.exists():
        log(f"结果根目录不存在: {PARSED_ROOT}")
        return
    if not dry_run:
        ensure_shard_dirs()
    names = [n for n in os.listdir(PARSED_ROOT) if SHA_RE.match(n)]
    log(f"平铺 sha256 目录: {len(names)}, 开始分片化...")
    moved = 0
    conflicts = 0
    for n in names:
        src = PARSED_ROOT / n
        dst = PARSED_ROOT / n[:2] / n
        if dry_run:
            continue
        st, _ = move_result_dir(src, dst)
        if st in ("moved", "merged"):
            moved += 1
        else:
            conflicts += 1
    log(f"分片化完成: moved={moved} conflict={conflicts} (dry_run={dry_run})")


def fetch_migratable(conn, limit=None):
    """取已完成顶层 sha256 任务, 结果仍在 mineru_outputs 下"""
    q = """
        SELECT task_id, file_name, result_path FROM tasks
        WHERE status='completed' AND parent_task_id IS NULL
    """
    rows = conn.execute(q).fetchall()
    out = []
    prefix = str(OUTPUT_DIR) + "/"
    for r in rows:
        m = SHA_FILE_RE.match(r["file_name"] or "")
        if not m:
            continue
        rp = r["result_path"] or ""
        if not rp.startswith(prefix):
            continue
        out.append((r["task_id"], m.group(1), rp))
    if limit:
        out = out[:limit]
    return out


def do_migrate(conn, dry_run=False, limit=None):
    """迁移 mineru_outputs/{md5}_{sha}/ → pdfs_parsed/{xx}/{sha}/, 更新 DB"""
    if not dry_run:
        ensure_shard_dirs()
    tasks = fetch_migratable(conn, limit)
    log(f"待迁移任务: {len(tasks)}")

    stats = {"moved": 0, "merged": 0, "missing": 0, "conflict": 0, "dst_missing": 0}
    db_updates = []
    t0 = time.time()
    for i, (tid, sha, rp) in enumerate(tasks):
        src = Path(rp)
        dst = PARSED_ROOT / sha[:2] / sha

        if dry_run:
            continue

        # 源目录已不在: 若目标存在, 修复 DB; 否则记录缺失
        if not src.exists():
            if dst.exists():
                db_updates.append((str(dst), tid))
                stats["dst_missing"] += 1
            else:
                stats["missing"] += 1
            continue

        st, _ = move_result_dir(src, dst)
        stats[st] = stats.get(st, 0) + 1
        if st in ("moved", "merged"):
            db_updates.append((str(dst), tid))
        elif st == "missing":
            if dst.exists():
                db_updates.append((str(dst), tid))

        if (i + 1) % 5000 == 0:
            db_exec_batch(conn, "UPDATE tasks SET result_path=? WHERE task_id=?", db_updates)
            db_updates = []
            rate = (i + 1) / (time.time() - t0)
            log(f"  进度 {i+1}/{len(tasks)}  {rate:.0f}/s  {stats}")

    if db_updates and not dry_run:
        db_exec_batch(conn, "UPDATE tasks SET result_path=? WHERE task_id=?", db_updates)

    dt = time.time() - t0
    log(f"迁移完成 ({dt:.0f}s): {stats}  DB 更新: {stats.get('dst_missing',0)+stats['moved']+stats['merged']}")


def cmd_daemon(conn, interval, limit_per_cycle):
    """常驻模式: 持续把新完成任务的结果同步到分片布局"""
    ensure_shard_dirs()
    log(f"daemon 启动: interval={interval}s, 每轮最多 {limit_per_cycle} 个")
    while True:
        try:
            pending = fetch_migratable(conn, limit_per_cycle)
            if pending:
                do_migrate(conn, limit=limit_per_cycle)
            else:
                log("空闲: 无待同步结果")
        except Exception as e:
            log(f"❌ daemon 周期异常: {e}")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="结果归档到 pdfs_parsed/{xx}/{sha256}/")
    ap.add_argument("--reshard-flat", action="store_true", help="平铺目录分片化")
    ap.add_argument("--migrate", action="store_true", help="一次性迁移已完成任务")
    ap.add_argument("--daemon", action="store_true", help="常驻同步模式")
    ap.add_argument("--interval", type=int, default=60, help="daemon 轮询间隔 (秒)")
    ap.add_argument("--limit", type=int, default=None, help="每轮最多处理数")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log(f"DB:       {DB_PATH}")
    log(f"OUTPUT:   {OUTPUT_DIR}")
    log(f"PARSED:   {PARSED_ROOT}")

    conn = db_connect()
    try:
        if args.reshard_flat:
            cmd_reshard_flat(conn, args.dry_run)
        if args.migrate:
            do_migrate(conn, args.dry_run, args.limit)
        if args.daemon:
            cmd_daemon(conn, args.interval, args.limit or 1000)
        if not (args.reshard_flat or args.migrate or args.daemon):
            ap.print_help()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
