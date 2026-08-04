#!/usr/bin/env python3
"""fix_orphan_merges.py — 手动合并因竞态 bug 遗漏的父任务

当 orphan recovery 在 children 完成前把父任务从 'processing' 重置为 'pending' 时，
on_child_task_completed 的 CAS (processing→merging) 会失败，导致合并永不触发。

本脚本找出这类父任务（status=pending 但 child_completed>=child_count），
用与 _merge_parent_task_results() 相同的逻辑手动合并。

用法:
  python scripts/fix_orphan_merges.py --dry-run   # 只列出，不执行
  python scripts/fix_orphan_merges.py              # 执行合并
"""
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

INSTANCE_ID = os.environ.get("INSTANCE_ID") or os.popen("hostname").read().strip()
DATA_DIR = Path(f"/share/wangjiong/databases/mineru_database/{INSTANCE_ID}")
DB_PATH = Path(os.environ.get("DATABASE_PATH", str(DATA_DIR / "mineru_tianshu.db")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", str(DATA_DIR / "mineru_outputs")))


def find_orphaned_parents(conn):
    """找出所有 child_completed >= child_count 但 status != 'completed' 的父任务"""
    rows = conn.execute(
        """
        SELECT task_id, file_name, file_path, child_count, child_completed, status
        FROM tasks
        WHERE is_parent = 1
          AND child_count > 0
          AND child_completed >= child_count
          AND status != 'completed'
        ORDER BY created_at
        """
    ).fetchall()
    return rows


def get_children(conn, parent_task_id):
    """获取子任务，按 chunk_info.start_page 排序"""
    rows = conn.execute(
        "SELECT task_id, status, result_path, options FROM tasks WHERE parent_task_id = ?",
        (parent_task_id,),
    ).fetchall()

    children = []
    for row in rows:
        child = dict(row)
        try:
            opts = json.loads(child.get("options") or "{}")
            child["start_page"] = opts.get("chunk_info", {}).get("start_page", 0)
        except Exception:
            child["start_page"] = 0
        children.append(child)

    children.sort(key=lambda x: x["start_page"])
    return children


def merge_one(conn, parent):
    """合并单个父任务的子任务结果"""
    pid = parent["task_id"]
    file_path = parent["file_path"]
    file_name = parent["file_name"]

    children = get_children(conn, pid)
    completed_children = [c for c in children if c["status"] == "completed"]
    if not completed_children:
        print(f"  ❌ {pid[:13]} — 无已完成的子任务，跳过")
        return False

    # 父任务输出目录 = OUTPUT_DIR / stem(file_path)
    parent_out = OUTPUT_DIR / Path(file_path).stem

    print(f"  🔀 {pid[:13]} → {parent_out.name}")
    print(f"     children: {len(completed_children)}/{len(children)} completed")
    for c in completed_children:
        print(f"       chunk p{c['start_page']}: {Path(c['result_path']).name}")

    # 如果已有合并结果，先备份
    if parent_out.exists() and (parent_out / "result.md").exists():
        print(f"     ⚠️  合并目录已存在 result.md，将覆盖")

    parent_out.mkdir(parents=True, exist_ok=True)

    # --- 合并 Markdown ---
    md_parts = []
    for child in completed_children:
        res_dir = Path(child["result_path"])
        md_files = list(res_dir.rglob("*.md"))
        # 优先 result.md，否则取第一个
        md_file = next((f for f in md_files if f.name == "result.md"), None)
        if not md_file:
            md_file = md_files[0] if md_files else None
        if md_file:
            md_parts.append(md_file.read_text(encoding="utf-8"))

    (parent_out / "result.md").write_text("\n\n\n\n".join(md_parts), encoding="utf-8")
    print(f"     ✅ result.md = {sum(len(p) for p in md_parts)} chars from {len(md_parts)} chunks")

    # --- 合并 JSON（修正 page_idx）---
    json_pages = []
    for child in completed_children:
        res_dir = Path(child["result_path"])
        json_file = next(
            (f for f in res_dir.rglob("*.json") if "result" in f.name or "content" in f.name),
            None,
        )
        if not json_file:
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            offset = child["start_page"] - 1  # chunk_info.start_page 是 1-based
            pages = []
            if isinstance(data, list):
                pages = data
            elif "pages" in data:
                pages = data["pages"]
            for p in pages:
                if isinstance(p, dict) and "page_idx" in p:
                    p["page_idx"] += offset
                json_pages.append(p)
        except Exception as e:
            print(f"     ⚠️  JSON merge failed for chunk p{child['start_page']}: {e}")

    if json_pages:
        (parent_out / "result.json").write_text(
            json.dumps(json_pages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"     ✅ result.json = {len(json_pages)} pages")

    # --- 复制源 PDF ---
    src_pdf = Path(file_path)
    if src_pdf.exists() and src_pdf.suffix.lower() == ".pdf":
        target = parent_out / src_pdf.name
        if not target.exists():
            shutil.copy2(src_pdf, target)
        print(f"     ✅ source PDF copied ({src_pdf.stat().st_size // 1024}KB)")

    # --- 更新 DB ---
    conn.execute(
        "UPDATE tasks SET status = 'completed', result_path = ?, completed_at = datetime('now') WHERE task_id = ?",
        (str(parent_out), pid),
    )
    conn.commit()
    print(f"     ✅ DB updated: status=completed, result_path set")

    return True


def main():
    dry_run = "--dry-run" in sys.argv

    print(f"DB:       {DB_PATH}")
    print(f"OUTPUT:   {OUTPUT_DIR}")
    print(f"INSTANCE: {INSTANCE_ID}")
    print()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    orphans = find_orphaned_parents(conn)
    print(f"找到 {len(orphans)} 个遗漏合并的父任务:")
    for o in orphans:
        print(f"  {o['task_id'][:13]}  status={o['status']}  children={o['child_completed']}/{o['child_count']}  {o['file_name'][:50]}")

    if dry_run:
        print("\n--dry-run 模式，不执行合并")
        return

    # 原子认领：把 status 从 pending 改为 processing，防止 worker 抢走重新拆分
    pids = [o["task_id"] for o in orphans]
    placeholders = ",".join("?" * len(pids))
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE tasks SET status = 'processing' WHERE task_id IN ({placeholders}) AND status = 'pending'",
        pids,
    )
    claimed = cursor.rowcount
    conn.commit()
    print(f"\n已认领 {claimed}/{len(pids)} 个父任务 (status→processing)")

    print(f"\n开始合并 {len(orphans)} 个父任务...\n")

    ok = 0
    fail = 0
    for parent in orphans:
        try:
            if merge_one(conn, parent):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ❌ {parent['task_id'][:13]} 合并失败: {e}")
            fail += 1
        print()

    conn.close()
    print(f"{'='*60}")
    print(f"合并完成: {ok} 成功 / {fail} 失败 / {len(orphans)} 总计")


if __name__ == "__main__":
    main()
