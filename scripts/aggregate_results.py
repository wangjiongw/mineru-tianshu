#!/usr/bin/env python3
"""
aggregate_results.py — 索引(默认)/汇聚 多实例 mineru_outputs 结果。

实例隔离后,各实例结果已在共享存储的 <DATA_ROOT>/<实例>/mineru_outputs/<hash>_<文件>/。
同集群下游可直接读这些目录,所以**默认只做【索引 + 去重报告】,不复制**(省空间):

  - 扫描各实例,生成 manifest.jsonl/.md(每条指向**原位置**) + duplicates.txt(跨实例重复,只报告)

需要物理汇聚时再加:
  --copy   物理复制(双份存储,适合导出到 /share 之外)
  --link   软链视图(不占额外空间,output 里是指向原位置的软链)

用法:
  # 默认:index + 去重报告(不复制)
  python scripts/aggregate_results.py \\
    --data-root /share/wangjiong/databases/mineru_database \\
    --output    /share/wangjiong/databases/aggregated

  # 物理汇聚(可选)
  python scripts/aggregate_results.py --data-root <DIR> --output <DIR> \\
    [--copy | --link] [--layout by-instance|flat] \\
    [--include md,full,json,model,images[,pdf|,all]] [--instances a,b] [--dry-run]

依赖:仅 Python 标准库。
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# include 类别 → 任务目录里的实际文件/子目录名(images 是目录)
FILE_MAP = {
    "md": ["result.md"],
    "full": ["full.md"],
    "json": ["result.json"],
    "model": ["mineru_model.json"],
    "images": ["images"],  # 子目录
    # "pdf" 特殊:匹配 *.pdf
}
DEFAULT_INCLUDE = "md,full,json,model,images"


def parse_include(s: str):
    items = [x.strip() for x in s.split(",") if x.strip()]
    if "all" in items:
        return list(FILE_MAP.keys()) + ["pdf"]
    return items


def iter_task_dirs(data_root: Path, instances):
    """yield (instance_name, task_dir_path)。instances=None 表示全部。"""
    if not data_root.is_dir():
        sys.exit(f"❌ data-root 不存在: {data_root}")
    if instances:
        names = instances
    else:
        names = sorted(d.name for d in data_root.iterdir() if d.is_dir() and (d / "mineru_outputs").is_dir())
    for inst in names:
        out_dir = data_root / inst / "mineru_outputs"
        if not out_dir.is_dir():
            print(f"⚠️  跳过实例 {inst}: 无 mineru_outputs 目录", file=sys.stderr)
            continue
        for td in sorted(out_dir.iterdir()):
            if td.is_dir():
                yield inst, td


def collect_files(task_dir: Path, include):
    """返回任务目录里可用且匹配 include 的 [(相对名, 绝对路径)]。"""
    files = []
    for cat in include:
        if cat == "pdf":
            files += [(p.name, p) for p in task_dir.glob("*.pdf")]
            continue
        names = FILE_MAP.get(cat)
        if names is None:
            print(f"⚠️  未知 include 类别 '{cat}',忽略", file=sys.stderr)
            continue
        for n in names:
            p = task_dir / n
            if p.exists():
                files.append((n, p))
    return files


def path_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def materialize(src: Path, dst: Path, mode: str):
    """按 copy / link 把 src 落到 dst。"""
    if mode == "copy":
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    else:  # link
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser(description="索引/汇聚多实例 mineru_outputs")
    ap.add_argument("--data-root", required=True, help="实例数据根(含 <实例>/mineru_outputs)")
    ap.add_argument("--output", required=True, help="manifest/duplicates 写入处;--copy/--link 时也是物理汇聚根目录")
    ap.add_argument("--instances", default=None, help="逗号分隔实例名;默认全部")
    ap.add_argument(
        "--include",
        default=DEFAULT_INCLUDE,
        help=f"类别,默认 '{DEFAULT_INCLUDE}';可选 md,full,json,model,images,pdf,all",
    )
    ap.add_argument("--layout", choices=["by-instance", "flat"], default="by-instance", help="仅 --copy/--link 时生效")
    ap.add_argument("--copy", action="store_true", help="物理复制(双份存储,导出场景)")
    ap.add_argument("--link", action="store_true", help="软链视图(不占额外空间,指向原位置)")
    ap.add_argument("--dry-run", action="store_true", help="只扫描统计,不写任何文件")
    args = ap.parse_args()

    if args.copy and args.link:
        sys.exit("❌ --copy 与 --link 互斥")
    mode = "copy" if args.copy else ("link" if args.link else "index")
    physical = mode in ("copy", "link")

    include = parse_include(args.include)
    data_root = Path(args.data_root)
    out = Path(args.output)
    instances = [x.strip() for x in args.instances.split(",")] if args.instances else None

    print(f"data-root : {data_root}")
    print(f"output    : {out}")
    print(f"mode      : {mode} | include: {include} | layout: {args.layout} | dry-run: {args.dry_run}\n")

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    manifest = []
    seen_keys = {}  # dir_name -> [instance,...]  重复检测
    flat_used = {}  # flat 布局: dir_name -> 已占用实例(冲突跳过)
    inst_stat = {}
    skipped = 0
    total = 0
    total_size = 0

    for inst, td in iter_task_dirs(data_root, instances):
        dname = td.name
        files = collect_files(td, include)
        if not files:
            skipped += 1
            continue

        dest = None
        if physical:
            if args.layout == "by-instance":
                dest = out / inst / dname
            else:  # flat
                if dname in flat_used:
                    print(f"⚠️  [skip] flat 冲突: {dname} 已来自 {flat_used[dname]},跳过 {inst}", file=sys.stderr)
                    skipped += 1
                    continue
                flat_used[dname] = inst
                dest = out / dname
            if not args.dry_run:
                for rel, src in files:
                    materialize(src, dest / rel, mode)

        size = sum(path_size(src) for _, src in files)
        manifest.append(
            {
                "instance": inst,
                "dir_name": dname,
                "source": str(td),  # 原位置(始终记录)
                "dest": str(dest) if dest else None,  # 仅 copy/link 时
                "files": [r for r, _ in files],
                "size": size,
                "mtime": datetime.fromtimestamp(td.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
        seen_keys.setdefault(dname, []).append(inst)
        st = inst_stat.setdefault(inst, {"tasks": 0, "size": 0})
        st["tasks"] += 1
        st["size"] += size
        total += 1
        total_size += size

    dups = {k: v for k, v in seen_keys.items() if len(v) > 1}

    # 写产物(manifest + duplicates;index 模式下 output 里只有这些,无实例副本)
    if not args.dry_run:
        (out / "manifest.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest),
            encoding="utf-8",
        )
        lines = ["# manifest", "", "| 实例 | 任务数 | 大小(原位) |", "|---|---:|---:|"]
        for inst, st in sorted(inst_stat.items()):
            lines.append(f"| {inst} | {st['tasks']} | {st['size'] // 1024} KB |")
        lines.append(f"| **合计** | **{total}** | **{total_size // 1024} KB** |")
        (out / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if dups:
            (out / "duplicates.txt").write_text(
                "\n".join(f"{k}  出现实例: {', '.join(v)}" for k, v in sorted(dups.items())) + "\n",
                encoding="utf-8",
            )

    print("=== 结果 ===")
    print(f"  实例数: {len(inst_stat)}")
    print(f"  任务数: {total}")
    if mode == "copy":
        print(f"  总大小: {total_size // 1024} KB(已物理复制到 output)")
    else:
        print(f"  总大小: {total_size // 1024} KB(原位,{mode} 模式不新增存储)")
    print(f"  跳过  : {skipped}")
    dup_note = f"(见 {out}/duplicates.txt)" if dups and not args.dry_run else ""
    print(f"  重复  : {len(dups)} {dup_note}")
    if mode == "index" and not args.dry_run:
        print(f"  → index 模式:未复制,manifest 指向原位置 {data_root}/<实例>/mineru_outputs/")
    if args.dry_run:
        print("  (dry-run,未写任何文件)")


if __name__ == "__main__":
    main()
