#!/usr/bin/env python3
"""
tianshu_status.py — MinerU Tianshu 命令行状态监控工具

用法:
  python scripts/tianshu_status.py status               # 服务总览
  python scripts/tianshu_status.py queue                # 队列统计
  python scripts/tianshu_status.py tasks                # 任务列表
  python scripts/tianshu_status.py tasks --status failed
  python scripts/tianshu_status.py tasks --search report.pdf
  python scripts/tianshu_status.py task <task_id>       # 单任务详情
  python scripts/tianshu_status.py watch                # 持续监控（默认30s刷新）
  python scripts/tianshu_status.py watch --interval 10

认证（任选一种）:
  --username admin --password admin123
  --token <jwt_or_api_key>
  环境变量: TIANSHU_TOKEN / TIANSHU_USER / TIANSHU_PASS
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    sys.exit("❌ 缺少依赖: pip install requests")

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

# ─────────────────────────── 配置默认值 ───────────────────────────

DEFAULTS = {
    "base_url": "http://localhost:8000",
    "vllm_base_port": 30025,
    "vllm_count": 8,
    "worker_base_port": 8101,
    "worker_count": 8,
}

STATUS_COLOR = {
    "completed": "green",
    "processing": "cyan",
    "pending": "yellow",
    "failed": "red",
    "unknown": "dim",
}

# ─────────────────────────── HTTP 工具 ───────────────────────────

class APIClient:
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def login(self, username: str, password: str) -> str:
        resp = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        resp.raise_for_status()
        self.token = resp.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        return self.token

    def get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        try:
            resp = self.session.get(
                f"{self.base_url}{path}", params=params, timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return None
        except Exception as e:
            return {"_error": str(e)}

    def probe(self, url: str) -> bool:
        """检查一个 URL 是否可达（不带认证）"""
        try:
            r = requests.get(url, timeout=3)
            return r.status_code < 500
        except Exception:
            return False

# ─────────────────────────── 数据获取 ───────────────────────────

def fetch_health(client: APIClient) -> Optional[dict]:
    return client.get("/api/v1/health")

def fetch_queue_stats(client: APIClient) -> Optional[dict]:
    r = client.get("/api/v1/queue/stats")
    return r.get("stats") if r else None

def fetch_tasks(client: APIClient, status: Optional[str] = None,
                search: Optional[str] = None, page: int = 1,
                page_size: int = 30, backend: Optional[str] = None) -> Optional[dict]:
    params: Dict[str, Any] = {"page": page, "page_size": page_size}
    if status:
        params["status"] = status
    if search:
        params["search"] = search
    if backend:
        params["backend"] = backend
    return client.get("/api/v1/queue/tasks", params=params)

def fetch_task_detail(client: APIClient, task_id: str) -> Optional[dict]:
    return client.get(f"/api/v1/tasks/{task_id}")

def probe_vllm(cfg: dict) -> List[bool]:
    results = []
    for i in range(cfg["vllm_count"]):
        host = cfg["base_url"].rsplit(":", 1)[0]
        port = cfg["vllm_base_port"] + i
        ok = requests.get(f"{host}:{port}/v1/models", timeout=2).status_code == 200 \
            if True else False
        try:
            ok = requests.get(f"{host}:{port}/v1/models", timeout=2).status_code == 200
        except Exception:
            ok = False
        results.append(ok)
    return results

def probe_workers(cfg: dict) -> List[bool]:
    results = []
    for i in range(cfg["worker_count"]):
        host = cfg["base_url"].rsplit(":", 1)[0]
        port = cfg["worker_base_port"] + i
        try:
            ok = requests.get(f"{host}:{port}/health", timeout=2).status_code < 500
        except Exception:
            ok = False
        results.append(ok)
    return results

# ─────────────────────────── 渲染工具 ───────────────────────────

def _c(text: str, color: str) -> str:
    """ANSI 颜色（fallback when no rich）"""
    codes = {"green": 32, "red": 31, "yellow": 33, "cyan": 36, "dim": 2, "bold": 1}
    code = codes.get(color, 0)
    return f"\033[{code}m{text}\033[0m"

def ok_x(flag: bool) -> str:
    return ("✅" if flag else "❌")

def status_badge(s: str) -> str:
    icons = {"completed": "✅", "processing": "⚙️ ", "pending": "⏳", "failed": "❌"}
    return f"{icons.get(s, '❓')} {s}"

def fmt_dt(dt_str: Optional[str]) -> str:
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M:%S")
    except Exception:
        return dt_str[:16]

def fmt_elapsed(started: Optional[str], ended: Optional[str] = None) -> str:
    if not started:
        return "—"
    try:
        s = datetime.fromisoformat(started.replace("Z", ""))
        e = datetime.fromisoformat(ended.replace("Z", "")) if ended else datetime.now()
        secs = int((e - s).total_seconds())
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs//60}m{secs%60:02d}s"
        else:
            return f"{secs//3600}h{(secs%3600)//60:02d}m"
    except Exception:
        return "—"

# ─────────────────────────── 命令：status ────────────────────────

def cmd_status(client: APIClient, cfg: dict):
    health = fetch_health(client)
    stats = fetch_queue_stats(client)
    vllm_status = probe_vllm(cfg)
    worker_status = probe_workers(cfg)

    if HAS_RICH:
        _render_status_rich(health, stats, vllm_status, worker_status, cfg)
    else:
        _render_status_plain(health, stats, vllm_status, worker_status, cfg)

def _render_status_rich(health, stats, vllm_status, worker_status, cfg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── API 健康 ──
    api_url = cfg["base_url"]
    api_ok = health and health.get("status") == "healthy"
    api_text = Text()
    api_text.append(f"  {'✅' if api_ok else '❌'} API Server  ", style="bold")
    api_text.append(f"{api_url}  ", style="dim")
    db_ok = health and health.get("database") == "connected"
    api_text.append(f"DB: {'✅' if db_ok else '❌'}  ", style="green" if db_ok else "red")
    console.print(Panel(api_text, title=f"[bold cyan]MinerU Tianshu  {now}[/]", border_style="cyan"))

    # ── VLLM ──
    vllm_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    vllm_table.add_column("NPU", width=5)
    vllm_table.add_column("Port", width=7)
    vllm_table.add_column("状态", width=10)
    ok_count = sum(vllm_status)
    for i, ok in enumerate(vllm_status):
        port = cfg["vllm_base_port"] + i
        vllm_table.add_row(
            str(i),
            str(port),
            Text("● 运行中", style="green") if ok else Text("● 离线", style="red"),
        )
    console.print(Panel(vllm_table,
        title=f"[bold magenta]VLLM 服务  {ok_count}/{cfg['vllm_count']}[/]",
        border_style="magenta"))

    # ── Workers ──
    worker_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold blue")
    worker_table.add_column("Worker", width=8)
    worker_table.add_column("Port", width=7)
    worker_table.add_column("状态", width=10)
    wok = sum(worker_status)
    for i, ok in enumerate(worker_status):
        port = cfg["worker_base_port"] + i
        worker_table.add_row(
            f"#{i}",
            str(port),
            Text("● 就绪", style="green") if ok else Text("● 离线", style="red"),
        )
    console.print(Panel(worker_table,
        title=f"[bold blue]Workers  {wok}/{cfg['worker_count']}[/]",
        border_style="blue"))

    # ── 队列 ──
    if stats:
        q = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        q.add_column("key", style="dim")
        q.add_column("val", style="bold")
        redis_on = stats.get("_redis_enabled", False)
        q.add_row("pending",    Text(str(stats.get("pending", 0)), style="yellow"))
        q.add_row("processing", Text(str(stats.get("processing", 0)), style="cyan"))
        q.add_row("completed",  Text(str(stats.get("completed", 0)), style="green"))
        q.add_row("failed",     Text(str(stats.get("failed", 0)), style="red"))
        q.add_row("Redis",      Text(f"{'启用' if redis_on else '未启用'} | pending={stats.get('_redis_pending',0)} processing={stats.get('_redis_processing',0)}", style="dim"))
        console.print(Panel(q, title="[bold green]队列统计[/]", border_style="green"))

def _render_status_plain(health, stats, vllm_status, worker_status, cfg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  MinerU Tianshu  {now}")
    print(f"{'='*60}")

    api_url = cfg["base_url"]
    api_ok = health and health.get("status") == "healthy"
    print(f"\n{'API Server':20} {ok_x(api_ok)}  {api_url}")

    ok_count = sum(vllm_status)
    print(f"\nVLLM ({ok_count}/{cfg['vllm_count']}):")
    for i, ok in enumerate(vllm_status):
        port = cfg["vllm_base_port"] + i
        print(f"  NPU {i}  port {port}  {ok_x(ok)}")

    wok = sum(worker_status)
    print(f"\nWorkers ({wok}/{cfg['worker_count']}):")
    for i, ok in enumerate(worker_status):
        port = cfg["worker_base_port"] + i
        print(f"  #{i}  port {port}  {ok_x(ok)}")

    if stats:
        print(f"\n队列统计:")
        for k in ("pending", "processing", "completed", "failed"):
            print(f"  {k:12} {stats.get(k, 0)}")

# ─────────────────────────── 命令：queue ─────────────────────────

def cmd_queue(client: APIClient):
    stats = fetch_queue_stats(client)
    if not stats:
        sys.exit("❌ 无法获取队列统计（API 不可达或未登录）")

    if HAS_RICH:
        table = Table(title="队列统计", box=box.ROUNDED, show_lines=True)
        table.add_column("状态", style="bold", width=14)
        table.add_column("数量", justify="right", width=8)
        table.add_column("说明", style="dim")

        descs = {
            "pending": "等待处理",
            "processing": "处理中",
            "completed": "已完成",
            "failed": "失败",
        }
        colors = STATUS_COLOR
        for k, desc in descs.items():
            v = stats.get(k, 0)
            table.add_row(
                Text(k, style=colors.get(k, "")),
                Text(str(v), style=colors.get(k, "")),
                desc,
            )

        redis_on = stats.get("_redis_enabled", False)
        table.add_row("", "", "")
        table.add_row(
            Text("Redis", style="dim"),
            Text("启用" if redis_on else "未启用", style="green" if redis_on else "red"),
            f"pending={stats.get('_redis_pending',0)}  processing={stats.get('_redis_processing',0)}",
        )
        console.print(table)
    else:
        print(f"\n{'─'*30}")
        print(f"{'状态':<14}{'数量':>6}")
        print(f"{'─'*30}")
        for k in ("pending", "processing", "completed", "failed"):
            print(f"{k:<14}{stats.get(k, 0):>6}")
        redis_on = stats.get("_redis_enabled", False)
        print(f"\nRedis: {'启用' if redis_on else '未启用'}  "
              f"pending={stats.get('_redis_pending',0)}  processing={stats.get('_redis_processing',0)}")

# ─────────────────────────── 命令：tasks ─────────────────────────

def cmd_tasks(client: APIClient, status: Optional[str], search: Optional[str],
              page: int, page_size: int, backend: Optional[str]):
    data = fetch_tasks(client, status=status, search=search,
                       page=page, page_size=page_size, backend=backend)
    if not data:
        sys.exit("❌ 无法获取任务列表")
    if "_error" in data:
        sys.exit(f"❌ {data['_error']}")

    total = data.get("total", 0)
    tasks = data.get("tasks", [])
    page_n = data.get("page", page)
    page_sz = data.get("page_size", page_size)
    title_parts = [f"任务列表  共 {total} 条"]
    if status:
        title_parts.append(f"status={status}")
    if search:
        title_parts.append(f"search={search}")
    title_parts.append(f"第 {page_n} 页 / 每页 {page_sz}")
    title = "  ".join(title_parts)

    if HAS_RICH:
        table = Table(title=title, box=box.ROUNDED, show_lines=False, expand=True)
        table.add_column("task_id",   width=10, style="dim")
        table.add_column("文件名",    min_width=20, max_width=40)
        table.add_column("状态",      width=12)
        table.add_column("引擎",      width=20, style="dim")
        table.add_column("创建时间",  width=14)
        table.add_column("耗时",      width=8, justify="right")
        table.add_column("错误",      min_width=10, max_width=35, style="red dim")

        for t in tasks:
            s = t.get("status", "unknown")
            elapsed = fmt_elapsed(t.get("started_at"), t.get("completed_at"))
            table.add_row(
                t.get("task_id", "")[:8],
                t.get("file_name", ""),
                Text(status_badge(s), style=STATUS_COLOR.get(s, "")),
                t.get("backend", ""),
                fmt_dt(t.get("created_at")),
                elapsed,
                (t.get("error_message") or "")[:60],
            )
        console.print(table)

        if total > page_n * page_sz:
            console.print(f"[dim]  还有更多，使用 --page {page_n+1} 查看下一页[/]")
    else:
        print(f"\n{title}")
        print(f"{'─'*100}")
        fmt = f"{'task_id':<10}  {'file_name':<38}  {'status':<12}  {'created_at':<16}  {'elapsed':>8}  error"
        print(fmt)
        print(f"{'─'*100}")
        for t in tasks:
            elapsed = fmt_elapsed(t.get("started_at"), t.get("completed_at"))
            err = (t.get("error_message") or "")[:50]
            print(f"{t.get('task_id','')[:8]:<10}  {t.get('file_name',''):<38}  "
                  f"{t.get('status',''):<12}  {fmt_dt(t.get('created_at')):<16}  "
                  f"{elapsed:>8}  {err}")
        if total > page_n * page_sz:
            print(f"\n  还有更多，使用 --page {page_n+1} 查看下一页")

# ─────────────────────────── 命令：task（单任务）─────────────────

def cmd_task_detail(client: APIClient, task_id: str):
    data = fetch_task_detail(client, task_id)
    if not data:
        sys.exit(f"❌ 任务不存在或无权访问: {task_id}")
    if "_error" in data:
        sys.exit(f"❌ {data['_error']}")

    s = data.get("status", "unknown")

    if HAS_RICH:
        rows = [
            ("task_id",     data.get("task_id", "")),
            ("file_name",   data.get("file_name", "")),
            ("状态",        Text(status_badge(s), style=STATUS_COLOR.get(s, ""))),
            ("引擎",        data.get("backend", "")),
            ("优先级",      str(data.get("priority", 0))),
            ("创建时间",    fmt_dt(data.get("created_at"))),
            ("开始时间",    fmt_dt(data.get("started_at"))),
            ("完成时间",    fmt_dt(data.get("completed_at"))),
            ("耗时",        fmt_elapsed(data.get("started_at"), data.get("completed_at"))),
            ("worker_id",   data.get("worker_id") or "—"),
            ("retry_count", str(data.get("retry_count", 0))),
            ("result_path", data.get("result_path") or "—"),
            ("error",       Text(data.get("error_message") or "—", style="red") if data.get("error_message") else "—"),
        ]
        if data.get("is_parent"):
            prog = data.get("subtask_progress", {})
            rows.append(("子任务进度",
                f"{prog.get('completed',0)}/{prog.get('total',0)}  ({prog.get('percentage',0)}%)"))

        table = Table(box=box.ROUNDED, show_header=False, title=f"任务详情  {task_id[:8]}")
        table.add_column("字段", style="dim", width=14)
        table.add_column("值")
        for k, v in rows:
            table.add_row(k, v)
        console.print(table)
    else:
        print(f"\n{'─'*60}")
        print(f"任务详情: {task_id}")
        print(f"{'─'*60}")
        for k, v in [
            ("file_name",   data.get("file_name", "")),
            ("status",      s),
            ("backend",     data.get("backend", "")),
            ("created_at",  fmt_dt(data.get("created_at"))),
            ("started_at",  fmt_dt(data.get("started_at"))),
            ("completed_at",fmt_dt(data.get("completed_at"))),
            ("耗时",        fmt_elapsed(data.get("started_at"), data.get("completed_at"))),
            ("worker_id",   data.get("worker_id") or "—"),
            ("retry_count", str(data.get("retry_count", 0))),
            ("result_path", data.get("result_path") or "—"),
            ("error",       data.get("error_message") or "—"),
        ]:
            print(f"  {k:<14} {v}")

# ─────────────────────────── 命令：watch ─────────────────────────

def _build_watch_panel(client: APIClient, cfg: dict) -> Any:
    """构建 watch 模式的渲染内容"""
    from rich.columns import Columns
    from rich.layout import Layout

    health = fetch_health(client)
    stats = fetch_queue_stats(client)
    api_ok = health and health.get("status") == "healthy"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 服务状态行
    svc_parts = []
    svc_parts.append(Text(f"{'✅' if api_ok else '❌'} API  ", style="green" if api_ok else "red"))
    if stats:
        for k, color in [("pending","yellow"),("processing","cyan"),("completed","green"),("failed","red")]:
            v = stats.get(k, 0)
            svc_parts.append(Text(f"{k}={v}  ", style=color if v > 0 else "dim"))
        redis_pend = stats.get("_redis_pending", 0)
        redis_proc = stats.get("_redis_processing", 0)
        svc_parts.append(Text(f"Redis(q={redis_pend} p={redis_proc})", style="dim"))

    svc_line = Text.assemble(*svc_parts)

    # 处理中任务
    data = fetch_tasks(client, status="processing", page_size=20)
    tasks = data.get("tasks", []) if data else []

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan",
                  title=None, expand=True, padding=(0,1))
    table.add_column("task_id",  width=10, style="dim")
    table.add_column("文件名",   min_width=24, max_width=42)
    table.add_column("引擎",     width=20, style="dim")
    table.add_column("开始",     width=14)
    table.add_column("已耗时",   width=8, justify="right")
    table.add_column("worker",   width=36, style="dim")

    if tasks:
        for t in tasks:
            table.add_row(
                t.get("task_id","")[:8],
                t.get("file_name",""),
                t.get("backend",""),
                fmt_dt(t.get("started_at")),
                fmt_elapsed(t.get("started_at")),
                (t.get("worker_id") or "")[-30:],
            )
    else:
        table.add_row("[dim]—[/]", "[dim]队列空闲，无处理中任务[/]", "", "", "", "")

    # 最近失败
    failed_data = fetch_tasks(client, status="failed", page_size=5)
    failed_tasks = failed_data.get("tasks", []) if failed_data else []

    failed_table = Table(box=box.SIMPLE, show_header=True, header_style="bold red",
                         title=None, expand=True, padding=(0,1))
    failed_table.add_column("task_id", width=10, style="dim")
    failed_table.add_column("文件名",  min_width=24, max_width=36)
    failed_table.add_column("时间",    width=14)
    failed_table.add_column("错误",    min_width=20)

    if failed_tasks:
        for t in failed_tasks:
            failed_table.add_row(
                t.get("task_id","")[:8],
                t.get("file_name",""),
                fmt_dt(t.get("completed_at")),
                Text((t.get("error_message") or "")[:80], style="red"),
            )
    else:
        failed_table.add_row("[dim]—[/]", "[dim]无失败任务[/]", "", "")

    content = Text.assemble(svc_line)

    from rich.console import Group
    return Panel(
        Group(
            svc_line,
            Text(""),
            Panel(table, title=f"[cyan]处理中任务 ({len(tasks)})[/]", border_style="cyan", padding=(0,1)),
            Panel(failed_table, title="[red]最近失败（最多5条）[/]", border_style="red", padding=(0,1)),
        ),
        title=f"[bold cyan]MinerU Tianshu 监控  {now}[/]",
        border_style="cyan",
    )

def cmd_watch(client: APIClient, cfg: dict, interval: int):
    if not HAS_RICH:
        # fallback: 简单循环
        while True:
            os.system("clear")
            cmd_status(client, cfg)
            cmd_queue(client)
            print(f"\n[每 {interval}s 刷新，Ctrl+C 退出]")
            time.sleep(interval)
        return

    console.print(f"[dim]每 {interval}s 刷新  Ctrl+C 退出[/]")
    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    panel = _build_watch_panel(client, cfg)
                except Exception as e:
                    panel = Panel(f"[red]刷新出错: {e}[/]", border_style="red")
                live.update(panel)
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]已退出监控[/]")

# ─────────────────────────── 入口 ────────────────────────────────

def build_client(args) -> APIClient:
    base_url = args.base_url or os.environ.get("TIANSHU_URL", DEFAULTS["base_url"])
    token = args.token or os.environ.get("TIANSHU_TOKEN")
    username = args.username or os.environ.get("TIANSHU_USER")
    password = args.password or os.environ.get("TIANSHU_PASS")

    client = APIClient(base_url, token=token)

    if not token:
        if not username or not password:
            sys.exit("❌ 需要认证：提供 --token 或 --username/--password，或设置环境变量 TIANSHU_TOKEN / TIANSHU_USER+TIANSHU_PASS")
        try:
            client.login(username, password)
        except Exception as e:
            sys.exit(f"❌ 登录失败: {e}")

    return client

def main():
    parser = argparse.ArgumentParser(
        description="MinerU Tianshu 命令行监控工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url",  default=None, help="API地址 (默认 http://localhost:8000)")
    parser.add_argument("--username",  default=None, help="用户名")
    parser.add_argument("--password",  default=None, help="密码")
    parser.add_argument("--token",     default=None, help="JWT token 或 API Key")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # status
    sub.add_parser("status", help="服务总览（API / VLLM / Workers / 队列）")

    # queue
    sub.add_parser("queue", help="队列统计（各状态任务数）")

    # tasks
    p_tasks = sub.add_parser("tasks", help="任务列表")
    p_tasks.add_argument("--status",    default=None,
        choices=["pending","processing","completed","failed"],
        help="按状态筛选")
    p_tasks.add_argument("--search",    default=None, help="按文件名或 task_id 搜索")
    p_tasks.add_argument("--backend",   default=None, help="按引擎筛选")
    p_tasks.add_argument("--page",      type=int, default=1)
    p_tasks.add_argument("--page-size", type=int, default=30)

    # task (单任务)
    p_task = sub.add_parser("task", help="单任务详情")
    p_task.add_argument("task_id", help="task_id")

    # watch
    p_watch = sub.add_parser("watch", help="持续监控（实时刷新）")
    p_watch.add_argument("--interval", type=int, default=30, help="刷新间隔秒数（默认30）")

    args = parser.parse_args()
    client = build_client(args)

    cfg = {
        "vllm_base_port":  int(os.environ.get("VLLM_BASE_PORT",  DEFAULTS["vllm_base_port"])),
        "vllm_count":      int(os.environ.get("VLLM_COUNT",      DEFAULTS["vllm_count"])),
        "worker_base_port":int(os.environ.get("WORKER_BASE_PORT",DEFAULTS["worker_base_port"])),
        "worker_count":    int(os.environ.get("WORKER_COUNT",    DEFAULTS["worker_count"])),
    }

    if args.cmd == "status":
        cmd_status(client, cfg)
    elif args.cmd == "queue":
        cmd_queue(client)
    elif args.cmd == "tasks":
        cmd_tasks(client,
            status=args.status, search=args.search,
            page=args.page, page_size=args.page_size,
            backend=args.backend)
    elif args.cmd == "task":
        cmd_task_detail(client, args.task_id)
    elif args.cmd == "watch":
        cmd_watch(client, cfg, args.interval)

if __name__ == "__main__":
    main()
