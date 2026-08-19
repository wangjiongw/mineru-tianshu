#!/usr/bin/env python3
"""
Read-only local performance evaluator for an 8-card MinerU Tianshu deployment.

The evaluator probes local health/metrics endpoints, npu-smi, /proc/stat, SQLite
task state, and logs without mutating service state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/share/wangjiong/databases/mineru_database")


def default_instance_id() -> str:
    return os.getenv("INSTANCE_ID") or socket.gethostname()


def default_instance_data_dir() -> Path:
    return DEFAULT_DATA_ROOT / default_instance_id()


def default_db_path() -> Path:
    return Path(os.getenv("DATABASE_PATH", str(default_instance_data_dir() / "mineru_tianshu.db")))


def default_log_paths() -> list[Path]:
    if os.getenv("LOG_DIR"):
        return [Path(os.environ["LOG_DIR"])]
    return [default_instance_data_dir() / "mineru_logs"]

DEFAULT_WORKER_PORTS = list(range(8101, 8109))
DEFAULT_VLLM_PORTS = list(range(30025, 30033))
EXPECTED_WORKER_COUNT = 8
EXPECTED_VLLM_COUNT = 8
DB_LOCK_TERMINAL_RE = re.compile(
    r"(?:\b(?:terminal|failed|failure|mark(?:ed)?\s+failed|status[=: ]+failed)\b.{0,240}database\s+is\s+locked)"
    r"|(?:database\s+is\s+locked.{0,240}\b(?:terminal|failed|failure|mark(?:ed)?\s+failed|status[=: ]+failed)\b)",
    re.IGNORECASE | re.DOTALL,
)
OOM_PREEMPTION_RE = re.compile(
    r"\b(?:oom|out\s+of\s+memory|memory\s+allocation\s+fail(?:ed|ure)?|acl_error_rt_memory_allocation|preempt(?:ion|ed)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Unknown:
    reason: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def status_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Unknown):
        return {"status": "unknown", "reason": value.reason}
    return {"status": "known", "value": value}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def http_get(url: str, timeout: float = 2.0) -> dict[str, Any]:
    req = Request(url, method="GET", headers={"User-Agent": "tianshu-local-efficiency-evaluator/1"})
    started = time.monotonic()
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read(256 * 1024).decode("utf-8", errors="replace")
            return {
                "ok": 200 <= resp.status < 300,
                "status_code": resp.status,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "body": body,
                "error": None,
            }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "body": "",
            "error": str(exc),
        }


def service_health_from_probe(probe: dict[str, Any]) -> str:
    if not probe["ok"]:
        return "unhealthy"
    body = probe.get("body") or ""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {}
    status = str(parsed.get("status", "")).lower()
    if status and status not in {"ok", "healthy", "running"}:
        return "unhealthy"
    return "healthy"


def collect_worker_health(host: str, ports: list[int]) -> list[dict[str, Any]]:
    workers = []
    for port in ports:
        url = f"http://{host}:{port}/health"
        probe = http_get(url)
        workers.append(
            {
                "port": port,
                "url": url,
                "status": service_health_from_probe(probe),
                "http_status": probe["status_code"],
                "latency_ms": probe["latency_ms"],
                "error": probe["error"],
            }
        )
    return workers


def collect_vllm(host: str, ports: list[int]) -> list[dict[str, Any]]:
    services = []
    for port in ports:
        health_url = f"http://{host}:{port}/health"
        metrics_url = f"http://{host}:{port}/metrics"
        health = http_get(health_url)
        metrics = http_get(metrics_url)
        services.append(
            {
                "port": port,
                "health_url": health_url,
                "metrics_url": metrics_url,
                "status": service_health_from_probe(health),
                "health_http_status": health["status_code"],
                "health_error": health["error"],
                "metrics_status": "known" if metrics["ok"] else "unknown",
                "metrics_http_status": metrics["status_code"],
                "metrics_error": metrics["error"],
                "metrics_summary": parse_vllm_metrics(metrics.get("body", "")) if metrics["ok"] else {},
            }
        )
    return services


def parse_vllm_metrics(text: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    wanted = {
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
        "vllm:kv_cache_usage_perc": "kv_cache_usage_percent",
        "vllm:num_preemptions_total": "num_preemptions_total",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = wanted.get(name)
        if not key:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
            if key == "kv_cache_usage_percent" and value <= 1.0:
                value *= 100.0
            summary[key] = value
        except (IndexError, ValueError):
            continue
    return summary


def parse_npu_smi(text: str) -> list[dict[str, float]]:
    cards: dict[int, dict[str, float]] = {}
    current_card: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|"):
            cols = [part.strip() for part in line.strip("|").split("|")]
            if len(cols) >= 2:
                first_match = re.match(r"\s*(\d+)", cols[0])
                first = int(first_match.group(1)) if first_match else -1
                if 0 <= first < 8 and ":" not in cols[1] and cols[1] != "0":
                    current_card = first
                    cards.setdefault(first, {"card_id": float(first)})
                    continue
                if current_card is not None and len(cols) >= 3 and ":" in cols[1]:
                    card = cards.setdefault(current_card, {"card_id": float(current_card)})
                    numeric_tail = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", " ".join(cols[2:]))]
                    if numeric_tail:
                        card["util_percent"] = numeric_tail[0]
                    if len(numeric_tail) >= 5 and numeric_tail[-1] > 0:
                        card["hbm_percent"] = numeric_tail[-2] * 100.0 / numeric_tail[-1]
                    continue

        lower = line.lower()
        if "npu-smi" in lower or "version" in lower:
            continue
        numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", line)]
        if not numbers:
            continue
        card_id = int(numbers[0])
        if not 0 <= card_id < 8:
            continue
        card = cards.setdefault(card_id, {"card_id": float(card_id)})
        if "hbm" in lower or "memory" in lower:
            if len(numbers) >= 2:
                card["hbm_percent"] = numbers[-1]
        if "util" in lower or "aicore" in lower or "ai core" in lower:
            if len(numbers) >= 2:
                card["util_percent"] = numbers[-1]
    return [cards[key] for key in sorted(cards)]


def collect_npu_smi() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {"status": "unknown", "reason": "npu-smi command not found", "cards": []}
    except Exception as exc:
        return {"status": "unknown", "reason": str(exc), "cards": []}
    if proc.returncode != 0:
        return {"status": "unknown", "reason": (proc.stderr or proc.stdout).strip(), "cards": []}
    return {"status": "known", "cards": parse_npu_smi(proc.stdout), "reason": None}


def read_cpu_times(path: Path = Path("/proc/stat")) -> dict[str, int] | Unknown:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except Exception as exc:
        return Unknown(str(exc))
    parts = first.split()
    if not parts or parts[0] != "cpu":
        return Unknown("missing aggregate cpu line")
    fields = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice"]
    values = [int(value) for value in parts[1:]]
    return dict(zip(fields, values))


def cpu_delta(start: dict[str, int] | Unknown, end: dict[str, int] | Unknown) -> dict[str, Any]:
    if isinstance(start, Unknown):
        return status_value(start)
    if isinstance(end, Unknown):
        return status_value(end)
    total_delta = sum(end.values()) - sum(start.values())
    if total_delta <= 0:
        return status_value(Unknown("non-positive cpu sample delta"))
    idle_delta = end.get("idle", 0) - start.get("idle", 0)
    iowait_delta = end.get("iowait", 0) - start.get("iowait", 0)
    return status_value(
        {
            "idle_percent": round(idle_delta * 100.0 / total_delta, 3),
            "iowait_percent": round(iowait_delta * 100.0 / total_delta, 3),
        }
    )


def sqlite_counts(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"status": "unknown", "reason": f"database not found: {db_path}", "counts": {}}
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return {"status": "unknown", "reason": str(exc), "counts": {}}
    counts = {row["status"]: int(row["count"]) for row in rows}
    return {"status": "known", "reason": None, "counts": counts}


def discover_log_files(paths: list[Path]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.is_file():
            files.append(path)
    return files, missing


def snapshot_log_files(paths: list[Path]) -> dict[str, Any]:
    files, missing = discover_log_files(paths)
    snapshot: dict[str, Any] = {"status": "known", "files": {}, "missing_paths": missing, "errors": []}
    if not files and missing:
        snapshot["status"] = "unknown"
        snapshot["reason"] = "no log paths found"
    for file_path in files:
        try:
            stat = file_path.stat()
        except OSError as exc:
            snapshot["errors"].append({"path": str(file_path), "error": str(exc)})
            continue
        snapshot["files"][str(file_path)] = {
            "path": str(file_path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
        }
    return snapshot


def count_pattern_in_bytes(data: bytes, pattern: re.Pattern[str]) -> int:
    text = data.decode("utf-8", errors="ignore")
    return len(pattern.findall(text))


def count_pattern_in_range(file_path: Path, offset: int, pattern: re.Pattern[str]) -> int:
    count = 0
    with file_path.open("rb") as handle:
        handle.seek(max(offset, 0))
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            count += count_pattern_in_bytes(chunk, pattern)
    return count


def count_log_delta(paths: list[Path], start_snapshot: dict[str, Any], pattern: re.Pattern[str]) -> dict[str, Any]:
    if start_snapshot["status"] != "known":
        return {"status": "unknown", "reason": start_snapshot.get("reason", "log snapshot unknown"), "count": None, "files": []}
    files, missing = discover_log_files(paths)
    if not files and missing:
        return {"status": "unknown", "reason": "no log paths found", "count": None, "files": []}

    count = 0
    scanned: list[dict[str, Any]] = []
    start_files = start_snapshot.get("files", {})
    for file_path in files:
        path_key = str(file_path)
        try:
            stat = file_path.stat()
        except OSError as exc:
            scanned.append({"path": path_key, "status": "unknown", "reason": str(exc)})
            continue
        previous = start_files.get(path_key)
        offset = 0
        mode = "new"
        if previous:
            same_file = previous["device"] == stat.st_dev and previous["inode"] == stat.st_ino
            if same_file and stat.st_size >= previous["size"]:
                offset = int(previous["size"])
                mode = "append"
            elif same_file:
                mode = "truncated"
            else:
                mode = "rotated"
        try:
            file_count = count_pattern_in_range(file_path, offset, pattern)
        except OSError as exc:
            scanned.append({"path": path_key, "status": "unknown", "reason": str(exc), "mode": mode})
            continue
        count += file_count
        scanned.append({"path": path_key, "mode": mode, "offset": offset, "size": stat.st_size, "count": file_count})
    return {"status": "known", "reason": None, "count": count, "files": scanned, "missing_paths": missing}



def summarize_log_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot["status"] != "known":
        return {"status": snapshot["status"], "reason": snapshot.get("reason")}
    sizes = [item["size"] for item in snapshot.get("files", {}).values()]
    return {
        "status": "known",
        "file_count": len(sizes),
        "total_size_bytes": sum(sizes),
        "missing_paths": snapshot.get("missing_paths", []),
        "error_count": len(snapshot.get("errors", [])),
    }


def summarize_log_delta(delta: dict[str, Any]) -> dict[str, Any]:
    if delta["status"] != "known":
        return {"status": delta["status"], "reason": delta.get("reason"), "count": None}
    files = delta.get("files", [])
    mode_counts: dict[str, int] = {}
    bytes_scanned = 0
    matches = []
    errors = []
    for item in files:
        mode = item.get("mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if "size" in item:
            bytes_scanned += max(0, int(item["size"]) - int(item.get("offset", 0)))
        if item.get("count"):
            matches.append({"path": item["path"], "count": item["count"], "mode": mode})
        if item.get("status") == "unknown":
            errors.append({"path": item["path"], "reason": item.get("reason")})
    return {
        "status": "known",
        "reason": None,
        "count": delta["count"],
        "file_count": len(files),
        "bytes_scanned": bytes_scanned,
        "mode_counts": mode_counts,
        "matches": matches,
        "errors": errors,
        "missing_paths": delta.get("missing_paths", []),
    }

def count_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    if start["status"] != "known":
        return {"status": "unknown", "reason": start["reason"], "delta": None}
    if end["status"] != "known":
        return {"status": "unknown", "reason": end["reason"], "delta": None}
    return {"status": "known", "reason": None, "delta": end["count"] - start["count"]}


def task_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    if start["status"] != "known":
        return {"status": "unknown", "reason": start["reason"]}
    if end["status"] != "known":
        return {"status": "unknown", "reason": end["reason"]}
    start_counts = start["counts"]
    end_counts = end["counts"]
    completed = end_counts.get("completed", 0) - start_counts.get("completed", 0)
    failed = end_counts.get("failed", 0) - start_counts.get("failed", 0)
    total_terminal = completed + failed
    failed_rate = None if total_terminal <= 0 else failed / total_terminal
    return {
        "status": "known",
        "completed_delta": completed,
        "failed_delta": failed,
        "terminal_delta": total_terminal,
        "failed_rate": failed_rate,
    }



def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0



def population_cv(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def summarize_vllm_metrics(vllm_samples: list[list[dict[str, Any]]]) -> dict[str, Any]:
    per_port: dict[int, dict[str, Any]] = {}
    all_kv: list[float] = []
    missing_kv_ports: set[int] = set()
    missing_preemption_ports: set[int] = set()
    for sample in vllm_samples:
        for service in sample:
            port = int(service["port"])
            port_data = per_port.setdefault(port, {"kv_cache_usage_percent": [], "preemptions": []})
            metrics = service.get("metrics_summary", {})
            if service.get("metrics_status") != "known":
                missing_kv_ports.add(port)
                missing_preemption_ports.add(port)
                continue
            if "kv_cache_usage_percent" in metrics:
                value = float(metrics["kv_cache_usage_percent"])
                port_data["kv_cache_usage_percent"].append(value)
                all_kv.append(value)
            else:
                missing_kv_ports.add(port)
            if "num_preemptions_total" in metrics:
                port_data["preemptions"].append(float(metrics["num_preemptions_total"]))
            else:
                missing_preemption_ports.add(port)

    port_rows = []
    total_preemption_delta = 0.0
    preemption_unknown = False
    for port in sorted(per_port):
        kv_values = per_port[port]["kv_cache_usage_percent"]
        preemptions = per_port[port]["preemptions"]
        preemption_delta = None
        if len(preemptions) >= 2:
            preemption_delta = preemptions[-1] - preemptions[0]
            total_preemption_delta += preemption_delta
        else:
            preemption_unknown = True
        port_rows.append(
            {
                "port": port,
                "kv_cache_p99_percent": percentile(kv_values, 0.99),
                "kv_cache_max_percent": max(kv_values) if kv_values else None,
                "preemption_delta": None if preemption_delta is None else int(preemption_delta),
                "samples": max(len(kv_values), len(preemptions)),
            }
        )

    kv_status = "known" if all_kv and not missing_kv_ports else "unknown"
    preemption_status = "known" if per_port and not missing_preemption_ports and not preemption_unknown else "unknown"
    return {
        "kv_cache": {
            "status": kv_status,
            "reason": None if kv_status == "known" else f"missing kv_cache_usage_perc for ports: {sorted(missing_kv_ports)}",
            "p99_percent": percentile(all_kv, 0.99),
            "max_percent": max(all_kv) if all_kv else None,
        },
        "preemptions": {
            "status": preemption_status,
            "reason": None if preemption_status == "known" else f"missing num_preemptions_total for ports: {sorted(missing_preemption_ports)}",
            "delta": int(total_preemption_delta) if preemption_status == "known" else None,
        },
        "ports": port_rows,
    }

def build_throughput_windows(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for index in range(1, len(snapshots)):
        start = snapshots[index - 1]
        end = snapshots[index]
        duration = end["elapsed_seconds"] - start["elapsed_seconds"]
        start_counts = start["snapshot"]
        end_counts = end["snapshot"]
        if duration <= 0 or start_counts["status"] != "known" or end_counts["status"] != "known":
            windows.append(
                {
                    "index": index,
                    "status": "unknown",
                    "duration_seconds": round(duration, 3),
                    "reason": "task snapshot unavailable or non-positive duration",
                    "completed_delta": None,
                    "successful_per_minute": None,
                }
            )
            continue
        completed = end_counts["counts"].get("completed", 0) - start_counts["counts"].get("completed", 0)
        windows.append(
            {
                "index": index,
                "status": "known",
                "start_elapsed_seconds": round(start["elapsed_seconds"], 3),
                "end_elapsed_seconds": round(end["elapsed_seconds"], 3),
                "duration_seconds": round(duration, 3),
                "completed_delta": completed,
                "successful_per_minute": round(completed / duration * 60.0, 6),
            }
        )
    return windows


def compute_throughput(
    task_start: dict[str, Any],
    task_end: dict[str, Any],
    elapsed_seconds: float,
    window_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    windows = build_throughput_windows(window_snapshots)
    known_window_rates = [window["successful_per_minute"] for window in windows if window["status"] == "known"]
    if elapsed_seconds >= 1800 and len(known_window_rates) >= 6:
        value = median(known_window_rates[:6])
        return {
            "successful_per_minute": None if value is None else round(value, 6),
            "method": "median_6x5m_windows",
            "window_seconds": 300,
            "windows_required": 6,
            "windows": windows,
        }

    delta = task_delta(task_start, task_end)
    completed_delta = delta.get("completed_delta") if delta["status"] == "known" else None
    value = None if completed_delta is None else completed_delta / elapsed_seconds * 60.0
    return {
        "successful_per_minute": None if value is None else round(value, 6),
        "method": "aggregate_short_run",
        "reason": "duration < 1800 seconds or fewer than 6 known 5-minute windows",
        "windows": windows,
    }

def load_baseline(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "unknown", "reason": "baseline path not provided"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unknown", "reason": str(exc)}
    throughput = (
        data.get("throughput", {}).get("successful_per_minute")
        or data.get("successful_throughput_per_minute")
        or data.get("successful_per_minute")
    )
    try:
        throughput = float(throughput)
    except (TypeError, ValueError):
        return {"status": "unknown", "reason": "baseline missing successful throughput"}
    return {
        "status": "known",
        "path": str(path),
        "successful_per_minute": throughput,
        "method": data.get("throughput", {}).get("method") or data.get("method") or "unknown",
        "windows": data.get("throughput", {}).get("windows", data.get("windows", [])),
    }



def baseline_from_result(result: dict[str, Any]) -> dict[str, Any]:
    throughput = result["throughput"]["successful_per_minute"]
    if throughput is None:
        return {"status": "unknown", "reason": "successful throughput unknown"}
    return {
        "schema_version": 1,
        "status": "known",
        "created_at": result["ended_at"],
        "source_duration_seconds": result["duration_seconds"],
        "throughput": {
            "successful_per_minute": throughput,
            "method": result["throughput"].get("method"),
            "windows": result["throughput"].get("windows", []),
        },
    }

def evaluate(result: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    worker_rows = result["services"]["workers"]
    vllm_rows = result["services"]["vllm"]
    unhealthy_workers = [row["port"] for row in worker_rows if row["status"] != "healthy"]
    unhealthy_vllm = [row["port"] for row in vllm_rows if row["status"] != "healthy"]
    worker_ports = {row["port"] for row in worker_rows}
    vllm_ports = {row["port"] for row in vllm_rows}
    services_passed = (
        len(worker_rows) == EXPECTED_WORKER_COUNT
        and len(worker_ports) == EXPECTED_WORKER_COUNT
        and len(vllm_rows) == EXPECTED_VLLM_COUNT
        and len(vllm_ports) == EXPECTED_VLLM_COUNT
        and not unhealthy_workers
        and not unhealthy_vllm
    )
    checks["services_healthy"] = {
        "passed": services_passed,
        "reason": (
            "8 workers and 8 vLLMs healthy"
            if services_passed
            else (
                f"workers={len(worker_rows)}/8 unique={len(worker_ports)}/8 "
                f"unhealthy={unhealthy_workers}; vllms={len(vllm_rows)}/8 "
                f"unique={len(vllm_ports)}/8 unhealthy={unhealthy_vllm}"
            )
        ),
    }

    baseline = result["baseline"]
    throughput = result["throughput"]["successful_per_minute"]
    if baseline["status"] != "known":
        checks["throughput_plus_20_percent"] = {"passed": False, "reason": baseline["reason"]}
    elif throughput is None:
        checks["throughput_plus_20_percent"] = {"passed": False, "reason": "successful throughput unknown"}
    else:
        required = baseline["successful_per_minute"] * 1.2
        checks["throughput_plus_20_percent"] = {
            "passed": throughput >= required,
            "reason": f"{throughput:.6g} >= required {required:.6g}",
        }

    failed_rate = result["tasks"]["delta"].get("failed_rate") if result["tasks"]["delta"]["status"] == "known" else None
    checks["failed_rate_lte_1_percent"] = {
        "passed": failed_rate is not None and failed_rate <= 0.01,
        "reason": "failed rate unknown" if failed_rate is None else f"{failed_rate:.6g} <= 0.01",
    }

    db_lock_evidence = result["db_lock_evidence"]
    locked = db_lock_evidence.get("combined_delta")
    checks["db_lock_terminal_failures_zero"] = {
        "passed": locked == 0 and db_lock_evidence["status"] == "known",
        "reason": db_lock_evidence.get("reason") if db_lock_evidence["status"] != "known" else f"{locked} == 0",
    }

    hbm_p99 = result["fleet"]["hbm_p99_percent"]
    checks["hbm_p99_lte_95_5_percent"] = {
        "passed": hbm_p99 is not None and hbm_p99 <= 95.5,
        "reason": "HBM p99 unknown" if hbm_p99 is None else f"{hbm_p99:.6g} <= 95.5",
    }

    hbm_max = result["fleet"].get("hbm_max_percent")
    diagnostics["hbm_max_lt_98_percent"] = {
        "available": hbm_max is not None,
        "meets_reference": hbm_max is not None and hbm_max < 98.0,
        "gating": False,
        "reason": "HBM max unknown" if hbm_max is None else f"{hbm_max:.6g} < 98",
    }

    util_avg = result["fleet"].get("util_avg_percent")
    diagnostics["fleet_aicore_util_avg_gte_35_percent"] = {
        "available": util_avg is not None,
        "meets_reference": util_avg is not None and util_avg >= 35.0,
        "gating": False,
        "reason": "fleet AICore util avg unknown" if util_avg is None else f"{util_avg:.6g} >= 35",
    }

    util_cv = result["fleet"].get("util_load_balance_cv")
    diagnostics["load_balance_cv_lte_0_30"] = {
        "available": util_cv is not None,
        "meets_reference": util_cv is not None and util_cv <= 0.30,
        "gating": False,
        "reason": "load-balance CV unknown or zero mean" if util_cv is None else f"{util_cv:.6g} <= 0.30",
    }

    kv_cache = result["vllm_metrics"]["kv_cache"]
    kv_p99 = kv_cache.get("p99_percent")
    diagnostics["kv_cache_p99_lte_70_percent"] = {
        "available": kv_cache["status"] == "known" and kv_p99 is not None,
        "meets_reference": kv_cache["status"] == "known" and kv_p99 is not None and kv_p99 <= 70.0,
        "gating": False,
        "reason": kv_cache.get("reason") if kv_cache["status"] != "known" else f"{kv_p99:.6g} <= 70",
    }

    preemptions = result["vllm_metrics"]["preemptions"]
    preemption_delta = preemptions.get("delta")
    checks["vllm_preemption_delta_zero"] = {
        "passed": preemptions["status"] == "known" and preemption_delta == 0,
        "reason": preemptions.get("reason") if preemptions["status"] != "known" else f"{preemption_delta} == 0",
    }

    oom_preemption = result["logs"]["oom_preemption"]
    checks["no_oom_or_preemption"] = {
        "passed": oom_preemption["status"] == "known" and oom_preemption["count"] == 0,
        "reason": oom_preemption["reason"] if oom_preemption["status"] != "known" else f"{oom_preemption['count']} matches",
    }

    cpu = result["cpu"]
    idle = cpu.get("value", {}).get("idle_percent") if cpu["status"] == "known" else None
    checks["cpu_idle_gte_30_percent"] = {
        "passed": idle is not None and idle >= 30.0,
        "reason": cpu.get("reason", "CPU idle unknown") if idle is None else f"{idle:.6g} >= 30",
    }
    iowait = cpu.get("value", {}).get("iowait_percent") if cpu["status"] == "known" else None
    diagnostics["cpu_iowait_lt_5_percent"] = {
        "available": iowait is not None,
        "meets_reference": iowait is not None and iowait < 5.0,
        "gating": False,
        "reason": cpu.get("reason", "CPU iowait unknown") if iowait is None else f"{iowait:.6g} < 5",
    }

    for name, check in checks.items():
        if not check["passed"]:
            reasons.append(f"{name}: {check['reason']}")
    return {"passed": not reasons, "checks": checks, "diagnostics": diagnostics, "reasons": reasons}


def reevaluate_report(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("evaluation report must contain a JSON object")
    original_evaluation = result.get("evaluation")
    result["evaluation"] = evaluate(result)
    result["reevaluation"] = {
        "source_report": str(path),
        "evaluated_at": utc_now(),
        "evidence_unchanged": True,
        "original_evaluation": original_evaluation,
    }
    return result


def summarize_fleet(npu_samples: list[dict[str, Any]]) -> dict[str, Any]:
    per_card: dict[int, dict[str, list[float]]] = {}
    for sample in npu_samples:
        if sample["status"] != "known":
            continue
        for card in sample["cards"]:
            card_id = int(card["card_id"])
            dest = per_card.setdefault(card_id, {"hbm_percent": [], "util_percent": []})
            for key in ("hbm_percent", "util_percent"):
                if key in card:
                    dest[key].append(float(card[key]))
    cards = []
    all_hbm = []
    all_util = []
    per_card_util_means = []
    for card_id in sorted(per_card):
        hbm = per_card[card_id]["hbm_percent"]
        util = per_card[card_id]["util_percent"]
        all_hbm.extend(hbm)
        all_util.extend(util)
        util_avg = round(sum(util) / len(util), 3) if util else None
        if util_avg is not None:
            per_card_util_means.append(util_avg)
        cards.append(
            {
                "card_id": card_id,
                "hbm_p99_percent": percentile(hbm, 0.99),
                "hbm_max_percent": max(hbm) if hbm else None,
                "util_avg_percent": util_avg,
                "samples": max(len(hbm), len(util)),
            }
        )
    util_avg_percent = round(sum(all_util) / len(all_util), 3) if all_util else None
    util_cv = population_cv(per_card_util_means)
    return {
        "cards": cards,
        "hbm_p99_percent": percentile(all_hbm, 0.99),
        "hbm_max_percent": max(all_hbm) if all_hbm else None,
        "util_avg_percent": util_avg_percent,
        "util_load_balance_cv": None if util_cv is None else round(util_cv, 6),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    start_monotonic = time.monotonic()
    start_cpu = read_cpu_times()
    task_start = sqlite_counts(args.database)
    log_start = snapshot_log_files(args.log_path)
    throughput_snapshots = [{"elapsed_seconds": 0.0, "snapshot": task_start}]
    next_task_sample_at = 300.0
    npu_samples: list[dict[str, Any]] = []
    worker_samples: list[list[dict[str, Any]]] = []
    vllm_samples: list[list[dict[str, Any]]] = []

    sample_count = 1 if args.sample else max(1, math.floor(args.duration / args.interval) + 1)
    for index in range(sample_count):
        worker_samples.append(collect_worker_health(args.host, args.worker_ports))
        vllm_samples.append(collect_vllm(args.host, args.vllm_ports))
        npu_samples.append(collect_npu_smi())
        elapsed_now = time.monotonic() - start_monotonic
        while not args.sample and elapsed_now >= next_task_sample_at and next_task_sample_at <= args.duration:
            throughput_snapshots.append({"elapsed_seconds": next_task_sample_at, "snapshot": sqlite_counts(args.database)})
            next_task_sample_at += 300.0
        if index + 1 < sample_count:
            sleep_for = min(args.interval, max(0.0, args.duration - elapsed_now))
            if sleep_for > 0:
                time.sleep(sleep_for)

    end_cpu = read_cpu_times()
    task_end = sqlite_counts(args.database)
    db_lock_log_delta = count_log_delta(args.log_path, log_start, DB_LOCK_TERMINAL_RE)
    oom_preemption = count_log_delta(args.log_path, log_start, OOM_PREEMPTION_RE)
    elapsed = max(time.monotonic() - start_monotonic, 0.001)
    if throughput_snapshots[-1]["elapsed_seconds"] < elapsed:
        throughput_snapshots.append({"elapsed_seconds": elapsed, "snapshot": task_end})
    delta = task_delta(task_start, task_end)
    throughput = compute_throughput(task_start, task_end, elapsed, throughput_snapshots)

    log_lock_delta = db_lock_log_delta.get("count") if db_lock_log_delta["status"] == "known" else None
    if log_lock_delta is None:
        db_lock_evidence = {"status": "unknown", "reason": db_lock_log_delta.get("reason", "log DB lock evidence unknown"), "combined_delta": None}
    else:
        db_lock_evidence = {
            "status": "known",
            "reason": None,
            "log_match_delta": log_lock_delta,
            "combined_delta": log_lock_delta,
        }

    result = {
        "schema_version": 1,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(elapsed, 3),
        "mode": "sample" if args.sample else "duration",
        "inputs": {
            "host": args.host,
            "worker_ports": args.worker_ports,
            "vllm_ports": args.vllm_ports,
            "database": str(args.database),
            "log_path": [str(path) for path in args.log_path],
        },
        "services": {
            "workers": worker_samples[-1],
            "vllm": vllm_samples[-1],
        },
        "fleet": summarize_fleet(npu_samples),
        "vllm_metrics": summarize_vllm_metrics(vllm_samples),
        "cpu": cpu_delta(start_cpu, end_cpu),
        "tasks": {
            "start": task_start,
            "end": task_end,
            "delta": delta,
        },
        "logs": {
            "start_snapshot": summarize_log_snapshot(log_start),
            "db_locked_terminal_delta": summarize_log_delta(db_lock_log_delta),
            "oom_preemption_delta": summarize_log_delta(oom_preemption),
            "oom_preemption": summarize_log_delta(oom_preemption),
        },
        "throughput": throughput,
        "db_lock_evidence": db_lock_evidence,
        "baseline": load_baseline(args.baseline),
    }
    result["evaluation"] = evaluate(result)
    return result


def parse_ports(raw: str) -> list[int]:
    ports: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only local MinerU efficiency evaluator")
    parser.add_argument("--duration", type=float, default=60.0, help="collection duration in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="sample interval in seconds")
    parser.add_argument("--baseline", type=Path, help="baseline JSON path")
    parser.add_argument("--record-baseline", type=Path, help="write this run's successful throughput baseline JSON to path")
    parser.add_argument("--evaluate-report", type=Path, help="re-evaluate an existing report without collecting new evidence")
    parser.add_argument("--sample", action="store_true", help="take one short sample for tests/smoke checks")
    parser.add_argument("--host", default="localhost", help="host for local HTTP probes")
    parser.add_argument("--database", type=Path, default=default_db_path())
    parser.add_argument("--log-path", type=Path, action="append", default=None, help="log file or directory to scan")
    parser.add_argument("--worker-ports", type=parse_ports, default=DEFAULT_WORKER_PORTS)
    parser.add_argument("--vllm-ports", type=parse_ports, default=DEFAULT_VLLM_PORTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate_report:
        if args.record_baseline:
            parser.error("--evaluate-report cannot be combined with --record-baseline")
        try:
            result = reevaluate_report(args.evaluate_report)
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            parser.error(f"cannot evaluate report: {exc}")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["evaluation"]["passed"] else 1
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    if args.log_path is None:
        args.log_path = default_log_paths()
    result = run_evaluation(args)
    if args.record_baseline:
        baseline = baseline_from_result(result)
        if baseline["status"] == "known":
            args.record_baseline.parent.mkdir(parents=True, exist_ok=True)
            args.record_baseline.write_text(json.dumps(baseline, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            result["recorded_baseline"] = {"status": "known", "path": str(args.record_baseline), "baseline": baseline}
        else:
            result["recorded_baseline"] = baseline
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if args.record_baseline:
        return 0 if result["recorded_baseline"]["status"] == "known" else 1
    return 0 if result["evaluation"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
