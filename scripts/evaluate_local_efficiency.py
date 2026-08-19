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
DEFAULT_STALE_PARENT_MERGE_THRESHOLD_SECONDS = 600.0
DB_LOCK_TERMINAL_RE = re.compile(
    r"(?:\b(?:terminal|failed|failure|mark(?:ed)?\s+failed|status[=: ]+failed)\b.{0,240}database\s+is\s+locked)"
    r"|(?:database\s+is\s+locked.{0,240}\b(?:terminal|failed|failure|mark(?:ed)?\s+failed|status[=: ]+failed)\b)",
    re.IGNORECASE | re.DOTALL,
)
OOM_PREEMPTION_RE = re.compile(
    r"\b(?:oom|out\s+of\s+memory|memory\s+allocation\s+fail(?:ed|ure)?|acl_error_rt_memory_allocation|preempt(?:ion|ed)?)\b",
    re.IGNORECASE,
)
API_HTTP_500_RE = re.compile(
    r"(?:\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b.{0,240}\b500\b)|(?:\bstatus(?:_code)?[=: ]+500\b)|(?:\bHTTP/\d(?:\.\d)?[\"]?\s+500\b)|(?:\b500\s+(?:Internal Server Error|ERROR)\b)",
    re.IGNORECASE,
)
PAGE_COUNT_KEYS = ("page_count", "total_pages", "pages")


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


def first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return None if match is None else float(match.group(0))


def parse_percent_or_ratio(text: str) -> float | None:
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    if "/" in text and len(numbers) >= 2 and numbers[-1] > 0:
        return numbers[-2] * 100.0 / numbers[-1]
    return numbers[-1]


def parse_npu_smi(text: str) -> list[dict[str, float]]:
    cards: dict[int, dict[str, float]] = {}
    current_card: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if "npu-smi" in lower or "version" in lower:
            continue
        if line.startswith("|"):
            cols = [part.strip() for part in line.strip("|").split("|")]
            if len(cols) >= 2:
                first_match = re.match(r"\s*(\d+)", cols[0])
                first = int(first_match.group(1)) if first_match else -1
                if 0 <= first < 8:
                    card = cards.setdefault(first, {"card_id": float(first)})
                    if len(cols) >= 3 and re.match(r"^\d+\s+\S+", cols[0]) and cols[1].lower() in {"ok", "warning", "fault", "unhealthy"}:
                        power_temp = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", cols[2])]
                        if len(power_temp) >= 2:
                            card["power_w"] = power_temp[0]
                            card["temperature_c"] = power_temp[1]
                            current_card = first
                            continue
                    positional_values = [first_number(col) for col in cols[1:5]]
                    if len(cols) >= 5 and all(value is not None for value in positional_values):
                        card["power_w"] = float(positional_values[0])
                        card["temperature_c"] = float(positional_values[1])
                        card["util_percent"] = float(positional_values[2])
                        card["hbm_percent"] = float(positional_values[3])
                        continue
                    parsed_named_metric = False
                    for col in cols[1:]:
                        col_lower = col.lower()
                        value = first_number(col)
                        if value is None:
                            continue
                        if "power" in col_lower or re.search(r"\b\d+(?:\.\d+)?\s*w\b", col_lower):
                            card["power_w"] = value
                            parsed_named_metric = True
                        elif "temp" in col_lower or "temperature" in col_lower or re.search(r"\b\d+(?:\.\d+)?\s*c\b", col_lower):
                            card["temperature_c"] = value
                            parsed_named_metric = True
                        elif "aicore" in col_lower or "ai core" in col_lower or "util" in col_lower:
                            card["util_percent"] = value
                            parsed_named_metric = True
                        elif "hbm" in col_lower or "memory" in col_lower:
                            hbm = parse_percent_or_ratio(col)
                            if hbm is not None:
                                card["hbm_percent"] = hbm
                                parsed_named_metric = True
                    if parsed_named_metric:
                        continue
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

        numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", line)]
        if not numbers:
            continue
        card_id = int(numbers[0])
        if not 0 <= card_id < 8:
            continue
        card = cards.setdefault(card_id, {"card_id": float(card_id)})
        if "hbm" in lower or "memory" in lower:
            hbm = parse_percent_or_ratio(line)
            if hbm is not None:
                card["hbm_percent"] = hbm
        if "util" in lower or "aicore" in lower or "ai core" in lower:
            if len(numbers) >= 2:
                card["util_percent"] = numbers[-1]
        if "power" in lower or re.search(r"\b\d+(?:\.\d+)?\s*w\b", lower):
            if len(numbers) >= 2:
                card["power_w"] = numbers[-1]
        if "temp" in lower or "temperature" in lower or re.search(r"\b\d+(?:\.\d+)?\s*c\b", lower):
            if len(numbers) >= 2:
                card["temperature_c"] = numbers[-1]
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


def parse_json_object(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def task_page_count(row: sqlite3.Row) -> float | None:
    data = parse_json_object(row["data"] if "data" in row.keys() else None)
    for key in PAGE_COUNT_KEYS:
        value = nested_get(data, ("metrics", key))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    options = parse_json_object(row["options"] if "options" in row.keys() else None)
    for key in PAGE_COUNT_KEYS:
        value = nested_get(options, ("chunk_info", key))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None



def float_row_value(row: sqlite3.Row, key: str) -> float | None:
    if key not in row.keys():
        return None
    value = row[key]
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def task_page_count_from_options(row: sqlite3.Row) -> float | None:
    options = parse_json_object(row["options"] if "options" in row.keys() else None)
    for key in PAGE_COUNT_KEYS:
        value = nested_get(options, ("chunk_info", key))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def task_page_count_light(row: sqlite3.Row) -> tuple[float | None, str | None]:
    page_count = float_row_value(row, "page_count")
    if page_count is not None:
        return page_count, "page_count"
    page_count = task_page_count_from_options(row)
    if page_count is not None:
        return page_count, "options.chunk_info"
    return None, None


def parse_timestamp(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            if fmt is None:
                parsed = datetime.fromisoformat(text)
            else:
                parsed = datetime.strptime(text, fmt)
            return parsed
        except ValueError:
            continue
    return None


def processing_seconds(row: sqlite3.Row) -> float | None:
    started = parse_timestamp(row["started_at"] if "started_at" in row.keys() else None)
    completed = parse_timestamp(row["completed_at"] if "completed_at" in row.keys() else None)
    if started is None or completed is None:
        return None
    value = (completed - started).total_seconds()
    return value if value >= 0 else None



def task_processing_seconds_light(row: sqlite3.Row) -> tuple[float | None, str | None]:
    seconds = float_row_value(row, "processing_seconds")
    if seconds is not None:
        return seconds, "processing_seconds"
    seconds = processing_seconds(row)
    if seconds is not None:
        return seconds, "timestamps"
    return None, None


def timestamp_age_seconds(value: Any, now: datetime) -> float | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        now_value = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        parsed_value = parsed
    else:
        now_value = now.replace(tzinfo=None)
        parsed_value = parsed.replace(tzinfo=None)
    age = (now_value - parsed_value).total_seconds()
    return age if age >= 0 else 0.0


def empty_parent_merge_backlog(status: str, reason: str, threshold_seconds: float) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "threshold_seconds": threshold_seconds,
        "total_merging_parents": None,
        "active_merging_parents": None,
        "recoverable_stale_merging_parents": None,
        "blocked_merging_parents": None,
        "stale_merging_parents": None,
        "oldest_age_seconds": None,
        "age_sources": [],
        "age_source": None,
    }


def row_value(row: sqlite3.Row, columns: set[str], key: str) -> Any:
    return row[key] if key in columns else None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_parent_merge_backlog(
    rows: list[sqlite3.Row],
    columns: set[str],
    stale_threshold_seconds: float = DEFAULT_STALE_PARENT_MERGE_THRESHOLD_SECONDS,
    now: datetime | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    required = {"status", "parent_task_id", "started_at"}
    missing = sorted(required - columns)
    if missing:
        return empty_parent_merge_backlog("unknown", f"task schema missing columns: {missing}", stale_threshold_seconds)
    now_value = now or datetime.now(timezone.utc)
    total = 0
    active = 0
    recoverable_stale = 0
    blocked = 0
    ages: list[float] = []
    age_sources: set[str] = set()
    notes: set[str] = set()
    for row in rows:
        if row["status"] != "merging" or row["parent_task_id"] not in (None, ""):
            continue
        total += 1
        attempts = int_or_none(row_value(row, columns, "merge_attempts"))
        if attempts is None:
            attempts = int_or_none(row_value(row, columns, "retry_count")) or 0
        merge_owner = row_value(row, columns, "merge_owner")
        merge_claimed_at = row_value(row, columns, "merge_claimed_at")
        started_at = row_value(row, columns, "started_at")
        age = None
        age_source = None
        if merge_claimed_at not in (None, ""):
            age = timestamp_age_seconds(merge_claimed_at, now_value)
            age_source = "merge_claimed_at"
        elif started_at not in (None, ""):
            age = timestamp_age_seconds(started_at, now_value)
            age_source = "started_at"
            if "merge_claimed_at" in columns:
                notes.add("merge_claimed_at NULL; using started_at for reporting")
            else:
                notes.add("merge_claimed_at missing; using started_at fallback")
        if age is not None:
            ages.append(age)
            if age_source:
                age_sources.add(age_source)
        if attempts >= max_attempts:
            blocked += 1
        elif merge_owner in (None, "") or merge_claimed_at in (None, ""):
            recoverable_stale += 1
        elif age is not None and age > stale_threshold_seconds:
            recoverable_stale += 1
        else:
            active += 1
    reason = None if not notes else "; ".join(sorted(notes))
    sorted_age_sources = sorted(age_sources)
    return {
        "status": "known",
        "reason": reason,
        "threshold_seconds": stale_threshold_seconds,
        "max_attempts": max_attempts,
        "total_merging_parents": total,
        "active_merging_parents": active,
        "recoverable_stale_merging_parents": recoverable_stale,
        "blocked_merging_parents": blocked,
        "stale_merging_parents": recoverable_stale,
        "oldest_age_seconds": None if not ages else round(max(ages), 3),
        "age_sources": sorted_age_sources,
        "age_source": sorted_age_sources[0] if len(sorted_age_sources) == 1 else ("mixed" if sorted_age_sources else None),
    }


def worker_group_key(worker_id: Any) -> str:
    if worker_id in (None, ""):
        return "unknown"
    text = str(worker_id)
    match = re.search(r"(?:group|worker)[_-]?(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\d+", text)
    return match.group(0) if match else text



def worker_group_key_light(row: sqlite3.Row) -> str:
    if "worker_group_index" in row.keys() and row["worker_group_index"] not in (None, ""):
        return str(row["worker_group_index"])
    return worker_group_key(row["worker_id"] if "worker_id" in row.keys() else None)


def empty_task_metrics(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "logical_pdf_completed": None,
        "leaf_completed": None,
        "completed_pages": None,
        "page_count_coverage": None,
        "page_count_source": None,
        "processing_seconds": None,
        "worker_groups": {},
    }


def compute_task_metrics(rows: list[sqlite3.Row], columns: set[str]) -> dict[str, Any]:
    required = {"parent_task_id", "options", "data", "started_at", "completed_at", "worker_id"}
    missing = sorted(required - columns)
    if missing:
        return empty_task_metrics("unknown", f"task schema missing columns: {missing}")

    child_counts: dict[str, int] = {}
    for row in rows:
        parent_id = row["parent_task_id"]
        if parent_id:
            child_counts[str(parent_id)] = child_counts.get(str(parent_id), 0) + 1

    logical_pdf_completed = 0
    leaf_completed = 0
    completed_pages = 0.0
    known_pages = 0
    processing_total = 0.0
    known_processing = 0
    worker_groups: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        task_id = str(row["task_id"])
        is_root = row["parent_task_id"] in (None, "")
        is_leaf = task_id not in child_counts
        if is_root:
            logical_pdf_completed += 1
        if not is_leaf:
            continue
        leaf_completed += 1
        page_count = task_page_count(row)
        if page_count is not None:
            completed_pages += page_count
            known_pages += 1
        seconds = processing_seconds(row)
        if seconds is not None:
            processing_total += seconds
            known_processing += 1
        group_key = worker_group_key(row["worker_id"])
        group = worker_groups.setdefault(
            group_key,
            {"tasks_completed": 0.0, "pages_completed": 0.0, "processing_seconds": 0.0},
        )
        group["tasks_completed"] += 1.0
        if page_count is not None:
            group["pages_completed"] += page_count
        if seconds is not None:
            group["processing_seconds"] += seconds

    return {
        "status": "known",
        "reason": None,
        "logical_pdf_completed": logical_pdf_completed,
        "leaf_completed": leaf_completed,
        "completed_pages": round(completed_pages, 6),
        "known_page_count_tasks": known_pages,
        "processing_seconds": round(processing_total, 6),
        "known_processing_seconds_tasks": known_processing,
        "worker_groups": {
            key: {
                "tasks_completed": int(value["tasks_completed"]),
                "pages_completed": round(value["pages_completed"], 6),
                "processing_seconds": round(value["processing_seconds"], 6),
            }
            for key, value in sorted(worker_groups.items())
        },
    }


def compute_task_metrics_from_leaf_rows(
    logical_pdf_completed: int,
    leaf_rows: Any,
    columns: set[str],
) -> dict[str, Any]:
    required = {"task_id", "parent_task_id"}
    missing = sorted(required - columns)
    if missing:
        return empty_task_metrics("unknown", f"task schema missing columns: {missing}")

    leaf_completed = 0
    completed_pages = 0.0
    known_pages = 0
    processing_total = 0.0
    known_processing = 0
    page_sources: set[str] = set()
    processing_sources: set[str] = set()
    worker_group_sources: set[str] = set()
    worker_groups: dict[str, dict[str, float]] = {}
    for row in leaf_rows:
        leaf_completed += 1
        page_count, page_source = task_page_count_light(row)
        if page_count is not None:
            completed_pages += page_count
            known_pages += 1
        if page_source:
            page_sources.add(page_source)
        seconds, processing_source = task_processing_seconds_light(row)
        if seconds is not None:
            processing_total += seconds
            known_processing += 1
        if processing_source:
            processing_sources.add(processing_source)
        group_key = worker_group_key_light(row)
        worker_group_sources.add("worker_group_index" if "worker_group_index" in row.keys() and row["worker_group_index"] not in (None, "") else "worker_id")
        group = worker_groups.setdefault(
            group_key,
            {"tasks_completed": 0.0, "pages_completed": 0.0, "processing_seconds": 0.0},
        )
        group["tasks_completed"] += 1.0
        if page_count is not None:
            group["pages_completed"] += page_count
        if seconds is not None:
            group["processing_seconds"] += seconds

    coverage = None if leaf_completed == 0 else round(known_pages / leaf_completed, 6)
    reason = None
    if leaf_completed and known_pages < leaf_completed:
        reason = "page counts only read from page_count/options.chunk_info; data.metrics not scanned during periodic snapshots"
    return {
        "status": "known",
        "reason": reason,
        "logical_pdf_completed": logical_pdf_completed,
        "leaf_completed": leaf_completed,
        "completed_pages": round(completed_pages, 6),
        "known_page_count_tasks": known_pages,
        "page_count_coverage": coverage,
        "page_count_sources": sorted(page_sources),
        "page_count_source": sorted(page_sources)[0] if len(page_sources) == 1 else ("mixed" if page_sources else None),
        "processing_seconds": round(processing_total, 6),
        "known_processing_seconds_tasks": known_processing,
        "processing_seconds_sources": sorted(processing_sources),
        "worker_group_sources": sorted(worker_group_sources),
        "worker_groups": {
            key: {
                "tasks_completed": int(value["tasks_completed"]),
                "pages_completed": round(value["pages_completed"], 6),
                "processing_seconds": round(value["processing_seconds"], 6),
            }
            for key, value in sorted(worker_groups.items())
        },
    }


def sqlite_counts(
    db_path: Path,
    stale_parent_threshold_seconds: float = DEFAULT_STALE_PARENT_MERGE_THRESHOLD_SECONDS,
    now: datetime | None = None,
    collect_task_metrics: bool = True,
    cohort_floor: int | None = None,
) -> dict[str, Any]:
    if not db_path.exists():
        result = {"status": "unknown", "reason": f"database not found: {db_path}", "counts": {}}
        result["metrics"] = empty_task_metrics("unknown", result["reason"])
        result["parent_merge_backlog"] = empty_parent_merge_backlog("unknown", result["reason"], stale_parent_threshold_seconds)
        return result
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
            if "status" not in columns:
                reason = "task schema missing status column"
                return {
                    "status": "unknown",
                    "reason": reason,
                    "counts": {},
                    "metrics": empty_task_metrics("unknown", reason),
                    "parent_merge_backlog": empty_parent_merge_backlog("unknown", reason, stale_parent_threshold_seconds),
                }

            cohort_where = "WHERE rowid > ?" if cohort_floor is not None else ""
            cohort_params = (cohort_floor,) if cohort_floor is not None else ()

            count_rows = conn.execute(
                f"SELECT status, COUNT(*) AS count FROM tasks {cohort_where} GROUP BY status",
                cohort_params,
            ).fetchall()
            counts = {row["status"]: int(row["count"]) for row in count_rows}

            metric_required = {"task_id", "parent_task_id"}
            if not collect_task_metrics:
                metrics = empty_task_metrics("unknown", "task metrics skipped for sample mode")
                metrics["page_count_coverage"] = None
                metrics["page_count_source"] = None
            elif metric_required.issubset(columns):
                logical_pdf_completed = int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*) AS count FROM tasks
                        WHERE status = 'completed'
                          AND (parent_task_id IS NULL OR parent_task_id = '')
                          {"AND rowid > ?" if cohort_floor is not None else ""}
                        """,
                        cohort_params,
                    ).fetchone()["count"]
                )
                leaf_columns = []
                for candidate in (
                    "page_count",
                    "processing_seconds",
                    "worker_group_index",
                    "worker_child_index",
                    "options",
                    "started_at",
                    "completed_at",
                    "worker_id",
                ):
                    if candidate in columns:
                        leaf_columns.append(candidate)
                selected_leaf_columns = ", ".join(leaf_columns) if leaf_columns else "task_id"
                leaf_rows = conn.execute(
                    f"""
                    SELECT {selected_leaf_columns}
                    FROM tasks AS task
                    WHERE status = 'completed'
                      {"AND task.rowid > ?" if cohort_floor is not None else ""}
                      AND NOT EXISTS (
                          SELECT 1 FROM tasks AS child WHERE child.parent_task_id = task.task_id
                      )
                    """,
                    cohort_params,
                )
                metrics = compute_task_metrics_from_leaf_rows(logical_pdf_completed, leaf_rows, columns)
            else:
                missing = sorted(metric_required - columns)
                metrics = empty_task_metrics("unknown", f"task schema missing columns: {missing}")

            backlog_required = {"status", "parent_task_id", "started_at"}
            if backlog_required.issubset(columns):
                backlog_columns = ["status", "parent_task_id", "started_at"]
                for optional in ("merge_claimed_at", "merge_owner", "merge_attempts", "retry_count"):
                    if optional in columns:
                        backlog_columns.append(optional)
                selected = ", ".join(backlog_columns)
                backlog_rows = conn.execute(
                    f"""
                    SELECT {selected} FROM tasks
                    WHERE status = 'merging' AND (parent_task_id IS NULL OR parent_task_id = '')
                    """
                ).fetchall()
                parent_merge_backlog = compute_parent_merge_backlog(
                    backlog_rows,
                    columns,
                    stale_parent_threshold_seconds,
                    now,
                )
            else:
                missing = sorted(backlog_required - columns)
                parent_merge_backlog = empty_parent_merge_backlog(
                    "unknown",
                    f"task schema missing columns: {missing}",
                    stale_parent_threshold_seconds,
                )
        finally:
            conn.close()
    except Exception as exc:
        return {
            "status": "unknown",
            "reason": str(exc),
            "counts": {},
            "metrics": empty_task_metrics("unknown", str(exc)),
            "parent_merge_backlog": empty_parent_merge_backlog("unknown", str(exc), stale_parent_threshold_seconds),
        }
    return {
        "status": "known",
        "reason": None,
        "counts": counts,
        "metrics": metrics,
        "parent_merge_backlog": parent_merge_backlog,
        "cohort_floor": cohort_floor,
    }


def sqlite_task_cohort_floor(db_path: Path) -> dict[str, Any]:
    semantics = (
        "duration task counts and typed metrics include only tasks inserted after evaluator start; "
        "start the evaluator before submitting the batch for accurate run throughput"
    )
    if not db_path.exists():
        return {
            "status": "unknown",
            "reason": f"database not found: {db_path}",
            "rowid_floor": None,
            "scope": "unknown",
            "semantics": semantics,
        }
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rowid_floor = conn.execute("SELECT MAX(rowid) AS max_rowid FROM tasks").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return {
            "status": "unknown",
            "reason": str(exc),
            "rowid_floor": None,
            "scope": "unknown",
            "semantics": semantics,
        }
    return {
        "status": "known",
        "reason": None,
        "rowid_floor": 0 if rowid_floor is None else int(rowid_floor),
        "scope": "rowid_gt_floor",
        "semantics": semantics,
    }


def sample_task_cohort() -> dict[str, Any]:
    return {
        "status": "known",
        "reason": None,
        "rowid_floor": None,
        "scope": "global_status_only_sample",
        "semantics": "sample mode reports global task status counts and skips typed completed-task metrics",
    }


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


def subtract_optional(start_value: Any, end_value: Any) -> float | None:
    if start_value is None or end_value is None:
        return None
    try:
        return float(end_value) - float(start_value)
    except (TypeError, ValueError):
        return None


def subtract_worker_groups(start_groups: dict[str, Any], end_groups: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for key in sorted(set(start_groups) | set(end_groups)):
        start_row = start_groups.get(key, {})
        end_row = end_groups.get(key, {})
        groups[key] = {
            "tasks_completed": int(subtract_optional(start_row.get("tasks_completed", 0), end_row.get("tasks_completed", 0)) or 0),
            "pages_completed": round(subtract_optional(start_row.get("pages_completed", 0), end_row.get("pages_completed", 0)) or 0.0, 6),
            "processing_seconds": round(subtract_optional(start_row.get("processing_seconds", 0), end_row.get("processing_seconds", 0)) or 0.0, 6),
        }
    return groups


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
    delta = {
        "status": "known",
        "completed_delta": completed,
        "failed_delta": failed,
        "terminal_delta": total_terminal,
        "failed_rate": failed_rate,
    }
    start_metrics = start.get("metrics", empty_task_metrics("unknown", "task metrics unavailable"))
    end_metrics = end.get("metrics", empty_task_metrics("unknown", "task metrics unavailable"))
    if start_metrics["status"] != "known" or end_metrics["status"] != "known":
        delta["metrics"] = {
            "status": "unknown",
            "reason": start_metrics.get("reason") or end_metrics.get("reason") or "task metrics unavailable",
        }
        return delta
    delta["logical_pdf_completed_delta"] = int(
        subtract_optional(start_metrics.get("logical_pdf_completed"), end_metrics.get("logical_pdf_completed")) or 0
    )
    delta["leaf_completed_delta"] = int(subtract_optional(start_metrics.get("leaf_completed"), end_metrics.get("leaf_completed")) or 0)
    delta["completed_pages_delta"] = round(
        subtract_optional(start_metrics.get("completed_pages"), end_metrics.get("completed_pages")) or 0.0,
        6,
    )
    delta["processing_seconds_delta"] = round(
        subtract_optional(start_metrics.get("processing_seconds"), end_metrics.get("processing_seconds")) or 0.0,
        6,
    )
    delta["worker_groups"] = subtract_worker_groups(start_metrics.get("worker_groups", {}), end_metrics.get("worker_groups", {}))
    delta["metrics"] = {"status": "known", "reason": None}
    return delta


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
        delta = task_delta(start_counts, end_counts) if duration > 0 else {"status": "unknown"}
        if duration <= 0 or start_counts["status"] != "known" or end_counts["status"] != "known" or delta["status"] != "known":
            windows.append(
                {
                    "index": index,
                    "status": "unknown",
                    "duration_seconds": round(duration, 3),
                    "reason": "task snapshot unavailable or non-positive duration",
                    "completed_delta": None,
                    "logical_pdf_completed_delta": None,
                    "completed_pages_delta": None,
                    "successful_per_minute": None,
                    "raw_successful_per_minute": None,
                    "logical_pdf_per_minute": None,
                    "pages_per_minute": None,
                }
            )
            continue
        raw_completed_delta = delta.get("completed_delta")
        raw_rate = None if raw_completed_delta is None else round(raw_completed_delta / duration * 60.0, 6)
        metrics_known = delta.get("metrics", {}).get("status") == "known"
        logical_delta = delta.get("logical_pdf_completed_delta") if metrics_known else None
        page_delta = delta.get("completed_pages_delta") if metrics_known else None
        logical_rate = None if logical_delta is None else round(logical_delta / duration * 60.0, 6)
        page_rate = None if page_delta is None else round(page_delta / duration * 60.0, 6)
        windows.append(
            {
                "index": index,
                "status": "known",
                "start_elapsed_seconds": round(start["elapsed_seconds"], 3),
                "end_elapsed_seconds": round(end["elapsed_seconds"], 3),
                "duration_seconds": round(duration, 3),
                "completed_delta": raw_completed_delta,
                "logical_pdf_completed_delta": logical_delta,
                "completed_pages_delta": page_delta,
                "successful_per_minute": logical_rate,
                "raw_successful_per_minute": raw_rate,
                "logical_pdf_per_minute": logical_rate,
                "pages_per_minute": page_rate,
                "metrics_status": "known" if metrics_known else "unknown",
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
    known_logical_rates = [window["logical_pdf_per_minute"] for window in windows if window["status"] == "known" and window["logical_pdf_per_minute"] is not None]
    known_page_rates = [window["pages_per_minute"] for window in windows if window["status"] == "known" and window["pages_per_minute"] is not None]
    if elapsed_seconds >= 1800 and len(known_logical_rates) >= 6:
        logical_value = median(known_logical_rates[:6])
        page_value = median(known_page_rates[:6]) if len(known_page_rates) >= 6 else None
        return {
            "successful_per_minute": None if logical_value is None else round(logical_value, 6),
            "raw_successful_per_minute": None,
            "logical_pdf_per_minute": None if logical_value is None else round(logical_value, 6),
            "pages_per_minute": None if page_value is None else round(page_value, 6),
            "method": "median_6x5m_windows",
            "window_seconds": 300,
            "windows_required": 6,
            "windows": windows,
        }

    delta = task_delta(task_start, task_end)
    if delta["status"] != "known":
        raw_completed_delta = None
        logical_delta = None
        page_delta = None
    else:
        raw_completed_delta = delta.get("completed_delta")
        metrics_known = delta.get("metrics", {}).get("status") == "known"
        logical_delta = delta.get("logical_pdf_completed_delta") if metrics_known else None
        page_delta = delta.get("completed_pages_delta") if metrics_known else None
    raw_value = None if raw_completed_delta is None else raw_completed_delta / elapsed_seconds * 60.0
    logical_value = None if logical_delta is None else logical_delta / elapsed_seconds * 60.0
    page_value = None if page_delta is None else page_delta / elapsed_seconds * 60.0
    return {
        "successful_per_minute": None if logical_value is None else round(logical_value, 6),
        "raw_successful_per_minute": None if raw_value is None else round(raw_value, 6),
        "logical_pdf_per_minute": None if logical_value is None else round(logical_value, 6),
        "pages_per_minute": None if page_value is None else round(page_value, 6),
        "method": "aggregate_short_run",
        "reason": "duration < 1800 seconds or fewer than 6 known 5-minute windows",
        "windows": windows,
    }


def summarize_energy(
    npu_sample_records: list[dict[str, Any]],
    expected_card_count: int = EXPECTED_WORKER_COUNT,
) -> dict[str, Any]:
    if len(npu_sample_records) < 2:
        return {"status": "unknown", "reason": "fewer than two NPU power samples", "fleet_energy_wh": None, "cards": []}
    card_energy: dict[int, float] = {card_id: 0.0 for card_id in range(expected_card_count)}
    integrated_intervals = 0
    for interval_index, (previous, current) in enumerate(zip(npu_sample_records, npu_sample_records[1:]), start=1):
        dt = float(current["elapsed_seconds"]) - float(previous["elapsed_seconds"])
        if dt <= 0:
            continue
        previous_sample = previous["sample"]
        current_sample = current["sample"]
        if previous_sample.get("status") != "known" or current_sample.get("status") != "known":
            return {
                "status": "unknown",
                "reason": f"NPU sample status unknown in interval {interval_index}",
                "fleet_energy_wh": None,
                "cards": [],
            }
        previous_cards = {int(card["card_id"]): card for card in previous_sample.get("cards", []) if "card_id" in card}
        current_cards = {int(card["card_id"]): card for card in current_sample.get("cards", []) if "card_id" in card}
        expected_ids = set(range(expected_card_count))
        missing_cards = sorted((expected_ids - set(previous_cards)) | (expected_ids - set(current_cards)))
        if missing_cards:
            return {
                "status": "unknown",
                "reason": f"missing NPU cards in power interval {interval_index}: {missing_cards}",
                "fleet_energy_wh": None,
                "cards": [],
            }
        missing_power = sorted(
            card_id
            for card_id in expected_ids
            if previous_cards[card_id].get("power_w") is None or current_cards[card_id].get("power_w") is None
        )
        if missing_power:
            return {
                "status": "unknown",
                "reason": f"missing NPU power_w in interval {interval_index}: {missing_power}",
                "fleet_energy_wh": None,
                "cards": [],
            }
        for card_id in sorted(expected_ids):
            start_power = float(previous_cards[card_id]["power_w"])
            end_power = float(current_cards[card_id]["power_w"])
            card_energy[card_id] += ((start_power + end_power) / 2.0) * dt / 3600.0
        integrated_intervals += 1
    if integrated_intervals == 0:
        return {"status": "unknown", "reason": "no positive-duration NPU power intervals", "fleet_energy_wh": None, "cards": []}
    cards = [{"card_id": card_id, "energy_wh": round(value, 6)} for card_id, value in sorted(card_energy.items())]
    return {
        "status": "known",
        "reason": None,
        "expected_card_count": expected_card_count,
        "integrated_intervals": integrated_intervals,
        "fleet_energy_wh": round(sum(card_energy.values()), 6),
        "cards": cards,
    }


def derive_efficiency(energy: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if energy.get("status") != "known" or energy.get("fleet_energy_wh") is None:
        return {
            "status": "unknown",
            "reason": energy.get("reason", "energy unknown"),
            "fleet_energy_wh_per_completed_page": None,
            "fleet_energy_wh_per_logical_pdf": None,
        }
    pages = delta.get("completed_pages_delta") if delta.get("status") == "known" else None
    logical = delta.get("logical_pdf_completed_delta") if delta.get("status") == "known" else None
    if not pages and not logical:
        return {
            "status": "unknown",
            "reason": "completed page and logical PDF deltas unavailable or zero",
            "fleet_energy_wh_per_completed_page": None,
            "fleet_energy_wh_per_logical_pdf": None,
        }
    fleet_wh = float(energy["fleet_energy_wh"])
    return {
        "status": "known",
        "reason": None,
        "fleet_energy_wh_per_completed_page": None if not pages else round(fleet_wh / float(pages), 6),
        "fleet_energy_wh_per_logical_pdf": None if not logical else round(fleet_wh / float(logical), 6),
    }


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_baseline(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "unknown", "reason": "baseline path not provided"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unknown", "reason": str(exc)}
    throughput_data = data.get("throughput", {}) if isinstance(data.get("throughput"), dict) else {}
    efficiency_data = data.get("efficiency", {}) if isinstance(data.get("efficiency"), dict) else {}
    logical = float_or_none(
        throughput_data.get("logical_pdf_per_minute")
        or throughput_data.get("successful_per_minute")
        or data.get("successful_throughput_per_minute")
        or data.get("successful_per_minute")
    )
    pages = float_or_none(throughput_data.get("pages_per_minute") or data.get("pages_per_minute"))
    energy_per_page = float_or_none(
        efficiency_data.get("fleet_energy_wh_per_completed_page")
        or data.get("fleet_energy_wh_per_completed_page")
        or data.get("energy_wh_per_completed_page")
    )
    energy_per_logical_pdf = float_or_none(
        efficiency_data.get("fleet_energy_wh_per_logical_pdf")
        or data.get("fleet_energy_wh_per_logical_pdf")
        or data.get("energy_wh_per_logical_pdf")
    )
    if logical is None and pages is None and energy_per_page is None and energy_per_logical_pdf is None:
        return {"status": "unknown", "reason": "baseline missing throughput and energy metrics"}
    return {
        "status": "known",
        "path": str(path),
        "successful_per_minute": logical,
        "logical_pdf_per_minute": logical,
        "pages_per_minute": pages,
        "fleet_energy_wh_per_completed_page": energy_per_page,
        "fleet_energy_wh_per_logical_pdf": energy_per_logical_pdf,
        "method": throughput_data.get("method") or data.get("method") or "unknown",
        "windows": throughput_data.get("windows", data.get("windows", [])),
    }


def baseline_from_result(result: dict[str, Any]) -> dict[str, Any]:
    logical = result["throughput"].get("logical_pdf_per_minute")
    pages = result["throughput"].get("pages_per_minute")
    efficiency = result.get("efficiency", {})
    energy_per_page = efficiency.get("fleet_energy_wh_per_completed_page")
    energy_per_logical_pdf = efficiency.get("fleet_energy_wh_per_logical_pdf")
    if logical is None and pages is None and energy_per_page is None and energy_per_logical_pdf is None:
        return {"status": "unknown", "reason": "baseline metrics unknown"}
    return {
        "schema_version": 2,
        "status": "known",
        "created_at": result["ended_at"],
        "source_duration_seconds": result["duration_seconds"],
        "throughput": {
            "successful_per_minute": logical,
            "logical_pdf_per_minute": logical,
            "pages_per_minute": pages,
            "method": result["throughput"].get("method"),
            "windows": result["throughput"].get("windows", []),
        },
        "efficiency": {
            "fleet_energy_wh_per_completed_page": energy_per_page,
            "fleet_energy_wh_per_logical_pdf": energy_per_logical_pdf,
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
    throughput = result["throughput"]
    logical_rate = throughput.get("logical_pdf_per_minute")
    pages_rate = throughput.get("pages_per_minute")
    if baseline["status"] != "known":
        checks["logical_pdf_per_minute_gte_90_percent_baseline"] = {"passed": False, "reason": baseline["reason"]}
        checks["pages_per_minute_gte_90_percent_baseline"] = {"passed": False, "reason": baseline["reason"]}
    else:
        baseline_logical = baseline.get("logical_pdf_per_minute")
        if baseline_logical is None:
            checks["logical_pdf_per_minute_gte_90_percent_baseline"] = {"passed": False, "reason": "baseline logical PDF throughput unknown"}
        elif logical_rate is None:
            checks["logical_pdf_per_minute_gte_90_percent_baseline"] = {"passed": False, "reason": "logical PDF throughput unknown"}
        else:
            required = baseline_logical * 0.9
            checks["logical_pdf_per_minute_gte_90_percent_baseline"] = {
                "passed": logical_rate >= required,
                "reason": f"{logical_rate:.6g} >= required {required:.6g}",
            }
        baseline_pages = baseline.get("pages_per_minute")
        if baseline_pages is None:
            checks["pages_per_minute_gte_90_percent_baseline"] = {"passed": False, "reason": "baseline page throughput unknown"}
        elif pages_rate is None:
            checks["pages_per_minute_gte_90_percent_baseline"] = {"passed": False, "reason": "page throughput unknown"}
        else:
            required = baseline_pages * 0.9
            checks["pages_per_minute_gte_90_percent_baseline"] = {
                "passed": pages_rate >= required,
                "reason": f"{pages_rate:.6g} >= required {required:.6g}",
            }

    efficiency = result.get("efficiency", {})
    energy_per_page = efficiency.get("fleet_energy_wh_per_completed_page")
    if baseline["status"] != "known":
        checks["energy_per_completed_page_lte_90_percent_baseline"] = {"passed": False, "reason": baseline["reason"]}
    elif baseline.get("fleet_energy_wh_per_completed_page") is None:
        checks["energy_per_completed_page_lte_90_percent_baseline"] = {"passed": False, "reason": "baseline energy/page unknown"}
    elif energy_per_page is None:
        checks["energy_per_completed_page_lte_90_percent_baseline"] = {"passed": False, "reason": "energy/page unknown"}
    else:
        required = baseline["fleet_energy_wh_per_completed_page"] * 0.9
        checks["energy_per_completed_page_lte_90_percent_baseline"] = {
            "passed": energy_per_page <= required,
            "reason": f"{energy_per_page:.6g} <= required {required:.6g}",
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
    checks["hbm_max_lt_98_percent"] = {
        "passed": hbm_max is not None and hbm_max < 98.0,
        "reason": "HBM max unknown" if hbm_max is None else f"{hbm_max:.6g} < 98",
    }
    diagnostics["hbm_max_lt_98_percent"] = {
        "available": hbm_max is not None,
        "meets_reference": hbm_max is not None and hbm_max < 98.0,
        "gating": True,
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

    parent_merge_backlog = result.get("parent_merge_backlog", {"status": "unknown", "reason": "parent merge backlog unavailable"})
    stale_merges = parent_merge_backlog.get(
        "recoverable_stale_merging_parents", parent_merge_backlog.get("stale_merging_parents")
    )
    checks["stale_parent_merges_zero"] = {
        "passed": parent_merge_backlog.get("status") == "known" and stale_merges == 0,
        "reason": parent_merge_backlog.get("reason")
        if parent_merge_backlog.get("status") != "known"
        else (
            f"{stale_merges} recoverable stale merging parents; "
            f"blocked={parent_merge_backlog.get('blocked_merging_parents')}; "
            f"oldest_age_seconds={parent_merge_backlog.get('oldest_age_seconds')}"
        ),
    }

    oom_preemption = result["logs"]["oom_preemption"]
    checks["no_oom_or_preemption"] = {
        "passed": oom_preemption["status"] == "known" and oom_preemption["count"] == 0,
        "reason": oom_preemption["reason"] if oom_preemption["status"] != "known" else f"{oom_preemption['count']} matches",
    }

    api_500 = result.get("logs", {}).get("api_500", {"status": "unknown", "reason": "API 500 log evidence unavailable", "count": None})
    checks["api_500_delta_zero"] = {
        "passed": api_500.get("status") == "known" and api_500.get("count") == 0,
        "reason": api_500.get("reason") if api_500.get("status") != "known" else f"{api_500.get('count')} matches",
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
    task_cohort = sample_task_cohort() if args.sample else sqlite_task_cohort_floor(args.database)
    cohort_floor = task_cohort["rowid_floor"] if task_cohort["status"] == "known" else None
    collect_task_metrics = not args.sample and task_cohort["status"] == "known"
    task_start = sqlite_counts(
        args.database,
        args.stale_parent_threshold_seconds,
        collect_task_metrics=collect_task_metrics,
        cohort_floor=cohort_floor,
    )
    log_start = snapshot_log_files(args.log_path)
    throughput_snapshots = [{"elapsed_seconds": 0.0, "snapshot": task_start}]
    next_task_sample_at = 300.0
    npu_samples: list[dict[str, Any]] = []
    npu_sample_records: list[dict[str, Any]] = []
    worker_samples: list[list[dict[str, Any]]] = []
    vllm_samples: list[list[dict[str, Any]]] = []

    sample_count = 1 if args.sample else max(1, math.floor(args.duration / args.interval) + 1)
    for index in range(sample_count):
        worker_samples.append(collect_worker_health(args.host, args.worker_ports))
        vllm_samples.append(collect_vllm(args.host, args.vllm_ports))
        npu_sample = collect_npu_smi()
        elapsed_now = time.monotonic() - start_monotonic
        npu_samples.append(npu_sample)
        npu_sample_records.append({"elapsed_seconds": elapsed_now, "sample": npu_sample})
        while not args.sample and elapsed_now >= next_task_sample_at and next_task_sample_at <= args.duration:
            throughput_snapshots.append({
                "elapsed_seconds": next_task_sample_at,
                "snapshot": sqlite_counts(
                    args.database,
                    args.stale_parent_threshold_seconds,
                    collect_task_metrics=collect_task_metrics,
                    cohort_floor=cohort_floor,
                ),
            })
            next_task_sample_at += 300.0
        if index + 1 < sample_count:
            sleep_for = min(args.interval, max(0.0, args.duration - elapsed_now))
            if sleep_for > 0:
                time.sleep(sleep_for)

    end_cpu = read_cpu_times()
    task_end = sqlite_counts(
        args.database,
        args.stale_parent_threshold_seconds,
        collect_task_metrics=collect_task_metrics,
        cohort_floor=cohort_floor,
    )
    db_lock_log_delta = count_log_delta(args.log_path, log_start, DB_LOCK_TERMINAL_RE)
    oom_preemption = count_log_delta(args.log_path, log_start, OOM_PREEMPTION_RE)
    api_500_delta = count_log_delta(args.log_path, log_start, API_HTTP_500_RE)
    elapsed = max(time.monotonic() - start_monotonic, 0.001)
    if throughput_snapshots[-1]["elapsed_seconds"] < elapsed:
        throughput_snapshots.append({"elapsed_seconds": elapsed, "snapshot": task_end})
    delta = task_delta(task_start, task_end)
    throughput = compute_throughput(task_start, task_end, elapsed, throughput_snapshots)
    energy = summarize_energy(npu_sample_records)
    efficiency = derive_efficiency(energy, delta)

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
        "schema_version": 2,
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
        "energy": energy,
        "efficiency": efficiency,
        "vllm_metrics": summarize_vllm_metrics(vllm_samples),
        "cpu": cpu_delta(start_cpu, end_cpu),
        "task_cohort": task_cohort,
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
            "api_500_delta": summarize_log_delta(api_500_delta),
            "api_500": summarize_log_delta(api_500_delta),
        },
        "throughput": throughput,
        "parent_merge_backlog": task_end.get(
            "parent_merge_backlog",
            empty_parent_merge_backlog("unknown", "parent merge backlog unavailable", args.stale_parent_threshold_seconds),
        ),
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
    parser.add_argument(
        "--stale-parent-threshold-seconds",
        type=float,
        default=DEFAULT_STALE_PARENT_MERGE_THRESHOLD_SECONDS,
        help="age threshold for stale merging parent tasks",
    )
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
