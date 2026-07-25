#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def login(base_url: str, username: str, password: str) -> str:
    url = f"{base_url}/api/v1/auth/login"
    resp = requests.post(url, json={"username": username, "password": password}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def safe_request(method: str, url: str, headers: Dict[str, str], params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        resp = requests.request(method, url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return {"ok": True, "status_code": resp.status_code, "data": resp.json()}
    except Exception as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        text = getattr(getattr(e, "response", None), "text", None)
        return {"ok": False, "status_code": status_code, "error": str(e), "response": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MinerU Tianshu API status as admin")
    parser.add_argument("--base-url", required=True, help="API base URL, e.g. http://localhost:8000")
    parser.add_argument("--username", help="Admin username")
    parser.add_argument("--password", help="Admin password")
    parser.add_argument("--token", help="Bearer token (skip login if provided)")
    parser.add_argument("--page-size", type=int, default=50, help="Task list page size")
    parser.add_argument("--max-tasks", type=int, default=50, help="Max tasks to fetch details for")
    parser.add_argument("--out", help="Write output JSON to file")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.token:
        token = args.token
        log("Using provided token")
    else:
        if not args.username or not args.password:
            raise SystemExit("Provide --token or --username/--password")
        log("Logging in")
        token = login(base_url, args.username, args.password)

    headers = {"Authorization": f"Bearer {token}"}

    log("Fetching health")
    health = safe_request("GET", f"{base_url}/api/v1/health", headers={})

    log("Fetching users")
    users = safe_request("GET", f"{base_url}/api/v1/auth/users", headers=headers)

    log("Fetching queue stats")
    queue_stats = safe_request("GET", f"{base_url}/api/v1/queue/stats", headers=headers)

    log("Fetching task list")
    tasks = safe_request(
        "GET",
        f"{base_url}/api/v1/queue/tasks",
        headers=headers,
        params={"page": 1, "page_size": args.page_size},
    )

    task_items: List[dict] = []
    if tasks.get("ok"):
        data = tasks.get("data") or {}
        task_items = data.get("tasks") or []

    detail_limit = min(args.max_tasks, len(task_items))
    log(f"Fetching task details ({detail_limit})")
    task_details = []
    for item in task_items[:detail_limit]:
        task_id = item.get("task_id")
        if not task_id:
            continue
        detail = safe_request(
            "GET",
            f"{base_url}/api/v1/tasks/{task_id}",
            headers=headers,
            params={"format": "both"},
        )
        task_details.append({"task_id": task_id, "detail": detail})

    output = {
        "generated_at": datetime.now().isoformat(),
        "base_url": base_url,
        "health": health,
        "users": users,
        "queue_stats": queue_stats,
        "tasks": tasks,
        "task_details": task_details,
    }

    result_json = json.dumps(output, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result_json)
        log(f"Output written: {args.out}")
    else:
        print(result_json)


if __name__ == "__main__":
    main()
