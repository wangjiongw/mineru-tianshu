"""
MinerU Tianshu - Task Scheduler (Optional)
天枢任务调度器（可选）
企业级 AI 数据预处理平台 - 任务调度服务

在 Worker 自动循环模式下，调度器主要用于：
1. 监控队列状态（默认5分钟一次）
2. 健康检查（默认15分钟一次）
3. 统计信息收集
4. 故障恢复（重置超时任务）

注意：
- 如果 workers 启用了自动循环模式（默认），则不需要调度器来触发任务处理
- Worker 已经主动工作，调度器只是偶尔检查系统状态
- 较长的间隔可以最小化系统开销，同时保持必要的监控能力
- 5分钟监控、15分钟健康检查对于自动运行的系统来说已经足够及时
"""

import asyncio
import aiohttp
import argparse
import json
import math
import signal
import sys
import os
import time
import sqlite3
from pathlib import Path
from loguru import logger

# 添加父目录到路径以确保能导入 task_db
sys.path.insert(0, str(Path(__file__).parent))

from task_db import TaskDB

DEFAULT_ORPHAN_RECOVERY_BATCH_SIZE = 500
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
MIN_STALE_TASK_TIMEOUT_MINUTES = 60


class TaskScheduler:
    """
    任务调度器（可选）

    职责（在 Worker 自动循环模式下）：
    1. 监控 SQLite 任务队列状态
    2. 健康检查 Workers
    3. 故障恢复（重置超时任务）
    4. 收集和展示统计信息
    """

    def __init__(
        self,
        litserve_url="http://localhost:8001/predict",
        monitor_interval=300,
        health_check_interval=900,
        stale_task_timeout=10,
        cleanup_old_files_days=7,
        cleanup_old_records_days=0,
        worker_auto_mode=True,
        orphan_recovery_apply=False,
        orphan_recovery_batch_size=DEFAULT_ORPHAN_RECOVERY_BATCH_SIZE,
        task_p99_seconds=None,
        heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ):
        """
        初始化调度器

        Args:
            litserve_url: LitServe Worker 的 URL
            monitor_interval: 监控间隔（秒，默认300秒=5分钟）
            health_check_interval: 健康检查间隔（秒，默认900秒=15分钟）
            stale_task_timeout: 超时任务重置时间（分钟）
            cleanup_old_files_days: 清理多少天前的结果文件（0=禁用，默认7天）
            cleanup_old_records_days: 清理多少天前的数据库记录（0=禁用，不推荐删除）
            worker_auto_mode: Worker 是否启用自动循环模式
            orphan_recovery_apply: 是否实际应用孤儿恢复（默认 dry-run）
            orphan_recovery_batch_size: 每轮孤儿恢复最多处理的任务数
            task_p99_seconds: 可选任务 P99 耗时，用于推导保守超时阈值
            heartbeat_interval_seconds: Worker 心跳间隔，用于推导保守超时阈值
        """
        self.litserve_url = litserve_url
        self.monitor_interval = monitor_interval
        self.health_check_interval = health_check_interval
        self.configured_stale_task_timeout = stale_task_timeout
        self.task_p99_seconds = self._optional_positive_float(
            task_p99_seconds if task_p99_seconds is not None else os.getenv("SCHEDULER_TASK_P99_SECONDS")
        )
        self.heartbeat_interval_seconds = self._optional_positive_float(
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else os.getenv("WORKER_HEARTBEAT_INTERVAL_SECONDS")
        ) or DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        self.heartbeat_freshness_seconds = math.ceil(3 * self.heartbeat_interval_seconds)
        self.stale_task_timeout = self._effective_stale_timeout_minutes(
            stale_task_timeout,
            task_p99_seconds=self.task_p99_seconds,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
        )
        self.cleanup_old_files_days = cleanup_old_files_days
        self.cleanup_old_records_days = cleanup_old_records_days
        self.worker_auto_mode = worker_auto_mode
        self.orphan_recovery_apply = orphan_recovery_apply
        self.orphan_recovery_batch_size = max(1, int(orphan_recovery_batch_size or DEFAULT_ORPHAN_RECOVERY_BATCH_SIZE))
        
        # 初始化数据库连接
        db_path = os.getenv("DATABASE_PATH")
        if db_path:
            self.db = TaskDB(db_path)
        else:
            self.db = TaskDB()
            
        self.running = True

    @staticmethod
    def _optional_positive_float(value):
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _effective_stale_timeout_minutes(
        cls,
        configured_minutes,
        *,
        task_p99_seconds=None,
        heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ):
        candidates = [MIN_STALE_TASK_TIMEOUT_MINUTES, int(configured_minutes or 0)]
        if task_p99_seconds:
            candidates.append(math.ceil((1.5 * task_p99_seconds) / 60))
        if heartbeat_interval_seconds:
            candidates.append(math.ceil((3 * heartbeat_interval_seconds) / 60))
        return max(candidates)

    @staticmethod
    def _parse_processing_entry(raw_data):
        if raw_data is None:
            return {}
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode()
        try:
            return json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            return {}

    def _read_processing_heartbeat(self, redis_queue, task_id):
        client = getattr(redis_queue, "client", None)
        if not client or not hasattr(client, "hget"):
            return {}
        return self._parse_processing_entry(client.hget(redis_queue.config.processing_key, task_id))

    def _has_fresh_heartbeat(self, redis_queue, task_id, now):
        heartbeat = self._read_processing_heartbeat(redis_queue, task_id)
        claimed_at = float(heartbeat.get("claimed_at") or 0)
        return bool(claimed_at and now - claimed_at <= self.heartbeat_freshness_seconds)

    def _iter_processing_entries(self, redis_queue):
        client = getattr(redis_queue, "client", None)
        if not client:
            return

        processing_key = redis_queue.config.processing_key
        if hasattr(client, "hscan_iter"):
            count = 0
            for item in client.hscan_iter(processing_key, count=self.orphan_recovery_batch_size):
                yield item
                count += 1
                if count >= self.orphan_recovery_batch_size:
                    return
            return

        if hasattr(client, "hscan"):
            cursor = 0
            count = 0
            while True:
                cursor, entries = client.hscan(processing_key, cursor=cursor, count=self.orphan_recovery_batch_size)
                for item in entries.items():
                    yield item
                    count += 1
                    if count >= self.orphan_recovery_batch_size:
                        return
                if not cursor:
                    return

        hlen = client.hlen(processing_key) if hasattr(client, "hlen") else None
        if hlen is not None and hlen > self.orphan_recovery_batch_size:
            logger.warning(
                f"Skipping Redis processing ghost scan: no HSCAN API and "
                f"hlen={hlen} exceeds batch_size={self.orphan_recovery_batch_size}"
            )
            return

        if hasattr(client, "hgetall"):
            for item in client.hgetall(processing_key).items():
                yield item

    @staticmethod
    def _task_columns(cursor):
        cursor.execute("PRAGMA table_info(tasks)")
        return {row["name"] for row in cursor.fetchall()}

    @staticmethod
    def _sqlite_time_age_seconds(timestamp_expr):
        return f"MAX(0, CAST((julianday('now') - julianday({timestamp_expr})) * 86400 AS INTEGER))"

    def _get_parent_merge_backlog_stats(self, stale_seconds=None, max_attempts=3):
        """Read a bounded parent-merge backlog snapshot without claiming or merging work."""
        stale_seconds = int(stale_seconds if stale_seconds is not None else self.stale_task_timeout * 60)
        max_attempts = int(max_attempts or 3)
        with self.db.get_cursor() as cursor:
            try:
                columns = self._task_columns(cursor)
            except sqlite3.OperationalError:
                return {}

            required = {"task_id", "status", "is_parent", "child_count", "created_at"}
            if not required.issubset(columns):
                return {}

            merge_owner = "p.merge_owner" if "merge_owner" in columns else "NULL"
            merge_claimed_at = "p.merge_claimed_at" if "merge_claimed_at" in columns else "NULL"
            merge_attempts = "COALESCE(p.merge_attempts, 0)" if "merge_attempts" in columns else "0"
            child_completed = "p.child_completed" if "child_completed" in columns else "0"
            child_failed = (
                "SUM(CASE WHEN c.status = 'failed' THEN 1 ELSE 0 END)"
                if "parent_task_id" in columns
                else "0"
            )
            real_child_completed = (
                "SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END)"
                if "parent_task_id" in columns
                else child_completed
            )
            join_clause = "LEFT JOIN tasks c ON c.parent_task_id = p.task_id" if "parent_task_id" in columns else ""
            if "merge_claimed_at" in columns and "started_at" in columns:
                age_source = "COALESCE(p.merge_claimed_at, p.started_at, p.created_at)"
            elif "started_at" in columns:
                age_source = "COALESCE(p.started_at, p.created_at)"
            else:
                age_source = "p.created_at"
            age_expr = self._sqlite_time_age_seconds(age_source)

            cursor.execute(
                f"""
                WITH parent_merge_backlog AS (
                    SELECT
                        p.task_id,
                        p.status,
                        p.child_count,
                        {merge_owner} AS merge_owner,
                        {merge_claimed_at} AS merge_claimed_at,
                        {merge_attempts} AS merge_attempts,
                        {child_failed} AS child_failed,
                        {real_child_completed} AS real_child_completed,
                        CASE
                            WHEN p.status = 'merging' THEN {age_expr}
                            ELSE NULL
                        END AS merging_age_seconds
                    FROM tasks p
                    {join_clause}
                    WHERE p.is_parent = 1
                      AND p.child_count > 0
                      AND p.status IN ('pending', 'processing', 'merging')
                    GROUP BY p.task_id
                )
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE
                        WHEN status IN ('pending', 'processing')
                         AND real_child_completed >= child_count
                        THEN 1 ELSE 0 END) AS ready,
                    SUM(CASE
                        WHEN status = 'merging'
                         AND real_child_completed >= child_count
                         AND child_failed = 0
                         AND merge_attempts < ?
                         AND (
                            merge_owner IS NULL
                            OR merge_claimed_at IS NULL
                            OR merge_claimed_at <= datetime('now', '-' || ? || ' seconds')
                         )
                        THEN 1 ELSE 0 END) AS recoverable,
                    SUM(CASE
                        WHEN status = 'merging'
                         AND real_child_completed >= child_count
                         AND child_failed = 0
                         AND merge_attempts < ?
                         AND merge_owner IS NOT NULL
                         AND merge_claimed_at IS NOT NULL
                         AND merge_claimed_at > datetime('now', '-' || ? || ' seconds')
                        THEN 1 ELSE 0 END) AS active_leased,
                    SUM(CASE
                        WHEN status = 'merging'
                         AND real_child_completed >= child_count
                         AND child_failed = 0
                         AND merge_attempts < ?
                         AND merge_owner IS NOT NULL
                         AND merge_claimed_at IS NOT NULL
                         AND merge_claimed_at <= datetime('now', '-' || ? || ' seconds')
                        THEN 1 ELSE 0 END) AS stale_leased,
                    SUM(CASE
                        WHEN status = 'merging'
                         AND (
                            real_child_completed < child_count
                            OR child_failed > 0
                            OR merge_attempts >= ?
                         )
                        THEN 1 ELSE 0 END) AS blocked,
                    MAX(merging_age_seconds) AS oldest_merging_age_seconds
                FROM parent_merge_backlog
                """,
                (max_attempts, stale_seconds, max_attempts, stale_seconds, max_attempts, stale_seconds, max_attempts),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            return {
                "total": int(row["total"] or 0),
                "ready": int(row["ready"] or 0),
                "recoverable": int(row["recoverable"] or 0),
                "active_leased": int(row["active_leased"] or 0),
                "stale_leased": int(row["stale_leased"] or 0),
                "blocked": int(row["blocked"] or 0),
                "oldest_merging_age_seconds": int(row["oldest_merging_age_seconds"] or 0),
            }

    def _recover_orphans_safely(self):
        """Plan or apply bounded orphan recovery without destructive defaults."""
        stats = {
            "dry_run": not self.orphan_recovery_apply,
            "batch_size": self.orphan_recovery_batch_size,
            "stale_minutes": self.stale_task_timeout,
            "heartbeat_freshness_seconds": self.heartbeat_freshness_seconds,
            "planned_sqlite_reset": 0,
            "planned_sqlite_failed": 0,
            "planned_parent_merging": 0,
            "planned_parent_already_merging": 0,
            "planned_redis_requeued": 0,
            "planned_ghosts_purged": 0,
            "applied_sqlite_reset": 0,
            "applied_sqlite_failed": 0,
            "applied_parent_merging": 0,
            "applied_redis_requeued": 0,
            "applied_ghosts_purged": 0,
            "fresh_heartbeat_protected": 0,
            "parent_waiting_protected": 0,
        }

        redis_queue = None
        try:
            from redis_queue import get_redis_queue

            redis_queue = get_redis_queue()
        except Exception as exc:
            logger.debug(f"Redis queue unavailable for scheduler orphan recovery: {exc}")

        stale_task_ids = set()
        fail_task_ids = set()
        parent_merging_ids = set()
        parent_already_merging_ids = set()
        fresh_protected_ids = set()
        parent_waiting_ids = set()
        now = time.time()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT task_id, worker_id, retry_count, is_parent, child_count, child_completed, status
                FROM tasks
                WHERE status IN ('processing', 'merging')
                  AND started_at < datetime('now', '-' || ? || ' minutes')
                ORDER BY started_at
                LIMIT ?
                """,
                (self.stale_task_timeout, self.orphan_recovery_batch_size),
            )
            rows = cursor.fetchall()

            for row in rows:
                task_id = row["task_id"]
                is_parent_waiting = (
                    row["is_parent"] == 1
                    and row["status"] == "processing"
                    and row["child_count"] > 0
                    and row["child_completed"] < row["child_count"]
                )
                if is_parent_waiting:
                    parent_waiting_ids.add(task_id)
                    stats["parent_waiting_protected"] += 1
                    continue

                if redis_queue and self._has_fresh_heartbeat(redis_queue, task_id, now):
                    fresh_protected_ids.add(task_id)
                    stats["fresh_heartbeat_protected"] += 1
                    continue

                is_parent_ready = (
                    row["is_parent"] == 1
                    and row["status"] in ("processing", "merging")
                    and row["child_count"] > 0
                    and (row["status"] == "merging" or row["child_completed"] >= row["child_count"])
                )
                if is_parent_ready:
                    if row["status"] == "merging":
                        parent_already_merging_ids.add(task_id)
                    else:
                        parent_merging_ids.add(task_id)
                elif row["retry_count"] >= 3:
                    fail_task_ids.add(task_id)
                else:
                    stale_task_ids.add(task_id)

            stats["planned_sqlite_reset"] = len(stale_task_ids)
            stats["planned_sqlite_failed"] = len(fail_task_ids)
            stats["planned_parent_merging"] = len(parent_merging_ids)
            stats["planned_parent_already_merging"] = len(parent_already_merging_ids)
            stats["planned_redis_requeued"] = len(stale_task_ids)

            if self.orphan_recovery_apply:
                if parent_merging_ids:
                    placeholders = ",".join("?" * len(parent_merging_ids))
                    cursor.execute(
                        f"""
                        UPDATE tasks
                        SET status = 'merging'
                        WHERE task_id IN ({placeholders})
                          AND status = 'processing'
                        """,
                        tuple(parent_merging_ids),
                    )
                    stats["applied_parent_merging"] = cursor.rowcount

                if stale_task_ids:
                    placeholders = ",".join("?" * len(stale_task_ids))
                    cursor.execute(
                        f"""
                        UPDATE tasks
                        SET status = 'pending',
                            worker_id = NULL,
                            retry_count = retry_count + 1
                        WHERE task_id IN ({placeholders})
                        """,
                        tuple(stale_task_ids),
                    )
                    stats["applied_sqlite_reset"] = cursor.rowcount

                if fail_task_ids:
                    placeholders = ",".join("?" * len(fail_task_ids))
                    cursor.execute(
                        f"""
                        UPDATE tasks
                        SET status = 'failed',
                            error_message = 'Task repeatedly crashed workers after retry limit.',
                            completed_at = datetime('now')
                        WHERE task_id IN ({placeholders})
                        """,
                        tuple(fail_task_ids),
                    )
                    stats["applied_sqlite_failed"] = cursor.rowcount

        if redis_queue and stale_task_ids:
            for task_id in stale_task_ids:
                if not self.orphan_recovery_apply:
                    continue
                try:
                    if redis_queue.fail(task_id, "scheduler", requeue=True):
                        stats["applied_redis_requeued"] += 1
                except Exception as exc:
                    logger.error(f"Failed to requeue orphan task {task_id}: {exc}")

        if redis_queue:
            ghost_candidates = []
            for task_id, raw_data in self._iter_processing_entries(redis_queue) or ():
                task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                if (
                    task_id in stale_task_ids
                    or task_id in fail_task_ids
                    or task_id in parent_merging_ids
                    or task_id in parent_already_merging_ids
                    or task_id in fresh_protected_ids
                    or task_id in parent_waiting_ids
                ):
                    continue
                heartbeat = self._parse_processing_entry(raw_data)
                claimed_at = float(heartbeat.get("claimed_at") or 0)
                if claimed_at and now - claimed_at <= self.heartbeat_freshness_seconds:
                    stats["fresh_heartbeat_protected"] += 1
                    continue
                ghost_candidates.append(task_id)

            active_processing_ids = set()
            if ghost_candidates:
                placeholders = ",".join("?" * len(ghost_candidates))
                with self.db.get_cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT task_id
                        FROM tasks
                        WHERE task_id IN ({placeholders})
                          AND status IN ('processing', 'merging')
                        """,
                        tuple(ghost_candidates),
                    )
                    active_processing_ids = {row["task_id"] for row in cursor.fetchall()}

            for task_id in ghost_candidates:
                if task_id in active_processing_ids:
                    continue
                stats["planned_ghosts_purged"] += 1
                if self.orphan_recovery_apply:
                    redis_queue.client.hdel(redis_queue.config.processing_key, task_id)
                    stats["applied_ghosts_purged"] += 1

        return stats

    async def check_worker_health(self, session: aiohttp.ClientSession):
        """
        检查 worker 健康状态
        """
        try:
            # 使用 /health 端点通常比 /predict 更轻量
            health_url = self.litserve_url.replace("/predict", "/health")
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json()
                    except Exception:
                        return {"status": "ok", "raw": await resp.text()}
                else:
                    # 如果 /health 不存在，尝试 POST /predict
                    async with session.post(
                        self.litserve_url, json={"action": "health"}, timeout=aiohttp.ClientTimeout(total=10)
                    ) as predict_resp:
                        if predict_resp.status == 200:
                            return await predict_resp.json()
                        else:
                            logger.error(f"Health check failed with status {predict_resp.status}")
                            return None

        except asyncio.TimeoutError:
            logger.warning("Health check timeout")
            return None
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return None

    async def schedule_loop(self):
        """
        主监控循环
        """
        logger.info("🔄 Task scheduler started")
        logger.info(f"    LitServe URL: {self.litserve_url}")
        logger.info(f"    Worker Mode: {'Auto-Loop' if self.worker_auto_mode else 'Scheduler-Driven'}")
        logger.info(f"    Monitor Interval: {self.monitor_interval}s")
        logger.info(f"    Health Check Interval: {self.health_check_interval}s")
        logger.info(
            f"    Stale Task Timeout: {self.stale_task_timeout}m "
            f"(configured={self.configured_stale_task_timeout}m, apply={self.orphan_recovery_apply}, "
            f"batch={self.orphan_recovery_batch_size})"
        )
        logger.info(f"    Cleanup Old Files: {self.cleanup_old_files_days} days")

        health_check_counter = 0
        stale_task_counter = 0
        cleanup_counter = 0

        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    # 1. 监控队列状态
                    try:
                        stats = self.db.get_queue_stats()
                        pending_count = stats.get("pending", 0)
                        processing_count = stats.get("processing", 0)
                        completed_count = stats.get("completed", 0)
                        failed_count = stats.get("failed", 0)

                        if pending_count > 0 or processing_count > 0:
                            logger.info(
                                f"📊 Queue: {pending_count} pending, {processing_count} processing, "
                                f"{completed_count} completed, {failed_count} failed"
                            )
                    except Exception as e:
                        logger.error(f"Failed to get queue stats: {e}")

                    # 2. 定期健康检查
                    health_check_counter += 1
                    if health_check_counter * self.monitor_interval >= self.health_check_interval:
                        health_check_counter = 0
                        logger.info("🏥 Performing health check...")
                        health_result = await self.check_worker_health(session)
                        if health_result:
                            logger.info(f"✅ Workers healthy: {health_result.get('status', 'ok')}")
                        else:
                            logger.warning("⚠️  Workers health check failed")

                    # 3. 定期恢复孤儿任务（SQLite + Redis 两层同步，含幽灵清理）
                    stale_task_counter += 1
                    if stale_task_counter * self.monitor_interval >= self.stale_task_timeout * 60:
                        stale_task_counter = 0
                        try:
                            stats = self._recover_orphans_safely()
                            merge_stats = self._get_parent_merge_backlog_stats()
                            planned = (
                                stats["planned_sqlite_reset"]
                                + stats["planned_sqlite_failed"]
                                + stats["planned_parent_merging"]
                                + stats["planned_parent_already_merging"]
                                + stats["planned_ghosts_purged"]
                            )
                            applied = (
                                stats["applied_sqlite_reset"]
                                + stats["applied_sqlite_failed"]
                                + stats["applied_parent_merging"]
                                + stats["applied_redis_requeued"]
                                + stats["applied_ghosts_purged"]
                            )
                            if (
                                planned
                                or applied
                                or stats["fresh_heartbeat_protected"]
                                or stats["parent_waiting_protected"]
                                or merge_stats.get("total", 0)
                            ):
                                logger.warning(
                                    f"⚠️  Orphan recovery {'dry-run' if stats['dry_run'] else 'apply'} "
                                    f"(stale > {stats['stale_minutes']}m, batch={stats['batch_size']}): "
                                    f"planned_reset={stats['planned_sqlite_reset']} "
                                    f"planned_failed={stats['planned_sqlite_failed']} "
                                    f"planned_parent_merging={stats['planned_parent_merging']} "
                                    f"planned_parent_already_merging={stats['planned_parent_already_merging']} "
                                    f"planned_ghosts={stats['planned_ghosts_purged']} "
                                    f"applied_reset={stats['applied_sqlite_reset']} "
                                    f"applied_failed={stats['applied_sqlite_failed']} "
                                    f"applied_requeued={stats['applied_redis_requeued']} "
                                    f"applied_ghosts={stats['applied_ghosts_purged']} "
                                    f"fresh_heartbeat_protected={stats['fresh_heartbeat_protected']} "
                                    f"parent_waiting_protected={stats['parent_waiting_protected']} "
                                    f"parent_merge_total={merge_stats.get('total', 0)} "
                                    f"parent_merge_ready={merge_stats.get('ready', 0)} "
                                    f"parent_merge_recoverable={merge_stats.get('recoverable', 0)} "
                                    f"parent_merge_active_leased={merge_stats.get('active_leased', 0)} "
                                    f"parent_merge_stale_leased={merge_stats.get('stale_leased', 0)} "
                                    f"parent_merge_blocked={merge_stats.get('blocked', 0)} "
                                    f"parent_merge_oldest_age_s={merge_stats.get('oldest_merging_age_seconds', 0)}"
                                )
                        except Exception as e:
                            logger.error(f"Failed to recover orphan tasks: {e}")

                    # 4. 定期清理旧任务文件
                    cleanup_counter += 1
                    # 每24小时清理一次
                    cleanup_interval_cycles = (24 * 3600) / max(1, self.monitor_interval)
                    if cleanup_counter >= cleanup_interval_cycles:
                        cleanup_counter = 0
                        if self.cleanup_old_files_days > 0:
                            try:
                                logger.info(f"🧹 Cleaning up tasks older than {self.cleanup_old_files_days} days...")
                                record_count = self.db.cleanup_old_task_records(days=self.cleanup_old_files_days)
                                if record_count > 0:
                                    logger.info(f"✅ Cleaned up {record_count} old tasks")
                            except Exception as e:
                                logger.error(f"Failed to cleanup old tasks: {e}")

                    # 等待下一次监控
                    await asyncio.sleep(self.monitor_interval)

                except Exception as e:
                    logger.error(f"Scheduler loop error: {e}")
                    await asyncio.sleep(self.monitor_interval)

        logger.info("⏹️  Task scheduler stopped")

    def start(self):
        """启动调度器"""
        logger.info("🚀 Starting MinerU Tianshu Task Scheduler...")

        # 设置信号处理
        def signal_handler(sig, frame):
            logger.info("\n🛑 Received stop signal, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 运行调度循环
        asyncio.run(self.schedule_loop())

    def stop(self):
        """停止调度器"""
        self.running = False


async def health_check(litserve_url: str) -> bool:
    """
    健康检查：验证 LitServe Worker 是否可用
    """
    try:
        async with aiohttp.ClientSession() as session:
            health_url = litserve_url.replace("/predict", "/health")
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinerU Tianshu Task Scheduler (Optional)")
    
    parser.add_argument("--litserve-url", type=str, default="http://localhost:8001/predict", help="LitServe worker URL")
    
    # ✅ 修复：同时支持 --monitor-interval 和 --interval (兼容 docker-compose)
    parser.add_argument(
        "--monitor-interval", type=int, default=300, help="Monitor interval in seconds (default: 300s = 5 minutes)"
    )
    parser.add_argument(
        "--interval", type=int, dest="monitor_interval", help="Alias for --monitor-interval"
    )
    
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=900,
        help="Health check interval in seconds (default: 900s = 15 minutes)",
    )
    parser.add_argument(
        "--stale-task-timeout", type=int, default=10, help="Timeout for stale tasks in minutes (default: 10)"
    )
    parser.add_argument(
        "--orphan-recovery-apply",
        action="store_true",
        help="Apply orphan recovery changes. Default is dry-run.",
    )
    parser.add_argument(
        "--orphan-recovery-batch-size",
        type=int,
        default=DEFAULT_ORPHAN_RECOVERY_BATCH_SIZE,
        help="Maximum orphan recovery records to inspect per cycle (default: 500)",
    )
    parser.add_argument(
        "--task-p99-seconds",
        type=float,
        help="Optional task P99 duration used to derive stale threshold",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        help="Worker heartbeat interval used to derive stale threshold (default: 30)",
    )
    parser.add_argument(
        "--cleanup-old-files-days",
        type=int,
        default=7,
        help="Delete result files older than N days (0=disable, default: 7)",
    )
    # 兼容旧参数
    parser.add_argument(
        "--cleanup-old-records-days",
        type=int,
        default=0,
        help="Delete DB records older than N days (deprecated)",
    )
    parser.add_argument("--wait-for-workers", action="store_true", help="Wait for workers to be ready before starting")
    parser.add_argument("--no-worker-auto-mode", action="store_true", help="Disable worker auto-loop mode assumption")

    args = parser.parse_args()

    # 等待 workers 就绪（可选）
    if args.wait_for_workers:
        logger.info("⏳ Waiting for LitServe workers to be ready...")
        import time

        max_retries = 30
        for i in range(max_retries):
            if asyncio.run(health_check(args.litserve_url)):
                logger.info("✅ LitServe workers are ready!")
                break
            time.sleep(2)
            if i == max_retries - 1:
                logger.error("❌ LitServe workers not responding, starting anyway...")

    # 创建并启动调度器
    scheduler = TaskScheduler(
        litserve_url=args.litserve_url,
        monitor_interval=args.monitor_interval,
        health_check_interval=args.health_check_interval,
        stale_task_timeout=args.stale_task_timeout,
        cleanup_old_files_days=args.cleanup_old_files_days,
        cleanup_old_records_days=args.cleanup_old_records_days,
        worker_auto_mode=not args.no_worker_auto_mode,
        orphan_recovery_apply=args.orphan_recovery_apply,
        orphan_recovery_batch_size=args.orphan_recovery_batch_size,
        task_p99_seconds=args.task_p99_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("👋 Scheduler interrupted by user")
