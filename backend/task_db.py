"""
MinerU Tianshu - SQLite Task Database Manager
天枢任务数据库管理器

负责任务的持久化存储、状态管理和原子性操作

架构说明 (Hybrid Queue):
    - SQLite: 任务元数据存储、历史记录、结果管理
    - Redis (可选): 高性能任务队列、优先级调度
    - 当 Redis 可用时，队列操作由 Redis 处理
    - 当 Redis 不可用时，自动回退到 SQLite

更新日志:
    - [新增] data 字段支持，用于存储 json_content 和 pdf_path 等扩展元数据
    - [修复] clear_failed_tasks 增加物理文件删除逻辑
"""

import sqlite3
import json
import uuid
import shutil
import os
from contextlib import contextmanager
from typing import Optional, List, Dict
from pathlib import Path
from loguru import logger

# 导入 Redis 队列（可选）
try:
    from redis_queue import get_redis_queue

    REDIS_QUEUE_AVAILABLE = True
except ImportError:
    REDIS_QUEUE_AVAILABLE = False

    def get_redis_queue():
        return None


class TaskDB:
    """任务数据库管理类"""

    def __init__(self, db_path=None):
        # 导入所需模块
        import os
        from pathlib import Path

        # 优先使用传入的路径，其次使用环境变量，最后使用默认路径
        if db_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent
            default_db = project_root / "data" / "db" / "mineru_tianshu.db"
            db_path = os.getenv("DATABASE_PATH", str(default_db))
            # 确保父目录存在
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            # 确保使用绝对路径
            db_path = str(Path(db_path).resolve())
        else:
            # 确保使用绝对路径
            db_path = str(Path(db_path).resolve())

        # 确保 db_path 是绝对路径字符串
        self.db_path = str(Path(db_path).resolve())
        self._init_db()

    def _get_conn(self):
        """获取数据库连接（每次创建新连接，避免 pickle 问题）

        并发安全说明：
            - 使用 check_same_thread=False 是安全的，因为：
              1. 每次调用都创建新连接，不跨线程共享
              2. 连接使用完立即关闭（在 get_cursor 上下文管理器中）
              3. 不使用连接池，避免线程间共享同一连接
            - timeout=30.0 防止死锁，如果锁等待超过30秒会抛出异常
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")  # 锁等待 30s，避免高并发下立刻报 database is locked
        return conn

    @contextmanager
    def get_cursor(self):
        """上下文管理器，自动提交和错误处理"""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()  # 关闭连接

    def _init_db(self):
        """初始化数据库表"""
        with self.get_cursor() as cursor:
            # 启用 WAL 模式：允许并发读写（8 workers + API + scheduler 同时写入不再互斥）
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA wal_autocheckpoint=1000")

            # 创建表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    backend TEXT DEFAULT 'pipeline',
                    options TEXT,
                    result_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    worker_id TEXT,
                    shared_read INTEGER DEFAULT 0,
                    file_hash TEXT,
                    lang TEXT,
                    method TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            """)

            # 创建基础索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_worker_id ON tasks(worker_id)")

            # 迁移：添加 parent_task_id 等字段（如果不存在）
            try:
                cursor.execute("SELECT parent_task_id FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                # 字段不存在，添加新字段
                logger.info("📊 Migrating database schema: adding parent-child task support")
                cursor.execute("ALTER TABLE tasks ADD COLUMN parent_task_id TEXT")
                cursor.execute("ALTER TABLE tasks ADD COLUMN is_parent INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE tasks ADD COLUMN child_count INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE tasks ADD COLUMN child_completed INTEGER DEFAULT 0")
                logger.info("✅ Parent-child task fields added")

            # 创建主子任务索引（字段存在后才创建）
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent_task ON tasks(parent_task_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_parent ON tasks(is_parent)")

            # 迁移：添加 user_id 字段（如果不存在）
            try:
                cursor.execute("SELECT user_id FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding user_id field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN user_id TEXT")
                logger.info("✅ user_id field added")

            # 迁移：添加 shared_read 字段（如果不存在）
            try:
                cursor.execute("SELECT shared_read FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding shared_read field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN shared_read INTEGER DEFAULT 0")
                logger.info("✅ shared_read field added")

            # 迁移：添加 file_hash/lang/method 字段（如果不存在）
            try:
                cursor.execute("SELECT file_hash FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding file_hash field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN file_hash TEXT")
                logger.info("✅ file_hash field added")

            try:
                cursor.execute("SELECT lang FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding lang field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN lang TEXT")
                logger.info("✅ lang field added")

            try:
                cursor.execute("SELECT method FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding method field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN method TEXT")
                logger.info("✅ method field added")

            # 迁移：添加 data 字段（如果不存在）
            try:
                cursor.execute("SELECT data FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding data field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN data TEXT")
                logger.info("✅ data field added")

            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_task_dedup ON tasks(file_hash, backend, lang, method)"
            )

    def get_task_by_dedup(self, file_hash: str, backend: str, lang: str, method: str) -> Optional[Dict]:
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM tasks
                WHERE file_hash = ? AND backend = ? AND lang = ? AND method = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (file_hash, backend, lang, method),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def set_task_shared_read(self, task_id: str) -> None:
        with self.get_cursor() as cursor:
            cursor.execute("UPDATE tasks SET shared_read = 1 WHERE task_id = ?", (task_id,))

    def create_task(
        self,
        file_name: str,
        file_path: str,
        backend: str = "pipeline",
        options: dict = None,
        priority: int = 0,
        user_id: str = None,
        file_hash: str = None,
        lang: str = None,
        method: str = None,
    ) -> Dict:
        """
        创建新任务（含去重）
        """
        task_id = str(uuid.uuid4())
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tasks (task_id, file_name, file_path, backend, options, priority, user_id, file_hash, lang, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        task_id,
                        file_name,
                        file_path,
                        backend,
                        json.dumps(options or {}),
                        priority,
                        user_id,
                        file_hash,
                        lang,
                        method,
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.get_task_by_dedup(file_hash, backend, lang, method)
            if not existing:
                raise

            if user_id and existing.get("user_id") != user_id and not existing.get("shared_read"):
                self.set_task_shared_read(existing["task_id"])
                existing["shared_read"] = 1

            status = existing.get("status")
            requeued = False
            if status in ("failed", "timeout"):
                if self.retry_task(existing["task_id"]):
                    status = "pending"
                    requeued = True
                    self._enqueue_to_redis(
                        existing["task_id"],
                        existing.get("priority", 0),
                        {
                            "file_name": existing.get("file_name"),
                            "backend": existing.get("backend"),
                        },
                    )

            return {
                "task_id": existing["task_id"],
                "status": status,
                "deduped": True,
                "requeued": requeued,
            }

        # 入队到 Redis（如果可用）
        self._enqueue_to_redis(
            task_id,
            priority,
            {
                "file_name": file_name,
                "backend": backend,
            },
        )

        return {
            "task_id": task_id,
            "status": "pending",
            "deduped": False,
            "requeued": False,
        }

    def _enqueue_to_redis(self, task_id: str, priority: int, task_data: dict = None) -> bool:
        """将任务加入 Redis 队列"""
        if not REDIS_QUEUE_AVAILABLE:
            return False

        redis_queue = get_redis_queue()
        if redis_queue:
            try:
                return redis_queue.enqueue(task_id, priority, task_data)
            except Exception as e:
                logger.warning(f"⚠️  Failed to enqueue to Redis, SQLite fallback active: {e}")
        return False

    def get_next_task(self, worker_id: str, max_retries: int = 3) -> Optional[Dict]:
        """
        获取下一个待处理任务（原子操作，防止并发冲突）
        """
        from loguru import logger

        # 尝试使用 Redis 队列（如果可用）
        task = self._get_next_task_redis(worker_id)
        if task is not None:
            return task

        # Redis 不可用或出错，回退到 SQLite
        for attempt in range(max_retries):
            try:
                with self.get_cursor() as cursor:
                    # 使用事务确保原子性
                    cursor.execute("BEGIN IMMEDIATE")

                    # 按优先级和创建时间获取任务
                    cursor.execute("""
                        SELECT * FROM tasks
                        WHERE status = 'pending'
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                    """)

                    task = cursor.fetchone()
                    if task:
                        task_id = task["task_id"]
                        # 立即标记为 processing，并确保状态仍是 pending
                        cursor.execute(
                            """
                            UPDATE tasks
                            SET status = 'processing',
                                started_at = CURRENT_TIMESTAMP,
                                worker_id = ?
                            WHERE task_id = ? AND status = 'pending'
                        """,
                            (worker_id, task_id),
                        )

                        # 检查是否更新成功（防止被其他 worker 抢走）
                        if cursor.rowcount == 0:
                            # 任务被其他进程抢走了，立即重试
                            if attempt == 0:  # 只在第一次尝试时记录日志
                                logger.debug(f"Task {task_id} was grabbed by another worker, retrying...")
                            continue

                        return dict(task)
                    else:
                        # 队列中没有待处理任务，返回 None
                        if attempt == 0:
                            # 检查是否有 pending 任务（用于诊断）
                            cursor.execute("SELECT COUNT(*) as count FROM tasks WHERE status = 'pending'")
                            pending_count = cursor.fetchone()["count"]
                            if pending_count > 0:
                                logger.warning(
                                    f"⚠️  Found {pending_count} pending tasks but failed to grab one "
                                    f"(attempt {attempt + 1}/{max_retries})"
                                )
                        return None

            except Exception as e:
                logger.error(f"❌ Error in get_next_task (attempt {attempt + 1}/{max_retries}): {e}")
                logger.exception(e)
                if attempt == max_retries - 1:
                    return None
                # 等待一小段时间后重试
                import time

                time.sleep(0.1)

        # 重试次数用尽，仍未获取到任务（高并发场景）
        logger.warning(f"⚠️  Failed to get task after {max_retries} attempts")
        return None

    def _get_next_task_redis(self, worker_id: str) -> Optional[Dict]:
        """从 Redis 队列获取下一个任务"""
        if not REDIS_QUEUE_AVAILABLE:
            return None

        redis_queue = get_redis_queue()
        if not redis_queue:
            return None

        try:
            # 从 Redis 获取任务 ID（阻塞式，1秒超时）
            task_id = redis_queue.dequeue(worker_id, timeout=1.0)
            if not task_id:
                return None

            # 从 SQLite 获取完整任务数据
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
                task = cursor.fetchone()

                if not task:
                    logger.error(f"❌ Task {task_id} found in Redis but not in SQLite")
                    redis_queue.fail(task_id, worker_id, requeue=False)
                    return None

                # 更新 SQLite 中的任务状态
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'processing',
                        started_at = CURRENT_TIMESTAMP,
                        worker_id = ?
                    WHERE task_id = ? AND status = 'pending'
                    """,
                    (worker_id, task_id),
                )

                if cursor.rowcount == 0:
                    logger.warning(f"⚠️  Task {task_id} status changed, skipping")
                    redis_queue.fail(task_id, worker_id, requeue=False)
                    return None

                logger.info(f"📤 [Redis] Task {task_id} claimed by worker {worker_id}")
                return dict(task)

        except Exception as e:
            logger.error(f"❌ Redis dequeue failed, falling back to SQLite: {e}")
            return None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result_path: str = None,
        error_message: str = None,
        worker_id: str = None,
        data: str = None,  # 新增：接收扩展数据（JSON字符串）
    ):
        """
        更新任务状态
        """
        with self.get_cursor() as cursor:
            success = False

            # 根据不同状态使用预定义的 SQL 模板
            if status == "completed":
                # 修复：写入 data 字段
                if worker_id:
                    sql = """
                        UPDATE tasks
                        SET status = ?,
                            completed_at = CURRENT_TIMESTAMP,
                            result_path = ?,
                            data = ?
                        WHERE task_id = ?
                        AND status IN ('processing', 'merging')
                        AND worker_id = ?
                    """
                    cursor.execute(sql, (status, result_path, data, task_id, worker_id))
                else:
                    sql = """
                        UPDATE tasks
                        SET status = ?,
                            completed_at = CURRENT_TIMESTAMP,
                            result_path = ?,
                            data = ?
                        WHERE task_id = ?
                        AND status IN ('processing', 'merging')
                    """
                    cursor.execute(sql, (status, result_path, data, task_id))

                success = cursor.rowcount > 0

            elif status == "failed":
                if worker_id:
                    sql = """
                        UPDATE tasks
                        SET status = ?,
                            completed_at = CURRENT_TIMESTAMP,
                            error_message = ?
                        WHERE task_id = ?
                        AND status IN ('processing', 'merging')
                        AND worker_id = ?
                    """
                    cursor.execute(sql, (status, error_message, task_id, worker_id))
                else:
                    sql = """
                        UPDATE tasks
                        SET status = ?,
                            completed_at = CURRENT_TIMESTAMP,
                            error_message = ?
                        WHERE task_id = ?
                        AND status IN ('processing', 'merging')
                    """
                    cursor.execute(sql, (status, error_message, task_id))

                success = cursor.rowcount > 0

            elif status == "cancelled":
                sql = """
                    UPDATE tasks
                    SET status = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE task_id = ?
                """
                cursor.execute(sql, (status, task_id))
                success = cursor.rowcount > 0

            elif status == "pending":
                sql = """
                    UPDATE tasks
                    SET status = ?,
                        worker_id = NULL,
                        started_at = NULL
                    WHERE task_id = ?
                """
                cursor.execute(sql, (status, task_id))
                success = cursor.rowcount > 0

            else:
                sql = """
                    UPDATE tasks
                    SET status = ?
                    WHERE task_id = ?
                """
                cursor.execute(sql, (status, task_id))
                success = cursor.rowcount > 0

            # 调试日志（仅在失败时）
            if not success and status in ["completed", "failed"]:
                from loguru import logger

                logger.debug(f"Status update failed: task_id={task_id}, status={status}, " f"worker_id={worker_id}")

            # 通知 Redis 任务完成/失败（清理 processing set）
            if success and status in ["completed", "failed"]:
                self._notify_redis_task_done(task_id, worker_id or "", status)

            return success

    def _notify_redis_task_done(self, task_id: str, worker_id: str, status: str):
        """通知 Redis 任务已完成/失败"""
        if not REDIS_QUEUE_AVAILABLE:
            return

        redis_queue = get_redis_queue()
        if redis_queue:
            try:
                if status == "completed":
                    redis_queue.complete(task_id, worker_id)
                else:
                    redis_queue.fail(task_id, worker_id, requeue=False)
            except Exception as e:
                logger.warning(f"⚠️  Failed to notify Redis about task {task_id}: {e}")

    def get_task(self, task_id: str) -> Optional[Dict]:
        """查询任务详情"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            task = cursor.fetchone()
            return dict(task) if task else None

    def get_queue_stats(self) -> Dict[str, int]:
        """获取队列统计信息"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM tasks
                GROUP BY status
            """)
            stats = {row["status"]: row["count"] for row in cursor.fetchall()}

        # 添加 Redis 队列统计（如果可用）
        if REDIS_QUEUE_AVAILABLE:
            redis_queue = get_redis_queue()
            if redis_queue:
                try:
                    redis_stats = redis_queue.get_stats()
                    stats["_redis_enabled"] = True
                    stats["_redis_pending"] = redis_stats.get("pending", 0)
                    stats["_redis_processing"] = redis_stats.get("processing", 0)
                except Exception as e:
                    stats["_redis_enabled"] = False
                    stats["_redis_error"] = str(e)
            else:
                stats["_redis_enabled"] = False
        else:
            stats["_redis_enabled"] = False

        return stats

    def get_tasks_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """根据状态获取任务列表"""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (status, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # 核心修复：物理删除文件逻辑
    # -------------------------------------------------------------------------
    def _delete_task_files(self, task_row):
        """辅助方法：安全删除任务的源文件和结果目录"""
        task_id = task_row["task_id"]
        
        # 1. 删除上传的源文件
        if task_row["file_path"]:
            try:
                fp = Path(task_row["file_path"])
                if fp.exists() and fp.is_file():
                    fp.unlink()
                    logger.debug(f"Deleted source file for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to delete source file for task {task_id}: {e}")
        
        # 2. 删除结果目录
        if task_row["result_path"]:
            try:
                rp = Path(task_row["result_path"])
                if rp.exists() and rp.is_dir():
                    shutil.rmtree(rp)
                    logger.debug(f"Deleted result dir for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to delete result dir for task {task_id}: {e}")

    def cleanup_old_task_records(self, days: int = 30):
        """清理旧任务"""
        with self.get_cursor() as cursor:
            # 先查询要删除的任务及其文件路径
            cursor.execute("""
                SELECT task_id, file_path, result_path FROM tasks
                WHERE completed_at < datetime('now', '-' || ? || ' days')
                AND status IN ('completed', 'failed')
            """, (days,))
            old_tasks = cursor.fetchall()
            
            # 删除所有相关文件
            for task in old_tasks:
                self._delete_task_files(task)
            
            # 删除数据库记录
            cursor.execute("""
                DELETE FROM tasks
                WHERE completed_at < datetime('now', '-' || ? || ' days')
                AND status IN ('completed', 'failed')
            """, (days,))
            
            return cursor.rowcount

    def reset_stale_tasks(self, timeout_minutes: int = 60) -> int:
        """重置超时的 processing 任务为 pending

        历史入口：保留签名兼容 API 路由 /admin/reset-stale。
        内部走 recover_orphans，同时清理 Redis（避免 SQLite/Redis 不同步）。
        """
        return self.recover_orphans(stale_minutes=timeout_minutes)["sqlite_reset"]

    def heartbeat(self, task_id: str, worker_id: str) -> bool:
        """转发心跳到 Redis processing hash

        Worker 处理任务期间周期性调用；进程崩溃则心跳停止，
        recover_orphans 会据此判定任务为孤儿并重排。
        """
        redis_queue = get_redis_queue()
        if not redis_queue:
            return False
        return redis_queue.heartbeat(task_id, worker_id)

    def recover_orphans(self, stale_minutes: int = 30, max_retries: int = 3) -> Dict:
        """统一孤儿任务恢复：同时处理 SQLite 和 Redis 两层

        1. SQLite 维度：扫描 started_at 超时的 processing/merging 任务。
           - retry_count < max_retries：重置为 pending（允许重试）。
           - retry_count >= max_retries：标记为 failed（毒丸任务兜底，避免无限循环）。
        2. Redis 维度：重试的 task 调 fail(requeue=True) 重入队列；
           放弃的 task 调 fail(requeue=False) 仅清除 processing hash。
        3. 幽灵清理：Redis processing hash 中存在但 SQLite 已无记录的条目直接 HDEL。

        Returns:
            {"sqlite_reset": N, "sqlite_failed": F, "redis_requeued": M, "ghosts_purged": K}
        """
        result = {"sqlite_reset": 0, "sqlite_failed": 0, "redis_requeued": 0, "ghosts_purged": 0}
        stale_task_ids: set = set()
        fail_task_ids: set = set()

        # 1. SQLite 维度：先查再改，拿到 task_id + worker_id + retry_count 供后续决策
        #    注意:status 包含 'merging' —— 合并过程中 worker 崩溃的父任务也需要恢复
        #    排除：is_parent=1 且 children 未全部完成的 processing 任务
        #    —— 这些父任务在正常等待 chunk 完成，不是孤儿，reset 会破坏合并流程
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT task_id, worker_id, retry_count, is_parent, child_count, child_completed
                FROM tasks
                WHERE status IN ('processing', 'merging')
                AND started_at < datetime('now', '-' || ? || ' minutes')
                AND NOT (
                    is_parent = 1
                    AND status = 'processing'
                    AND child_count > 0
                    AND child_completed < child_count
                )
                """,
                (stale_minutes,),
            )
            stale_rows = cursor.fetchall()

            # 单独处理：父任务 children 全部完成但仍卡在 processing（合并竞态遗漏）
            # → 设为 merging 而非 pending（park 住，防止 get_next_task 抢走重新拆分）
            stuck_parents = [
                r for r in stale_rows
                if r["is_parent"] == 1
                and r["child_count"] > 0
                and r["child_completed"] >= r["child_count"]
            ]
            if stuck_parents:
                stuck_ids = [r["task_id"] for r in stuck_parents]
                placeholders = ",".join("?" * len(stuck_ids))
                cursor.execute(
                    f"UPDATE tasks SET status = 'merging' WHERE task_id IN ({placeholders})",
                    stuck_ids,
                )
                logger.warning(
                    f"🔧 recover_orphans: {len(stuck_ids)} parent task(s) had all children "
                    f"completed but merge never fired (race condition). Set to 'merging' "
                    f"(parked safely). Run scripts/fix_orphan_merges.py to merge them."
                )
                # 从 stale_rows 中移除，不走下面的 reset/failed 逻辑
                stuck_set = set(stuck_ids)
                stale_rows = [r for r in stale_rows if r["task_id"] not in stuck_set]

            if stale_rows:
                for row in stale_rows:
                    if row["retry_count"] >= max_retries:
                        fail_task_ids.add(row["task_id"])
                    else:
                        stale_task_ids.add(row["task_id"])

                # 重试：status=pending, retry_count+1
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
                    result["sqlite_reset"] = cursor.rowcount

                # 放弃：status=failed, 记录错误, 不改 worker_id（留证据）
                if fail_task_ids:
                    placeholders = ",".join("?" * len(fail_task_ids))
                    cursor.execute(
                        f"""
                        UPDATE tasks
                        SET status = 'failed',
                            error_message = 'Task repeatedly crashed workers (native SIGSEGV/SIGABRT) after '
                                            || retry_count || ' retries. Likely a poison task.',
                            completed_at = datetime('now')
                        WHERE task_id IN ({placeholders})
                        """,
                        tuple(fail_task_ids),
                    )
                    result["sqlite_failed"] = cursor.rowcount

        # 2. Redis 维度
        redis_queue = get_redis_queue()
        if redis_queue and (stale_task_ids or fail_task_ids):
            for row in stale_rows:
                tid = row["task_id"]
                old_worker = row["worker_id"] or "unknown"
                requeue = tid in stale_task_ids
                try:
                    if redis_queue.fail(tid, old_worker, requeue=requeue):
                        if requeue:
                            result["redis_requeued"] += 1
                        else:
                            logger.warning(
                                f"☠️  recover_orphans: task {tid} permanently failed "
                                f"(exceeded {max_retries} retries, likely poison task)"
                            )
                except Exception as e:
                    logger.error(f"❌ recover_orphans: redis fail({tid}) failed: {e}")

        # 3. 幽灵清理：Redis processing hash 中有、但 SQLite 已无此 task_id 的条目
        if redis_queue:
            try:
                processing = redis_queue.client.hgetall(redis_queue.config.processing_key)
                for tid in list(processing.keys()):
                    if tid not in stale_task_ids and tid not in fail_task_ids:
                        # 再确认一次 SQLite 是否真无此任务
                        with self.get_cursor() as cursor:
                            cursor.execute(
                                "SELECT 1 FROM tasks WHERE task_id = ? AND status = 'processing'",
                                (tid,),
                            )
                            if cursor.fetchone() is None:
                                logger.warning(
                                    f"👻 recover_orphans: purging ghost task {tid} "
                                    f"(in Redis processing but absent/stale in SQLite)"
                                )
                                redis_queue.client.hdel(redis_queue.config.processing_key, tid)
                                result["ghosts_purged"] += 1
            except Exception as e:
                logger.error(f"❌ recover_orphans: ghost scan failed: {e}")

        if any(result.values()):
            logger.info(
                f"🔄 recover_orphans: reset={result['sqlite_reset']} "
                f"failed={result['sqlite_failed']} "
                f"requeued={result['redis_requeued']} ghosts={result['ghosts_purged']}"
            )
        return result

    # -------------------------------------------------------------------------
    # 新增功能：清理失败任务 (包含物理文件删除)
    # -------------------------------------------------------------------------
    def clear_failed_tasks(self) -> int:
        """
        一键清理所有失败的任务
        执行步骤: 1.查询路径 -> 2.删除磁盘文件 -> 3.删除数据库记录
        """
        with self.get_cursor() as cursor:
            # 1. 查询所有 failed 任务
            cursor.execute("SELECT task_id, file_path, result_path FROM tasks WHERE status = 'failed'")
            failed_tasks = cursor.fetchall()
            
            count = 0
            # 2. 物理删除
            for task in failed_tasks:
                self._delete_task_files(task)
                count += 1
            
            # 3. 数据库删除
            cursor.execute("DELETE FROM tasks WHERE status = 'failed'")
            logger.info(f"🧹 Cleared {cursor.rowcount} failed tasks (files deleted for {count} tasks)")
            return cursor.rowcount

    # ============================================================================
    # 主子任务支持 (Parent-Child Task Support)
    # ============================================================================

    def create_parent_task(
        self,
        file_name: str,
        file_path: str,
        backend: str = "pipeline",
        options: dict = None,
        priority: int = 0,
        user_id: str = None,
    ) -> str:
        """创建主任务"""
        task_id = str(uuid.uuid4())
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO tasks (
                    task_id, file_name, file_path, backend, options,
                    status, priority, user_id, is_parent, child_count
                ) VALUES (?, ?, ?, ?, ?, 'processing', ?, ?, 1, 0)
            """,
                (task_id, file_name, file_path, backend, json.dumps(options or {}), priority, user_id),
            )
        logger.info(f"📋 Created parent task: {task_id}")
        return task_id

    def convert_to_parent_task(self, task_id: str, child_count: int = 0):
        """将普通任务转换为父任务

        child_count=0 表示初始化父任务（拆分开始）：此时同步重置 child_completed=0，
        并清除上一轮残留的子任务（防止重试场景下旧子任务成为孤儿）。
        """
        with self.get_cursor() as cursor:
            if child_count == 0:
                # 初置：清除上一轮残留子任务 + 清零完成计数，防止重试场景下旧值残留导致合并提前触发
                cursor.execute("DELETE FROM tasks WHERE parent_task_id = ?", (task_id,))
                deleted = cursor.rowcount
                cursor.execute(
                    """
                    UPDATE tasks
                    SET is_parent = 1, child_count = 0, child_completed = 0, status = 'processing'
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )
                if deleted:
                    logger.info(f"🧹 Cleared {deleted} stale children from previous split of {task_id}")
            else:
                cursor.execute(
                    """
                    UPDATE tasks
                    SET is_parent = 1, child_count = ?, status = 'processing'
                    WHERE task_id = ?
                    """,
                    (child_count, task_id),
                )
        logger.info(f"🔄 Converted task {task_id} to parent task with {child_count} children")

    def create_child_task(
        self,
        parent_task_id: str,
        file_name: str,
        file_path: str,
        backend: str = "pipeline",
        options: dict = None,
        priority: int = 0,
        user_id: str = None,
    ) -> str:
        """创建子任务"""
        task_id = str(uuid.uuid4())
        with self.get_cursor() as cursor:
            # 创建子任务
            cursor.execute(
                """
                INSERT INTO tasks (
                    task_id, parent_task_id, file_name, file_path,
                    backend, options, status, priority, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
                (
                    task_id,
                    parent_task_id,
                    file_name,
                    file_path,
                    backend,
                    json.dumps(options or {}),
                    priority,
                    user_id,
                ),
            )

            # 更新父任务的子任务计数
            cursor.execute(
                """
                UPDATE tasks
                SET child_count = child_count + 1
                WHERE task_id = ?
            """,
                (parent_task_id,),
            )

        # 入 Redis 队列,让 Redis-first worker 能及时拉到子任务
        self._enqueue_to_redis(task_id, priority, {
            "file_name": file_name,
            "backend": backend,
            "parent_task_id": parent_task_id,
        })

        logger.debug(f"📄 Created child task: {task_id} (parent: {parent_task_id})")
        return task_id

    def on_child_task_completed(self, child_task_id: str) -> Optional[str]:
        """子任务完成回调。返回 parent_task_id 表示当前 worker 赢得合并权。"""
        with self.get_cursor() as cursor:
            cursor.execute("BEGIN IMMEDIATE")

            # 获取父任务ID
            cursor.execute(
                """
                SELECT parent_task_id FROM tasks WHERE task_id = ?
            """,
                (child_task_id,),
            )
            row = cursor.fetchone()

            if not row or not row["parent_task_id"]:
                return None  # 不是子任务

            parent_task_id = row["parent_task_id"]

            # 更新父任务的完成计数
            cursor.execute(
                """
                UPDATE tasks
                SET child_completed = child_completed + 1
                WHERE task_id = ?
            """,
                (parent_task_id,),
            )

            # 检查是否所有子任务都完成了
            cursor.execute(
                """
                SELECT child_count, child_completed, file_name, status
                FROM tasks WHERE task_id = ?
            """,
                (parent_task_id,),
            )
            parent = cursor.fetchone()

            if parent and parent["child_completed"] >= parent["child_count"]:
                # 原子 CAS：processing/pending → merging。
                # processing = 正常流程；pending = orphan recovery 曾重置父任务，
                #   但 children 已全部完成，仍需合并。两种状态都允许 CAS。
                # merging/completed = 另一个 worker 已认领合并，跳过。
                if parent["status"] in ("processing", "pending"):
                    cursor.execute(
                        "UPDATE tasks SET status = 'merging' "
                        "WHERE task_id = ? AND status IN ('processing', 'pending')",
                        (parent_task_id,),
                    )
                    if cursor.rowcount > 0:
                        logger.info(
                            f"🎉 All subtasks completed for parent task {parent_task_id} "
                            f"({parent['child_completed']}/{parent['child_count']}, "
                            f"was {parent['status']}) - {parent['file_name']}"
                        )
                        return parent_task_id
                    else:
                        logger.info(f"↩️ Merge already claimed by another worker for {parent_task_id}")
                else:
                    logger.info(f"⏭️ Parent {parent_task_id} status={parent['status']}, skip merge claim")

            if parent:
                logger.info(
                    f"⏳ Subtask progress: {parent['child_completed']}/{parent['child_count']} "
                    f"for parent task {parent_task_id}"
                )

        return None

    def on_child_task_failed(self, child_task_id: str, error_message: str):
        """子任务失败回调"""
        with self.get_cursor() as cursor:
            # 获取父任务ID
            cursor.execute(
                """
                SELECT parent_task_id FROM tasks WHERE task_id = ?
            """,
                (child_task_id,),
            )
            row = cursor.fetchone()

            if not row or not row["parent_task_id"]:
                return  # 不是子任务

            parent_task_id = row["parent_task_id"]

            # 标记父任务为失败
            cursor.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = ?
                WHERE task_id = ?
                AND status = 'processing'
            """,
                (f"Subtask {child_task_id} failed: {error_message}", parent_task_id),
            )

            if cursor.rowcount > 0:
                logger.error(f"❌ Parent task {parent_task_id} marked as failed due to subtask failure")

    def get_task_with_children(self, task_id: str) -> Optional[Dict]:
        """获取任务及其所有子任务"""
        with self.get_cursor() as cursor:
            # 获取主任务
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            parent_row = cursor.fetchone()

            if not parent_row:
                return None

            parent = dict(parent_row)

            # 如果是主任务，获取所有子任务
            if parent.get("is_parent"):
                cursor.execute(
                    """
                    SELECT * FROM tasks
                    WHERE parent_task_id = ?
                    ORDER BY created_at
                """,
                    (task_id,),
                )
                children = [dict(row) for row in cursor.fetchall()]
                parent["children"] = children
            else:
                parent["children"] = []

            return parent

    def get_child_tasks(self, parent_task_id: str) -> List[Dict]:
        """获取父任务的所有子任务"""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM tasks
                WHERE parent_task_id = ?
                ORDER BY created_at
            """,
                (parent_task_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ========================================================================
    # 新增功能：重试、清理、暂停、恢复、清理缓存
    # ========================================================================

    def retry_task(self, task_id: str) -> bool:
        """
        重试任务：将任务状态重置为 pending，清空错误和时间，重试次数 +1
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks 
                SET status = 'pending', 
                    error_message = NULL, 
                    started_at = NULL, 
                    completed_at = NULL, 
                    worker_id = NULL,
                    retry_count = retry_count + 1
                WHERE task_id = ?
                """,
                (task_id,)
            )
            return cursor.rowcount > 0

    def pause_task(self, task_id: str) -> bool:
        """
        暂停任务：仅允许暂停处于 pending（排队中）的任务
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks 
                SET status = 'paused' 
                WHERE task_id = ? AND status = 'pending'
                """,
                (task_id,)
            )
            return cursor.rowcount > 0

    def resume_task(self, task_id: str) -> bool:
        """
        恢复任务：将 paused 状态的任务重新放回 pending 队列
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks 
                SET status = 'pending' 
                WHERE task_id = ? AND status = 'paused'
                """,
                (task_id,)
            )
            return cursor.rowcount > 0

    def clear_task_cache(self, task_id: str) -> bool:
        """
        清理任务缓存：保留数据库历史记录，但将 result_path 标记为已清理
        """
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks 
                SET result_path = 'CLEARED' 
                WHERE task_id = ?
                """,
                (task_id,)
            )
            return cursor.rowcount > 0


if __name__ == "__main__":
    # 测试代码
    db = TaskDB("test_tianshu.db")

    # 创建测试任务
    task_result = db.create_task(
        file_name="test.pdf",
        file_path="/tmp/test.pdf",
        backend="pipeline",
        options={"lang": "ch", "formula_enable": True},
        priority=1,
    )
    print(f"Created task: {task_result['task_id']}")

    # 查询任务
    task = db.get_task(task_result["task_id"])
    print(f"Task details: {task}")

    # 获取统计
    stats = db.get_queue_stats()
    print(f"Queue stats: {stats}")

    # 清理测试数据库
    Path("test_tianshu.db").unlink(missing_ok=True)
    print("Test completed!")
