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
import errno
import fcntl
import hashlib
import random
import tempfile
import time
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


_SQLITE_LOCK_DEADLINE_SECONDS = float(os.getenv("SQLITE_LOCK_DEADLINE_SECONDS", "45"))
_SQLITE_BUSY_BASE_SLEEP_SECONDS = float(os.getenv("SQLITE_BUSY_BASE_SLEEP_SECONDS", "0.02"))
_SQLITE_BUSY_MAX_SLEEP_SECONDS = float(os.getenv("SQLITE_BUSY_MAX_SLEEP_SECONDS", "0.5"))
_SQLITE_BUSY_ERRORS = ("database is locked", "database table is locked", "SQLITE_BUSY", "SQLITE_LOCKED")


def _sqlite_queue_fallback_enabled() -> bool:
    return os.getenv("SQLITE_QUEUE_FALLBACK", "true").lower() not in {"0", "false", "no", "off"}


_REDIS_MAINTENANCE_SENTINEL = object()
_REDIS_COORDINATION_FAILURE_SENTINEL = object()


def _is_sqlite_busy(exc: Exception) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc)
    return any(token in message for token in _SQLITE_BUSY_ERRORS)


class RetryableSQLiteWriteTimeout(sqlite3.OperationalError):
    """Raised when the local SQLite write lock cannot be acquired in time."""


class _SQLiteWriteLock:
    def __init__(self, db_path: str, deadline_seconds: float = _SQLITE_LOCK_DEADLINE_SECONDS):
        digest = hashlib.sha256(str(Path(db_path).resolve()).encode()).hexdigest()[:16]
        self.path = Path(tempfile.gettempdir()) / f"mineru_tianshu_sqlite_{digest}.lock"
        self.deadline_seconds = deadline_seconds
        self._fh = None
        self.deadline = time.monotonic() + deadline_seconds

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+")
        attempt = 0
        while True:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self.deadline
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN) or time.monotonic() >= self.deadline:
                    raise RetryableSQLiteWriteTimeout(f"database is locked: timed out waiting for {self.path}") from exc
                sleep_for = min(_SQLITE_BUSY_MAX_SLEEP_SECONDS, _SQLITE_BUSY_BASE_SLEEP_SECONDS * (2 ** attempt))
                time.sleep(random.uniform(0, sleep_for))
                attempt += 1

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


class _LazyLockingCursor:
    WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER", "BEGIN IMMEDIATE", "BEGIN EXCLUSIVE", "VACUUM")
    WRITE_PRAGMAS = ("PRAGMA JOURNAL_MODE", "PRAGMA WAL_CHECKPOINT")

    def __init__(self, cursor, db_path: str):
        self._cursor = cursor
        self._db_path = db_path
        self._lock = None
        self._deadline = time.monotonic() + _SQLITE_LOCK_DEADLINE_SECONDS

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    @property
    def has_write_lock(self) -> bool:
        return self._lock is not None

    @property
    def deadline(self) -> float:
        return self._deadline

    def close_lock(self) -> None:
        if self._lock is not None:
            self._lock.__exit__(None, None, None)
            self._lock = None

    def _needs_write_lock(self, sql: str) -> bool:
        normalized = sql.lstrip().upper()
        return normalized.startswith(self.WRITE_PREFIXES) or normalized.startswith(self.WRITE_PRAGMAS)

    def _ensure_write_lock(self) -> None:
        if self._lock is None:
            self._lock = _SQLiteWriteLock(self._db_path)
            self._deadline = self._lock.__enter__()

    def _retry(self, func, *args, **kwargs):
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_busy(exc) or time.monotonic() >= self._deadline:
                    raise
                sleep_for = min(_SQLITE_BUSY_MAX_SLEEP_SECONDS, _SQLITE_BUSY_BASE_SLEEP_SECONDS * (2 ** attempt))
                time.sleep(random.uniform(0, sleep_for))
                attempt += 1

    def execute(self, sql, *args, **kwargs):
        if self._needs_write_lock(sql):
            self._ensure_write_lock()
        return self._retry(self._cursor.execute, sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        if self._needs_write_lock(sql):
            self._ensure_write_lock()
        return self._retry(self._cursor.executemany, sql, *args, **kwargs)

    def executescript(self, sql, *args, **kwargs):
        if any(self._needs_write_lock(part) for part in sql.split(";")):
            self._ensure_write_lock()
        return self._retry(self._cursor.executescript, sql, *args, **kwargs)


class TaskDB:
    """任务数据库管理类"""

    def __init__(self, db_path=None, initialize: bool = True, read_only: bool = False):
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
        self.read_only = read_only
        if initialize:
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
        if self.read_only:
            db_uri = Path(self.db_path).as_uri() + "?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False, timeout=30.0)
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")  # 锁等待 30s，避免高并发下立刻报 database is locked
        return conn

    @contextmanager
    def get_cursor(self):
        """上下文管理器，自动提交和错误处理"""
        conn = self._get_conn()
        cursor = _LazyLockingCursor(conn.cursor(), self.db_path)
        try:
            yield cursor
            while True:
                try:
                    conn.commit()
                    break
                except sqlite3.OperationalError as exc:
                    if not cursor.has_write_lock or not _is_sqlite_busy(exc) or time.monotonic() >= cursor.deadline:
                        raise
                    time.sleep(random.uniform(0, _SQLITE_BUSY_MAX_SLEEP_SECONDS))
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close_lock()
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
                    preserve_all_artifacts INTEGER DEFAULT 0,
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

            try:
                cursor.execute("SELECT preserve_all_artifacts FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding artifact policy field")
                cursor.execute(
                    "ALTER TABLE tasks ADD COLUMN preserve_all_artifacts INTEGER DEFAULT 0"
                )
                logger.info("✅ Artifact policy field added")

            # 迁移：添加 data 字段（如果不存在）
            try:
                cursor.execute("SELECT data FROM tasks LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("📊 Migrating database schema: adding data field")
                cursor.execute("ALTER TABLE tasks ADD COLUMN data TEXT")
                logger.info("✅ data field added")

            for column_name, definition in (
                ("merge_owner", "TEXT"),
                ("merge_claimed_at", "TIMESTAMP"),
                ("merge_attempts", "INTEGER DEFAULT 0"),
                ("merge_error", "TEXT"),
                ("page_count", "INTEGER"),
                ("processing_seconds", "REAL"),
                ("worker_group_index", "INTEGER"),
                ("worker_child_index", "INTEGER"),
            ):
                try:
                    cursor.execute(f"SELECT {column_name} FROM tasks LIMIT 1")
                except sqlite3.OperationalError:
                    logger.info(f"📊 Migrating database schema: adding {column_name} field")
                    cursor.execute(f"ALTER TABLE tasks ADD COLUMN {column_name} {definition}")

            # Artifact policy changes the physical result set and must be part
            # of logical-task deduplication. Rebuild only the legacy index so
            # concurrent API/worker startup does not perform repeated DDL.
            cursor.execute("PRAGMA index_info(idx_task_dedup)")
            dedup_columns = [row["name"] for row in cursor.fetchall()]
            expected_dedup_columns = [
                "file_hash",
                "backend",
                "lang",
                "method",
                "preserve_all_artifacts",
            ]
            if dedup_columns and dedup_columns != expected_dedup_columns:
                cursor.execute("DROP INDEX idx_task_dedup")
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_task_dedup
                ON tasks(file_hash, backend, lang, method, preserve_all_artifacts)
                """
            )

            if os.getenv("MINERU_AUTO_CREATE_QUEUE_INDEXES", "false").lower() in {"1", "true", "yes", "on"}:
                logger.warning("MINERU_AUTO_CREATE_QUEUE_INDEXES enabled; building queue indexes during startup")
                self.ensure_queue_indexes(cursor)

    def ensure_queue_indexes(self, cursor=None) -> None:
        """Create optional queue/reconciliation indexes during an explicit maintenance window."""
        statements = [
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_pending_claim
            ON tasks(priority DESC, created_at ASC, task_id)
            WHERE status = 'pending'
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_active_started
            ON tasks(started_at, task_id)
            WHERE status IN ('processing', 'merging')
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_parent_status_nonnull
            ON tasks(parent_task_id, status)
            WHERE parent_task_id IS NOT NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_terminal_completed_at
            ON tasks(completed_at, status)
            WHERE completed_at IS NOT NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_parent_merge_recovery
            ON tasks(merge_claimed_at, merge_attempts, task_id)
            WHERE is_parent = 1 AND child_count > 0 AND status = 'merging'
            """,
        ]
        if cursor is not None:
            for statement in statements:
                cursor.execute(statement)
            return
        with self.get_cursor() as locked_cursor:
            for statement in statements:
                locked_cursor.execute(statement)

    def get_task_by_dedup(self, file_hash: str, backend: str, lang: str, method: str, preserve_all_artifacts: bool = False) -> Optional[Dict]:
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM tasks
                WHERE file_hash = ? AND backend = ? AND lang = ? AND method = ?
                  AND preserve_all_artifacts = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (file_hash, backend, lang, method, int(preserve_all_artifacts)),
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
        preserve_all_artifacts = bool((options or {}).get("preserve_all_artifacts", False))
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tasks (
                        task_id, file_name, file_path, backend, options, priority,
                        user_id, file_hash, lang, method, preserve_all_artifacts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        int(preserve_all_artifacts),
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.get_task_by_dedup(file_hash, backend, lang, method, preserve_all_artifacts)
            if not existing:
                raise

            if user_id and existing.get("user_id") != user_id and not existing.get("shared_read"):
                self.set_task_shared_read(existing["task_id"])
                existing["shared_read"] = 1

            status = existing.get("status")
            requeued = False
            retry_info = None
            if status in ("failed", "timeout"):
                retry_info = self.retry_failed_logical_task(existing["task_id"])
                status = retry_info["status"]
                requeued = bool(retry_info["requeued_task_ids"])

            result = {
                "task_id": existing["task_id"],
                "status": status,
                "deduped": True,
                "requeued": requeued,
            }
            if retry_info is not None:
                result["retry_info"] = retry_info
            return result

        # 入队到 Redis（如果可用）
        enqueue_payload = {
            "file_name": file_name,
            "backend": backend,
        }
        enqueued = self._enqueue_to_redis(task_id, priority, enqueue_payload)
        enqueue_failed = REDIS_QUEUE_AVAILABLE and not enqueued and not _sqlite_queue_fallback_enabled()
        if enqueue_failed:
            with self.get_cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'failed',
                        error_message = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE task_id = ? AND status = 'pending'
                    """,
                    ("Initial enqueue failed: Redis queue did not accept task", task_id),
                )

        return {
            "task_id": task_id,
            "status": "failed" if enqueue_failed else "pending",
            "deduped": False,
            "requeued": False,
            "enqueue_failed": enqueue_failed,
            "enqueue_failed_task_ids": [task_id] if enqueue_failed else [],
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
        if task is _REDIS_MAINTENANCE_SENTINEL:
            return None
        if task is _REDIS_COORDINATION_FAILURE_SENTINEL:
            return None
        if task is not None:
            return task

        # Redis 不可用或出错，按开关决定是否回退到 SQLite（默认启用）
        if not _sqlite_queue_fallback_enabled():
            return None

        for attempt in range(max_retries):
            try:
                with self.get_cursor() as cursor:
                    # 使用事务确保原子性
                    cursor.execute("BEGIN IMMEDIATE")

                    # 按优先级和创建时间获取任务
                    cursor.execute("""
                        SELECT * FROM tasks
                        WHERE status = 'pending'
                          AND NOT (is_parent = 1 AND child_count > 0)
                        ORDER BY priority DESC, created_at ASC, task_id
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
                              AND NOT (is_parent = 1 AND child_count > 0)
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

    def _cleanup_failed_redis_claim(self, redis_queue, task_id: str, worker_id: str) -> None:
        try:
            redis_queue.fail(task_id, worker_id, requeue=True)
        except Exception as cleanup_error:
            logger.error(f"❌ Failed to clean up Redis claim {task_id}: {cleanup_error}")

    def _get_next_task_redis(self, worker_id: str) -> Optional[Dict]:
        """从 Redis 队列获取下一个任务"""
        if not REDIS_QUEUE_AVAILABLE:
            return None

        redis_queue = get_redis_queue()
        if not redis_queue:
            return None

        task_id = None
        try:
            # 从 Redis 原子 claim 任务 ID，并区分空队列与 Redis 不可用
            if hasattr(redis_queue, "claim"):
                claim = redis_queue.claim(worker_id, timeout=1.0)
                if claim.status == "unavailable":
                    logger.error(f"❌ Redis unavailable during claim: {claim.error}")
                    return None
                if claim.status == "maintenance":
                    logger.info("Redis queue claims paused for maintenance")
                    return _REDIS_MAINTENANCE_SENTINEL
                if claim.status == "empty":
                    return None
                task_id = claim.task_id
            else:
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
                    return _REDIS_COORDINATION_FAILURE_SENTINEL

                # 更新 SQLite 中的任务状态
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'processing',
                        started_at = CURRENT_TIMESTAMP,
                        worker_id = ?
                    WHERE task_id = ? AND status = 'pending'
                      AND NOT (is_parent = 1 AND child_count > 0)
                    """,
                    (worker_id, task_id),
                )

                if cursor.rowcount == 0:
                    logger.warning(f"⚠️  Task {task_id} status changed or is protected, skipping")
                    is_split_parent = bool(task["is_parent"] and task["child_count"] > 0)
                    redis_queue.fail(task_id, worker_id, requeue=(task["status"] == "pending" and not is_split_parent))
                    return _REDIS_COORDINATION_FAILURE_SENTINEL

                logger.info(f"📤 [Redis] Task {task_id} claimed by worker {worker_id}")
                return dict(task)

        except Exception as e:
            logger.error(f"❌ Redis/SQLite claim coordination failed: {e}")
            if task_id:
                self._cleanup_failed_redis_claim(redis_queue, task_id, worker_id)
            return _REDIS_COORDINATION_FAILURE_SENTINEL

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result_path: str = None,
        error_message: str = None,
        worker_id: str = None,
        data: str = None,  # 新增：接收扩展数据（JSON字符串）
        page_count: int = None,
        processing_seconds: float = None,
        worker_group_index: int = None,
        worker_child_index: int = None,
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
                            data = ?,
                            page_count = ?,
                            processing_seconds = ?,
                            worker_group_index = ?,
                            worker_child_index = ?
                        WHERE task_id = ?
                        AND status IN ('processing', 'merging')
                        AND worker_id = ?
                    """
                    cursor.execute(
                        sql,
                        (
                            status,
                            result_path,
                            data,
                            page_count,
                            processing_seconds,
                            worker_group_index,
                            worker_child_index,
                            task_id,
                            worker_id,
                        ),
                    )
                else:
                    sql = """
                        UPDATE tasks
                        SET status = ?,
                            completed_at = CURRENT_TIMESTAMP,
                            result_path = ?,
                            data = ?,
                            page_count = ?,
                            processing_seconds = ?,
                            worker_group_index = ?,
                            worker_child_index = ?
                        WHERE task_id = ?
                        AND status IN ('processing', 'merging')
                    """
                    cursor.execute(
                        sql,
                        (
                            status,
                            result_path,
                            data,
                            page_count,
                            processing_seconds,
                            worker_group_index,
                            worker_child_index,
                            task_id,
                        ),
                    )

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
        #    merging parent tasks are owned exclusively by the merge reconciler.
        #    排除：is_parent=1 且 children 未全部完成的 processing 任务
        #    —— 这些父任务在正常等待 chunk 完成，不是孤儿，reset 会破坏合并流程
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT task_id, worker_id, retry_count, is_parent, child_count, child_completed
                FROM tasks
                WHERE status = 'processing'
                AND started_at < datetime('now', '-' || ? || ' minutes')
                AND NOT (
                    is_parent = 1
                    AND child_count > 0
                )
                """,
                (stale_minutes,),
            )
            stale_rows = cursor.fetchall()

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
                    SET is_parent = 1,
                        child_count = 0,
                        child_completed = 0,
                        status = 'processing',
                        merge_owner = NULL,
                        merge_claimed_at = NULL,
                        merge_attempts = 0,
                        merge_error = NULL
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
                    SET is_parent = 1,
                        child_count = ?,
                        status = 'processing',
                        merge_owner = NULL,
                        merge_claimed_at = NULL,
                        merge_attempts = 0,
                        merge_error = NULL
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

    def _count_completed_children(self, cursor, parent_task_id: str) -> int:
        cursor.execute(
            """
            SELECT COUNT(*) AS completed_count
            FROM tasks
            WHERE parent_task_id = ? AND status = 'completed'
            """,
            (parent_task_id,),
        )
        row = cursor.fetchone()
        return int(row["completed_count"] if row else 0)

    def claim_parent_merge(
        self,
        parent_task_id: str,
        merge_owner: str,
        stale_seconds: int = None,
        max_attempts: int = None,
        force: bool = False,
    ) -> bool:
        """Atomically claim a finalizable parent merge for one owner."""
        with self.get_cursor() as cursor:
            cursor.execute("BEGIN IMMEDIATE")
            completed_children = self._count_completed_children(cursor, parent_task_id)
            cursor.execute(
                """
                UPDATE tasks
                SET child_completed = ?,
                    status = 'merging',
                    merge_owner = ?,
                    merge_claimed_at = CURRENT_TIMESTAMP,
                    merge_attempts = COALESCE(merge_attempts, 0) + 1,
                    merge_error = NULL
                WHERE task_id = ?
                  AND is_parent = 1
                  AND child_count > 0
                  AND child_count <= ?
                  AND status != 'completed'
                  AND (
                        ? = 1
                     OR status IN ('processing', 'pending')
                     OR (
                        status = 'merging'
                        AND (
                            merge_owner IS NULL
                            OR merge_claimed_at IS NULL
                            OR (? IS NOT NULL AND merge_claimed_at <= datetime('now', '-' || ? || ' seconds'))
                        )
                     )
                  )
                  AND (? IS NULL OR COALESCE(merge_attempts, 0) < ?)
                """,
                (
                    completed_children,
                    merge_owner,
                    parent_task_id,
                    completed_children,
                    1 if force else 0,
                    stale_seconds,
                    stale_seconds,
                    max_attempts,
                    max_attempts,
                ),
            )
            return cursor.rowcount > 0

    def complete_parent_merge(self, parent_task_id: str, result_path: str, merge_owner: str, data: str = None) -> bool:
        """Complete a merge only for the owner that claimed it."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    result_path = ?,
                    data = COALESCE(?, data),
                    merge_owner = NULL,
                    merge_claimed_at = NULL,
                    merge_error = NULL
                WHERE task_id = ?
                  AND status = 'merging'
                  AND merge_owner = ?
                """,
                (result_path, data, parent_task_id, merge_owner),
            )
            return cursor.rowcount > 0

    def release_parent_merge(self, parent_task_id: str, merge_owner: str, error_message: str = None) -> bool:
        """Release a claimed merge lease without making it claimable as a normal task."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks
                SET merge_owner = NULL,
                    merge_claimed_at = NULL,
                    merge_error = ?
                WHERE task_id = ?
                  AND status = 'merging'
                  AND merge_owner = ?
                """,
                (error_message, parent_task_id, merge_owner),
            )
            return cursor.rowcount > 0

    def block_parent_merge(
        self, parent_task_id: str, merge_owner: str, error_message: str = None, max_attempts: int = 3
    ) -> bool:
        """Pin a repeatedly failing merge as reconciler-blocked while keeping status='merging'."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE tasks
                SET error_message = ?,
                    merge_error = ?,
                    merge_attempts = MAX(COALESCE(merge_attempts, 0), ?),
                    merge_owner = NULL,
                    merge_claimed_at = NULL
                WHERE task_id = ?
                  AND status = 'merging'
                  AND merge_owner = ?
                """,
                (error_message, error_message, max_attempts, parent_task_id, merge_owner),
            )
            return cursor.rowcount > 0

    def list_parent_merge_candidates(
        self,
        stale_seconds: int = 300,
        max_attempts: int = 3,
        limit: int = 100,
        task_id: str = None,
        force: bool = False,
    ) -> List[Dict]:
        """Return parent merge recovery candidates classified by current DB evidence."""
        with self.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(tasks)")
            columns = {row["name"] for row in cursor.fetchall()}
            merge_attempts_expr = "COALESCE(p.merge_attempts, 0)" if "merge_attempts" in columns else "0"
            where = "WHERE p.is_parent = 1 AND p.child_count > 0 AND p.status IN ('processing', 'pending', 'merging')"
            params = []
            if task_id:
                where += " AND p.task_id = ?"
                params.append(task_id)
            cursor.execute(
                f"""
                SELECT p.*,
                       SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) AS real_child_completed,
                       SUM(CASE WHEN c.status = 'failed' THEN 1 ELSE 0 END) AS child_failed
                FROM tasks p
                LEFT JOIN tasks c ON c.parent_task_id = p.task_id
                {where}
                GROUP BY p.task_id
                ORDER BY
                    CASE WHEN {merge_attempts_expr} >= ? THEN 1 ELSE 0 END,
                    p.created_at
                LIMIT ?
                """,
                (*params, max_attempts, limit),
            )
            rows = []
            for row in cursor.fetchall():
                item = dict(row)
                completed = int(item.get("real_child_completed") or 0)
                child_count = int(item.get("child_count") or 0)
                attempts = int(item.get("merge_attempts") or 0)
                status = item.get("status")
                is_stale = False
                if item.get("merge_claimed_at"):
                    cursor.execute(
                        "SELECT ? <= datetime('now', '-' || ? || ' seconds') AS stale",
                        (item["merge_claimed_at"], stale_seconds),
                    )
                    is_stale = bool(cursor.fetchone()["stale"])
                has_open_merge_lease = bool(item.get("merge_owner")) and bool(item.get("merge_claimed_at"))
                if attempts >= max_attempts or item.get("child_failed"):
                    classification = "blocked"
                elif completed >= child_count and status in ("processing", "pending"):
                    classification = "finalizable"
                elif completed >= child_count and status == "merging" and (force or not has_open_merge_lease or is_stale):
                    classification = "remergeable"
                else:
                    classification = "blocked"
                item["classification"] = classification
                item["real_child_completed"] = completed
                item["child_failed"] = int(item.get("child_failed") or 0)
                rows.append(item)
            return rows

    def list_completed_parent_tasks(self, limit: int = 100, task_id: str = None) -> List[Dict]:
        """List completed split parents eligible for an explicit artifact rebuild."""
        with self.get_cursor() as cursor:
            where = "WHERE p.is_parent = 1 AND p.child_count > 0 AND p.status = 'completed'"
            params = []
            if task_id:
                where += " AND p.task_id = ?"
                params.append(task_id)
            cursor.execute(
                f"""
                SELECT p.*,
                       SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) AS real_child_completed
                FROM tasks p
                LEFT JOIN tasks c ON c.parent_task_id = p.task_id
                {where}
                GROUP BY p.task_id
                ORDER BY p.completed_at DESC, p.created_at DESC
                LIMIT ?
                """,
                (*params, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def on_child_task_completed(self, child_task_id: str, merge_owner: str = None) -> Optional[str]:
        """子任务完成回调。返回 parent_task_id 表示当前 owner 赢得合并权。"""
        merge_owner = merge_owner or "unknown-merge-owner"
        with self.get_cursor() as cursor:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                SELECT parent_task_id, status FROM tasks WHERE task_id = ?
                """,
                (child_task_id,),
            )
            row = cursor.fetchone()
            if not row or not row["parent_task_id"]:
                return None
            if row["status"] != "completed":
                logger.info(f"⏭️ Child {child_task_id} status={row['status']}, skip merge callback")
                return None

            parent_task_id = row["parent_task_id"]
            completed_children = self._count_completed_children(cursor, parent_task_id)
            cursor.execute(
                """
                UPDATE tasks
                SET child_completed = ?
                WHERE task_id = ?
                """,
                (completed_children, parent_task_id),
            )
            cursor.execute(
                """
                SELECT child_count, file_name, status
                FROM tasks WHERE task_id = ?
                """,
                (parent_task_id,),
            )
            parent = cursor.fetchone()

        if parent and completed_children >= parent["child_count"] and parent["child_count"] > 0:
            if self.claim_parent_merge(parent_task_id, merge_owner):
                logger.info(
                    f"🎉 All subtasks completed for parent task {parent_task_id} "
                    f"({completed_children}/{parent['child_count']}, was {parent['status']}) - {parent['file_name']}"
                )
                return parent_task_id
            logger.info(f"↩️ Merge already claimed by another owner for {parent_task_id}")
        elif parent:
            logger.info(f"⏳ Subtask progress: {completed_children}/{parent['child_count']} for parent task {parent_task_id}")
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
        重试任务：兼容旧 bool 契约，同时避免拆分父任务被错误入队。
        """
        result = self.retry_failed_logical_task(task_id)
        if result["mode"] == "missing" or result.get("enqueue_failed_task_ids"):
            return False
        return bool(result.get("requeued_task_ids")) or result["mode"] in {
            "split_ready_to_merge",
            "split_awaiting_children",
        }

    def _mark_retry_enqueue_failed(self, failed_task_ids: List[str], parent_task_id: str = None) -> None:
        if not failed_task_ids:
            return
        placeholders = ",".join("?" for _ in failed_task_ids)
        message = "Retry enqueue failed: Redis queue did not accept task"
        with self.get_cursor() as cursor:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                f"""
                UPDATE tasks
                SET status = 'failed',
                    error_message = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    worker_id = NULL
                WHERE task_id IN ({placeholders})
                  AND status = 'pending'
                """,
                [message, *failed_task_ids],
            )
            if parent_task_id:
                completed_children = self._count_completed_children(cursor, parent_task_id)
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'failed',
                        error_message = ?,
                        completed_at = CURRENT_TIMESTAMP,
                        worker_id = NULL,
                        child_completed = ?,
                        merge_owner = NULL,
                        merge_claimed_at = NULL,
                        merge_error = NULL
                    WHERE task_id = ?
                    """,
                    (f"Retry enqueue failed for {len(failed_task_ids)} child task(s)", completed_children, parent_task_id),
                )

    def retry_failed_logical_task(self, task_id: str) -> Dict:
        """Retry a failed logical task without enqueueing an existing split parent.

        Split parents are orchestration rows: when one child failed, retrying the
        parent itself would be claimed by workers and then skipped because the
        child rows already exist. Instead, only failed/timeout children are made
        pending again and enqueued after the transaction commits.
        """
        requeued_tasks = []
        parent_task_id = None
        with self.get_cursor() as cursor:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            task = cursor.fetchone()
            if not task:
                return {
                    "task_id": task_id,
                    "mode": "missing",
                    "status": None,
                    "requeued_task_ids": [],
                    "requeued_child_count": 0,
                    "enqueue_failed_task_ids": [],
                }

            is_split_parent = bool(task["is_parent"] and task["child_count"] > 0)
            if not is_split_parent:
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending',
                        error_message = NULL,
                        started_at = NULL,
                        completed_at = NULL,
                        worker_id = NULL,
                        retry_count = retry_count + CASE
                            WHEN error_message LIKE 'Retry enqueue failed:%' THEN 0
                            ELSE 1
                        END
                    WHERE task_id = ?
                      AND status IN ('failed', 'timeout')
                    """,
                    (task_id,),
                )
                if cursor.rowcount > 0:
                    requeued_tasks.append(
                        {
                            "task_id": task_id,
                            "priority": task["priority"],
                            "file_name": task["file_name"],
                            "backend": task["backend"],
                            "parent_task_id": task["parent_task_id"],
                        }
                    )
                    status = "pending"
                else:
                    status = task["status"]
                mode = "leaf"
            else:
                parent_task_id = task_id
                completed_children = self._count_completed_children(cursor, task_id)
                cursor.execute(
                    """
                    SELECT task_id, priority, file_name, backend, parent_task_id
                    FROM tasks
                    WHERE parent_task_id = ?
                      AND status IN ('failed', 'timeout')
                    ORDER BY created_at, task_id
                    """,
                    (task_id,),
                )
                failed_children = [dict(row) for row in cursor.fetchall()]
                failed_child_ids = [child["task_id"] for child in failed_children]

                if failed_child_ids:
                    placeholders = ",".join("?" for _ in failed_child_ids)
                    cursor.execute(
                        f"""
                        UPDATE tasks
                        SET status = 'pending',
                            error_message = NULL,
                            started_at = NULL,
                            completed_at = NULL,
                            worker_id = NULL,
                            result_path = NULL,
                            retry_count = retry_count + CASE
                                WHEN error_message LIKE 'Retry enqueue failed:%' THEN 0
                                ELSE 1
                            END
                        WHERE task_id IN ({placeholders})
                          AND status IN ('failed', 'timeout')
                        """,
                        failed_child_ids,
                    )
                    requeued_tasks.extend(failed_children)
                    mode = "split_children_requeued"
                elif completed_children >= int(task["child_count"] or 0):
                    mode = "split_ready_to_merge"
                else:
                    mode = "split_awaiting_children"

                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'processing',
                        child_completed = ?,
                        error_message = NULL,
                        completed_at = NULL,
                        merge_owner = NULL,
                        merge_claimed_at = NULL,
                        merge_error = NULL,
                        worker_id = NULL
                    WHERE task_id = ?
                    """,
                    (completed_children, task_id),
                )
                status = "processing"

        enqueue_failed_task_ids = []
        successful_requeued_tasks = []
        for item in requeued_tasks:
            payload = {
                "file_name": item.get("file_name"),
                "backend": item.get("backend"),
            }
            if item.get("parent_task_id"):
                payload["parent_task_id"] = item.get("parent_task_id")
            enqueued = self._enqueue_to_redis(item["task_id"], item.get("priority", 0), payload)
            if REDIS_QUEUE_AVAILABLE and not enqueued:
                enqueue_failed_task_ids.append(item["task_id"])
            else:
                successful_requeued_tasks.append(item)

        if enqueue_failed_task_ids:
            self._mark_retry_enqueue_failed(enqueue_failed_task_ids, parent_task_id=parent_task_id)
            status = "failed"

        return {
            "task_id": task_id,
            "mode": mode,
            "status": status,
            "requeued_task_ids": [item["task_id"] for item in successful_requeued_tasks],
            "requeued_child_count": sum(1 for item in successful_requeued_tasks if item.get("parent_task_id") == task_id),
            "enqueue_failed_task_ids": enqueue_failed_task_ids,
        }

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
