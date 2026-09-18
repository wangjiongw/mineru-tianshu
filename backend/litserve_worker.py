"""
MinerU Tianshu - LitServe Worker
天枢 LitServe Worker

企业级 AI 数据预处理平台 - GPU Worker
支持文档、图片、音频、视频等多模态数据处理
使用 LitServe 实现 GPU 资源的自动负载均衡
Worker 主动循环拉取任务并处理

优化日志 (2026-02-16):
1. [并发] 强制限制 workers_per_device=1 (可通过 MAX_CONCURRENT_TASKS 调整)，防止爆显存
2. [修复] 强制回写源 PDF 到 output 目录，解决前端无法预览源文件的问题
3. [修复] PaddleOCR/MinerU 返回结果中补全 json_content 和 pdf_path 以支持双向定位
4. [性能] 移除单次任务后的强制显存清理 (clean_memory)，依赖引擎的智能休眠机制
5. [稳定] 增强 VLLM 容器互斥切换的健壮性
"""

import os
import json
import sys
import time
import threading
import signal
import atexit
import shutil
import socket
import multiprocessing
import requests
import warnings
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# ==============================================================================
# 1. LitServe MCP Patch (Disable Internal MCP)
# ==============================================================================
try:
    import litserve.mcp as ls_mcp

    if not hasattr(ls_mcp, "MCPServer"):
        class DummyMCPServer:
            def __init__(self, *args, **kwargs): pass
        ls_mcp.MCPServer = DummyMCPServer
        if "litserve.mcp" in sys.modules:
            sys.modules["litserve.mcp"].MCPServer = DummyMCPServer

    if not hasattr(ls_mcp, "StreamableHTTPSessionManager"):
        class DummyStreamableHTTPSessionManager:
            def __init__(self, *args, **kwargs): pass
        ls_mcp.StreamableHTTPSessionManager = DummyStreamableHTTPSessionManager
        if "litserve.mcp" in sys.modules:
            sys.modules["litserve.mcp"].StreamableHTTPSessionManager = DummyStreamableHTTPSessionManager

    class DummyMCPConnector:
        """完全禁用 LitServe 内置 MCP 的 Dummy 实现"""
        def __init__(self, *args, **kwargs):
            self.mcp_server = None
            self.session_manager = None
            self.request_handler = None

        @asynccontextmanager
        async def lifespan(self, app):
            yield

        def connect_mcp_server(self, *args, **kwargs):
            pass

    ls_mcp._LitMCPServerConnector = DummyMCPConnector
    if "litserve.mcp" in sys.modules:
        sys.modules["litserve.mcp"]._LitMCPServerConnector = DummyMCPConnector

except Exception as e:
    warnings.warn(f"Failed to patch litserve.mcp (MCP will be disabled): {e}")

import litserve as ls
from litserve.connector import check_cuda_with_nvidia_smi
from loguru import logger
from output_normalizer import normalize_output

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
from task_db import TaskDB
from parent_merge import build_completion_payload, count_result_pages, merge_parent_task_results
from utils import parse_list_arg
import importlib.util


def configure_onnxruntime_thread_defaults(ort_module=None) -> bool:
    """Bound per-session ORT pools unless the caller configured them."""
    if ort_module is None:
        try:
            import onnxruntime as ort_module
        except ImportError:
            return False

    session_class = ort_module.InferenceSession
    if getattr(session_class, "_mineru_thread_defaults_patched", False):
        return True

    def positive_env(name: str, default: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(value, 1)

    intra_threads = positive_env("MINERU_INTRA_OP_NUM_THREADS", 4)
    inter_threads = positive_env("MINERU_INTER_OP_NUM_THREADS", 1)
    original_init = session_class.__init__

    def bounded_init(self, path_or_bytes, sess_options=None, *args, **kwargs):
        options = sess_options or ort_module.SessionOptions()
        if options.intra_op_num_threads <= 0:
            options.intra_op_num_threads = intra_threads
        if options.inter_op_num_threads <= 0:
            options.inter_op_num_threads = inter_threads
        return original_init(self, path_or_bytes, options, *args, **kwargs)

    session_class.__init__ = bounded_init
    session_class._mineru_thread_defaults_patched = True
    return True


configure_onnxruntime_thread_defaults()

# ==============================================================================
# 2. Dependency Checks & Global Configurations
# ==============================================================================
def check_dependency(module_name: str, display_name: str) -> bool:
    """Helper to check if a module is installed and log the result."""
    available = importlib.util.find_spec(module_name) is not None
    icon = "✅" if available else "ℹ️ "
    msg = "available" if available else "not available (optional)"
    logger.info(f"{icon} {display_name} {msg}")
    return available

# Check optional dependencies
MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
    logger.info("✅ MarkItDown available")
except ImportError:
    logger.info("ℹ️  MarkItDown not available (optional)")

PADDLEOCR_VL_AVAILABLE = check_dependency("paddleocr_vl", "PaddleOCR-VL")
PADDLEOCR_VL_VLLM_AVAILABLE = check_dependency("paddleocr_vl_vllm", "PaddleOCR-VL-VLLM")
MINERU_PIPELINE_AVAILABLE = check_dependency("mineru_pipeline", "MinerU Pipeline")
SENSEVOICE_AVAILABLE = check_dependency("audio_engines", "SenseVoice")
VIDEO_ENGINE_AVAILABLE = check_dependency("video_engines", "Video Engine")
WATERMARK_REMOVAL_AVAILABLE = check_dependency("remove_watermark", "Watermark Removal")

FORMAT_ENGINES_AVAILABLE = False
try:
    from format_engines import FormatEngineRegistry, FASTAEngine, GenBankEngine
    FormatEngineRegistry.register(FASTAEngine())
    FormatEngineRegistry.register(GenBankEngine())
    FORMAT_ENGINES_AVAILABLE = True
    logger.info(f"✅ Format Engines available: {', '.join(FormatEngineRegistry.get_supported_extensions())}")
except ImportError as e:
    logger.info(f"ℹ️  Format Engines not available: {e}")


# ==============================================================================
# 3. VLLM Container Controller
# ==============================================================================
VLLM_ENDPOINT_STRATEGY_LOCAL = "local"
VLLM_ENDPOINT_STRATEGY_RING3 = "ring3"
VLLM_ENDPOINT_STRATEGIES = {VLLM_ENDPOINT_STRATEGY_LOCAL, VLLM_ENDPOINT_STRATEGY_RING3}
RING3_ENDPOINT_COUNT = 8


def normalize_vllm_endpoint_strategy(strategy: Optional[str]) -> str:
    normalized = (strategy or VLLM_ENDPOINT_STRATEGY_LOCAL).strip().lower()
    if normalized in VLLM_ENDPOINT_STRATEGIES:
        return normalized
    return VLLM_ENDPOINT_STRATEGY_LOCAL


def parse_worker_group_index(value: Optional[str]) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def map_vllm_endpoint_index(
    endpoint_count: int,
    worker_group_index: int,
    child_index: int = 0,
    strategy: str = VLLM_ENDPOINT_STRATEGY_LOCAL,
) -> int:
    if endpoint_count <= 0:
        raise ValueError("endpoint_count must be positive")

    if normalize_vllm_endpoint_strategy(strategy) != VLLM_ENDPOINT_STRATEGY_RING3:
        return 0

    return (max(int(worker_group_index or 0), 0) + max(int(child_index or 0), 0)) % endpoint_count


def is_transient_sqlite_lock_error(error: Exception) -> bool:
    message = str(error).lower()
    return "database is locked" in message or "database table is locked" in message or "database schema is locked" in message


def is_worker_drain_requested(drain_file: Optional[str]) -> bool:
    return bool(drain_file) and Path(drain_file).exists()


def worker_activity_marker_name(worker_id: str, pid: int) -> str:
    safe_worker_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in worker_id)
    return f"{safe_worker_id}-{pid}.json"


def build_worker_activity_payload(task_id: str, worker_id: str, pid: int, started_at: float) -> dict:
    return {
        "task_id": task_id,
        "worker_id": worker_id,
        "pid": pid,
        "started_at": started_at,
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
    }


class VLLMController:
    """管理 vLLM Docker 容器的互斥启动"""

    def __init__(self):
        self._client_initialized = False
        self._client = None

    def _get_client(self):
        """按需获取并缓存 Docker 客户端。"""
        if self._client_initialized:
            return self._client

        self._client_initialized = True
        enabled = os.getenv("VLLM_DOCKER_CONTROLLER_ENABLED", "true").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None

        try:
            import docker
            self._client = docker.from_env()
        except Exception as e:
            logger.warning(f"⚠️  Docker client init failed: {e}")
        return self._client

    def ensure_service(self, target_container: str, conflict_container: str):
        """
        确保目标容器运行，并关闭冲突容器 (严格互斥逻辑)
        """
        client = self._get_client()
        if not client:
            return
        
        try:
            # 1. 检查并关闭冲突容器
            try:
                conflict = client.containers.get(conflict_container)
                if conflict.status == 'running':
                    logger.info(f"🛑 Stopping conflicting service {conflict_container} to free VRAM...")
                    conflict.stop()
                    time.sleep(2) # 等待释放
                    logger.info(f"✅ Service {conflict_container} stopped.")
            except Exception:
                pass

            # 2. 检查并启动目标容器
            try:
                target = client.containers.get(target_container)
                if target.status == 'running':
                    return
                
                logger.info(f"🚀 Starting service {target_container} (Manual/Cold Start)...")
                target.start()
                
                # 等待服务健康 (简单轮询)
                for _ in range(30):
                    time.sleep(1)
                    target.reload()
                    if target.status == 'running':
                        break
                logger.info(f"✅ Service {target_container} started.")
                
            except Exception as e:
                logger.error(f"❌ Failed to start target container {target_container}: {e}")
                raise e
        finally:
            try:
                client.close()
            except:
                pass


# ==============================================================================
# 4. MinerU Worker API
# ==============================================================================
class MinerUWorkerAPI(ls.LitAPI):
    def __init__(
        self,
        paddleocr_vl_vllm_api_list=None,
        mineru_vllm_api_list=None,
        output_dir=None,
        poll_interval=0.5,
        enable_worker_loop=True,
        paddleocr_vl_vllm_engine_enabled=False,
        workers_per_device=1,
        worker_group_index=None,
    ):
        super().__init__()
        
        # 路径配置
        project_root = Path(__file__).parent.parent
        default_output = project_root / "data" / "output"
        self.output_dir = output_dir or os.getenv("OUTPUT_PATH", str(default_output))
        
        # 运行配置
        self.poll_interval = poll_interval
        self.enable_worker_loop = enable_worker_loop
        self.workers_per_device = max(int(workers_per_device or 1), 1)
        self.worker_group_index = parse_worker_group_index(
            worker_group_index if worker_group_index is not None else os.getenv("WORKER_GROUP_INDEX")
        )
        self.worker_child_index = None
        self.vllm_endpoint_strategy = normalize_vllm_endpoint_strategy(os.getenv("VLLM_ENDPOINT_STRATEGY"))
        
        # API 配置
        self.paddleocr_vl_vllm_engine_enabled = paddleocr_vl_vllm_engine_enabled
        self.paddleocr_vl_vllm_api_list = paddleocr_vl_vllm_api_list or []
        self.mineru_vllm_api_list = mineru_vllm_api_list or []
        
        # 进程间共享计数器
        ctx = multiprocessing.get_context("spawn")
        self._global_worker_counter = ctx.Value("i", 0)

        # 初始化控制器
        self.vllm_controller = VLLMController()

    def _select_vllm_endpoint(self, endpoints: list, child_index: int) -> str:
        selected_index = map_vllm_endpoint_index(
            endpoint_count=len(endpoints),
            worker_group_index=self.worker_group_index,
            child_index=child_index,
            strategy=self.vllm_endpoint_strategy,
        )
        if self.vllm_endpoint_strategy != VLLM_ENDPOINT_STRATEGY_RING3:
            return endpoints[selected_index]

        for offset in range(len(endpoints)):
            candidate = endpoints[(selected_index + offset) % len(endpoints)]
            if self._is_vllm_endpoint_healthy(candidate):
                if offset:
                    logger.warning(
                        f"⚠️ Ring3 vLLM endpoint {endpoints[selected_index]} is unhealthy; falling back to {candidate}"
                    )
                return candidate

        logger.warning(f"⚠️ No healthy ring3 vLLM endpoints detected; using assigned endpoint {endpoints[selected_index]}")
        return endpoints[selected_index]

    def _is_vllm_endpoint_healthy(self, endpoint: str) -> bool:
        health_url = endpoint.rstrip("/")
        if health_url.endswith("/v1"):
            health_url = health_url[:-3]
        health_url = f"{health_url}/health"

        try:
            response = requests.get(health_url, timeout=1.0)
            return response.status_code < 500
        except Exception:
            return False

    def setup(self, device):
        """初始化 Worker (每个 GPU 进程调用一次)"""
        with self._global_worker_counter.get_lock():
            child_index = self._global_worker_counter.value
            self._global_worker_counter.value += 1
        self.worker_child_index = child_index
        
        logger.info(f"🔢 [Init] Worker group #{self.worker_group_index}, child #{child_index} (on {device})")
        
        # API 分配
        self.paddleocr_vl_vllm_api = None
        if self.paddleocr_vl_vllm_engine_enabled and self.paddleocr_vl_vllm_api_list:
            assigned_api = self._select_vllm_endpoint(self.paddleocr_vl_vllm_api_list, child_index)
            self.paddleocr_vl_vllm_api = assigned_api
            logger.info(f"🔧 Worker group #{self.worker_group_index}, child #{child_index} assigned Paddle OCR VL API: {assigned_api}")

        self.mineru_vllm_api = None
        if self.mineru_vllm_api_list:
            assigned_mineru_api = self._select_vllm_endpoint(self.mineru_vllm_api_list, child_index)
            self.mineru_vllm_api = assigned_mineru_api
            logger.info(f"🔧 Worker group #{self.worker_group_index}, child #{child_index} assigned MinerU VLLM API: {assigned_mineru_api}")

        # 设置 CUDA 隔离
        if "cuda:" in str(device):
            gpu_id = str(device).split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
            logger.info(f"🎯 [GPU Isolation] Set CUDA_VISIBLE_DEVICES={gpu_id}")

        # 配置模型源
        model_source = os.getenv("MODEL_DOWNLOAD_SOURCE", "auto").lower()
        if model_source in ["modelscope", "auto"]:
            try:
                importlib.util.find_spec("modelscope")
                os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
            except ImportError:
                if model_source == "modelscope":
                    logger.warning("⚠️  ModelScope not available, falling back to HuggingFace")

        if model_source == "huggingface":
            hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        # 设备配置
        self.device = device
        if "cuda" in str(device):
            self.accelerator = "cuda"
            self.engine_device = "cuda:0"
        else:
            self.accelerator = "cpu"
            self.engine_device = "cpu"

        # MinerU VRAM 设置
        from mineru.utils.model_utils import get_vram
        if os.getenv("MINERU_VIRTUAL_VRAM_SIZE", None) is None:
            if self.accelerator == "cuda":
                try:
                    vram = round(get_vram("cuda:0"))
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(vram)
                except Exception:
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "8"
            else:
                os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "1"

        # 初始化数据库
        db_path_env = os.getenv("DATABASE_PATH")
        if db_path_env:
            db_path = Path(db_path_env).resolve()
        else:
            project_root = Path(__file__).parent.parent
            db_path = (project_root / "data" / "db" / "mineru_tianshu.db").resolve()
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_db = TaskDB(str(db_path))

        # 初始化状态
        self.running = True
        self.current_task_id = None
        hostname = socket.gethostname()
        pid = os.getpid()
        self.worker_id = f"tianshu-{hostname}-{device}-{pid}"
        self.worker_drain_file = os.getenv("WORKER_DRAIN_FILE")
        self.worker_activity_dir = os.getenv("WORKER_ACTIVITY_DIR")
        self.worker_activity_marker = None
        if self.worker_activity_dir:
            self.worker_activity_marker = (
                Path(self.worker_activity_dir) / worker_activity_marker_name(self.worker_id, pid)
            )

        # 引擎占位符
        self.markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None
        self.mineru_pipeline_engine = None
        self.paddleocr_vl_engine = None
        self.paddleocr_vl_vllm_engine = None
        self.sensevoice_engine = None
        self.video_engine = None
        self.watermark_handler = None

        logger.info(f"🚀 Worker Setup Complete: {self.worker_id}")

        if WATERMARK_REMOVAL_AVAILABLE and self.accelerator == "cuda":
            try:
                from remove_watermark.pdf_watermark_handler import PDFWatermarkHandler
                self.watermark_handler = PDFWatermarkHandler(device="cuda:0", use_lama=True)
                logger.info(f"✅ Watermark engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to init watermark engine: {e}")

        if self.enable_worker_loop:
            # 心跳线程：周期性通知 Redis "我还活着"，崩溃后心跳停止 → recover_orphans 判定为孤儿
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def _heartbeat_loop(self):
        """后台心跳：每 30s 向 Redis 更新当前任务的 claimed_at"""
        HEARTBEAT_INTERVAL = 30
        while not self._heartbeat_stop.is_set():
            try:
                tid = self.current_task_id
                if tid:
                    self.task_db.heartbeat(tid, self.worker_id)
            except Exception as e:
                logger.warning(f"💓 {self.worker_id} heartbeat error: {e}")
            self._heartbeat_stop.wait(HEARTBEAT_INTERVAL)

    def _is_drain_requested(self) -> bool:
        return is_worker_drain_requested(self.worker_drain_file)

    def _write_worker_activity(self, task_id: str):
        if not self.worker_activity_marker:
            return

        marker_path = Path(self.worker_activity_marker)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_worker_activity_payload(
            task_id=task_id,
            worker_id=self.worker_id,
            pid=os.getpid(),
            started_at=time.time(),
        )
        tmp_path = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, marker_path)

    def _remove_worker_activity(self):
        if not self.worker_activity_marker:
            return

        try:
            Path(self.worker_activity_marker).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove worker activity marker {self.worker_activity_marker}: {e}")

    def _begin_task_activity(self, task_id: str):
        self.current_task_id = task_id
        try:
            self._write_worker_activity(task_id)
        except Exception as e:
            logger.warning(f"⚠️ {self.worker_id} failed to write activity marker for task {task_id}: {e}")

    def _finish_task_activity(self):
        self._remove_worker_activity()
        self.current_task_id = None

    def _worker_loop(self):
        logger.info(f"🔁 {self.worker_id} started task polling loop")
        loop_count = 0
        last_stats_log = 0

        while self.running:
            try:
                loop_count += 1
                if self._is_drain_requested():
                    if loop_count - last_stats_log >= 20:
                        logger.info(f"⏸️ {self.worker_id} drain requested; skipping new task claims")
                        last_stats_log = loop_count
                    time.sleep(self.poll_interval)
                    continue

                task = self.task_db.get_next_task(worker_id=self.worker_id)

                if task:
                    task_id = task["task_id"]
                    self._begin_task_activity(task_id)
                    logger.info(f"📥 {self.worker_id} pulled task: {task_id}")

                    try:
                        process_state = self._process_task(task)
                        if process_state == "deferred":
                            logger.warning(f"⏳ {self.worker_id} deferred terminal commit for task: {task_id}")
                        else:
                            logger.info(f"✅ {self.worker_id} completed task: {task_id}")
                    except Exception as e:
                        logger.error(f"❌ {self.worker_id} failed task {task_id}: {e}")
                        logger.exception(e)
                    finally:
                        self._finish_task_activity()
                else:
                    if loop_count - last_stats_log >= 20:
                        try:
                            stats = self.task_db.get_queue_stats()
                            if loop_count % 100 == 0:
                                logger.info(f"💤 {self.worker_id} idle. Queue stats: {stats}")
                        except: pass
                        last_stats_log = loop_count
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                time.sleep(self.poll_interval)

    def _process_task(self, task: dict):
        """处理任务"""
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}"))
        parent_task_id = task.get("parent_task_id")
        backend = task.get("backend", "auto")
        task_started_at = time.monotonic()

        try:
            # 1. 智能服务切换
            paddle_container = "tianshu-vllm-paddleocr"
            mineru_container = "tianshu-vllm-mineru"

            if backend == "paddleocr-vl-vllm" and self.paddleocr_vl_vllm_api:
                self.vllm_controller.ensure_service(target_container=paddle_container, conflict_container=mineru_container)
            elif backend in ["vlm-auto-engine", "hybrid-auto-engine", "auto"] and self.mineru_vllm_api:
                self.vllm_controller.ensure_service(target_container=mineru_container, conflict_container=paddle_container)

            file_ext = Path(file_path).suffix.lower()

            # 2. 预处理
            if file_ext in [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"] and options.get("convert_office_to_pdf", False):
                try:
                    pdf_path = self._convert_office_to_pdf(file_path)
                    file_path = pdf_path
                    file_ext = ".pdf"
                    logger.info(f"✅ Office converted to PDF: {pdf_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Office conversion failed, falling back: {e}")

            # 3. PDF 拆分
            if file_ext == ".pdf" and not parent_task_id:
                if self._should_split_pdf(task_id, file_path, task, options):
                    return

            # 4. 去水印
            if file_ext == ".pdf" and options.get("remove_watermark", False) and self.watermark_handler:
                try:
                    cleaned_path = self._preprocess_remove_watermark(file_path, options)
                    file_path = str(cleaned_path)
                except Exception as e:
                    logger.warning(f"⚠️ Watermark removal failed: {e}")

            # 5. 引擎路由
            result = None

            if backend == "sensevoice":
                if not SENSEVOICE_AVAILABLE: raise ValueError("SenseVoice not available")
                result = self._process_audio(file_path, options)

            elif backend == "video":
                if not VIDEO_ENGINE_AVAILABLE: raise ValueError("Video engine not available")
                result = self._process_video(file_path, options)

            elif backend == "paddleocr-vl":
                if not PADDLEOCR_VL_AVAILABLE: raise ValueError("PaddleOCR-VL not available")
                result = self._process_with_paddleocr_vl(file_path, options)

            elif backend == "paddleocr-vl-vllm":
                if not PADDLEOCR_VL_VLLM_AVAILABLE: raise ValueError("PaddleOCR-VL-VLLM not available")
                result = self._process_with_paddleocr_vl_vllm(file_path, options)

            elif "pipeline" in backend or "vlm-" in backend or "hybrid-" in backend:
                if not MINERU_PIPELINE_AVAILABLE: raise ValueError("MinerU Pipeline not available")
                options["parse_mode"] = backend
                result = self._process_with_mineru(file_path, options)

            elif backend == "auto":
                if FORMAT_ENGINES_AVAILABLE and FormatEngineRegistry.is_supported(file_path):
                    result = self._process_with_format_engine(file_path, options)
                elif file_ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"] and SENSEVOICE_AVAILABLE:
                    result = self._process_audio(file_path, options)
                elif file_ext in [".mp4", ".avi", ".mkv", ".mov"] and VIDEO_ENGINE_AVAILABLE:
                    result = self._process_video(file_path, options)
                elif file_ext in [".pdf", ".png", ".jpg", ".jpeg"] and MINERU_PIPELINE_AVAILABLE:
                    if self.mineru_vllm_api:
                        options["parse_mode"] = "hybrid-auto-engine"
                        options.setdefault("effort", "high")
                        logger.info(f"🔄 [Auto] vLLM available → hybrid-auto-engine (effort=high) for {file_ext}")
                    else:
                        options["parse_mode"] = "pipeline"
                        logger.info(f"🔄 [Auto] vLLM unavailable → pipeline fallback for {file_ext}")
                    result = self._process_with_mineru(file_path, options)
                elif self.markitdown:
                    result = self._process_with_markitdown(file_path)
                else:
                    raise ValueError(f"Unsupported file type for Auto mode: {file_ext}")

            else:
                if FORMAT_ENGINES_AVAILABLE:
                    engine = FormatEngineRegistry.get_engine(backend)
                    if engine:
                        result = self._process_with_format_engine(file_path, options, engine_name=backend)
                    else:
                        raise ValueError(f"Unknown backend: {backend}")
                else:
                    raise ValueError(f"Unknown backend: {backend}")

            if not result:
                raise ValueError("No result generated by engine")

            # 6. 保存完整结果到数据库 (包含 json_content 和 pdf_path)
            completed = self._complete_task_after_compute(task_id, result, options=options, started_at=task_started_at)
            if not completed:
                return "deferred"

            # 7. 合并子任务
            if parent_task_id:
                parent_id_to_merge = self.task_db.on_child_task_completed(task_id, merge_owner=self.worker_id)
                if parent_id_to_merge:
                    try:
                        self._merge_parent_task_results(parent_id_to_merge)
                    except Exception as e:
                        error_message = str(e)
                        if not self.task_db.release_parent_merge(parent_id_to_merge, self.worker_id, error_message):
                            logger.warning(f"⚠️ Failed to release parent merge claim for {parent_id_to_merge}: {error_message}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.task_db.update_task_status(task_id, "failed", error_message=error_msg, worker_id=self.worker_id)
            if parent_task_id:
                self.task_db.on_child_task_failed(task_id, error_msg)
            raise

    def _complete_task_after_compute(self, task_id: str, result: dict, options: dict = None, started_at: float = None) -> bool:
        options = options or {}
        page_count = count_result_pages(result)
        if page_count is None:
            page_count = options.get("chunk_info", {}).get("page_count")
        try:
            page_count = int(page_count) if page_count is not None else None
        except (TypeError, ValueError):
            page_count = None
        processing_seconds = None
        if started_at is not None:
            processing_seconds = max(time.monotonic() - started_at, 0.0)
        payload = build_completion_payload(
            result,
            page_count=page_count,
            worker_group_index=self.worker_group_index,
            worker_child_index=self.worker_child_index,
            processing_seconds=processing_seconds,
        )

        update_kwargs = {
            "task_id": task_id,
            "status": "completed",
            "result_path": result["result_path"],
            "error_message": None,
            "worker_id": self.worker_id,
            "data": payload,
            "page_count": page_count,
            "processing_seconds": processing_seconds,
            "worker_group_index": self.worker_group_index,
            "worker_child_index": self.worker_child_index,
        }
        retry_safe_update = self._get_retry_safe_task_status_updater()
        if retry_safe_update:
            try:
                return bool(retry_safe_update(**update_kwargs))
            except Exception as e:
                if not is_transient_sqlite_lock_error(e):
                    raise
                logger.warning(f"⚠️ Completion commit for task {task_id} was deferred after SQLite lock: {e}")
                return False

        attempts = max(int(os.getenv("TASKDB_COMPLETION_RETRY_ATTEMPTS", "3")), 1)
        retry_delay = max(float(os.getenv("TASKDB_COMPLETION_RETRY_DELAY_SECONDS", "0.2")), 0)
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                return bool(self.task_db.update_task_status(**update_kwargs))
            except Exception as e:
                if not is_transient_sqlite_lock_error(e):
                    raise
                last_error = e
                if attempt < attempts and retry_delay:
                    time.sleep(retry_delay)

        logger.warning(
            f"⚠️ Completion commit for task {task_id} deferred after {attempts} SQLite lock retry attempt(s): {last_error}"
        )
        return False

    def _get_retry_safe_task_status_updater(self):
        for method_name in (
            "update_task_status_retry_safe",
            "update_task_status_with_retry",
            "safe_update_task_status",
        ):
            method = getattr(self.task_db, method_name, None)
            if callable(method):
                return method
        return None

    # -------------------------------------------------------------------------
    # Helper: 确保 PDF 存在于 Output 目录
    # -------------------------------------------------------------------------
    def _ensure_pdf_in_output(self, file_path: str, output_dir: Path, preferred_name: str = None) -> str:
        """
        确保输出目录中有 PDF 文件，供前端预览使用。
        返回相对于 output_dir 的文件名。
        """
        output_dir = Path(output_dir)
        source_file = Path(file_path)
        
        # 1. 如果源文件不是 PDF (可能是图片)，尝试找转换后的 PDF
        if source_file.suffix.lower() != ".pdf":
             # 检查是否有 layout.pdf
            layout_pdfs = list(output_dir.glob("*_layout.pdf"))
            if layout_pdfs:
                return layout_pdfs[0].name
            return None

        # 2. 如果源文件是 PDF
        # 优先查找 MinerU 生成的带布局信息的 PDF
        layout_pdfs = list(output_dir.glob("*_layout.pdf"))
        if layout_pdfs:
            return layout_pdfs[0].name
        
        # 3. 如果没有布局 PDF，则复制源 PDF 到输出目录
        target_name = preferred_name or source_file.name
        target_path = output_dir / target_name
        
        if not target_path.exists():
            try:
                shutil.copy2(source_file, target_path)
                logger.info(f"📄 Copied source PDF to output: {target_name}")
            except Exception as e:
                logger.warning(f"Failed to copy source PDF: {e}")
                return None
        
        return target_name

    # -------------------------------------------------------------------------
    # Engine Processor Implementations
    # -------------------------------------------------------------------------
    def _process_with_mineru(self, file_path: str, options: dict) -> dict:
        if self.mineru_pipeline_engine is None:
            from mineru_pipeline import MinerUPipelineEngine
            self.mineru_pipeline_engine = MinerUPipelineEngine(
                device=self.engine_device,
                vlm_api_base=self.mineru_vllm_api
            )

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        if "http-client" in options.get("parse_mode", "") and self.mineru_vllm_api:
            options.setdefault("server_url", self.mineru_vllm_api.replace("/v1", ""))

        result = self.mineru_pipeline_engine.parse(file_path, output_path=str(output_dir), options=options)
        
        actual_output = Path(result["result_path"])
        normalize_output(
            actual_output,
            handle_method="mineru",
            preserve_intermediate_files=bool(options.get("preserve_all_artifacts", False)),
        )

        # 扁平化目录结构
        if actual_output.resolve() != output_dir.resolve():
            try:
                for item in actual_output.iterdir():
                    dest = output_dir / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: dest.unlink()
                    shutil.move(str(item), str(dest))
                shutil.rmtree(actual_output)
            except Exception as e:
                logger.warning(f"Flattening warning: {e}")
        
        # Split PDFs are ephemeral inputs. Compact child results do not need a
        # second copy; the parent publishes the authoritative source PDF.
        is_compact_child = bool(options.get("chunk_info")) and not bool(
            options.get("preserve_all_artifacts", False)
        )
        pdf_path = None if is_compact_child else self._ensure_pdf_in_output(file_path, output_dir)

        canonical_markdown = output_dir / "full.md"
        canonical_json = output_dir / "result.json"
        canonical_model = output_dir / "mineru_model.json"
        markdown_content = (
            canonical_markdown.read_text(encoding="utf-8")
            if canonical_markdown.is_file() else result.get("markdown", "")
        )
        json_content = result.get("json_content")
        model_content = None
        try:
            if canonical_json.is_file():
                json_content = json.loads(canonical_json.read_text(encoding="utf-8"))
            if canonical_model.is_file():
                model_content = json.loads(canonical_model.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not reload canonical MinerU JSON outputs: {exc}")

        return {
            "result_path": str(output_dir),
            "content": markdown_content,
            "json_content": json_content,
            "mineru_model_content": model_content,
            "total_pages": len(model_content) if isinstance(model_content, list) else None,
            "pdf_path": pdf_path,
            "markdown_file": str(canonical_markdown) if canonical_markdown.is_file() else result.get("markdown_file"),
        }

    def _process_with_paddleocr_vl(self, file_path: str, options: dict) -> dict:
        if self.accelerator == "cpu":
            raise RuntimeError("PaddleOCR-VL requires GPU")
            
        if self.paddleocr_vl_engine is None:
            from paddleocr_vl import PaddleOCRVLEngine
            self.paddleocr_vl_engine = PaddleOCRVLEngine(device="cuda:0", model_name="PaddleOCR-VL-1.5")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.paddleocr_vl_engine.parse(file_path, output_path=str(output_dir), **options)
        
        pdf_path = self._ensure_pdf_in_output(file_path, output_dir)
        normalize_output(output_dir)
        
        return {
            "result_path": str(output_dir), 
            "content": result.get("markdown", ""),
            "json_content": result.get("json_content"), # 必须传递
            "pdf_path": pdf_path
        }

    def _process_with_paddleocr_vl_vllm(self, file_path: str, options: dict) -> dict:
        if self.accelerator == "cpu":
            raise RuntimeError("PaddleOCR-VL-VLLM requires GPU")

        if self.paddleocr_vl_vllm_engine is None:
            from paddleocr_vl_vllm import PaddleOCRVLVLLMEngine
            self.paddleocr_vl_vllm_engine = PaddleOCRVLVLLMEngine(
                device="cuda:0",
                vllm_api_base=self.paddleocr_vl_vllm_api,
                model_name="PaddleOCR-VL-1.5-0.9B"
            )

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.paddleocr_vl_vllm_engine.parse(file_path, output_path=str(output_dir), **options)
        
        # [修复] 复制源文件以便预览 (关键修复)
        pdf_path = self._ensure_pdf_in_output(file_path, output_dir)
        
        normalize_output(output_dir, handle_method="paddleocr-vl")
        
        return {
            "result_path": str(output_dir), 
            "content": result.get("markdown", ""),
            "json_content": result.get("json_content"), # 关键：支持右侧高亮
            "pdf_path": pdf_path # 关键：支持左侧预览
        }

    def _process_audio(self, file_path: str, options: dict) -> dict:
        if self.sensevoice_engine is None:
            from audio_engines import SenseVoiceEngine
            self.sensevoice_engine = SenseVoiceEngine(device=self.engine_device)

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.sensevoice_engine.parse(
            audio_path=file_path,
            output_path=str(output_dir),
            language=options.get("lang", "auto"),
            use_itn=options.get("use_itn", True),
            enable_speaker_diarization=options.get("enable_speaker_diarization", False)
        )
        normalize_output(output_dir)
        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_video(self, file_path: str, options: dict) -> dict:
        if self.video_engine is None:
            from video_engines import VideoProcessingEngine
            self.video_engine = VideoProcessingEngine(device=self.engine_device)

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.video_engine.parse(
            video_path=file_path,
            output_path=str(output_dir),
            language=options.get("lang", "auto"),
            use_itn=options.get("use_itn", True),
            keep_audio=options.get("keep_audio", False),
            enable_keyframe_ocr=options.get("enable_keyframe_ocr", False),
            ocr_backend=options.get("ocr_backend", "paddleocr-vl"),
            keep_keyframes=options.get("keep_keyframes", False)
        )
        
        (output_dir / f"{Path(file_path).stem}_video_analysis.md").write_text(result["markdown"], encoding="utf-8")
        normalize_output(output_dir)
        return {"result_path": str(output_dir), "content": result["markdown"]}

    def _process_with_markitdown(self, file_path: str) -> dict:
        if not self.markitdown:
            raise RuntimeError("MarkItDown not available")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.markitdown.convert(file_path)
        markdown_content = result.text_content

        if Path(file_path).suffix.lower() == ".docx":
            try:
                from utils.docx_image_extractor import extract_images_from_docx, append_images_to_markdown
                images_dir = output_dir / "images"
                images = extract_images_from_docx(file_path, str(images_dir))
                if images:
                    markdown_content = append_images_to_markdown(markdown_content, images)
            except Exception as e:
                logger.warning(f"DOCX image extraction failed: {e}")

        (output_dir / f"{Path(file_path).stem}_markitdown.md").write_text(markdown_content, encoding="utf-8")
        normalize_output(output_dir)
        
        # MarkItDown 也可以尝试生成 PDF 预览 (如果源文件是 PDF)
        pdf_path = self._ensure_pdf_in_output(file_path, output_dir)
        
        return {"result_path": str(output_dir), "content": markdown_content, "pdf_path": pdf_path}

    def _process_with_format_engine(self, file_path: str, options: dict, engine_name: Optional[str] = None) -> dict:
        lang = options.get("language", "en")
        
        if engine_name:
            engine = FormatEngineRegistry.get_engine(engine_name)
        else:
            engine = FormatEngineRegistry.get_engine_by_extension(file_path)
            
        if not engine:
            raise ValueError("No format engine available")

        result = engine.parse(file_path, options={"language": lang})
        
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        (output_dir / "result.md").write_text(result["markdown"], encoding="utf-8")
        (output_dir / "result.json").write_text(json.dumps(result["json_content"], indent=2, ensure_ascii=False), encoding="utf-8")
        
        normalize_output(output_dir)
        return {
            "result_path": str(output_dir),
            "content": result["content"],
            "json_content": result["json_content"]
        }

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _convert_office_to_pdf(self, file_path: str) -> str:
        input_file = Path(file_path)
        final_pdf = input_file.parent / f"{input_file.stem}.pdf"
        if final_pdf.exists(): final_pdf.unlink()

        try:
            with tempfile.TemporaryDirectory(prefix="libreoffice_") as temp_dir:
                temp_path = Path(temp_dir)
                temp_input = temp_path / input_file.name
                shutil.copy2(input_file, temp_input)

                cmd = [
                    "libreoffice", "--headless", "--convert-to", "pdf",
                    "--outdir", str(temp_path), str(temp_input)
                ]
                subprocess.run(cmd, check=True, timeout=120, capture_output=True)
                
                temp_pdf = temp_path / f"{input_file.stem}.pdf"
                if not temp_pdf.exists(): raise RuntimeError("PDF output missing")
                shutil.move(str(temp_pdf), str(final_pdf))
                return str(final_pdf)
        except Exception as e:
            raise RuntimeError(f"Office conversion failed: {e}")

    def _preprocess_remove_watermark(self, file_path: str, options: dict) -> Path:
        if not self.watermark_handler: raise RuntimeError("Watermark handler missing")
        output_file = Path(self.output_dir) / f"{Path(file_path).stem}_no_watermark.pdf"
        
        kwargs = {}
        for k in ["auto_detect", "force_scanned", "remove_text", "remove_images", 
                  "remove_annotations", "watermark_keywords", "watermark_dpi", 
                  "watermark_conf_threshold", "watermark_dilation"]:
            if k in options: kwargs[k.replace("watermark_", "")] = options[k]

        return self.watermark_handler.remove_watermark(input_path=file_path, output_path=str(output_file), **kwargs)

    def _should_split_pdf(self, task_id, file_path, task, options):
        """Prepare a split task atomically; raise instead of parsing a half-parent."""
        if os.getenv("PDF_SPLIT_ENABLED", "true").lower() != "true":
            options["start_page_id"] = options.get("start_page") or 0
            options["end_page_id"] = options.get("end_page")
            return False

        if task.get("is_parent") and int(task.get("child_count") or 0) > 0:
            logger.warning(
                f"⚠️ Existing split parent {task_id} will be handled by child/merge reconciliation"
            )
            return True
        current_task = self.task_db.get_task(task_id) if hasattr(self.task_db, "get_task") else task
        if current_task and current_task.get("is_parent") and int(current_task.get("child_count") or 0) > 0:
            logger.warning(
                f"⚠️ Existing split parent {task_id} will be handled by child/merge reconciliation"
            )
            return True

        from utils.pdf_utils import get_pdf_page_count, split_pdf_file

        threshold = int(os.getenv("PDF_SPLIT_THRESHOLD_PAGES", "50"))
        chunk_size = int(os.getenv("PDF_SPLIT_CHUNK_SIZE", "50"))
        total_pages = get_pdf_page_count(Path(file_path))
        start_page = options.get("start_page")
        end_page = options.get("end_page")
        start_page = 0 if start_page is None else int(start_page)
        end_page = total_pages - 1 if end_page is None else int(end_page)
        if start_page < 0 or end_page < start_page or end_page >= total_pages:
            raise ValueError(
                f"invalid inclusive page range {start_page}-{end_page} for {total_pages} pages"
            )

        effective_pages = end_page - start_page + 1
        if effective_pages <= threshold:
            options["start_page_id"] = start_page
            options["end_page_id"] = end_page
            return False

        split_root = Path(self.output_dir) / "splits"
        split_root.mkdir(parents=True, exist_ok=True)
        split_dir = split_root / task_id
        staging_dir = split_root / f".{task_id}.split-{os.getpid()}-{uuid.uuid4().hex}"
        logger.info(
            f"🔀 Splitting PDF range {start_page + 1}-{end_page + 1} "
            f"({effective_pages}/{total_pages} pages)"
        )
        try:
            chunks = split_pdf_file(
                Path(file_path), staging_dir, chunk_size, task_id,
                start_page=start_page, end_page=end_page,
            )
            if not chunks:
                raise RuntimeError("PDF split produced no chunks")
            for index, chunk in enumerate(chunks):
                chunk["index"] = index
                chunk["file_name"] = (
                    f"{Path(file_path).stem}_p{chunk['start_page']}-{chunk['end_page']}.pdf"
                )

            # Publish the fully validated file set before committing task rows.
            # Preserve an orphaned pre-DB directory for operator inspection.
            if split_dir.exists():
                orphan = split_root / f".{task_id}.orphan-{uuid.uuid4().hex}"
                os.replace(split_dir, orphan)
                logger.warning(f"Preserved orphaned split directory at {orphan}")
            os.replace(staging_dir, split_dir)
            for chunk in chunks:
                chunk["path"] = str(split_dir / Path(chunk["path"]).name)

            parent_options = dict(options)
            parent_options["split_info"] = {
                "schema_version": 1,
                "source_total_pages": total_pages,
                "start_page": start_page,
                "end_page": end_page,
                "processed_pages": effective_pages,
                "chunk_size": chunk_size,
                "child_count": len(chunks),
            }
            self.task_db.create_split_children_atomic(
                task_id,
                chunks,
                backend=task.get("backend", "auto"),
                parent_options=parent_options,
                priority=task.get("priority", 0),
                user_id=task.get("user_id"),
            )
            logger.info(f"✂️ Split into {len(chunks)} validated subtasks")
            return True
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _merge_parent_task_results(self, parent_task_id):
        return merge_parent_task_results(
            task_db=self.task_db,
            parent_task_id=parent_task_id,
            output_dir=self.output_dir,
            merge_owner=self.worker_id,
            ensure_pdf_in_output=self._ensure_pdf_in_output,
            cleanup_child_files=False,
        )

    def _cleanup_child_task_files(self, children):
        # Retained for compatibility with tests or external callers.
        for child in children:
            if child.get("status") != "completed":
                continue
            try:
                if child.get("file_path"):
                    Path(child["file_path"]).unlink(missing_ok=True)
            except Exception:
                pass

    # LitServe Interfaces
    def decode_request(self, request): return request.get("action", "health")
    def predict(self, action):
        if action == "health":
            return {"status": "healthy", "worker_id": self.worker_id}
        elif action == "poll":
            if self.enable_worker_loop:
                return {"status": "skipped", "message": "Auto-loop active"}
            if self._is_drain_requested():
                return {"status": "draining"}
            task = self.task_db.pull_task()
            if task:
                task_id = task["task_id"]
                self._begin_task_activity(task_id)
                try:
                    process_state = self._process_task(task)
                    if process_state == "deferred":
                        return {"status": "deferred", "task_id": task_id}
                    return {"status": "completed", "task_id": task_id}
                except Exception as e:
                    return {"status": "failed", "error": str(e)}
                finally:
                    self._finish_task_activity()
            return {"status": "empty"}
        return {"status": "error", "message": "Invalid action"}
    def encode_response(self, response): return response
    def teardown(self):
        self.running = False
        if hasattr(self, "worker_thread"): self.worker_thread.join(timeout=2)


def start_litserve_workers(
    output_dir=None, accelerator="auto", devices="auto", workers_per_device=1,
    port=8001, poll_interval=0.5, enable_worker_loop=True,
    paddleocr_vl_vllm_engine_enabled=False, paddleocr_vl_vllm_api_list=[],
    mineru_vllm_api_list=[], worker_group_index=None
):
    def resolve_auto_accelerator():
        try:
            from importlib.metadata import distribution
            distribution("torch")
            if check_cuda_with_nvidia_smi() > 0: return "cuda"
        except: pass
        return "cpu"

    if output_dir is None:
        output_dir = os.getenv("OUTPUT_PATH", str(Path(__file__).parent.parent / "data" / "output"))

    if accelerator == "auto":
        accelerator = resolve_auto_accelerator()

    logger.info(f"🚀 Starting Worker | Acc: {accelerator} | Devices: {devices} | Out: {output_dir}")

    api = MinerUWorkerAPI(
        output_dir=output_dir,
        poll_interval=poll_interval,
        enable_worker_loop=enable_worker_loop,
        paddleocr_vl_vllm_engine_enabled=paddleocr_vl_vllm_engine_enabled,
        paddleocr_vl_vllm_api_list=paddleocr_vl_vllm_api_list,
        mineru_vllm_api_list=mineru_vllm_api_list,
        workers_per_device=workers_per_device,
        worker_group_index=worker_group_index,
    )

    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        timeout=False,
    )

    def graceful_shutdown(signum=None, frame=None):
        if hasattr(api, "teardown"): api.teardown()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(lambda: api.teardown() if hasattr(api, "teardown") else None)

    server.run(port=port, generate_client_file=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--disable-worker-loop", action="store_true")
    parser.add_argument("--paddleocr-vl-vllm-engine-enabled", action="store_true")
    parser.add_argument("--paddleocr-vl-vllm-api-list", type=parse_list_arg, default=[])
    parser.add_argument("--mineru-vllm-api-list", type=parse_list_arg, default=[])
    args = parser.parse_args()

    # Env Var Fallbacks
    devices = args.devices
    if devices == "auto":
        env_dev = os.getenv("CUDA_VISIBLE_DEVICES")
        if env_dev: devices = env_dev

    port = args.port
    if port == 8001:
        port = int(os.getenv("WORKER_PORT", "8001"))

    # [核心修复] 强制并发控制：优先使用环境变量，默认为 1
    # 用户可以在 .env 中设置 MAX_CONCURRENT_TASKS=1 来限制
    env_workers = os.getenv("MAX_CONCURRENT_TASKS", "1")
    workers_per_device = int(env_workers)
    
    logger.info(f"⚙️  Concurrency Config: workers_per_device={workers_per_device} (env: {env_workers})")

    start_litserve_workers(
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=devices,
        workers_per_device=workers_per_device, # 使用处理后的变量
        port=port,
        poll_interval=args.poll_interval,
        enable_worker_loop=not args.disable_worker_loop,
        paddleocr_vl_vllm_engine_enabled=args.paddleocr_vl_vllm_engine_enabled,
        paddleocr_vl_vllm_api_list=args.paddleocr_vl_vllm_api_list,
        mineru_vllm_api_list=args.mineru_vllm_api_list,
    )
