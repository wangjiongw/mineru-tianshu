import importlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


def _install_import_stubs():
    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    litserve = types.ModuleType("litserve")

    class LitAPI:
        def __init__(self, *args, **kwargs):
            pass

    class LitServer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    litserve.LitAPI = LitAPI
    litserve.LitServer = LitServer

    litserve_mcp = types.ModuleType("litserve.mcp")
    litserve.mcp = litserve_mcp

    connector = types.ModuleType("litserve.connector")
    connector.check_cuda_with_nvidia_smi = lambda: 0

    output_normalizer = types.ModuleType("output_normalizer")
    output_normalizer.normalize_output = lambda *args, **kwargs: None

    utils = types.ModuleType("utils")
    utils.parse_list_arg = lambda value: value

    redis_queue = types.ModuleType("redis_queue")
    redis_queue.get_redis_queue = lambda: None

    requests = types.ModuleType("requests")

    class Response:
        status_code = 200

    def get(*args, **kwargs):
        raise RuntimeError("requests stub should be monkeypatched by tests")

    requests.Response = Response
    requests.get = get

    markitdown = types.ModuleType("markitdown")

    class MarkItDown:
        pass

    markitdown.MarkItDown = MarkItDown

    loguru = types.ModuleType("loguru")

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

        def success(self, *args, **kwargs):
            pass

    loguru.logger = Logger()

    sys.modules.setdefault("litserve", litserve)
    sys.modules.setdefault("litserve.mcp", litserve_mcp)
    sys.modules.setdefault("litserve.connector", connector)
    sys.modules.setdefault("output_normalizer", output_normalizer)
    sys.modules.setdefault("utils", utils)
    sys.modules.setdefault("redis_queue", redis_queue)
    sys.modules.setdefault("requests", requests)
    sys.modules.setdefault("markitdown", markitdown)
    sys.modules.setdefault("loguru", loguru)

    real_find_spec = importlib.util.find_spec
    unavailable_optional_modules = {
        "paddleocr_vl",
        "paddleocr_vl_vllm",
        "mineru_pipeline",
        "audio_engines",
        "video_engines",
        "remove_watermark",
        "format_engines",
    }

    def find_spec_stub(name, package=None):
        if name in unavailable_optional_modules:
            return None
        return real_find_spec(name, package)

    importlib.util.find_spec = find_spec_stub


_install_import_stubs()
worker = importlib.import_module("backend.litserve_worker")


def test_worker_import_binds_output_normalizer_stub():
    assert worker.normalize_output is sys.modules["output_normalizer"].normalize_output


class FakeTaskDB:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []

    def update_task_status(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return True


class OwnershipTaskDB:
    def __init__(self, owner_worker_id):
        self.owner_worker_id = owner_worker_id
        self.status = "processing"
        self.calls = []

    def update_task_status(self, task_id, status, **kwargs):
        self.calls.append({"task_id": task_id, "status": status, **kwargs})
        if kwargs.get("worker_id") != self.owner_worker_id:
            return False
        if status in {"completed", "failed"}:
            self.status = status
        return True


class WorkerEndpointStrategyTest(unittest.TestCase):
    def test_docker_controller_can_be_disabled_and_caches_result(self):
        os.environ["VLLM_DOCKER_CONTROLLER_ENABLED"] = "false"
        try:
            controller = worker.VLLMController()
            self.assertIsNone(controller._get_client())
            self.assertIsNone(controller._get_client())
        finally:
            os.environ.pop("VLLM_DOCKER_CONTROLLER_ENABLED", None)

        self.assertTrue(controller._client_initialized)

    def test_onnxruntime_defaults_bound_unconfigured_session_pools(self):
        class FakeSessionOptions:
            def __init__(self):
                self.intra_op_num_threads = 0
                self.inter_op_num_threads = 0

        class FakeInferenceSession:
            def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
                self.path_or_bytes = path_or_bytes
                self.sess_options = sess_options

        fake_ort = types.SimpleNamespace(
            InferenceSession=FakeInferenceSession,
            SessionOptions=FakeSessionOptions,
        )
        os.environ["MINERU_INTRA_OP_NUM_THREADS"] = "4"
        os.environ["MINERU_INTER_OP_NUM_THREADS"] = "1"
        try:
            self.assertTrue(worker.configure_onnxruntime_thread_defaults(fake_ort))
            session = fake_ort.InferenceSession("model.onnx")
        finally:
            os.environ.pop("MINERU_INTRA_OP_NUM_THREADS", None)
            os.environ.pop("MINERU_INTER_OP_NUM_THREADS", None)

        self.assertEqual(session.sess_options.intra_op_num_threads, 4)
        self.assertEqual(session.sess_options.inter_op_num_threads, 1)

    def test_onnxruntime_defaults_preserve_explicit_session_pool_sizes(self):
        class FakeSessionOptions:
            def __init__(self):
                self.intra_op_num_threads = 0
                self.inter_op_num_threads = 0

        class FakeInferenceSession:
            def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
                self.sess_options = sess_options

        fake_ort = types.SimpleNamespace(
            InferenceSession=FakeInferenceSession,
            SessionOptions=FakeSessionOptions,
        )
        worker.configure_onnxruntime_thread_defaults(fake_ort)
        explicit = FakeSessionOptions()
        explicit.intra_op_num_threads = 7
        explicit.inter_op_num_threads = 2

        session = fake_ort.InferenceSession("model.onnx", explicit)

        self.assertEqual(session.sess_options.intra_op_num_threads, 7)
        self.assertEqual(session.sess_options.inter_op_num_threads, 2)

    def test_local_strategy_uses_first_configured_endpoint(self):
        self.assertEqual(
            worker.map_vllm_endpoint_index(
                endpoint_count=8,
                worker_group_index=7,
                child_index=2,
                strategy="local",
            ),
            0,
        )

    def test_ring3_strategy_maps_group_plus_child_for_all_groups_and_children(self):
        cases = {
            (0, 0): 0,
            (0, 2): 2,
            (3, 2): 5,
            (7, 0): 7,
            (7, 1): 0,
            (7, 2): 1,
        }

        for (group_index, child_index), expected_endpoint in cases.items():
            with self.subTest(group_index=group_index, child_index=child_index):
                self.assertEqual(
                    worker.map_vllm_endpoint_index(
                        endpoint_count=8,
                        worker_group_index=group_index,
                        child_index=child_index,
                        strategy="ring3",
                    ),
                    expected_endpoint,
                )

    def test_ring3_endpoint_selection_falls_forward_to_healthy_endpoint(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False, workers_per_device=3, worker_group_index=7)
        endpoints = [f"http://vllm-{i}:30025/v1" for i in range(8)]
        api.vllm_endpoint_strategy = "ring3"
        api._is_vllm_endpoint_healthy = lambda endpoint: endpoint == endpoints[2]

        self.assertEqual(api._select_vllm_endpoint(endpoints, child_index=2), endpoints[2])

    def test_worker_group_index_can_be_read_from_environment(self):
        os.environ["WORKER_GROUP_INDEX"] = "6"
        try:
            api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        finally:
            os.environ.pop("WORKER_GROUP_INDEX", None)

        self.assertEqual(api.worker_group_index, 6)

    def test_completion_retries_transient_sqlite_lock_as_completed(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.task_db = FakeTaskDB([sqlite3.OperationalError("database is locked")])
        os.environ["TASKDB_COMPLETION_RETRY_ATTEMPTS"] = "2"
        os.environ["TASKDB_COMPLETION_RETRY_DELAY_SECONDS"] = "0"
        try:
            completed = api._complete_task_after_compute("task-1", {"result_path": "/tmp/out", "content": "ok"})
        finally:
            os.environ.pop("TASKDB_COMPLETION_RETRY_ATTEMPTS", None)
            os.environ.pop("TASKDB_COMPLETION_RETRY_DELAY_SECONDS", None)

        self.assertTrue(completed)
        self.assertEqual([call["status"] for call in api.task_db.calls], ["completed", "completed"])

    def test_completion_defers_after_repeated_transient_sqlite_locks(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.task_db = FakeTaskDB([
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database is locked"),
        ])
        os.environ["TASKDB_COMPLETION_RETRY_ATTEMPTS"] = "2"
        os.environ["TASKDB_COMPLETION_RETRY_DELAY_SECONDS"] = "0"
        try:
            completed = api._complete_task_after_compute("task-2", {"result_path": "/tmp/out", "content": "ok"})
        finally:
            os.environ.pop("TASKDB_COMPLETION_RETRY_ATTEMPTS", None)
            os.environ.pop("TASKDB_COMPLETION_RETRY_DELAY_SECONDS", None)

        self.assertFalse(completed)
        self.assertEqual([call["status"] for call in api.task_db.calls], ["completed", "completed"])

    def test_completion_reraises_non_transient_database_errors(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.task_db = FakeTaskDB([sqlite3.OperationalError("no such table: tasks")])

        with self.assertRaises(sqlite3.OperationalError):
            api._complete_task_after_compute("task-3", {"result_path": "/tmp/out", "content": "ok"})

    def test_stale_worker_cannot_complete_task_owned_by_another_worker(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.task_db = OwnershipTaskDB(owner_worker_id="worker-b")

        completed = api._complete_task_after_compute("task-4", {"result_path": "/tmp/out", "content": "ok"})

        self.assertFalse(completed)
        self.assertEqual(api.task_db.status, "processing")
        self.assertEqual(api.task_db.calls[-1]["worker_id"], "worker-a")

    def test_current_worker_can_complete_owned_task(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-b"
        api.task_db = OwnershipTaskDB(owner_worker_id="worker-b")

        completed = api._complete_task_after_compute("task-5", {"result_path": "/tmp/out", "content": "ok"})

        self.assertTrue(completed)
        self.assertEqual(api.task_db.status, "completed")
        self.assertEqual(api.task_db.calls[-1]["worker_id"], "worker-b")

    def test_completion_payload_includes_backward_compatible_metrics(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False, worker_group_index=4)
        api.worker_id = "worker-b"
        api.worker_child_index = 2
        api.task_db = OwnershipTaskDB(owner_worker_id="worker-b")

        completed = api._complete_task_after_compute(
            "task-metrics",
            {
                "result_path": "/tmp/out",
                "content": "ok",
                "pdf_path": "/tmp/out/source.pdf",
                "json_content": {"pages": [{"page_idx": 0}, {"page_idx": 1}]},
            },
            options={"chunk_info": {"page_count": 9}},
            started_at=time.monotonic() - 1.0,
        )

        self.assertTrue(completed)
        payload = json.loads(api.task_db.calls[-1]["data"])
        self.assertEqual(payload["markdown"], "ok")
        self.assertEqual(payload["pdf_path"], "/tmp/out/source.pdf")
        self.assertEqual(payload["metrics"]["page_count"], 2)
        self.assertEqual(payload["metrics"]["worker_group_index"], 4)
        self.assertEqual(payload["metrics"]["worker_child_index"], 2)
        self.assertGreaterEqual(payload["metrics"]["processing_seconds"], 1.0)
        call = api.task_db.calls[-1]
        self.assertEqual(call["page_count"], 2)
        self.assertEqual(call["worker_group_index"], 4)
        self.assertEqual(call["worker_child_index"], 2)
        self.assertGreaterEqual(call["processing_seconds"], 1.0)

    def test_stale_worker_cannot_fail_task_owned_by_another_worker(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.task_db = OwnershipTaskDB(owner_worker_id="worker-b")

        with self.assertRaises(ValueError):
            api._process_task({"task_id": "task-6", "file_path": "/tmp/input.bin", "options": "{}", "backend": "unknown"})

        self.assertEqual(api.task_db.status, "processing")
        self.assertEqual(api.task_db.calls[-1]["status"], "failed")
        self.assertEqual(api.task_db.calls[-1]["worker_id"], "worker-a")

    def test_current_worker_can_fail_owned_task(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-b"
        api.task_db = OwnershipTaskDB(owner_worker_id="worker-b")

        with self.assertRaises(ValueError):
            api._process_task({"task_id": "task-7", "file_path": "/tmp/input.bin", "options": "{}", "backend": "unknown"})

        self.assertEqual(api.task_db.status, "failed")
        self.assertEqual(api.task_db.calls[-1]["worker_id"], "worker-b")

    def test_should_split_pdf_refuses_existing_parent_with_children(self):
        class NoResplitTaskDB:
            def get_task(self, task_id):
                raise AssertionError("task row already proves this is an existing split parent")

            def convert_to_parent_task(self, *args, **kwargs):
                raise AssertionError("existing split parent must not be converted again")

            def create_child_task(self, *args, **kwargs):
                raise AssertionError("existing split parent must not create replacement children")

        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.task_db = NoResplitTaskDB()

        self.assertTrue(
            api._should_split_pdf(
                "parent-task",
                "/tmp/parent.pdf",
                {"task_id": "parent-task", "is_parent": 1, "child_count": 2},
                {},
            )
        )




class DrainActivityTest(unittest.TestCase):
    def test_activity_marker_name_sanitizes_worker_id(self):
        self.assertEqual(
            worker.worker_activity_marker_name("worker host/cuda:0", 123),
            "worker_host_cuda_0-123.json",
        )

    def test_activity_payload_is_unit_testable(self):
        payload = worker.build_worker_activity_payload(
            task_id="task-9",
            worker_id="worker-a",
            pid=123,
            started_at=1_700_000_000,
        )

        self.assertEqual(payload["task_id"], "task-9")
        self.assertEqual(payload["worker_id"], "worker-a")
        self.assertEqual(payload["pid"], 123)
        self.assertEqual(payload["started_at"], 1_700_000_000)
        self.assertEqual(payload["started_at_iso"], "2023-11-14T22:13:20Z")

    def test_activity_marker_is_written_atomically_and_removed(self):
        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        with tempfile.TemporaryDirectory() as tmp_dir:
            marker_path = Path(tmp_dir) / "worker-a.json"
            api.worker_activity_marker = marker_path

            api._write_worker_activity("task-10")
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], "task-10")
            self.assertEqual(payload["worker_id"], "worker-a")
            self.assertFalse(list(Path(tmp_dir).glob("*.tmp")))

            api._remove_worker_activity()
            self.assertFalse(marker_path.exists())

    def test_worker_loop_drain_skips_claiming_tasks(self):
        class NoClaimTaskDB:
            def get_next_task(self, worker_id):
                raise AssertionError("draining workers must not claim tasks")

        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.running = True
        api.poll_interval = 0
        api.task_db = NoClaimTaskDB()

        def drain_once():
            api.running = False
            return True

        api._is_drain_requested = drain_once
        api._worker_loop()

    def test_poll_drain_skips_claiming_tasks(self):
        class NoClaimTaskDB:
            def pull_task(self):
                raise AssertionError("draining poll must not claim tasks")

        api = worker.MinerUWorkerAPI(enable_worker_loop=False)
        api.worker_id = "worker-a"
        api.task_db = NoClaimTaskDB()
        api._is_drain_requested = lambda: True

        self.assertEqual(api.predict("poll"), {"status": "draining"})




if __name__ == "__main__":
    unittest.main()
