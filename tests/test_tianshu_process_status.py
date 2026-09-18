import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tianshu.sh"


def run_bash(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "TIANSHU_SH_SOURCE_ONLY": "1",
            "TIANSHU_PROC_ROOT": str(tmp_path / "proc"),
            "INSTANCE_ID": "ignored-instance",
            "TIANSHU_INSTANCE_ID": "pytest-instance",
        }
    )
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {script}'],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def fake_proc(tmp_path: Path, pid: int, state: str, argv: list[str]) -> None:
    proc_dir = tmp_path / "proc" / str(pid)
    proc_dir.mkdir(parents=True)
    (proc_dir / "status").write_text(f"Name:\ttest\nState:\t{state} (test)\n")
    (proc_dir / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
    instance_dir = "/share/wangjiong/databases/mineru_database/pytest-instance"
    environ = {
        "INSTANCE_ID": "pytest-instance",
        "DATABASE_PATH": f"{instance_dir}/mineru_tianshu.db",
        "OUTPUT_PATH": f"{instance_dir}/mineru_outputs",
        "UPLOAD_PATH": f"{instance_dir}/mineru_uploads",
        "API_PORT": "8000",
        "MCP_PORT": "8002",
    }
    (proc_dir / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environ.items()) + b"\0"
    )
    (proc_dir / "stat").write_text(
        f"{pid} (test) {state} " + " ".join(["0"] * 18) + f" {pid * 100}\n"
    )


class TianshuProcessStatusTests(unittest.TestCase):
    def test_pid_matches_rejects_zombie_even_when_cmdline_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 101, "Z", ["python", "litserve_worker.py", "--port", "8101"])

            result = run_bash('pid_matches 101 "python.*litserve_worker.py" 8101', tmp_path)

        self.assertEqual(result.returncode, 1)

    def test_pid_matches_rejects_wrong_cmdline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 102, "S", ["python", "unrelated.py", "--port", "8101"])

            result = run_bash('pid_matches 102 "python.*litserve_worker.py" 8101', tmp_path)

        self.assertEqual(result.returncode, 1)

    def test_pid_matches_requires_expected_port_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 103, "S", ["python", "litserve_worker.py", "--port", "8102"])

            result = run_bash('pid_matches 103 "python.*litserve_worker.py" 8101', tmp_path)

        self.assertEqual(result.returncode, 1)

    def test_worker_is_running_validates_instance_pid_cmdline_and_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 104, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            (worker_dir / "worker_2.pid").write_text("104")

            result = run_bash(
                f'WORKER_LOG_DIR="{worker_dir}" WORKER_BASE_PORT=8101 WORKER_NUM_INSTANCES=8 worker_is_running 2',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_is_running_rejects_stale_wrong_port_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 105, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8104"])
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            (worker_dir / "worker_2.pid").write_text("105")

            result = run_bash(
                f'WORKER_LOG_DIR="{worker_dir}" WORKER_BASE_PORT=8101 WORKER_NUM_INSTANCES=8 worker_is_running 2',
                tmp_path,
            )

        self.assertEqual(result.returncode, 1)


    def test_stop_worker_process_tree_terms_only_validated_same_port_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 201, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            fake_proc(tmp_path, 202, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103", "claim"])
            fake_proc(tmp_path, 203, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8104", "claim"])
            fake_proc(tmp_path, 204, "S", ["/env/bin/python", "unrelated.py", "--port", "8103"])
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            pid_file = worker_dir / "worker_2.pid"
            pid_file.write_text("201")

            result = run_bash(
                'ps() { '
                'if [ "$1" = "-eo" ] && [ "$2" = "pid=,ppid=" ]; then '
                'printf "201 1\n202 201\n203 1\n204 1\n"; '
                'elif [ "$1" = "-eo" ] && [ "$2" = "pid=" ]; then '
                'printf "201\n202\n203\n204\n"; fi; }; '
                'kill() { echo "kill:$*"; '
                'case "$1" in 201|202) rm -rf "$TIANSHU_PROC_ROOT/$1";; '
                '-9) rm -rf "$TIANSHU_PROC_ROOT/$2";; esac; }; '
                f'stop_worker_process_tree "{pid_file}" 8103',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kill:202", result.stdout)
        self.assertIn("kill:201", result.stdout)
        self.assertNotIn("kill:203", result.stdout)
        self.assertNotIn("kill:204", result.stdout)

    def test_stop_worker_process_tree_stops_captured_spawn_style_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 401, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            fake_proc(tmp_path, 402, "S", ["/env/bin/python", "-c", "from multiprocessing.spawn import spawn_main"])
            fake_proc(tmp_path, 403, "S", ["/env/bin/python", "-c", "resource_tracker"])
            fake_proc(tmp_path, 404, "S", ["/env/bin/python", "unrelated.py", "--port", "8103"])
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            pid_file = worker_dir / "worker_2.pid"
            pid_file.write_text("401")

            result = run_bash(
                'ps() { '
                'if [ "$1" = "-eo" ] && [ "$2" = "pid=,ppid=" ]; then '
                'printf "401 1\n402 401\n403 402\n404 1\n"; '
                'elif [ "$1" = "-eo" ] && [ "$2" = "pid=" ]; then '
                'printf "401\n402\n403\n404\n"; fi; }; '
                'kill() { echo "kill:$*"; '
                'case "$1" in 401|402|403) rm -rf "$TIANSHU_PROC_ROOT/$1";; '
                '-9) rm -rf "$TIANSHU_PROC_ROOT/$2";; esac; }; '
                f'stop_worker_process_tree "{pid_file}" 8103',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kill:401", result.stdout)
        self.assertIn("kill:402", result.stdout)
        self.assertIn("kill:403", result.stdout)
        self.assertNotIn("kill:404", result.stdout)
        self.assertFalse(pid_file.exists())

    def test_stop_worker_process_tree_fails_if_exact_port_process_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 301, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            pid_file = worker_dir / "worker_2.pid"
            pid_file.write_text("999")

            result = run_bash(
                'ps() { '
                'if [ "$1" = "-eo" ] && [ "$2" = "pid=,ppid=" ]; then :; '
                'elif [ "$1" = "-eo" ] && [ "$2" = "pid=" ]; then printf "301\n"; fi; }; '
                'kill() { echo "kill:$*"; }; '
                f'stop_worker_process_tree "{pid_file}" 8103',
                tmp_path,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("kill:301", result.stdout)
        self.assertIn("kill:-9 301", result.stdout)
        self.assertFalse(pid_file.exists())

    def test_worker_candidate_scan_uses_pgrep_instead_of_all_processes(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("worker_candidate_pids()")
        end = text.index("stop_worker_process_tree()", start)
        implementation = text[start:end]

        self.assertIn("pgrep -f '[l]itserve_worker.py'", implementation)
        self.assertNotIn("ps -eo pid=", implementation)


    def test_wait_vllm_instance_ready_fails_fast_when_pid_exits_even_if_http_would_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vllm_dir = tmp_path / "vllm"
            calls = tmp_path / "curl-calls"
            vllm_dir.mkdir()
            (vllm_dir / "vllm_npu2.pid").write_text("701")
            result = run_bash(
                f'VLLM_LOG_DIR="{vllm_dir}" VLLM_BASE_PORT=30025 VLLM_NUM_INSTANCES=8; '
                f'curl() {{ printf "%s\n" "$*" >> "{calls}"; return 0; }}; '
                'sleep() { :; }; '
                'wait_vllm_instance_ready 2 900',
                tmp_path,
            )

        self.assertEqual(result.returncode, 1)
        self.assertFalse(calls.exists())

    def test_stop_vllm_process_tree_stops_captured_descendants_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vllm_dir = tmp_path / "vllm"
            vllm_dir.mkdir()
            fake_proc(tmp_path, 701, "S", ["/usr/bin/vllm", "serve", "/share/wangjiong/model_zoo/modelscope/models/OpenDataLab/MinerU2___5-Pro-2605-1___2B", "--port", "30027"])
            fake_proc(tmp_path, 702, "S", ["python", "-c", "VLLM::EngineCore"])
            fake_proc(tmp_path, 703, "S", ["python", "-c", "resource_tracker"])
            fake_proc(tmp_path, 704, "S", ["python", "unrelated.py"])
            pid_file = vllm_dir / "vllm_npu2.pid"
            pid_file.write_text("701")

            result = run_bash(
                'ps() { '
                'if [ "$1" = "-eo" ] && [ "$2" = "pid=,ppid=" ]; then '
                'printf "701 1\n702 701\n703 702\n704 1\n"; fi; }; '
                'kill() { echo "kill:$*"; '
                'case "$1" in 701|702|703) rm -rf "$TIANSHU_PROC_ROOT/$1";; '
                '-9) rm -rf "$TIANSHU_PROC_ROOT/$2";; esac; }; '
                f'stop_vllm_process_tree "{pid_file}" 30027',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kill:701", result.stdout)
        self.assertIn("kill:702", result.stdout)
        self.assertIn("kill:703", result.stdout)
        self.assertNotIn("kill:704", result.stdout)
        self.assertFalse(pid_file.exists())

    def test_stop_vllm_process_tree_reports_captured_descendant_survivor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vllm_dir = tmp_path / "vllm"
            vllm_dir.mkdir()
            fake_proc(tmp_path, 711, "S", ["/usr/bin/vllm", "serve", "/share/wangjiong/model_zoo/modelscope/models/OpenDataLab/MinerU2___5-Pro-2605-1___2B", "--port", "30027"])
            fake_proc(tmp_path, 712, "S", ["python", "-c", "VLLM::EngineCore"])
            pid_file = vllm_dir / "vllm_npu2.pid"
            pid_file.write_text("711")

            result = run_bash(
                'ps() { '
                'if [ "$1" = "-eo" ] && [ "$2" = "pid=,ppid=" ]; then '
                'printf "711 1\n712 711\n"; fi; }; '
                'kill() { echo "kill:$*"; case "$1" in 711) rm -rf "$TIANSHU_PROC_ROOT/711";; esac; }; '
                f'stop_vllm_process_tree "{pid_file}" 30027',
                tmp_path,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("kill:712", result.stdout)
        self.assertFalse(pid_file.exists())

    def test_worker_vllm_api_list_defaults_to_local_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash('VLLM_BASE_PORT=30025 VLLM_NUM_INSTANCES=8 worker_vllm_api_list 2', tmp_path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '["http://localhost:30027/v1"]')

    def test_worker_vllm_api_list_ring3_uses_all_active_local_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'VLLM_BASE_PORT=30025 VLLM_NUM_INSTANCES=8 VLLM_ENDPOINT_STRATEGY=ring3 worker_vllm_api_list 2',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            '["http://localhost:30025/v1","http://localhost:30026/v1","http://localhost:30027/v1","http://localhost:30028/v1"]',
        )


    def test_worker_launch_exports_group_index_strategy_and_drain_paths(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('WORKER_GROUP_INDEX="$i"', text)
        self.assertIn('WORKER_DRAIN_FILE="$(worker_drain_file "$i")"', text)
        self.assertIn('WORKER_ACTIVITY_DIR="$(worker_activity_dir "$i")"', text)
        self.assertIn('VLLM_ENDPOINT_STRATEGY="${VLLM_ENDPOINT_STRATEGY:-local}"', text)
        self.assertIn('--mineru-vllm-api-list "$vllm_api_list"', text)

    def test_drain_sets_disabled_marker_without_stopping_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime_dir = tmp_path / "runtime"
            result = run_bash(
                f'WORKER_RUNTIME_DIR="{runtime_dir}" WORKER_NUM_INSTANCES=8 DRAIN_WAIT_SECONDS=6 drain_worker_instance 3',
                tmp_path,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((runtime_dir / "worker_3.drain").exists())
            self.assertTrue((runtime_dir / "worker_3.disabled").exists())
            self.assertTrue((runtime_dir / "worker_3_activity").is_dir())



    def test_start_does_not_clear_runtime_state_before_running_check(self) -> None:
        text = SCRIPT.read_text()
        start = text.index('start_worker_instance()')
        running = text.index('if worker_is_running "$i"', start)
        clear_drain = text.index('rm -f "$(worker_drain_file "$i")"', start)

        self.assertLess(running, clear_drain)
        self.assertNotIn('rm -rf "$(worker_activity_dir "$i")"', text)
        self.assertIn('find "$(worker_activity_dir "$i")" -mindepth 1 -type f -delete', text)

    def test_drain_wait_requires_grace_and_two_empty_checks(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('local grace="${DRAIN_GRACE_SECONDS:-3}"', text)
        self.assertIn('local stable_interval="${DRAIN_STABLE_INTERVAL_SECONDS:-2}"', text)
        self.assertIn('sleep "$stable_interval"', text)

    def test_worker_drain_complete_uses_empty_activity_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime_dir = tmp_path / "runtime"
            activity_dir = runtime_dir / "worker_3_activity"
            activity_dir.mkdir(parents=True)

            idle = run_bash(f'WORKER_RUNTIME_DIR="{runtime_dir}" worker_drain_complete 3', tmp_path)
            (activity_dir / "task.json").write_text("{}")
            busy = run_bash(f'WORKER_RUNTIME_DIR="{runtime_dir}" worker_drain_complete 3', tmp_path)

        self.assertEqual(idle.returncode, 0, idle.stderr)
        self.assertEqual(busy.returncode, 1)

    def test_restart_does_not_stop_after_unverified_drain_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'drain_worker_instance() { return 2; }; '
                'stop_worker_instance() { echo stop; }; '
                'start_worker_instance() { echo start; }; '
                'restart_worker_instance 3',
                tmp_path,
            )

        self.assertEqual(result.returncode, 2)
        self.assertNotIn("stop\n", result.stdout)
        self.assertNotIn("start\n", result.stdout)

    def test_local_efficiency_defaults_match_proven_c8_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'printf "%s\n" "$VLLM_GPU_MEMORY_UTILIZATION"; '
                'printf "%s\n" "$VLLM_PERFORMANCE_MODE"; '
                'printf "%s\n" "$VLLM_MAX_NUM_BATCHED_TOKENS"; '
                'worker_max_concurrent_tasks_for 0',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["0.60", "throughput", "4096", "8"])

    def test_indexed_env_resolvers_use_specific_then_global_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'VLLM_GPU_MEMORY_UTILIZATION=0.80 '
                'VLLM_GPU_MEMORY_UTILIZATION_NPU4=0.72 '
                'MAX_CONCURRENT_TASKS=3 '
                'MAX_CONCURRENT_TASKS_WORKER4=1; '
                'vllm_gpu_memory_utilization_for 4; '
                'vllm_gpu_memory_utilization_for 5; '
                'worker_max_concurrent_tasks_for 4; '
                'worker_max_concurrent_tasks_for 5',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["0.72", "0.80", "1", "3"])

    def test_vllm_performance_mode_uses_per_npu_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'VLLM_PERFORMANCE_MODE=balanced '
                'VLLM_PERFORMANCE_MODE_NPU4=throughput; '
                'vllm_performance_mode_for 4; '
                'vllm_performance_mode_for 5',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["throughput", "balanced"])
        self.assertEqual(SCRIPT.read_text().count('--performance-mode ${performance_mode}'), 1)

    def test_vllm_max_batched_tokens_uses_per_npu_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'VLLM_MAX_NUM_BATCHED_TOKENS=4096 '
                'VLLM_MAX_NUM_BATCHED_TOKENS_NPU4=8192; '
                'vllm_max_num_batched_tokens_for 4; '
                'vllm_max_num_batched_tokens_for 5',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["8192", "4096"])
        self.assertEqual(SCRIPT.read_text().count('--max-num-batched-tokens ${max_num_batched_tokens}'), 1)

    def test_restart_compute_refuses_ring3_without_stopping_services(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'drain_worker_instance() { echo drain; return 0; }; '
                'stop_worker_instance() { echo stop_worker; }; '
                'stop_vllm_instance() { echo stop_vllm; }; '
                'start_vllm_instance() { echo start_vllm; }; '
                'start_worker_instance() { echo start_worker; }; '
                'VLLM_ENDPOINT_STRATEGY=ring3 restart_compute_instance 3',
                tmp_path,
            )

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("drain", result.stdout)
        self.assertNotIn("stop_worker", result.stdout)
        self.assertNotIn("stop_vllm", result.stdout)
        self.assertNotIn("start_vllm", result.stdout)
        self.assertNotIn("start_worker", result.stdout)

    def test_restart_compute_does_not_stop_after_unverified_drain_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'init_dirs() { :; }; source_ascend_env() { :; }; check_instance_conflict() { :; }; '
                'drain_worker_instance() { echo drain; return 2; }; '
                'stop_worker_instance() { echo stop_worker; }; '
                'stop_vllm_instance() { echo stop_vllm; }; '
                'start_vllm_instance() { echo start_vllm; }; '
                'start_worker_instance() { echo start_worker; }; '
                'VLLM_ENDPOINT_STRATEGY=local restart_compute_instance 3',
                tmp_path,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("drain", result.stdout)
        self.assertNotIn("stop_worker", result.stdout)
        self.assertNotIn("stop_vllm", result.stdout)
        self.assertNotIn("start_vllm", result.stdout)
        self.assertNotIn("start_worker", result.stdout)

    def test_restart_compute_sequence_after_verified_drain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'init_dirs() { :; }; source_ascend_env() { :; }; check_instance_conflict() { :; }; '
                'drain_worker_instance() { echo drain; return 0; }; '
                'stop_worker_instance() { echo stop_worker; }; '
                'stop_vllm_instance() { echo stop_vllm; }; '
                'start_vllm_instance() { echo start_vllm; }; '
                'start_worker_instance() { echo start_worker; }; '
                'VLLM_ENDPOINT_STRATEGY=local restart_compute_instance 3',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [line for line in result.stdout.splitlines() if not line.startswith("\x1b[")],
            ["drain", "stop_worker", "stop_vllm", "start_vllm", "start_worker"],
        )

    def test_restart_compute_cli_routes_instance_and_force_flag(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('compute)', text)
        self.assertIn('restart_compute_instance "$instance_index" "$force_flag"', text)

    def test_watchdog_skips_disabled_or_draining_workers(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('WORKER_RUNTIME_DIR=', text)
        self.assertIn('worker_${idx}.disabled', text)
        self.assertIn('worker_${idx}.drain', text)
        self.assertIn('return 0', text)


    def test_compute_supervisor_owns_and_cleans_both_children(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('start_vllm_instance "$i"', implementation)
        self.assertIn('start_worker_instance "$i"', implementation)
        self.assertIn('vllm_pid="$(cat "$(vllm_pid_file "$i")"', implementation)
        self.assertIn('worker_pid="$(cat "$(worker_pid_file "$i")"', implementation)
        self.assertIn('trap cleanup_compute_supervisor EXIT', implementation)
        self.assertIn('wait "$worker_pid"', implementation)
        self.assertIn('wait "$vllm_pid"', implementation)

    def test_compute_supervisor_restarts_owned_pair_after_child_exit(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('failed_component="VLLM"', implementation)
        self.assertIn('failed_component="Worker"', implementation)
        self.assertIn('stop_owned_compute', implementation)
        self.assertIn('supervisor restarting owned pair after ${backoff}s', implementation)
        self.assertIn('SUPERVISOR_MAX_BACKOFF', implementation)

    def test_compute_supervisor_probes_vllm_and_worker_health(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('vllm_http_healthy "$i"', implementation)
        self.assertIn('worker_http_healthy "$i"', implementation)
        self.assertIn('COMPUTE_SUPERVISOR_HEALTH_FAILURE_THRESHOLD', implementation)
        self.assertIn('consecutive health probe failures', implementation)


    def test_wait_worker_instance_ready_waits_for_health_while_process_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker_dir = tmp_path / "worker"
            calls = tmp_path / "curl-calls"
            counter = tmp_path / "curl-count"
            worker_dir.mkdir()
            fake_proc(tmp_path, 601, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            (worker_dir / "worker_2.pid").write_text("601")
            result = run_bash(
                f'WORKER_LOG_DIR="{worker_dir}" WORKER_BASE_PORT=8101 WORKER_NUM_INSTANCES=8 WORKER_READY_POLL_SECONDS=1 WORKER_READY_SUCCESS_THRESHOLD=1; '
                f'curl() {{ n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n + 1)); printf "%s" "$n" > "{counter}"; printf "%s\n" "$*" >> "{calls}"; [ "$n" -ge 3 ]; }}; '
                'sleep() { :; }; '
                'wait_worker_instance_ready 2 5',
                tmp_path,
            )
            recorded = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(recorded), 3)
        self.assertTrue(all('http://localhost:8103/health' in call for call in recorded))

    def test_wait_worker_instance_ready_requires_stable_consecutive_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker_dir = tmp_path / "worker"
            calls = tmp_path / "curl-calls"
            counter = tmp_path / "curl-count"
            worker_dir.mkdir()
            fake_proc(tmp_path, 603, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            (worker_dir / "worker_2.pid").write_text("603")
            result = run_bash(
                f'WORKER_LOG_DIR="{worker_dir}" WORKER_BASE_PORT=8101 WORKER_NUM_INSTANCES=8 WORKER_READY_POLL_SECONDS=1 WORKER_READY_SUCCESS_THRESHOLD=3; '
                f'curl() {{ n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n + 1)); printf "%s" "$n" > "{counter}"; printf "%s\n" "$*" >> "{calls}"; case "$n" in 1|4) return 1 ;; *) return 0 ;; esac; }}; '
                'sleep() { :; }; '
                'wait_worker_instance_ready 2 10',
                tmp_path,
            )
            recorded = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(recorded), 7)

    def test_wait_worker_instance_ready_fails_if_worker_process_exits_before_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker_dir = tmp_path / "worker"
            calls = tmp_path / "curl-calls"
            worker_dir.mkdir()
            fake_proc(tmp_path, 602, "S", ["/env/bin/python", "litserve_worker.py", "--port", "8103"])
            (worker_dir / "worker_2.pid").write_text("602")
            result = run_bash(
                f'WORKER_LOG_DIR="{worker_dir}" WORKER_BASE_PORT=8101 WORKER_NUM_INSTANCES=8 WORKER_READY_POLL_SECONDS=1; '
                f'curl() {{ printf "%s\n" "$*" >> "{calls}"; rm -rf "$TIANSHU_PROC_ROOT/602"; return 1; }}; '
                'sleep() { :; }; '
                'wait_worker_instance_ready 2 5',
                tmp_path,
            )
            recorded = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(recorded), 1)

    def test_compute_supervisor_waits_for_worker_ready_before_health_failure_loop(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('WORKER_READY_TIMEOUT', implementation)
        self.assertIn('wait_worker_instance_ready "$i" "$WORKER_READY_TIMEOUT"', implementation)
        self.assertLess(
            implementation.index('wait_worker_instance_ready "$i" "$WORKER_READY_TIMEOUT"'),
            implementation.index('local vllm_health_failures=0'),
        )
        self.assertLess(
            implementation.index('wait_worker_instance_ready "$i" "$WORKER_READY_TIMEOUT"'),
            implementation.index('worker_http_healthy "$i" "$health_probe_timeout"'),
        )

    def test_compute_supervisor_emits_periodic_heartbeat(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('COMPUTE_SUPERVISOR_HEARTBEAT_SECONDS', implementation)
        self.assertIn('supervisor heartbeat: VLLM PID', implementation)

    def test_compute_health_probes_are_bounded_http_requests(self) -> None:
        text = SCRIPT.read_text()
        ready_start = text.index("vllm_http_ready()")
        healthy_start = text.index("vllm_http_healthy()", ready_start)
        vllm_end = text.index("vllm_is_running()", healthy_start)
        ready_probe = text[ready_start:healthy_start]
        healthy_probe = text[healthy_start:vllm_end]
        worker_start = text.index("worker_http_healthy()")
        worker_end = text.index("check_worker_dependencies()", worker_start)
        worker_probe = text[worker_start:worker_end]

        self.assertIn('curl -fsS --max-time "$timeout"', ready_probe)
        self.assertIn('/v1/models', ready_probe)
        self.assertIn('curl -fsS --max-time "$timeout"', healthy_probe)
        self.assertIn('/health', healthy_probe)
        self.assertIn('curl -fsS --max-time "$timeout"', worker_probe)
        self.assertIn('/health', worker_probe)

    def test_compute_health_probes_target_assigned_ports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calls = tmp_path / "curl-calls"
            result = run_bash(
                f'curl() {{ printf "%s\\n" "$*" >> "{calls}"; }}; '
                'vllm_http_ready 2 7; vllm_http_healthy 2 7; worker_http_healthy 2 7',
                tmp_path,
            )

            recorded = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            recorded,
            [
                "-fsS --max-time 7 http://localhost:30027/v1/models",
                "-fsS --max-time 7 http://localhost:30027/health",
                "-fsS --max-time 7 http://localhost:8103/health",
            ],
        )

    def test_compute_supervisor_requires_explicit_safe_replacement(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('SUPERVISE_COMPUTE_REPLACE:-false', implementation)
        self.assertIn('drain_worker_instance "$i"', implementation)
        self.assertLess(
            implementation.index('drain_worker_instance "$i"'),
            implementation.index('stop_worker_instance "$i"'),
        )
        self.assertLess(
            implementation.index('stop_worker_instance "$i"'),
            implementation.index('stop_vllm_instance "$i"'),
        )

    def test_node_supervisor_owns_control_and_active_compute_supervisors(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_node()")
        end = text.index("supervise_compute_instance()", start)
        implementation = text[start:end]

        self.assertIn('start_redis', implementation)
        self.assertIn('start_api', implementation)
        self.assertIn('cmd_supervise control &', implementation)
        self.assertIn('supervise_compute_instance "$i" &', implementation)
        self.assertIn('for i in $(active_instance_indices)', implementation)
        self.assertIn('wait -n', implementation)
        self.assertIn('cleanup_node_supervisor', implementation)

    def test_supervise_cli_routes_compute_instance(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('supervise) cmd_supervise "$2" "$3"', text)
        self.assertIn('supervise_compute_instance "$instance_index"', text)
        self.assertIn('supervise_node', text)
        self.assertIn('control|compute|node', text)

    def test_rustfs_upload_is_disabled_by_default_but_overridable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default = run_bash('printf "%s" "$RUSTFS_ENABLED"', tmp_path)
            override = run_bash(
                'RUSTFS_ENABLED=true; source "{}"; printf "%s" "$RUSTFS_ENABLED"'.format(SCRIPT),
                tmp_path,
            )

        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout, "false")
        self.assertEqual(override.returncode, 0, override.stderr)
        self.assertEqual(override.stdout, "true")
        self.assertIn('RUSTFS_ENABLED="$RUSTFS_ENABLED"', SCRIPT.read_text())

    def test_docker_controller_is_disabled_for_host_managed_vllm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default = run_bash('printf "%s" "$VLLM_DOCKER_CONTROLLER_ENABLED"', tmp_path)
            override = run_bash(
                'VLLM_DOCKER_CONTROLLER_ENABLED=true; source "{}"; printf "%s" "$VLLM_DOCKER_CONTROLLER_ENABLED"'.format(SCRIPT),
                tmp_path,
            )

        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout, "false")
        self.assertEqual(override.returncode, 0, override.stderr)
        self.assertEqual(override.stdout, "true")
        self.assertIn(
            'VLLM_DOCKER_CONTROLLER_ENABLED="$VLLM_DOCKER_CONTROLLER_ENABLED"',
            SCRIPT.read_text(),
        )

    def test_queue_envs_are_instance_scoped_and_exported(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('REDIS_CLAIM_MAINTENANCE_KEY="tianshu:claim_maintenance:${INSTANCE_ID}"', text)
        self.assertIn('REDIS_CLAIM_PAUSE_KEY="tianshu:claim_pause:${INSTANCE_ID}"', text)
        self.assertIn('SQLITE_QUEUE_FALLBACK="${SQLITE_QUEUE_FALLBACK:-false}"', text)
        self.assertNotIn('record_worker_maintenance', text)
        self.assertGreaterEqual(text.count('REDIS_CLAIM_MAINTENANCE_KEY="$REDIS_CLAIM_MAINTENANCE_KEY"'), 3)
        self.assertGreaterEqual(text.count('REDIS_CLAIM_PAUSE_KEY="$REDIS_CLAIM_PAUSE_KEY"'), 3)
        self.assertGreaterEqual(text.count('SQLITE_QUEUE_FALLBACK="$SQLITE_QUEUE_FALLBACK"'), 3)

    def test_sqlite_queue_fallback_defaults_false_but_allows_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default = run_bash('printf "%s" "$SQLITE_QUEUE_FALLBACK"', tmp_path)
            override = run_bash('SQLITE_QUEUE_FALLBACK=true; source "{}"; printf "%s" "$SQLITE_QUEUE_FALLBACK"'.format(SCRIPT), tmp_path)

        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout, "false")
        self.assertEqual(override.returncode, 0, override.stderr)
        self.assertEqual(override.stdout, "true")

    def test_watchdog_repairs_by_instance_not_global_restart(self) -> None:
        text = SCRIPT.read_text()

        self.assertIn('bash "$TIANSHU_SCRIPT" start worker "$idx"', text)
        self.assertNotIn('bash "$TIANSHU_SCRIPT" start worker >>', text)
        self.assertNotIn('pkill -f "litserve_worker.py', text)


    def test_worker_runtime_config_is_persisted_and_read_without_sourcing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "runtime_config"
            result = run_bash(
                f'RUNTIME_CONFIG_DIR="{config_dir}"; '
                'configure_worker_instance 2 --hybrid-batch-ratio 8 --max-concurrent-tasks 12; '
                'worker_hybrid_batch_ratio_for 2; '
                'worker_max_concurrent_tasks_for 2',
                tmp_path,
            )
            config_text = (config_dir / "worker_2.env").read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config_text, ["MINERU_HYBRID_BATCH_RATIO=8", "MAX_CONCURRENT_TASKS=12"])
        self.assertEqual(result.stdout.splitlines()[-2:], ["8", "12"])

    def test_worker_runtime_config_rejects_invalid_values_and_unknown_cli_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "runtime_config"
            bad_ratio = run_bash(
                f'RUNTIME_CONFIG_DIR="{config_dir}" configure_worker_instance 2 --hybrid-batch-ratio 3 --max-concurrent-tasks 12',
                tmp_path,
            )
            bad_tasks = run_bash(
                f'RUNTIME_CONFIG_DIR="{config_dir}" configure_worker_instance 2 --hybrid-batch-ratio 4 --max-concurrent-tasks 17',
                tmp_path,
            )
            bad_flag = run_bash(
                f'RUNTIME_CONFIG_DIR="{config_dir}" configure_worker_instance 2 --hybrid-batch-ratio 4 --source /tmp/x --max-concurrent-tasks 8',
                tmp_path,
            )

        self.assertNotEqual(bad_ratio.returncode, 0)
        self.assertNotEqual(bad_tasks.returncode, 0)
        self.assertNotEqual(bad_flag.returncode, 0)
        self.assertFalse((config_dir / "worker_2.env").exists())

    def test_worker_runtime_config_parser_whitelists_keys_without_source_or_eval(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("worker_runtime_config_value()")
        end = text.index("worker_hybrid_batch_ratio_for()", start)
        implementation = text[start:end]

        self.assertIn('MINERU_HYBRID_BATCH_RATIO|MAX_CONCURRENT_TASKS', implementation)
        self.assertNotIn('source "$', implementation)
        self.assertNotIn('eval ', implementation)

    def test_worker_start_exports_runtime_tuning_and_logs_revision(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("start_worker_instance()")
        end = text.index("start_workers()", start)
        implementation = text[start:end]

        self.assertIn('worker_hybrid_display="auto"', implementation)
        self.assertIn('worker_hybrid_env=(env -u MINERU_HYBRID_BATCH_RATIO)', implementation)
        self.assertIn('worker_hybrid_env=(env "MINERU_HYBRID_BATCH_RATIO=$worker_hybrid_ratio")', implementation)
        self.assertIn('worker_config_revision="$(worker_config_revision_for "$i")"', implementation)
        self.assertIn('MAX_CONCURRENT_TASKS="$worker_max_tasks"', implementation)
        self.assertIn('"${worker_hybrid_env[@]}" nohup', implementation)
        self.assertIn('config_revision=${worker_config_revision}', implementation)


    def test_worker_start_unsets_hybrid_ratio_when_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime_dir = tmp_path / "runtime"
            worker_dir = tmp_path / "worker"
            env_capture = tmp_path / "env-capture"
            worker_dir.mkdir()
            result = run_bash(
                f'WORKER_RUNTIME_DIR="{runtime_dir}" WORKER_LOG_DIR="{worker_dir}"; '
                f'RUNTIME_CONFIG_DIR="{tmp_path / "runtime_config"}"; '
                f'env() {{ printf "%s\\n" "$*" > "{env_capture}"; }}; '
                'check_worker_dependencies() { return 0; }; '
                'worker_is_running() { [ -f "$(worker_pid_file 2)" ]; }; '
                'sleep() { :; }; '
                'start_worker_instance 2; '
                f'for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "{env_capture}" ] && break; /bin/sleep 0.05; done',
                tmp_path,
            )
            captured_env = env_capture.read_text()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('hybrid_batch_ratio=auto', result.stdout)
        self.assertIn('-u MINERU_HYBRID_BATCH_RATIO nohup', captured_env)

    def test_compute_supervisor_running_requires_matching_supervise_compute_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime_dir = tmp_path / "runtime"
            runtime_dir.mkdir()
            fake_proc(tmp_path, 501, "S", ["bash", "scripts/tianshu.sh", "supervise", "compute", "3"])
            fake_proc(tmp_path, 502, "S", ["bash", "scripts/tianshu.sh", "supervise", "compute", "4"])
            ok = run_bash(
                f'WORKER_RUNTIME_DIR="{runtime_dir}"; printf "501\n" > "$(compute_supervisor_pid_file 3)"; compute_supervisor_is_running 3',
                tmp_path,
            )
            wrong_index = run_bash(
                f'WORKER_RUNTIME_DIR="{runtime_dir}"; printf "502\n" > "$(compute_supervisor_pid_file 3)"; compute_supervisor_is_running 3',
                tmp_path,
            )

        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(wrong_index.returncode, 1)

    def test_restart_worker_uses_supervisor_request_when_compute_supervisor_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'compute_supervisor_is_running() { return 0; }; '
                'request_supervised_worker_restart() { echo "request:$1:$2"; }; '
                'drain_worker_instance() { echo drain; }; '
                'stop_worker_instance() { echo stop; }; '
                'start_worker_instance() { echo start; }; '
                'restart_worker_instance 3 --force',
                tmp_path,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("request:3:--force", result.stdout)
        self.assertNotIn("drain", result.stdout)
        self.assertNotIn("stop", result.stdout)
        self.assertNotIn("start", result.stdout)

    def test_supervised_worker_restart_handler_restarts_worker_without_stopping_vllm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime_dir = tmp_path / "runtime"
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            request = runtime_dir / "worker_3.restart.request"
            runtime_dir.mkdir()
            request.write_text("rev1\n--force\n123\n")
            result = run_bash(
                f'WORKER_RUNTIME_DIR="{runtime_dir}" WORKER_LOG_DIR="{worker_dir}"; '
                'vllm_pid=777; worker_pid=111; '
                'drain_worker_instance() { echo drain; return 2; }; '
                'stop_worker_instance() { echo stop_worker; }; '
                'start_worker_instance() { echo start_worker; printf "222\n" > "$(worker_pid_file 3)"; }; '
                'wait_worker_instance_ready() { echo wait_ready; return 0; }; '
                'stop_vllm_instance() { echo stop_vllm; }; '
                'start_vllm_instance() { echo start_vllm; }; '
                'handle_supervised_worker_restart_request 3; printf "worker_pid=%s\n" "$worker_pid"',
                tmp_path,
            )
            ack = (runtime_dir / "worker_3.restart.ack").read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("drain", result.stdout)
        self.assertIn("stop_worker", result.stdout)
        self.assertIn("start_worker", result.stdout)
        self.assertIn("wait_ready", result.stdout)
        self.assertIn("worker_pid=222", result.stdout)
        self.assertNotIn("stop_vllm", result.stdout)
        self.assertNotIn("start_vllm", result.stdout)
        self.assertEqual(ack[:2], ["rev1", "ok"])


    def test_supervised_worker_restart_handler_fails_until_worker_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime_dir = tmp_path / "runtime"
            worker_dir = tmp_path / "worker"
            worker_dir.mkdir()
            runtime_dir.mkdir()
            (runtime_dir / "worker_3.restart.request").write_text("rev2\n--force\n123\n")
            result = run_bash(
                f'WORKER_RUNTIME_DIR="{runtime_dir}" WORKER_LOG_DIR="{worker_dir}" WORKER_READY_TIMEOUT=9; '
                'vllm_pid=777; worker_pid=111; worker_health_failures=2; '
                'drain_worker_instance() { echo drain; return 0; }; '
                'stop_worker_instance() { echo stop_worker; }; '
                'start_worker_instance() { echo start_worker; printf "222\n" > "$(worker_pid_file 3)"; }; '
                'wait_worker_instance_ready() { echo wait_ready:$2; return 1; }; '
                'handle_supervised_worker_restart_request 3; rc=$?; printf "rc=%s worker_pid=%s failures=%s\n" "$rc" "$worker_pid" "$worker_health_failures"; exit "$rc"',
                tmp_path,
            )
            ack = (runtime_dir / "worker_3.restart.ack").read_text().splitlines()

        self.assertEqual(result.returncode, 75)
        self.assertIn("wait_ready:9", result.stdout)
        self.assertIn("rc=75 worker_pid=222 failures=2", result.stdout)
        self.assertEqual(ack[:2], ["rev2", "failed"])
        self.assertEqual(ack[3], "worker-not-ready")

    def test_compute_supervisor_pair_recovers_when_supervised_restart_not_ready(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("supervise_compute_instance()")
        end = text.index("cmd_supervise()", start)
        implementation = text[start:end]

        self.assertIn('restart_request_rc=$?', implementation)
        self.assertIn('failure_reason="supervised restart did not reach ready"', implementation)
        self.assertLess(
            implementation.index('handle_supervised_worker_restart_request "$i"'),
            implementation.index('if ! pid_matches "$worker_pid"'),
        )
        self.assertLess(
            implementation.index('failure_reason="supervised restart did not reach ready"'),
            implementation.index('if ! pid_matches "$worker_pid"'),
        )

    def test_reconciler_is_a_managed_persistent_service(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("start_parent_merge_reconciler()")
        end = text.index("cmd_configure()", start)
        implementation = text[start:end]
        cmd_start = text[text.index("cmd_start()"):text.index("cmd_stop()")]
        cmd_stop = text[text.index("cmd_stop()"):text.index("cmd_restart()")]
        cmd_status = text[text.index("cmd_status()"):text.index("cmd_logs()")]
        cmd_supervise = text[text.index("cmd_supervise()"):text.index("cmd_status()")]

        self.assertIn('reconcile_parent_merges.py', implementation)
        self.assertIn('--apply --watch --report-json', implementation)
        self.assertIn('--child-retention-hours', implementation)
        self.assertIn('start_parent_merge_reconciler', cmd_start)
        self.assertIn('stop_parent_merge_reconciler', cmd_stop)
        self.assertIn('status_parent_merge_reconciler', cmd_status)
        self.assertIn('start_parent_merge_reconciler', cmd_supervise)


    def test_instance_defaults_to_fixed_production_id_and_ignores_generic_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_bash('printf "%s\n%s\n" "$INSTANCE_ID" "$INSTANCE_DATA_DIR"', Path(tmp))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "pytest-instance",
                "/share/wangjiong/databases/mineru_database/pytest-instance",
            ],
        )

    def test_default_active_instances_are_zero_through_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_bash('active_instance_indices', Path(tmp))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0 1 2 3")

    def test_api_status_rejects_healthy_http_when_pid_identity_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_dir = tmp_path / "api"
            api_dir.mkdir()
            (api_dir / "api.pid").write_text("999")
            result = run_bash(
                f'API_LOG_DIR="{api_dir}"; curl() {{ return 0; }}; port_listener_pids() {{ :; }}; status_api',
                tmp_path,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("未运行", result.stdout)

    def test_api_status_reports_owned_listener_with_stale_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_proc(tmp_path, 801, "S", ["/env/bin/python", "api_server.py"])
            api_dir = tmp_path / "api"
            api_dir.mkdir()
            (api_dir / "api.pid").write_text("999")
            result = run_bash(
                f'API_LOG_DIR="{api_dir}"; port_listener_pids() {{ echo 801; }}; status_api',
                tmp_path,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("PID 文件缺失/失效", result.stdout)

    def test_restart_core_stops_controllers_before_api_and_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_bash(
                'init_dirs() { :; }; source_ascend_env() { :; }; validate_active_instance_indices() { :; }; '
                'check_instance_conflict() { :; }; status_redis() { :; }; vllm_is_running() { :; }; vllm_http_ready() { :; }; '
                'stop_watchdog() { echo stop_watchdog; }; stop_scheduler() { echo stop_scheduler; }; '
                'stop_parent_merge_reconciler() { echo stop_reconciler; }; stop_api() { echo stop_api; }; '
                'mark_worker_drain() { echo mark_$1; }; wait_worker_drain() { echo wait_$1; }; '
                'stop_worker_instance() { echo stop_worker_$1; }; start_api() { echo start_api; }; '
                'start_worker_instance() { echo start_worker_$1; }; wait_worker_instance_ready() { :; }; '
                'start_parent_merge_reconciler() { echo start_reconciler; }; start_scheduler() { echo start_scheduler; }; '
                'start_watchdog() { echo start_watchdog; }; cmd_status() { echo strict_$1; }; restart_core',
                tmp_path,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertLess(lines.index("stop_watchdog"), lines.index("stop_api"))
        self.assertLess(lines.index("stop_api"), lines.index("mark_0"))
        self.assertLess(lines.index("stop_worker_3"), lines.index("start_api"))
        self.assertIn("strict_--strict", lines)

    def test_extract_build_id_handles_api_and_worker_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_bash(
                "printf '%s\\n%s\\n' "
                "'{\"build_id\":\"api-build\"}' "
                "'{\"output\":{\"version\":{\"build_id\":\"worker-build\"}}}' "
                "| while IFS= read -r payload; do printf '%s' \"$payload\" | extract_build_id; done",
                Path(tmp),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["api-build", "worker-build"])

    def test_status_versions_accepts_matching_api_and_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_bash(
                'TIANSHU_BUILD_ID="2.0.0+abcdef123456"; '
                'get_api_build_id() { echo "2.0.0+abcdef123456"; }; '
                'get_worker_build_id() { echo "2.0.0+abcdef123456"; }; '
                'status_versions',
                Path(tmp),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("API: 2.0.0+abcdef123456", result.stdout)
        self.assertIn("Worker #3: 2.0.0+abcdef123456", result.stdout)

    def test_status_versions_rejects_mixed_worker_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_bash(
                'TIANSHU_BUILD_ID="2.0.0+abcdef123456"; '
                'get_api_build_id() { echo "2.0.0+abcdef123456"; }; '
                'get_worker_build_id() { '
                '  if [ "$1" = "2" ]; then echo "2.0.0+oldoldoldold"; '
                '  else echo "2.0.0+abcdef123456"; fi; '
                '}; status_versions',
                Path(tmp),
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Worker #2: 2.0.0+oldoldoldold", result.stdout)
        self.assertIn("期望 2.0.0+abcdef123456", result.stdout)

    def test_status_strict_includes_version_consistency(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        cmd_status = text[text.index("cmd_status()"):text.index("cmd_logs()")]
        self.assertIn("status_versions || failed=1", cmd_status)

if __name__ == "__main__":
    unittest.main()
