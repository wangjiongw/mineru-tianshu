import argparse
import io
import json
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

import scripts.evaluate_local_efficiency as evaluator


class EvaluateLocalEfficiencyTests(unittest.TestCase):
    def test_parse_npu_smi_extracts_hbm_and_utilization(self):
        text = """
        NPU 0 HBM Usage 64 %
        NPU 0 AICore Utilization 71 %
        NPU 1 HBM Usage 96 %
        NPU 1 AICore Utilization 88 %
        """

        cards = evaluator.parse_npu_smi(text)

        self.assertEqual(
            cards,
            [
                {"card_id": 0.0, "hbm_percent": 64.0, "util_percent": 71.0},
                {"card_id": 1.0, "hbm_percent": 96.0, "util_percent": 88.0},
            ],
        )

    def test_parse_vllm_metrics_reads_kv_fraction_and_preemptions(self):
        metrics = """
        # HELP vllm:kv_cache_usage_perc KV cache usage
        vllm:kv_cache_usage_perc{gpu="0"} 0.42
        vllm:num_preemptions_total 3
        vllm:num_requests_running 7
        """

        parsed = evaluator.parse_vllm_metrics(metrics)

        self.assertEqual(parsed["kv_cache_usage_percent"], 42.0)
        self.assertEqual(parsed["num_preemptions_total"], 3.0)
        self.assertEqual(parsed["requests_running"], 7.0)

    def test_summarize_vllm_metrics_computes_kv_p99_and_preemption_delta(self):
        samples = [
            [
                {"port": 30025, "metrics_status": "known", "metrics_summary": {"kv_cache_usage_percent": 40.0, "num_preemptions_total": 2.0}},
                {"port": 30026, "metrics_status": "known", "metrics_summary": {"kv_cache_usage_percent": 50.0, "num_preemptions_total": 1.0}},
            ],
            [
                {"port": 30025, "metrics_status": "known", "metrics_summary": {"kv_cache_usage_percent": 60.0, "num_preemptions_total": 2.0}},
                {"port": 30026, "metrics_status": "known", "metrics_summary": {"kv_cache_usage_percent": 80.0, "num_preemptions_total": 3.0}},
            ],
        ]

        summary = evaluator.summarize_vllm_metrics(samples)

        self.assertAlmostEqual(summary["kv_cache"]["p99_percent"], 79.4)
        self.assertEqual(summary["preemptions"]["delta"], 2)

    def test_summarize_vllm_metrics_fails_closed_when_metrics_missing(self):
        samples = [[{"port": 30025, "metrics_status": "known", "metrics_summary": {}}]]

        summary = evaluator.summarize_vllm_metrics(samples)

        self.assertEqual(summary["kv_cache"]["status"], "unknown")
        self.assertEqual(summary["preemptions"]["status"], "unknown")

    def test_fleet_summary_computes_util_cv_and_hbm_max(self):
        samples = [
            {"status": "known", "cards": [
                {"card_id": 0.0, "hbm_percent": 90.0, "util_percent": 40.0},
                {"card_id": 1.0, "hbm_percent": 97.0, "util_percent": 60.0},
            ]}
        ]

        summary = evaluator.summarize_fleet(samples)

        self.assertEqual(summary["hbm_max_percent"], 97.0)
        self.assertEqual(summary["util_avg_percent"], 50.0)
        self.assertEqual(summary["util_load_balance_cv"], 0.2)

    def test_task_delta_counts_terminal_status_without_error_message_scan(self):
        start = {"status": "known", "counts": {"completed": 10, "failed": 1}}
        end = {"status": "known", "counts": {"completed": 22, "failed": 2}}

        delta = evaluator.task_delta(start, end)

        self.assertEqual(delta["completed_delta"], 12)
        self.assertEqual(delta["failed_delta"], 1)
        self.assertEqual(delta["terminal_delta"], 13)
        self.assertEqual(delta["failed_rate"], 1 / 13)
        self.assertNotIn("db_locked_terminal_failures_delta", delta)

    def test_sqlite_counts_reads_database_without_writing(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE tasks (status TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('completed')")
        conn.execute("INSERT INTO tasks VALUES ('failed')")
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(db_path)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertNotIn("db_locked_terminal_failures", result["counts"])

    def test_evaluate_fails_unknown_metrics_instead_of_false_pass(self):
        result = {
            "services": {
                "workers": [{"port": 8101, "status": "healthy"}],
                "vllm": [{"port": 30025, "status": "healthy"}],
            },
            "baseline": {"status": "unknown", "reason": "baseline path not provided"},
            "throughput": {"successful_per_minute": None},
            "tasks": {"delta": {"status": "unknown", "reason": "database not found"}},
            "db_lock_evidence": {"status": "unknown", "reason": "database not found", "combined_delta": None},
            "fleet": {"hbm_p99_percent": None, "hbm_max_percent": None, "util_avg_percent": None, "util_load_balance_cv": None},
            "vllm_metrics": {
                "kv_cache": {"status": "unknown", "reason": "missing kv", "p99_percent": None},
                "preemptions": {"status": "unknown", "reason": "missing preemptions", "delta": None},
            },
            "logs": {"oom_preemption": {"status": "unknown", "reason": "no log paths found", "count": None}},
            "cpu": {"status": "unknown", "reason": "missing cpu"},
        }

        outcome = evaluator.evaluate(result)

        self.assertFalse(outcome["passed"])
        self.assertFalse(outcome["checks"]["throughput_plus_20_percent"]["passed"])
        self.assertFalse(outcome["checks"]["hbm_p99_lte_95_5_percent"]["passed"])
        self.assertFalse(outcome["checks"]["cpu_idle_gte_30_percent"]["passed"])
        self.assertFalse(outcome["checks"]["vllm_preemption_delta_zero"]["passed"])
        self.assertFalse(outcome["diagnostics"]["cpu_iowait_lt_5_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["kv_cache_p99_lte_70_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["hbm_max_lt_98_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["fleet_aicore_util_avg_gte_35_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["load_balance_cv_lte_0_30"]["available"])

    def test_run_evaluation_uses_mocks_and_stable_schema(self):
        tmp_path = Path(self.create_temp_dir())
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"throughput": {"successful_per_minute": 10}}), encoding="utf-8")
        log_path = tmp_path / "service.log"
        log_path.write_text("", encoding="utf-8")
        task_snapshots = iter(
            [
                {"status": "known", "reason": None, "counts": {"completed": 100, "failed": 2}},
                {"status": "known", "reason": None, "counts": {"completed": 124, "failed": 2}},
            ]
        )
        cpu_snapshots = iter(
            [
                {"user": 0, "nice": 0, "system": 0, "idle": 100, "iowait": 0},
                {"user": 10, "nice": 0, "system": 10, "idle": 180, "iowait": 0},
            ]
        )

        with mock.patch.object(evaluator, "sqlite_counts", side_effect=lambda _path: next(task_snapshots)), \
            mock.patch.object(evaluator, "read_cpu_times", side_effect=lambda: next(cpu_snapshots)), \
            mock.patch.object(evaluator, "collect_worker_health", side_effect=lambda host, ports: [{"port": port, "status": "healthy"} for port in ports]), \
            mock.patch.object(
                evaluator,
                "collect_vllm",
                side_effect=lambda host, ports: [
                    {
                        "port": port,
                        "status": "healthy",
                        "metrics_status": "known",
                        "metrics_summary": {"kv_cache_usage_percent": 50.0, "num_preemptions_total": 0.0},
                    }
                    for port in ports
                ],
            ), \
            mock.patch.object(
                evaluator,
                "collect_npu_smi",
                side_effect=lambda: {
                    "status": "known",
                    "reason": None,
                    "cards": [{"card_id": 0.0, "hbm_percent": 70.0, "util_percent": 20.0}],
                },
            ), \
            mock.patch.object(evaluator, "time", FakeTime):
            args = argparse.Namespace(
                duration=1.0,
                interval=1.0,
                baseline=baseline,
                record_baseline=None,
                sample=False,
                host="localhost",
                database=tmp_path / "tasks.db",
                log_path=[log_path],
                worker_ports=list(range(8101, 8109)),
                vllm_ports=list(range(30025, 30033)),
            )

            result = evaluator.run_evaluation(args)

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["throughput"]["successful_per_minute"], 480.0)
        self.assertEqual(result["throughput"]["method"], "aggregate_short_run")
        self.assertTrue(result["evaluation"]["passed"])
        self.assertTrue(result["evaluation"]["checks"]["vllm_preemption_delta_zero"]["passed"])
        self.assertTrue(result["evaluation"]["diagnostics"]["kv_cache_p99_lte_70_percent"]["meets_reference"])
        self.assertTrue(result["evaluation"]["diagnostics"]["hbm_max_lt_98_percent"]["meets_reference"])
        self.assertFalse(result["evaluation"]["diagnostics"]["fleet_aicore_util_avg_gte_35_percent"]["meets_reference"])
        self.assertTrue(result["evaluation"]["diagnostics"]["load_balance_cv_lte_0_30"]["meets_reference"])
        self.assertTrue(result["evaluation"]["diagnostics"]["cpu_iowait_lt_5_percent"]["meets_reference"])
        self.assertEqual(result["services"]["workers"][0]["port"], 8101)

    def test_reevaluate_report_uses_existing_evidence_and_canonical_gates(self):
        tmp_path = Path(self.create_temp_dir())
        report_path = tmp_path / "report.json"
        report = {
            "services": {
                "workers": [{"port": port, "status": "healthy"} for port in range(8101, 8109)],
                "vllm": [{"port": port, "status": "healthy"} for port in range(30025, 30033)],
            },
            "baseline": {"status": "known", "successful_per_minute": 10.0},
            "throughput": {"successful_per_minute": 12.0},
            "tasks": {"delta": {"status": "known", "failed_rate": 0.0}},
            "db_lock_evidence": {"status": "known", "combined_delta": 0},
            "fleet": {
                "hbm_p99_percent": 90.0,
                "hbm_max_percent": 92.0,
                "util_avg_percent": 20.0,
                "util_load_balance_cv": 0.1,
            },
            "vllm_metrics": {
                "kv_cache": {"status": "known", "p99_percent": 5.0},
                "preemptions": {"status": "known", "delta": 0},
            },
            "logs": {"oom_preemption": {"status": "known", "count": 0}},
            "cpu": {"status": "known", "value": {"idle_percent": 60.0, "iowait_percent": 0.0}},
            "evaluation": {"passed": False, "reasons": ["stale policy"]},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")

        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = evaluator.main(["--evaluate-report", str(report_path)])

        result = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(result["evaluation"]["passed"])
        self.assertFalse(result["evaluation"]["diagnostics"]["fleet_aicore_util_avg_gte_35_percent"]["meets_reference"])
        self.assertTrue(result["reevaluation"]["evidence_unchanged"])
        self.assertEqual(result["reevaluation"]["original_evaluation"], report["evaluation"])

    def test_log_delta_counts_only_appended_terminal_db_lock_matches(self):
        tmp_path = Path(self.create_temp_dir())
        log_path = tmp_path / "worker.log"
        log_path.write_text(
            "old failed terminal database is locked\nold out of memory\n",
            encoding="utf-8",
        )
        snapshot = evaluator.snapshot_log_files([log_path])

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("new failed terminal database is locked\n")
            handle.write("plain database is locked without terminal failure wording\n")

        result = evaluator.count_log_delta([log_path], snapshot, evaluator.DB_LOCK_TERMINAL_RE)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["files"][0]["mode"], "append")

    def test_log_delta_handles_truncation_without_counting_preexisting_matches(self):
        tmp_path = Path(self.create_temp_dir())
        log_path = tmp_path / "worker.log"
        log_path.write_text("old failed terminal database is locked\n", encoding="utf-8")
        snapshot = evaluator.snapshot_log_files([log_path])

        log_path.write_text("failed database is locked\n", encoding="utf-8")

        result = evaluator.count_log_delta([log_path], snapshot, evaluator.DB_LOCK_TERMINAL_RE)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["files"][0]["mode"], "truncated")

    def test_baseline_from_result_records_successful_throughput(self):
        result = {
            "ended_at": "2026-08-18T00:00:00Z",
            "duration_seconds": 60.0,
            "throughput": {
                "successful_per_minute": 42.5,
                "method": "median_6x5m_windows",
                "windows": [{"successful_per_minute": 42.5}],
            },
        }

        baseline = evaluator.baseline_from_result(result)

        self.assertEqual(baseline["throughput"]["successful_per_minute"], 42.5)
        self.assertEqual(baseline["throughput"]["method"], "median_6x5m_windows")
        self.assertEqual(baseline["throughput"]["windows"], [{"successful_per_minute": 42.5}])
        self.assertEqual(baseline["source_duration_seconds"], 60.0)

    def test_throughput_uses_median_of_six_5_minute_windows_and_ignores_outlier(self):
        counts = [0, 50, 100, 150, 200, 250, 1250]
        snapshots = [
            {"elapsed_seconds": index * 300.0, "snapshot": {"status": "known", "counts": {"completed": count}}}
            for index, count in enumerate(counts)
        ]
        start = snapshots[0]["snapshot"]
        end = snapshots[-1]["snapshot"]

        throughput = evaluator.compute_throughput(start, end, 1800.0, snapshots)

        self.assertEqual(throughput["method"], "median_6x5m_windows")
        self.assertEqual(throughput["successful_per_minute"], 10.0)
        self.assertEqual([window["successful_per_minute"] for window in throughput["windows"]], [10.0, 10.0, 10.0, 10.0, 10.0, 200.0])

    def test_throughput_short_run_retains_aggregate_with_explicit_method(self):
        start = {"status": "known", "counts": {"completed": 10}}
        end = {"status": "known", "counts": {"completed": 40}}
        snapshots = [
            {"elapsed_seconds": 0.0, "snapshot": start},
            {"elapsed_seconds": 60.0, "snapshot": end},
        ]

        throughput = evaluator.compute_throughput(start, end, 60.0, snapshots)

        self.assertEqual(throughput["method"], "aggregate_short_run")
        self.assertEqual(throughput["successful_per_minute"], 30.0)

    def create_temp_dir(self):
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return temp_dir.name


class FakeTime:
    current = 0.0

    @classmethod
    def monotonic(cls):
        cls.current += 1.0
        return cls.current

    @staticmethod
    def sleep(_seconds):
        return None


if __name__ == "__main__":
    unittest.main()
