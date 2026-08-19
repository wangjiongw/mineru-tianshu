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
            "throughput": {"successful_per_minute": None, "logical_pdf_per_minute": None, "pages_per_minute": None},
            "efficiency": {"fleet_energy_wh_per_completed_page": None},
            "tasks": {"delta": {"status": "unknown", "reason": "database not found"}},
            "db_lock_evidence": {"status": "unknown", "reason": "database not found", "combined_delta": None},
            "fleet": {"hbm_p99_percent": None, "hbm_max_percent": None, "util_avg_percent": None, "util_load_balance_cv": None},
            "vllm_metrics": {
                "kv_cache": {"status": "unknown", "reason": "missing kv", "p99_percent": None},
                "preemptions": {"status": "unknown", "reason": "missing preemptions", "delta": None},
            },
            "logs": {
                "oom_preemption": {"status": "unknown", "reason": "no log paths found", "count": None},
                "api_500": {"status": "unknown", "reason": "no log paths found", "count": None},
            },
            "parent_merge_backlog": {"status": "unknown", "reason": "database not found", "stale_merging_parents": None},
            "cpu": {"status": "unknown", "reason": "missing cpu"},
        }

        outcome = evaluator.evaluate(result)

        self.assertFalse(outcome["passed"])
        self.assertFalse(outcome["checks"]["logical_pdf_per_minute_gte_90_percent_baseline"]["passed"])
        self.assertFalse(outcome["checks"]["pages_per_minute_gte_90_percent_baseline"]["passed"])
        self.assertFalse(outcome["checks"]["energy_per_completed_page_lte_90_percent_baseline"]["passed"])
        self.assertFalse(outcome["checks"]["hbm_p99_lte_95_5_percent"]["passed"])
        self.assertFalse(outcome["checks"]["cpu_idle_gte_30_percent"]["passed"])
        self.assertFalse(outcome["checks"]["vllm_preemption_delta_zero"]["passed"])
        self.assertFalse(outcome["checks"]["stale_parent_merges_zero"]["passed"])
        self.assertFalse(outcome["checks"]["api_500_delta_zero"]["passed"])
        self.assertFalse(outcome["diagnostics"]["cpu_iowait_lt_5_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["kv_cache_p99_lte_70_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["hbm_max_lt_98_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["fleet_aicore_util_avg_gte_35_percent"]["available"])
        self.assertFalse(outcome["diagnostics"]["load_balance_cv_lte_0_30"]["available"])

    def test_run_evaluation_uses_mocks_and_stable_schema(self):
        tmp_path = Path(self.create_temp_dir())
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps({
                "throughput": {"successful_per_minute": 10, "logical_pdf_per_minute": 10, "pages_per_minute": 100},
                "efficiency": {"fleet_energy_wh_per_completed_page": 1.0},
            }),
            encoding="utf-8",
        )
        log_path = tmp_path / "service.log"
        log_path.write_text("", encoding="utf-8")
        task_snapshots = iter(
            [
                {
                    "status": "known",
                    "reason": None,
                    "counts": {"completed": 100, "failed": 2},
                    "metrics": {
                        "status": "known",
                        "logical_pdf_completed": 100,
                        "completed_pages": 1000,
                        "processing_seconds": 10.0,
                        "worker_groups": {"0": {"tasks_completed": 10, "pages_completed": 1000, "processing_seconds": 10.0}},
                    },
                    "parent_merge_backlog": {
                        "status": "known",
                        "reason": None,
                        "threshold_seconds": 600.0,
                        "total_merging_parents": 0,
                        "active_merging_parents": 0,
                        "recoverable_stale_merging_parents": 0,
                        "blocked_merging_parents": 0,
                        "stale_merging_parents": 0,
                        "oldest_age_seconds": None,
                        "age_source": "merge_claimed_at",
                    },
                },
                {
                    "status": "known",
                    "reason": None,
                    "counts": {"completed": 124, "failed": 2},
                    "metrics": {
                        "status": "known",
                        "logical_pdf_completed": 124,
                        "completed_pages": 1120,
                        "processing_seconds": 20.0,
                        "worker_groups": {"0": {"tasks_completed": 34, "pages_completed": 1120, "processing_seconds": 20.0}},
                    },
                    "parent_merge_backlog": {
                        "status": "known",
                        "reason": None,
                        "threshold_seconds": 600.0,
                        "total_merging_parents": 0,
                        "active_merging_parents": 0,
                        "recoverable_stale_merging_parents": 0,
                        "blocked_merging_parents": 0,
                        "stale_merging_parents": 0,
                        "oldest_age_seconds": None,
                        "age_source": "merge_claimed_at",
                    },
                },
            ]
        )
        cpu_snapshots = iter(
            [
                {"user": 0, "nice": 0, "system": 0, "idle": 100, "iowait": 0},
                {"user": 10, "nice": 0, "system": 10, "idle": 180, "iowait": 0},
            ]
        )

        with mock.patch.object(evaluator, "sqlite_task_cohort_floor", return_value={
            "status": "known",
            "reason": None,
            "rowid_floor": 99,
            "scope": "rowid_gt_floor",
            "source": "auto_before_collection",
            "semantics": "duration task counts and typed metrics include only tasks inserted after evaluator start; start the evaluator before submitting the batch for accurate run throughput",
        }), \
            mock.patch.object(evaluator, "sqlite_counts", side_effect=lambda *_args, **_kwargs: next(task_snapshots)), \
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
                    "cards": [
                        {"card_id": float(card_id), "hbm_percent": 70.0, "util_percent": 20.0, "power_w": 100.0}
                        for card_id in range(8)
                    ],
                },
            ), \
            mock.patch.object(evaluator, "time", FakeTime):
            args = argparse.Namespace(
                duration=2.0,
                interval=1.0,
                baseline=baseline,
                record_baseline=None,
                sample=False,
                host="localhost",
                database=tmp_path / "tasks.db",
                log_path=[log_path],
                worker_ports=list(range(8101, 8109)),
                vllm_ports=list(range(30025, 30033)),
                stale_parent_threshold_seconds=600.0,
            )

            result = evaluator.run_evaluation(args)

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["task_cohort"]["rowid_floor"], 99)
        self.assertEqual(result["task_cohort"]["scope"], "rowid_gt_floor")
        self.assertEqual(result["task_cohort"]["source"], "auto_before_collection")
        self.assertIn("start the evaluator before submitting the batch", result["task_cohort"]["semantics"])
        self.assertEqual(result["throughput"]["successful_per_minute"], 480.0)
        self.assertEqual(result["throughput"]["logical_pdf_per_minute"], 480.0)
        self.assertEqual(result["throughput"]["pages_per_minute"], 2400.0)
        self.assertEqual(result["throughput"]["method"], "aggregate_short_run")
        self.assertTrue(result["evaluation"]["passed"])
        self.assertAlmostEqual(result["energy"]["fleet_energy_wh"], 0.222222)
        self.assertAlmostEqual(result["efficiency"]["fleet_energy_wh_per_completed_page"], 0.001852)
        self.assertEqual(result["tasks"]["delta"]["worker_groups"]["0"]["pages_completed"], 120.0)
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
            "baseline": {
                "status": "known",
                "successful_per_minute": 10.0,
                "logical_pdf_per_minute": 10.0,
                "pages_per_minute": 100.0,
                "fleet_energy_wh_per_completed_page": 1.0,
            },
            "throughput": {"successful_per_minute": 12.0, "logical_pdf_per_minute": 12.0, "pages_per_minute": 95.0},
            "efficiency": {"fleet_energy_wh_per_completed_page": 0.8},
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
            "logs": {"oom_preemption": {"status": "known", "count": 0}, "api_500": {"status": "known", "count": 0}},
            "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0, "blocked_merging_parents": 0, "oldest_age_seconds": None},
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

    def test_api_500_log_delta_counts_only_new_http_500_lines(self):
        tmp_path = Path(self.create_temp_dir())
        log_path = tmp_path / "api.log"
        log_path.write_text('GET /api/tasks HTTP/1.1" 500\n', encoding="utf-8")
        snapshot = evaluator.snapshot_log_files([log_path])

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write('POST /api/tasks HTTP/1.1" 500\n')
            handle.write('GET /api/tasks HTTP/1.1" 200\n')
            handle.write('status_code=500 route=/api/submit\n')
            handle.write('get excited about a summary of approximately 500 words\n')

        result = evaluator.count_log_delta([log_path], snapshot, evaluator.API_HTTP_500_RE)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["count"], 2)

    def test_parent_merge_backlog_counts_stale_merging_parents_from_merge_claimed_at(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT,
                merge_claimed_at TEXT,
                merge_owner TEXT
            )
            """
        )
        rows = [
            ("fresh-parent", "merging", None, "{}", "{}", "2026-08-19 00:00:00", None, "worker_group_0", "2026-08-19 00:09:00", "worker_group_0"),
            ("stale-parent", "merging", None, "{}", "{}", "2026-08-19 00:00:00", None, "worker_group_1", "2026-08-19 00:00:00", "worker_group_1"),
            ("child", "merging", "stale-parent", "{}", "{}", "2026-08-19 00:00:00", None, "worker_group_1", "2026-08-19 00:00:00", "worker_group_1"),
        ]
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(
            db_path,
            stale_parent_threshold_seconds=600.0,
            now=evaluator.parse_timestamp("2026-08-19 00:11:00"),
        )

        backlog = result["parent_merge_backlog"]
        self.assertEqual(backlog["status"], "known")
        self.assertIsNone(backlog["reason"])
        self.assertEqual(backlog["total_merging_parents"], 2)
        self.assertEqual(backlog["active_merging_parents"], 1)
        self.assertEqual(backlog["recoverable_stale_merging_parents"], 1)
        self.assertEqual(backlog["blocked_merging_parents"], 0)
        self.assertEqual(backlog["stale_merging_parents"], 1)
        self.assertEqual(backlog["oldest_age_seconds"], 660.0)
        self.assertEqual(backlog["age_source"], "merge_claimed_at")

    def test_parent_merge_backlog_falls_back_to_started_at_for_legacy_schema(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("parent", "merging", None, "{}", "{}", "2026-08-19 00:00:00", None, "worker_group_0"),
        )
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(
            db_path,
            stale_parent_threshold_seconds=600.0,
            now=evaluator.parse_timestamp("2026-08-19 00:05:00"),
        )

        backlog = result["parent_merge_backlog"]
        self.assertEqual(backlog["status"], "known")
        self.assertIn("using started_at fallback", backlog["reason"])
        self.assertEqual(backlog["active_merging_parents"], 0)
        self.assertEqual(backlog["recoverable_stale_merging_parents"], 1)
        self.assertEqual(backlog["blocked_merging_parents"], 0)
        self.assertEqual(backlog["oldest_age_seconds"], 300.0)
        self.assertEqual(backlog["age_source"], "started_at")

    def test_parent_merge_backlog_marks_migrated_null_leases_recoverable_stale(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT,
                merge_claimed_at TEXT,
                merge_owner TEXT,
                merge_attempts INTEGER
            )
            """
        )
        rows = [
            ("historical-1", "merging", None, "{}", "{}", "2026-08-19 00:00:00", None, None, None, None, 0),
            ("historical-2", "merging", None, "{}", "{}", "2026-08-19 00:01:00", None, None, None, None, 2),
            ("blocked", "merging", None, "{}", "{}", "2026-08-19 00:02:00", None, None, None, None, 3),
            ("owned-active", "merging", None, "{}", "{}", "2026-08-19 00:00:00", None, "worker_group_0", "2026-08-19 00:09:30", "worker_group_0", 1),
            ("owned-stale", "merging", None, "{}", "{}", "2026-08-19 00:00:00", None, "worker_group_1", "2026-08-19 00:00:00", "worker_group_1", 1),
        ]
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(
            db_path,
            stale_parent_threshold_seconds=600.0,
            now=evaluator.parse_timestamp("2026-08-19 00:11:00"),
        )

        backlog = result["parent_merge_backlog"]
        self.assertEqual(backlog["status"], "known")
        self.assertEqual(backlog["total_merging_parents"], 5)
        self.assertEqual(backlog["active_merging_parents"], 1)
        self.assertEqual(backlog["recoverable_stale_merging_parents"], 3)
        self.assertEqual(backlog["blocked_merging_parents"], 1)
        self.assertEqual(backlog["stale_merging_parents"], 3)
        self.assertEqual(backlog["age_source"], "mixed")
        self.assertEqual(backlog["age_sources"], ["merge_claimed_at", "started_at"])
        self.assertIn("merge_claimed_at NULL", backlog["reason"])

    def test_evaluate_gates_parent_merge_backlog_and_api_500_zero_unknown_nonzero(self):
        result = {
            "services": {
                "workers": [{"port": port, "status": "healthy"} for port in range(8101, 8109)],
                "vllm": [{"port": port, "status": "healthy"} for port in range(30025, 30033)],
            },
            "baseline": {
                "status": "known",
                "logical_pdf_per_minute": 100.0,
                "successful_per_minute": 100.0,
                "pages_per_minute": 1000.0,
                "fleet_energy_wh_per_completed_page": 2.0,
            },
            "throughput": {"successful_per_minute": 100.0, "logical_pdf_per_minute": 100.0, "pages_per_minute": 1000.0},
            "efficiency": {"fleet_energy_wh_per_completed_page": 1.0},
            "tasks": {"delta": {"status": "known", "failed_rate": 0.0}},
            "db_lock_evidence": {"status": "known", "combined_delta": 0},
            "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0, "blocked_merging_parents": 0, "oldest_age_seconds": None},
            "fleet": {"hbm_p99_percent": 90.0, "hbm_max_percent": 92.0, "util_avg_percent": 50.0, "util_load_balance_cv": 0.1},
            "vllm_metrics": {"kv_cache": {"status": "known", "p99_percent": 5.0}, "preemptions": {"status": "known", "delta": 0}},
            "logs": {"oom_preemption": {"status": "known", "count": 0}, "api_500": {"status": "known", "count": 0}},
            "cpu": {"status": "known", "value": {"idle_percent": 60.0, "iowait_percent": 0.0}},
        }

        outcome = evaluator.evaluate(result)
        self.assertTrue(outcome["checks"]["stale_parent_merges_zero"]["passed"])
        self.assertTrue(outcome["checks"]["api_500_delta_zero"]["passed"])

        result["parent_merge_backlog"]["recoverable_stale_merging_parents"] = 1
        outcome = evaluator.evaluate(result)
        self.assertFalse(outcome["checks"]["stale_parent_merges_zero"]["passed"])

        result["parent_merge_backlog"] = {"status": "unknown", "reason": "legacy schema", "recoverable_stale_merging_parents": None}
        result["logs"]["api_500"] = {"status": "unknown", "reason": "no logs", "count": None}
        outcome = evaluator.evaluate(result)
        self.assertFalse(outcome["checks"]["stale_parent_merges_zero"]["passed"])
        self.assertFalse(outcome["checks"]["api_500_delta_zero"]["passed"])

        result["parent_merge_backlog"] = {"status": "known", "recoverable_stale_merging_parents": 0, "blocked_merging_parents": 0, "oldest_age_seconds": None}
        result["logs"]["api_500"] = {"status": "known", "count": 2}
        outcome = evaluator.evaluate(result)
        self.assertFalse(outcome["checks"]["api_500_delta_zero"]["passed"])

    def test_baseline_from_result_records_successful_throughput(self):
        result = {
            "ended_at": "2026-08-18T00:00:00Z",
            "duration_seconds": 60.0,
            "throughput": {
                "successful_per_minute": 42.5,
                "logical_pdf_per_minute": 42.5,
                "pages_per_minute": 425.0,
                "method": "median_6x5m_windows",
                "windows": [{"successful_per_minute": 42.5}],
            },
            "efficiency": {"fleet_energy_wh_per_completed_page": 0.5, "fleet_energy_wh_per_logical_pdf": 5.0},
        }

        baseline = evaluator.baseline_from_result(result)

        self.assertEqual(baseline["schema_version"], 2)
        self.assertEqual(baseline["throughput"]["successful_per_minute"], 42.5)
        self.assertEqual(baseline["throughput"]["pages_per_minute"], 425.0)
        self.assertEqual(baseline["efficiency"]["fleet_energy_wh_per_completed_page"], 0.5)
        self.assertEqual(baseline["throughput"]["method"], "median_6x5m_windows")
        self.assertEqual(baseline["throughput"]["windows"], [{"successful_per_minute": 42.5}])
        self.assertEqual(baseline["source_duration_seconds"], 60.0)

    def test_successful_per_minute_uses_logical_rate_when_raw_completed_delta_differs(self):
        start = {
            "status": "known",
            "counts": {"completed": 100},
            "metrics": {
                "status": "known",
                "logical_pdf_completed": 10,
                "completed_pages": 100,
                "processing_seconds": 0.0,
                "worker_groups": {},
            },
        }
        end = {
            "status": "known",
            "counts": {"completed": 120},
            "metrics": {
                "status": "known",
                "logical_pdf_completed": 12,
                "completed_pages": 120,
                "processing_seconds": 0.0,
                "worker_groups": {},
            },
        }
        snapshots = [
            {"elapsed_seconds": 0.0, "snapshot": start},
            {"elapsed_seconds": 60.0, "snapshot": end},
        ]

        throughput = evaluator.compute_throughput(start, end, 60.0, snapshots)

        self.assertEqual(throughput["raw_successful_per_minute"], 20.0)
        self.assertEqual(throughput["logical_pdf_per_minute"], 2.0)
        self.assertEqual(throughput["successful_per_minute"], 2.0)
        self.assertEqual(throughput["windows"][0]["raw_successful_per_minute"], 20.0)
        self.assertEqual(throughput["windows"][0]["logical_pdf_per_minute"], 2.0)
        self.assertEqual(throughput["windows"][0]["successful_per_minute"], 2.0)

    def test_throughput_keeps_logical_unknown_when_task_metrics_unknown(self):
        start = {
            "status": "known",
            "counts": {"completed": 10},
            "metrics": evaluator.empty_task_metrics("unknown", "task metrics skipped for sample mode"),
        }
        end = {
            "status": "known",
            "counts": {"completed": 25},
            "metrics": evaluator.empty_task_metrics("unknown", "task metrics skipped for sample mode"),
        }
        snapshots = [
            {"elapsed_seconds": 0.0, "snapshot": start},
            {"elapsed_seconds": 60.0, "snapshot": end},
        ]

        throughput = evaluator.compute_throughput(start, end, 60.0, snapshots)

        self.assertIsNone(throughput["successful_per_minute"])
        self.assertEqual(throughput["raw_successful_per_minute"], 15.0)
        self.assertIsNone(throughput["logical_pdf_per_minute"])
        self.assertIsNone(throughput["pages_per_minute"])
        self.assertIsNone(throughput["windows"][0]["logical_pdf_per_minute"])
        self.assertIsNone(throughput["windows"][0]["pages_per_minute"])
        self.assertEqual(throughput["windows"][0]["metrics_status"], "unknown")

    def test_baseline_from_result_does_not_record_raw_status_throughput_as_logical(self):
        result = {
            "ended_at": "2026-08-18T00:00:00Z",
            "duration_seconds": 60.0,
            "throughput": {
                "successful_per_minute": None,
                "raw_successful_per_minute": 15.0,
                "logical_pdf_per_minute": None,
                "pages_per_minute": None,
                "method": "aggregate_short_run",
                "windows": [],
            },
            "efficiency": {"fleet_energy_wh_per_completed_page": None, "fleet_energy_wh_per_logical_pdf": None},
        }

        baseline = evaluator.baseline_from_result(result)

        self.assertEqual(baseline["status"], "unknown")
        self.assertIn("baseline metrics unknown", baseline["reason"])

    def test_throughput_uses_median_of_six_5_minute_windows_and_ignores_outlier(self):
        counts = [0, 50, 100, 150, 200, 250, 1250]
        snapshots = [
            {
                "elapsed_seconds": index * 300.0,
                "snapshot": {
                    "status": "known",
                    "counts": {"completed": count},
                    "metrics": {
                        "status": "known",
                        "logical_pdf_completed": count,
                        "completed_pages": count * 10,
                        "processing_seconds": 0.0,
                        "worker_groups": {},
                    },
                },
            }
            for index, count in enumerate(counts)
        ]
        start = snapshots[0]["snapshot"]
        end = snapshots[-1]["snapshot"]

        throughput = evaluator.compute_throughput(start, end, 1800.0, snapshots)

        self.assertEqual(throughput["method"], "median_6x5m_windows")
        self.assertEqual(throughput["successful_per_minute"], 10.0)
        self.assertEqual([window["successful_per_minute"] for window in throughput["windows"]], [10.0, 10.0, 10.0, 10.0, 10.0, 200.0])

    def test_throughput_short_run_retains_aggregate_with_explicit_method(self):
        start = {
            "status": "known",
            "counts": {"completed": 10},
            "metrics": {"status": "known", "logical_pdf_completed": 10, "completed_pages": 100, "processing_seconds": 0.0, "worker_groups": {}},
        }
        end = {
            "status": "known",
            "counts": {"completed": 40},
            "metrics": {"status": "known", "logical_pdf_completed": 40, "completed_pages": 400, "processing_seconds": 0.0, "worker_groups": {}},
        }
        snapshots = [
            {"elapsed_seconds": 0.0, "snapshot": start},
            {"elapsed_seconds": 60.0, "snapshot": end},
        ]

        throughput = evaluator.compute_throughput(start, end, 60.0, snapshots)

        self.assertEqual(throughput["method"], "aggregate_short_run")
        self.assertEqual(throughput["successful_per_minute"], 30.0)

    def test_parse_npu_smi_extracts_power_temperature_aicore_and_hbm_table(self):
        text = """
        | NPU | Power(W) | Temp(C) | AICore(%) | HBM(%) |
        | 0   | 310.5    | 62      | 71        | 64     |
        | 1   | 290 W    | 58 C    | 88 %      | 96 %   |
        """

        cards = evaluator.parse_npu_smi(text)

        self.assertEqual(cards[0]["power_w"], 310.5)
        self.assertEqual(cards[0]["temperature_c"], 62.0)
        self.assertEqual(cards[0]["util_percent"], 71.0)
        self.assertEqual(cards[0]["hbm_percent"], 64.0)
        self.assertEqual(cards[1]["power_w"], 290.0)
        self.assertEqual(cards[1]["temperature_c"], 58.0)
        self.assertEqual(cards[1]["util_percent"], 88.0)
        self.assertEqual(cards[1]["hbm_percent"], 96.0)

    def test_parse_npu_smi_extracts_power_temp_from_real_two_row_910b2_format(self):
        rows = ["| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |"]
        expected = []
        for card_id in range(8):
            power = 96.8 + card_id
            temp = 34 + card_id
            util = card_id * 10
            hbm_used = 60135 - card_id * 1000
            rows.append(f"| {card_id} 910B2 | OK | {power:.1f} {temp} 0/0 |")
            rows.append(f"| 0 | 0000:C{card_id + 1}:00.0 | {util} 0/0 {hbm_used}/65536 |")
            expected.append((card_id, power, float(temp), float(util), hbm_used * 100.0 / 65536.0))
        rows.extend(
            [
                "| NPU ID | Process ID | Process name | Memory(MB) |",
                "| 0 | 12345 | python | 2048 |",
            ]
        )

        cards = evaluator.parse_npu_smi("\n".join(rows))

        self.assertEqual(len(cards), 8)
        for card, (card_id, power, temp, util, hbm) in zip(cards, expected):
            self.assertEqual(card["card_id"], float(card_id))
            self.assertEqual(card["power_w"], power)
            self.assertEqual(card["temperature_c"], temp)
            self.assertEqual(card["util_percent"], util)
            self.assertAlmostEqual(card["hbm_percent"], hbm)

    def test_energy_integrates_expected_eight_card_power_over_sample_times(self):
        records = [
            {"elapsed_seconds": 0.0, "sample": {"status": "known", "cards": [{"card_id": float(card_id), "power_w": 100.0 + card_id} for card_id in range(8)]}},
            {"elapsed_seconds": 5.0, "sample": {"status": "known", "cards": [{"card_id": float(card_id), "power_w": 110.0 + card_id} for card_id in range(8)]}},
            {"elapsed_seconds": 10.0, "sample": {"status": "known", "cards": [{"card_id": float(card_id), "power_w": 120.0 + card_id} for card_id in range(8)]}},
        ]

        energy = evaluator.summarize_energy(records)

        self.assertEqual(energy["status"], "known")
        self.assertEqual(len(energy["cards"]), 8)
        self.assertAlmostEqual(energy["cards"][0]["energy_wh"], 0.305556)
        self.assertAlmostEqual(energy["cards"][7]["energy_wh"], 0.325)
        self.assertAlmostEqual(energy["fleet_energy_wh"], 2.522222)

    def test_energy_is_unknown_when_expected_card_is_missing(self):
        records = [
            {"elapsed_seconds": 0.0, "sample": {"status": "known", "cards": [{"card_id": float(card_id), "power_w": 100.0} for card_id in range(8)]}},
            {"elapsed_seconds": 5.0, "sample": {"status": "known", "cards": [{"card_id": float(card_id), "power_w": 100.0} for card_id in range(7)]}},
        ]

        energy = evaluator.summarize_energy(records)
        efficiency = evaluator.derive_efficiency(energy, {"status": "known", "completed_pages_delta": 10, "logical_pdf_completed_delta": 1})

        self.assertEqual(energy["status"], "unknown")
        self.assertIn("missing NPU cards", energy["reason"])
        self.assertEqual(efficiency["status"], "unknown")
        self.assertIsNone(efficiency["fleet_energy_wh_per_completed_page"])

    def test_energy_is_unknown_when_power_is_missing_mid_run(self):
        good_cards = [{"card_id": float(card_id), "power_w": 100.0} for card_id in range(8)]
        missing_power_cards = [dict(card) for card in good_cards]
        missing_power_cards[3].pop("power_w")
        records = [
            {"elapsed_seconds": 0.0, "sample": {"status": "known", "cards": good_cards}},
            {"elapsed_seconds": 5.0, "sample": {"status": "known", "cards": missing_power_cards}},
            {"elapsed_seconds": 10.0, "sample": {"status": "known", "cards": good_cards}},
        ]

        energy = evaluator.summarize_energy(records)
        efficiency = evaluator.derive_efficiency(energy, {"status": "known", "completed_pages_delta": 10, "logical_pdf_completed_delta": 1})

        self.assertEqual(energy["status"], "unknown")
        self.assertIn("missing NPU power_w", energy["reason"])
        self.assertEqual(efficiency["status"], "unknown")
        self.assertIsNone(efficiency["fleet_energy_wh_per_completed_page"])

    def test_sqlite_counts_does_not_materialize_live_tasks_table_or_blob_data(self):
        source = Path(evaluator.__file__).read_text(encoding="utf-8")

        self.assertNotIn("SELECT * FROM tasks", source)
        self.assertNotIn("SELECT task_id, options, data", source)
        self.assertNotIn("SELECT options, data", source)

    def test_sqlite_counts_can_skip_completed_leaf_metrics_for_sample_mode(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        real_connect = sqlite3.connect
        captured_sql = []

        class SpyConnection(sqlite3.Connection):
            def execute(self, sql, parameters=(), /):
                captured_sql.append(" ".join(str(sql).split()))
                return super().execute(sql, parameters)

        conn = real_connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("task-1", "completed", None, "{}", json.dumps({"markdown": "x" * (1024 * 1024)}), None, None, None),
        )
        conn.commit()
        conn.close()

        def connect_spy(*args, **kwargs):
            kwargs["factory"] = SpyConnection
            return real_connect(*args, **kwargs)

        with mock.patch.object(evaluator.sqlite3, "connect", side_effect=connect_spy):
            result = evaluator.sqlite_counts(db_path, collect_task_metrics=False)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["metrics"]["status"], "unknown")
        self.assertIn("sample mode", result["metrics"]["reason"])
        self.assertFalse(any("FROM tasks AS task" in sql for sql in captured_sql))
        self.assertFalse(any(" data" in sql.lower() for sql in captured_sql))

    def test_sqlite_counts_streams_completed_leaf_rows_without_fetchall_or_data(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        real_connect = sqlite3.connect
        captured_sql = []

        class NoFetchAllCursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def __iter__(self):
                return iter(self.cursor)

            def fetchall(self):
                raise AssertionError("completed leaf query must not fetchall")

            def __getattr__(self, name):
                return getattr(self.cursor, name)

        class SpyConnection(sqlite3.Connection):
            def execute(self, sql, parameters=(), /):
                captured_sql.append(" ".join(str(sql).split()))
                cursor = super().execute(sql, parameters)
                compact = " ".join(str(sql).split()).lower()
                if "from tasks as task" in compact and "status = 'completed'" in compact:
                    assert "data" not in compact
                    return NoFetchAllCursor(cursor)
                return cursor

        conn = real_connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                page_count INTEGER,
                processing_seconds REAL,
                worker_group_index INTEGER,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "task-1",
                "completed",
                None,
                json.dumps({"chunk_info": {"page_count": 2}}),
                json.dumps({"markdown": "x" * (1024 * 1024)}),
                11,
                7.5,
                3,
                "2026-08-19 00:00:00",
                "2026-08-19 00:00:09",
                "worker_group_0",
            ),
        )
        conn.commit()
        conn.close()

        def connect_spy(*args, **kwargs):
            kwargs["factory"] = SpyConnection
            return real_connect(*args, **kwargs)

        with mock.patch.object(evaluator.sqlite3, "connect", side_effect=connect_spy):
            result = evaluator.sqlite_counts(db_path)

        self.assertEqual(result["metrics"]["completed_pages"], 11.0)
        self.assertEqual(result["metrics"]["processing_seconds"], 7.5)
        self.assertEqual(result["metrics"]["worker_groups"]["3"]["tasks_completed"], 1)
        leaf_queries = [sql for sql in captured_sql if "FROM tasks AS task" in sql]
        self.assertEqual(len(leaf_queries), 1)
        self.assertNotIn("data", leaf_queries[0].lower())

    def test_sqlite_counts_separates_root_logical_pdfs_and_leaf_pages(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT
            )
            """
        )
        rows = [
            ("parent", "completed", None, "{}", "{}", "2026-08-19 00:00:00", "2026-08-19 00:00:10", "worker_group_0"),
            ("child-1", "completed", "parent", json.dumps({"chunk_info": {"page_count": 10}}), "{}", "2026-08-19 00:00:00", "2026-08-19 00:00:04", "worker_group_0"),
            ("child-2", "completed", "parent", json.dumps({"chunk_info": {"page_count": 15}}), "{}", "2026-08-19 00:00:00", "2026-08-19 00:00:06", "worker_group_1"),
            ("standalone", "completed", None, "{}", json.dumps({"metrics": {"page_count": 3}}), "2026-08-19 00:00:00", "2026-08-19 00:00:03", "worker_group_1"),
        ]
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(db_path)

        self.assertEqual(result["metrics"]["status"], "known")
        self.assertEqual(result["metrics"]["logical_pdf_completed"], 2)
        self.assertEqual(result["metrics"]["leaf_completed"], 3)
        self.assertEqual(result["metrics"]["completed_pages"], 25.0)
        self.assertEqual(result["metrics"]["known_page_count_tasks"], 2)
        self.assertAlmostEqual(result["metrics"]["page_count_coverage"], 2 / 3, places=6)
        self.assertIn("data.metrics not scanned", result["metrics"]["reason"])
        self.assertEqual(result["metrics"]["worker_groups"]["0"]["pages_completed"], 10.0)
        self.assertEqual(result["metrics"]["worker_groups"]["1"]["pages_completed"], 15.0)
        self.assertEqual(result["metrics"]["worker_groups"]["1"]["processing_seconds"], 9.0)

    def test_sqlite_counts_cohort_floor_ignores_old_rows_and_keeps_backlog_global(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                page_count INTEGER,
                processing_seconds REAL,
                worker_group_index INTEGER,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT,
                merge_claimed_at TEXT,
                merge_owner TEXT,
                merge_attempts INTEGER
            )
            """
        )
        old_rows = [
            ("old-root", "completed", None, "{}", "{}", 999, 99.0, 7, "2026-08-19 00:00:00", "2026-08-19 00:01:39", "worker_group_7", None, None, 0),
            ("old-merging", "merging", None, "{}", "{}", None, None, None, "2026-08-19 00:00:00", None, None, None, None, 0),
        ]
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", old_rows)
        conn.commit()
        conn.close()

        cohort = evaluator.sqlite_task_cohort_floor(db_path)
        self.assertEqual(cohort["status"], "known")
        self.assertEqual(cohort["source"], "auto_before_collection")
        start = evaluator.sqlite_counts(
            db_path,
            now=evaluator.parse_timestamp("2026-08-19 00:11:00"),
            cohort_floor=cohort["rowid_floor"],
        )

        conn = sqlite3.connect(db_path)
        new_rows = [
            ("new-parent", "completed", None, "{}", "{}", 100, 20.0, 9, "2026-08-19 00:10:00", "2026-08-19 00:10:20", "worker_group_9", None, None, 0),
            ("new-child-1", "completed", "new-parent", "{}", "{}", 10, 4.0, 0, "2026-08-19 00:10:00", "2026-08-19 00:10:04", "worker_group_0", None, None, 0),
            ("new-child-2", "completed", "new-parent", json.dumps({"chunk_info": {"page_count": 15}}), "{}", None, 6.0, 1, "2026-08-19 00:10:00", "2026-08-19 00:10:06", "worker_group_1", None, None, 0),
        ]
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", new_rows)
        conn.commit()
        conn.close()

        end = evaluator.sqlite_counts(
            db_path,
            now=evaluator.parse_timestamp("2026-08-19 00:11:00"),
            cohort_floor=cohort["rowid_floor"],
        )
        delta = evaluator.task_delta(start, end)
        throughput = evaluator.compute_throughput(
            start,
            end,
            60.0,
            [{"elapsed_seconds": 0.0, "snapshot": start}, {"elapsed_seconds": 60.0, "snapshot": end}],
        )

        self.assertEqual(start["counts"], {})
        self.assertEqual(start["metrics"]["logical_pdf_completed"], 0)
        self.assertEqual(end["counts"]["completed"], 3)
        self.assertNotIn("merging", end["counts"])
        self.assertEqual(end["metrics"]["logical_pdf_completed"], 1)
        self.assertEqual(end["metrics"]["leaf_completed"], 2)
        self.assertEqual(end["metrics"]["completed_pages"], 25.0)
        self.assertEqual(end["metrics"]["worker_groups"]["0"]["pages_completed"], 10.0)
        self.assertEqual(end["metrics"]["worker_groups"]["1"]["pages_completed"], 15.0)
        self.assertEqual(end["parent_merge_backlog"]["total_merging_parents"], 1)
        self.assertEqual(delta["completed_delta"], 3)
        self.assertEqual(delta["logical_pdf_completed_delta"], 1)
        self.assertEqual(delta["completed_pages_delta"], 25.0)
        self.assertEqual(throughput["raw_successful_per_minute"], 3.0)
        self.assertEqual(throughput["logical_pdf_per_minute"], 1.0)
        self.assertEqual(throughput["successful_per_minute"], 1.0)
        self.assertEqual(throughput["pages_per_minute"], 25.0)

    def test_sqlite_counts_does_not_scan_completion_data_metrics_page_count(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "task-1",
                "completed",
                None,
                json.dumps({"chunk_info": {"page_count": 2}}),
                json.dumps({"metrics": {"page_count": 9}}),
                "2026-08-19 00:00:00",
                "2026-08-19 00:00:09",
                "worker_group_0",
            ),
        )
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(db_path)

        self.assertEqual(result["metrics"]["completed_pages"], 2.0)
        self.assertEqual(result["metrics"]["known_page_count_tasks"], 1)
        self.assertEqual(result["metrics"]["page_count_coverage"], 1.0)
        self.assertEqual(result["metrics"]["worker_groups"]["0"]["pages_completed"], 2.0)

    def test_sqlite_counts_reports_coverage_when_page_count_exists_only_in_data(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        large_payload = "x" * (1024 * 1024)
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                parent_task_id TEXT,
                options TEXT,
                data TEXT,
                started_at TEXT,
                completed_at TEXT,
                worker_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "task-1",
                "completed",
                None,
                "{}",
                json.dumps({"metrics": {"page_count": 99}, "markdown": large_payload}),
                "2026-08-19 00:00:00",
                "2026-08-19 00:00:09",
                "worker_group_0",
            ),
        )
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(db_path)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["metrics"]["leaf_completed"], 1)
        self.assertEqual(result["metrics"]["completed_pages"], 0.0)
        self.assertEqual(result["metrics"]["known_page_count_tasks"], 0)
        self.assertEqual(result["metrics"]["page_count_coverage"], 0.0)
        self.assertIn("data.metrics not scanned", result["metrics"]["reason"])

    def test_sqlite_counts_reports_unknown_metrics_for_old_schema(self):
        tmp_path = Path(self.create_temp_dir())
        db_path = tmp_path / "tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE tasks (status TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('completed')")
        conn.commit()
        conn.close()

        result = evaluator.sqlite_counts(db_path)

        self.assertEqual(result["status"], "known")
        self.assertEqual(result["counts"]["completed"], 1)
        self.assertEqual(result["metrics"]["status"], "unknown")
        self.assertIn("missing columns", result["metrics"]["reason"])

    def test_evaluate_applies_energy_and_90_percent_baseline_gates(self):
        result = {
            "services": {
                "workers": [{"port": port, "status": "healthy"} for port in range(8101, 8109)],
                "vllm": [{"port": port, "status": "healthy"} for port in range(30025, 30033)],
            },
            "baseline": {
                "status": "known",
                "logical_pdf_per_minute": 100.0,
                "successful_per_minute": 100.0,
                "pages_per_minute": 1000.0,
                "fleet_energy_wh_per_completed_page": 2.0,
            },
            "throughput": {"successful_per_minute": 90.0, "logical_pdf_per_minute": 90.0, "pages_per_minute": 900.0},
            "efficiency": {"fleet_energy_wh_per_completed_page": 1.8},
            "tasks": {"delta": {"status": "known", "failed_rate": 0.0}},
            "db_lock_evidence": {"status": "known", "combined_delta": 0},
            "fleet": {"hbm_p99_percent": 95.5, "hbm_max_percent": 97.9, "util_avg_percent": 50.0, "util_load_balance_cv": 0.1},
            "vllm_metrics": {"kv_cache": {"status": "known", "p99_percent": 5.0}, "preemptions": {"status": "known", "delta": 0}},
            "logs": {"oom_preemption": {"status": "known", "count": 0}, "api_500": {"status": "known", "count": 0}},
            "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0, "blocked_merging_parents": 0, "oldest_age_seconds": None},
            "cpu": {"status": "known", "value": {"idle_percent": 60.0, "iowait_percent": 0.0}},
        }

        outcome = evaluator.evaluate(result)

        self.assertTrue(outcome["passed"])
        result["efficiency"]["fleet_energy_wh_per_completed_page"] = 1.81
        outcome = evaluator.evaluate(result)
        self.assertFalse(outcome["checks"]["energy_per_completed_page_lte_90_percent_baseline"]["passed"])

    def test_run_evaluation_sample_skips_task_metrics_collection(self):
        tmp_path = Path(self.create_temp_dir())
        log_path = tmp_path / "service.log"
        log_path.write_text("", encoding="utf-8")
        task_snapshots = iter(
            [
                {"status": "known", "reason": None, "counts": {"completed": 100}, "metrics": evaluator.empty_task_metrics("unknown", "task metrics skipped for sample mode"), "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0}},
                {"status": "known", "reason": None, "counts": {"completed": 100}, "metrics": evaluator.empty_task_metrics("unknown", "task metrics skipped for sample mode"), "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0}},
            ]
        )
        sqlite_mock = mock.Mock(side_effect=lambda *_args, **_kwargs: next(task_snapshots))
        cpu_snapshots = iter(
            [
                {"user": 0, "nice": 0, "system": 0, "idle": 100, "iowait": 0},
                {"user": 0, "nice": 0, "system": 0, "idle": 200, "iowait": 0},
            ]
        )

        with mock.patch.object(evaluator, "sqlite_counts", sqlite_mock), \
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
                    "cards": [{"card_id": float(card_id), "hbm_percent": 70.0, "util_percent": 20.0, "power_w": 100.0} for card_id in range(8)],
                },
            ), \
            mock.patch.object(evaluator, "time", FakeTime):
            args = argparse.Namespace(
                duration=60.0,
                interval=5.0,
                baseline=None,
                record_baseline=None,
                sample=True,
                host="localhost",
                database=tmp_path / "tasks.db",
                log_path=[log_path],
                worker_ports=list(range(8101, 8109)),
                vllm_ports=list(range(30025, 30033)),
                stale_parent_threshold_seconds=600.0,
            )

            result = evaluator.run_evaluation(args)

        self.assertEqual(result["mode"], "sample")
        self.assertEqual(result["task_cohort"]["rowid_floor"], None)
        self.assertEqual(result["task_cohort"]["scope"], "global_status_only_sample")
        self.assertEqual(result["task_cohort"]["source"], "sample")
        self.assertEqual(sqlite_mock.call_count, 2)
        for call in sqlite_mock.call_args_list:
            self.assertIs(call.kwargs["collect_task_metrics"], False)
        self.assertEqual(result["tasks"]["start"]["metrics"]["status"], "unknown")

    def test_run_evaluation_uses_explicit_cohort_floor_without_auto_lookup(self):
        tmp_path = Path(self.create_temp_dir())
        log_path = tmp_path / "service.log"
        log_path.write_text("", encoding="utf-8")
        task_snapshots = iter(
            [
                {
                    "status": "known",
                    "reason": None,
                    "counts": {},
                    "metrics": {"status": "known", "logical_pdf_completed": 0, "completed_pages": 0, "processing_seconds": 0.0, "worker_groups": {}},
                    "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0},
                },
                {
                    "status": "known",
                    "reason": None,
                    "counts": {"completed": 3},
                    "metrics": {"status": "known", "logical_pdf_completed": 1, "completed_pages": 25, "processing_seconds": 10.0, "worker_groups": {}},
                    "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0},
                },
            ]
        )
        sqlite_mock = mock.Mock(side_effect=lambda *_args, **_kwargs: next(task_snapshots))
        cpu_snapshots = iter(
            [
                {"user": 0, "nice": 0, "system": 0, "idle": 100, "iowait": 0},
                {"user": 0, "nice": 0, "system": 0, "idle": 200, "iowait": 0},
            ]
        )

        with mock.patch.object(evaluator, "sqlite_task_cohort_floor", side_effect=AssertionError("auto floor should not run")), \
            mock.patch.object(evaluator, "sqlite_counts", sqlite_mock), \
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
                    "cards": [{"card_id": float(card_id), "hbm_percent": 70.0, "util_percent": 20.0, "power_w": 100.0} for card_id in range(8)],
                },
            ), \
            mock.patch.object(evaluator, "time", FakeTime):
            args = argparse.Namespace(
                duration=1.0,
                interval=1.0,
                baseline=None,
                record_baseline=None,
                sample=False,
                cohort_floor=123,
                host="localhost",
                database=tmp_path / "tasks.db",
                log_path=[log_path],
                worker_ports=list(range(8101, 8109)),
                vllm_ports=list(range(30025, 30033)),
                stale_parent_threshold_seconds=600.0,
            )

            result = evaluator.run_evaluation(args)

        self.assertEqual(result["task_cohort"]["rowid_floor"], 123)
        self.assertEqual(result["task_cohort"]["source"], "explicit_cli")
        self.assertEqual(result["task_cohort"]["scope"], "rowid_gt_floor")
        self.assertIn("reuse the same --cohort-floor", result["task_cohort"]["semantics"])
        for call in sqlite_mock.call_args_list:
            self.assertEqual(call.kwargs["cohort_floor"], 123)

    def test_main_rejects_negative_cohort_floor(self):
        with self.assertRaises(SystemExit) as raised, mock.patch("sys.stderr", io.StringIO()):
            evaluator.main(["--cohort-floor", "-1"])

        self.assertEqual(raised.exception.code, 2)

    def test_main_rejects_sample_with_cohort_floor(self):
        with self.assertRaises(SystemExit) as raised, mock.patch("sys.stderr", io.StringIO()):
            evaluator.main(["--sample", "--cohort-floor", "1"])

        self.assertEqual(raised.exception.code, 2)

    def test_duration_sampling_stops_on_wall_clock_instead_of_fixed_count(self):
        tmp_path = Path(self.create_temp_dir())
        log_path = tmp_path / "service.log"
        log_path.write_text("", encoding="utf-8")
        sqlite_mock = mock.Mock(
            return_value={
                "status": "known",
                "reason": None,
                "counts": {},
                "metrics": {"status": "known", "logical_pdf_completed": 0, "completed_pages": 0, "processing_seconds": 0.0, "worker_groups": {}},
                "parent_merge_backlog": {"status": "known", "recoverable_stale_merging_parents": 0},
            }
        )
        npu_mock = mock.Mock(
            return_value={
                "status": "known",
                "reason": None,
                "cards": [{"card_id": float(card_id), "hbm_percent": 70.0, "util_percent": 20.0, "power_w": 100.0} for card_id in range(8)],
            }
        )
        cpu_snapshots = iter(
            [
                {"user": 0, "nice": 0, "system": 0, "idle": 100, "iowait": 0},
                {"user": 0, "nice": 0, "system": 0, "idle": 200, "iowait": 0},
            ]
        )

        SlowSamplingTime.current = 0.0
        SlowSamplingTime.calls = 0

        with mock.patch.object(evaluator, "sqlite_counts", sqlite_mock), \
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
            mock.patch.object(evaluator, "collect_npu_smi", npu_mock), \
            mock.patch.object(evaluator, "time", SlowSamplingTime):
            args = argparse.Namespace(
                duration=10.0,
                interval=5.0,
                baseline=None,
                record_baseline=None,
                sample=False,
                cohort_floor=0,
                host="localhost",
                database=tmp_path / "tasks.db",
                log_path=[log_path],
                worker_ports=list(range(8101, 8109)),
                vllm_ports=list(range(30025, 30033)),
                stale_parent_threshold_seconds=600.0,
            )

            result = evaluator.run_evaluation(args)

        self.assertEqual(npu_mock.call_count, 2)
        self.assertGreaterEqual(result["duration_seconds"], 10.0)
        self.assertLess(result["duration_seconds"], 20.0)

    def create_temp_dir(self):
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return temp_dir.name


class SlowSamplingTime:
    current = 0.0
    calls = 0

    @classmethod
    def monotonic(cls):
        cls.calls += 1
        if cls.calls == 1:
            return cls.current
        if cls.calls <= 3:
            cls.current += 6.0
        return cls.current

    @classmethod
    def sleep(cls, seconds):
        cls.current += seconds


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
