"""Behavioral fixtures for Nsight correlation and report accounting (no GPU required)."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.bench.flash_next_performance import Hardware, Work, main, markdown, read_report, union_ns

PID = 17 << 24
TID = PID + 3
HARDWARE = Hardware("fixture", "synthetic rates", "theoretical", 1, 1, 1, 1)
TAG = "ninfer.work/1|moe.nvfp4|1|0|0|100|200|300|400|0"


def make_trace(path):
    db = sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, graphNodeId INTEGER, originalGraphNodeId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER, correlationId INTEGER, graphNodeId INTEGER);
    ''')
    db.execute("INSERT INTO StringIds VALUES (1, ?)", (TAG,))
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)", [
        (10, 100, TID, None, 1),  # graph construction, outside measurement
        (5, 105, TID, "ninfer.region/1|target.verify", None),
        (1000, 10000, TID, "ninfer.region/1|measured", None),
        (1010, 9990, TID, "decode", None),
        (6000, 6100, TID, TAG, None),  # eager Op launches asynchronously
        (8000, 8200, TID, "ninfer.host/1|ple.gather", None),
    ])
    db.executemany("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?, ?, ?, ?, ?)", [
        (20, 21, TID, 101, None), (40, 41, TID, 102, None),
        (199, 199, TID, 201, None), (200, 201, TID, 201, 101),
        (200, 200, TID, 202, None), (201, 202, TID, 202, 102),
    ])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)", [
        (1200, 1250, TID, 10), (3000, 3050, TID, 11),
        (6020, 6030, TID, 12), (6040, 6050, TID, 13),
        (7000, 7010, TID, 14),
        (1200, 1210, (18 << 24) + 1, 10),  # another process, same correlation ID
    ])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)", [
        (1300, 1800, PID, 0, 10, 201), (1600, 2100, PID, 0, 10, 202),
        (3100, 3600, PID, 0, 11, 201), (3600, 4100, PID, 0, 11, 202),
        (6200, 6300, PID, 0, 12, None), (6400, 6500, PID, 0, 13, None),
        (7100, 7200, PID, 0, 14, None),
        (500, 900, PID, 0, 1, 201),  # warmup is excluded
        (1300, 8000, 18 << 24, 0, 10, 201),
    ])
    db.commit()
    return db


class PerformanceReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.sqlite"
        self.db = make_trace(self.path)
        self.addCleanup(self.db.close)

    def report(self):
        self.db.commit()
        return read_report(self.path, HARDWARE)

    def test_graph_clone_replay_and_eager_work_count_once_per_call(self):
        report = self.report()
        self.assertEqual(sum(r["calls"] for r in report["stages"]), 3)
        graph = next(r for r in report["stages"] if r["role"] == "target.verify")
        self.assertEqual(graph["calls"], 2)
        self.assertAlmostEqual(graph["gpu_work_ms"], .002)
        self.assertAlmostEqual(graph["gpu_busy_ms"], .0018)
        self.assertAlmostEqual(graph["estimated_ms_min"], .0002)
        self.assertEqual(graph["phase"], "decode")
        self.assertAlmostEqual(report["measurement"]["gpu_work_ms"], .0023)
        self.assertAlmostEqual(report["measurement"]["gpu_busy_ms"], .0021)
        self.assertAlmostEqual(report["measurement"]["unattributed_gpu_work_ms"], .0001)
        self.assertEqual(report["host"][0]["host_work_ms"], .0002)
        self.assertIn("not an achievable optimum", markdown(report))
        json.dumps(report, allow_nan=False)

    def test_missing_graph_construction_is_unattributed_not_eager(self):
        self.db.execute("DELETE FROM CUDA_GRAPH_NODE_EVENTS")
        report = self.report()
        self.assertEqual(report["measurement"]["unattributed_graph_activities"], 4)
        self.assertEqual(sum(r["calls"] for r in report["stages"]), 1)

    def test_partial_graph_cannot_claim_full_op_efficiency(self):
        self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=10 AND graphNodeId=202")
        graph = next(r for r in self.report()["stages"] if r["role"] == "target.verify")
        self.assertEqual(graph["partial_calls"], 1)
        self.assertIsNone(graph["estimated_ms_min"])
        self.assertIsNone(graph["estimate_efficiency_pct_max"])

    def test_no_measurement_rejects_startup_as_inference(self):
        self.db.execute("DELETE FROM NVTX_EVENTS WHERE text='ninfer.region/1|measured'")
        with self.assertRaisesRegex(ValueError, "measured range"):
            self.report()

    def test_truncated_gpu_activity_is_rejected(self):
        self.db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=10001 WHERE correlationId=14")
        with self.assertRaisesRegex(ValueError, "crosses"):
            self.report()

    def test_graph_only_decode_is_not_silently_omitted(self):
        self.db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER)")
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (8500, 9000, ?, 99)", (PID,))
        with self.assertRaisesRegex(ValueError, "graph-only"):
            self.report()

    def test_compute_floor_and_expert_traffic_interval(self):
        work = Work.parse(TAG)
        self.assertEqual(HARDWARE.estimate_ms(work), (.0001, .0002))
        self.assertEqual(HARDWARE.estimate_ms(Work("dense", 1, 1, 0, 0, 0, 1000000, 0, 0)), (.001, .001))
        with self.assertRaises(ValueError):
            Hardware("gpu", "source", "theoretical", float("nan"), 1, 1, 1)
        with self.assertRaises(ValueError):
            Work.parse("ninfer.work/1|bad|0|0|0|200|100|0|0|0")

    def test_overlap_union_does_not_sum_parallel_time(self):
        self.assertEqual(union_ns([(0, 8), (2, 3), (6, 10), (12, 15)]), 13)

    def test_cli_preserves_benchmark_metadata_and_writes_both_formats(self):
        directory = Path(self.temp.name)
        hardware = directory / "hardware.json"
        hardware.write_text(json.dumps(HARDWARE.__dict__))
        benchmark = directory / "benchmark.json"
        metadata = {"artifact_type": "ninfer_bench_report", "schema_version": 16,
                    "environment": {"gpu_name": "fixture"},
                    "load": {"architecture": "Qwen3_8FlashNextForCausalLM", "formats": ["bf16", "nvfp4"]},
                    "tests": [{"decode_output_tok_s_mean": 123, "requested_output_tokens": 33}]}
        benchmark.write_text(json.dumps(metadata))
        output = directory / "result.json"
        with patch("sys.argv", ["report", str(self.path), "--hardware", str(hardware),
                                "--benchmark", str(benchmark), "--output", str(output)]):
            main()
        report = json.loads(output.read_text())
        self.assertEqual(report["benchmark"], metadata)
        self.assertEqual(report["schema_version"], 1)
        self.assertIn("Unattributed GPU work", output.with_suffix(".md").read_text())


if __name__ == "__main__":
    unittest.main()
