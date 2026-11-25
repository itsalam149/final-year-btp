"""
Unit Tests for Hardware Profiler & Telemetry Engine
Department of CSE, Delhi Technological University (DTU) | Group GP159

Lead Optimization Engineer: Ekansh Agarwal (23/CS/149)
"""

import os
import unittest
import tempfile

from mobile_agent.profiler.hardware_telemetry import (
    HardwareProfiler,
    HardwareTelemetrySnapshot,
    ADBTelemetryBridge,
)


class TestHardwareProfiler(unittest.TestCase):

    def setUp(self):
        self.profiler = HardwareProfiler()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

    def test_single_inference_profiling_q4_k_m(self):
        snapshot = self.profiler.profile_inference_run(
            model_name="Llama-3.2-1B-Instruct",
            quantization_format="Q4_K_M",
            context_tokens=512,
            generated_tokens=40,
            threads=4
        )
        self.assertEqual(snapshot.weight_ram_mb, 804.0)
        self.assertEqual(snapshot.kv_cache_ram_mb, 64.0)
        self.assertEqual(snapshot.total_ram_footprint_mb, 924.0)
        self.assertAlmostEqual(snapshot.tokens_per_second, 28.4, places=1)
        self.assertGreater(snapshot.time_to_first_token_ms, 0.0)
        self.assertGreater(snapshot.energy_joules_per_token, 0.0)
        self.assertLess(snapshot.battery_drop_pct_per_100_queries, 1.0)
        self.assertFalse(snapshot.is_throttled)
        self.assertEqual(snapshot.throttling_tier, "NONE")

    def test_quantization_comparison_memory_reduction(self):
        snap_fp16 = self.profiler.profile_inference_run(
            model_name="Llama-3.2-1B-Instruct",
            quantization_format="FP16"
        )
        snap_q4km = self.profiler.profile_inference_run(
            model_name="Llama-3.2-1B-Instruct",
            quantization_format="Q4_K_M"
        )
        # Q4_K_M should achieve >60% memory reduction compared to FP16
        reduction_pct = (1.0 - (snap_q4km.weight_ram_mb / snap_fp16.weight_ram_mb)) * 100.0
        self.assertGreater(reduction_pct, 65.0)

        # Energy per token should be significantly lower on Q4_K_M
        self.assertLess(snap_q4km.energy_joules_per_token, snap_fp16.energy_joules_per_token)

    def test_context_scaling_telemetry(self):
        snap_128 = self.profiler.profile_inference_run(
            model_name="Llama-3.2-1B-Instruct",
            quantization_format="Q4_K_M",
            context_tokens=128
        )
        snap_2048 = self.profiler.profile_inference_run(
            model_name="Llama-3.2-1B-Instruct",
            quantization_format="Q4_K_M",
            context_tokens=2048
        )
        # 2048 tokens KV cache must be exactly 16x of 128 tokens
        self.assertEqual(snap_128.kv_cache_ram_mb, 16.0)
        self.assertEqual(snap_2048.kv_cache_ram_mb, 256.0)
        self.assertGreater(snap_2048.prompt_eval_ms, snap_128.prompt_eval_ms)

    def test_thermal_throttling_model(self):
        # At t=0s, no throttling
        snap_initial = self.profiler.profile_inference_run(
            continuous_run_elapsed_sec=0.0
        )
        self.assertFalse(snap_initial.is_throttled)
        self.assertEqual(snap_initial.thermal_junction_temp_c, 26.5)

        # At t=240s of continuous saturation, heavy DVFS throttling should engage (>42C)
        snap_heated = self.profiler.profile_inference_run(
            continuous_run_elapsed_sec=240.0
        )
        self.assertTrue(snap_heated.is_throttled)
        self.assertGreaterEqual(snap_heated.thermal_junction_temp_c, 42.0)
        self.assertEqual(snap_heated.throttling_tier, "HEAVY_DVFS")
        self.assertLess(snap_heated.tokens_per_second, snap_initial.tokens_per_second)

    def test_adb_bridge_fallback(self):
        bridge = ADBTelemetryBridge(target_package="com.dtu.mobileagent")
        battery = bridge.query_live_battery()
        self.assertIn("level", battery)
        self.assertIn("temperature_c", battery)
        self.assertGreater(battery["level"], 0)

        mem = bridge.query_process_memory_mb()
        self.assertGreater(mem, 0.0)

    def test_hardware_suite_export(self):
        suite_res = self.profiler.run_hardware_suite(output_dir=self.temp_dir)
        self.assertGreater(suite_res["total_runs"], 10)
        self.assertTrue(os.path.exists(suite_res["csv_path"]))
        self.assertTrue(os.path.exists(suite_res["report_path"]))
        self.assertGreater(os.path.getsize(suite_res["csv_path"]), 100)
        self.assertGreater(os.path.getsize(suite_res["report_path"]), 500)


if __name__ == "__main__":
    unittest.main()
