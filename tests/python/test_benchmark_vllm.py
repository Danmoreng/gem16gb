import argparse
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "benchmark_vllm.py"
SPEC = importlib.util.spec_from_file_location("benchmark_vllm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_vllm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_vllm)


class BenchmarkVllmTest(unittest.TestCase):
    def test_lengths(self):
        self.assertEqual(benchmark_vllm.lengths("128, 2048,8192"), [128, 2048, 8192])
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark_vllm.lengths("128,128")

    def test_prompt_tokens_are_deterministic_and_bounded(self):
        first = benchmark_vllm.prompt_tokens(100, 7)
        self.assertEqual(first, benchmark_vllm.prompt_tokens(100, 7))
        self.assertEqual(len(first), 100)
        self.assertTrue(all(1000 <= token < 10000 for token in first))

    def test_summary_retains_all_samples(self):
        result = benchmark_vllm.summarize([1.0, 2.0, 3.0])
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["samples"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
