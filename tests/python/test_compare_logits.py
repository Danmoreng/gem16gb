from __future__ import annotations

from array import array
import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "compare_logits.py"
SPEC = importlib.util.spec_from_file_location("compare_logits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_logits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_logits)


class CompareLogitsTest(unittest.TestCase):
    def test_step_reports_rank_and_logprob_delta(self) -> None:
        logits = array("f", [0.0, 3.0, 2.0, -1.0])
        maximum = 3.0
        normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in logits)
        )
        reference = [
            {"token_id": 1, "logprob": 3.0 - normalizer},
            {"token_id": 2, "logprob": 2.0 - normalizer},
        ]
        result = compare_logits.summarize_step(logits, reference, 1)
        self.assertTrue(result["top1_agreement"])
        self.assertEqual(result["reference_top1_engine_rank"], 1)
        self.assertAlmostEqual(result["maximum_reference_top20_logprob_delta"], 0.0)

    def test_step_detects_different_top1(self) -> None:
        logits = array("f", [0.0, 2.0, 3.0])
        reference = [{"token_id": 1, "logprob": -1.0}]
        result = compare_logits.summarize_step(logits, reference, 1)
        self.assertFalse(result["top1_agreement"])
        self.assertEqual(result["reference_top1_engine_rank"], 2)

    def test_llama_fixture_shape_is_normalized(self) -> None:
        prompt = {
            "llama_cpp_output_token_ids": [1],
            "llama_cpp_top_logprobs": [
                {"top_logprobs": [{"id": 1, "logprob": -0.1}]}
            ],
        }
        result = compare_logits.compare_llama(
            [array("f", [0.0, 2.0, 1.0])], prompt
        )
        self.assertTrue(result["all_top1_agree"])


if __name__ == "__main__":
    unittest.main()
