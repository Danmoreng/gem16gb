from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/golden/vllm-gemma4-12b-nvfp4.json"
REVISION = "b1f649734b34aa5575b03d186abd1b9be3d0d5c4"


class GoldenFixtureTest(unittest.TestCase):
    def test_reference_fixture_is_internally_consistent(self) -> None:
        document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["checkpoint"]["revision"], REVISION)
        self.assertEqual(document["execution"]["batch_size"], 1)
        self.assertEqual(document["execution"]["cpu_offload_gb"], 0)
        self.assertTrue(document["execution"]["text_only"])
        self.assertEqual(
            [prompt["id"] for prompt in document["prompts"]],
            [
                "exact_blue_no_thinking",
                "sky_sentence_no_thinking",
                "integer_product_thinking",
            ],
        )

        for prompt in document["prompts"]:
            output_ids = prompt["output_token_ids"]
            positions = prompt["top_logprobs"]
            self.assertEqual(len(output_ids), len(positions), prompt["id"])
            self.assertGreater(len(output_ids), 0, prompt["id"])
            for sampled_token, entries in zip(output_ids, positions):
                by_id = {entry["token_id"]: entry for entry in entries}
                self.assertIn(sampled_token, by_id)
                self.assertEqual(by_id[sampled_token]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
