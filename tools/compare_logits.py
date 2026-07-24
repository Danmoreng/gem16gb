#!/usr/bin/env python3
"""Compare native full logits with the committed vLLM top-logprob fixture."""

from __future__ import annotations

import argparse
from array import array
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Any


class ComparisonError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/golden/vllm-gemma4-12b-nvfp4.json"),
    )
    parser.add_argument("--prompt-id", default="exact_blue_no_thinking")
    parser.add_argument("--vocabulary", type=int, default=262144)
    parser.add_argument("--llama-quality", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_prompt(document: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    prompts = document.get("prompts")
    if not isinstance(prompts, list):
        raise ComparisonError("golden fixture has no prompt list")
    for prompt in prompts:
        if isinstance(prompt, dict) and prompt.get("id") == prompt_id:
            return prompt
    raise ComparisonError(f"golden fixture has no prompt named {prompt_id!r}")


def read_logits(path: Path, vocabulary: int) -> list[array]:
    if vocabulary <= 0:
        raise ComparisonError("--vocabulary must be positive")
    payload = path.read_bytes()
    step_bytes = vocabulary * 4
    if not payload or len(payload) % step_bytes != 0:
        raise ComparisonError(
            f"logit dump size {len(payload)} is not a positive multiple of {step_bytes}"
        )
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return [
        values[offset : offset + vocabulary]
        for offset in range(0, len(values), vocabulary)
    ]


def summarize_step(
    logits: array, reference_entries: list[dict[str, Any]], sampled_token: int
) -> dict[str, Any]:
    if not logits:
        raise ComparisonError("empty logit step")
    maximum = max(logits)
    log_sum_exp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    engine_top_ids = heapq.nlargest(
        min(20, len(logits)), range(len(logits)), key=logits.__getitem__
    )
    sampled_value = logits[sampled_token]
    engine_rank = 1 + sum(
        value > sampled_value or (value == sampled_value and token_id < sampled_token)
        for token_id, value in enumerate(logits)
    )
    comparisons = []
    for entry in reference_entries:
        token_id = entry.get("token_id")
        reference_logprob = entry.get("logprob")
        if (
            not isinstance(token_id, int)
            or token_id < 0
            or token_id >= len(logits)
            or not isinstance(reference_logprob, (int, float))
        ):
            raise ComparisonError("golden top-logprob entry is malformed")
        engine_logprob = float(logits[token_id]) - log_sum_exp
        comparisons.append(
            {
                "token_id": token_id,
                "reference_logprob": float(reference_logprob),
                "engine_logprob": engine_logprob,
                "absolute_delta": abs(engine_logprob - float(reference_logprob)),
            }
        )
    return {
        "reference_top1_token_id": sampled_token,
        "engine_top1_token_id": engine_top_ids[0],
        "top1_agreement": engine_top_ids[0] == sampled_token,
        "reference_top1_engine_rank": engine_rank,
        "engine_top20_token_ids": engine_top_ids,
        "reference_top20_overlap": len(
            set(engine_top_ids) & {entry["token_id"] for entry in reference_entries}
        ),
        "reference_top_logprob_comparisons": comparisons,
        "maximum_reference_top20_logprob_delta": max(
            comparison["absolute_delta"] for comparison in comparisons
        ),
    }


def compare(logit_steps: list[array], prompt: dict[str, Any]) -> dict[str, Any]:
    output_ids = prompt.get("output_token_ids")
    reference_steps = prompt.get("top_logprobs")
    if (
        not isinstance(output_ids, list)
        or not isinstance(reference_steps, list)
        or len(output_ids) != len(reference_steps)
    ):
        raise ComparisonError("golden output/logprob steps are malformed")
    if len(logit_steps) > len(output_ids):
        raise ComparisonError("logit dump has more steps than the golden fixture")
    steps = []
    for index, logits in enumerate(logit_steps):
        entries = reference_steps[index]
        if not isinstance(entries, list) or not entries:
            raise ComparisonError(f"golden step {index} has no top-logprob entries")
        summary = summarize_step(logits, entries, output_ids[index])
        summary["step"] = index
        steps.append(summary)
    return {
        "schema_version": 1,
        "status": "diagnostic",
        "reference_scope": "committed_vllm_top20_logprobs",
        "prompt_id": prompt.get("id"),
        "steps_compared": len(steps),
        "all_top1_agree": all(step["top1_agreement"] for step in steps),
        "steps": steps,
    }


def compare_llama(logit_steps: list[array], prompt: dict[str, Any]) -> dict[str, Any]:
    output_ids = prompt.get("llama_cpp_output_token_ids")
    positions = prompt.get("llama_cpp_top_logprobs")
    if not isinstance(output_ids, list) or not isinstance(positions, list):
        raise ComparisonError("llama.cpp quality prompt is malformed")
    if len(logit_steps) > len(output_ids) or len(logit_steps) > len(positions):
        raise ComparisonError("logit dump has more steps than the llama.cpp fixture")
    steps = []
    for index, logits in enumerate(logit_steps):
        position = positions[index]
        entries = position.get("top_logprobs") if isinstance(position, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ComparisonError(f"llama.cpp step {index} has no top logprobs")
        normalized = [
            {"token_id": entry.get("id"), "logprob": entry.get("logprob")}
            for entry in entries
            if isinstance(entry, dict)
        ]
        summary = summarize_step(logits, normalized, output_ids[index])
        summary["step"] = index
        steps.append(summary)
    return {
        "reference_scope": "llama_cpp_top20_logprobs",
        "steps_compared": len(steps),
        "all_top1_agree": all(step["top1_agreement"] for step in steps),
        "steps": steps,
    }


def main() -> int:
    args = parse_args()
    try:
        golden = json.loads(args.golden.read_text(encoding="utf-8"))
        if not isinstance(golden, dict):
            raise ComparisonError("golden fixture must contain an object")
        prompt = select_prompt(golden, args.prompt_id)
        logit_steps = read_logits(args.logits, args.vocabulary)
        result = compare(logit_steps, prompt)
        if args.llama_quality is not None:
            llama_document = json.loads(
                args.llama_quality.read_text(encoding="utf-8")
            )
            if not isinstance(llama_document, dict):
                raise ComparisonError("llama.cpp quality fixture must contain an object")
            result["llama_cpp"] = compare_llama(
                logit_steps, select_prompt(llama_document, args.prompt_id)
            )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ComparisonError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
