#!/usr/bin/env python3
"""Capture llama.cpp token/logprob output and compare it with a golden fixture."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--gguf-sha256", required=True)
    parser.add_argument("--llama-cpp-commit", required=True)
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    captures = []
    exact_generation_count = 0
    agreed_tokens = 0
    compared_tokens = 0

    for prompt in reference["prompts"]:
        expected_ids = prompt["output_token_ids"]
        request_payload = {
            "messages": prompt["messages"],
            "temperature": 0.0,
            "max_tokens": len(expected_ids),
            "seed": reference["execution"]["seed"],
            "logprobs": True,
            "top_logprobs": reference["execution"]["top_logprobs"],
            "chat_template_kwargs": {
                "enable_thinking": prompt["enable_thinking"],
            },
            "stream": False,
        }
        response = post_json(
            args.endpoint.rstrip("/") + "/v1/chat/completions", request_payload
        )
        choice = response["choices"][0]
        content_logprobs = choice["logprobs"]["content"]
        actual_ids = [int(step["id"]) for step in content_logprobs]
        common = min(len(expected_ids), len(actual_ids))
        token_matches = [
            actual_ids[index] == expected_ids[index] for index in range(common)
        ]
        exact_generation = actual_ids == expected_ids
        exact_generation_count += int(exact_generation)
        agreed_tokens += sum(token_matches)
        compared_tokens += max(len(expected_ids), len(actual_ids))

        step_comparisons = []
        for index in range(common):
            ref_top = prompt["top_logprobs"][index]
            llama_top = content_logprobs[index]["top_logprobs"]
            ref_ids = {int(item["token_id"]) for item in ref_top}
            llama_ids = {int(item["id"]) for item in llama_top}
            ref_selected = next(
                item for item in ref_top if int(item["token_id"]) == expected_ids[index]
            )
            step_comparisons.append(
                {
                    "index": index,
                    "expected_token_id": expected_ids[index],
                    "actual_token_id": actual_ids[index],
                    "token_match": token_matches[index],
                    "top20_overlap_count": len(ref_ids & llama_ids),
                    "selected_logprob_reference": ref_selected["logprob"],
                    "selected_logprob_llama_cpp": content_logprobs[index]["logprob"],
                    "selected_logprob_delta": (
                        content_logprobs[index]["logprob"] - ref_selected["logprob"]
                    ),
                }
            )

        message = choice["message"]
        captures.append(
            {
                "id": prompt["id"],
                "enable_thinking": prompt["enable_thinking"],
                "request": request_payload,
                "reference_output_text": prompt["output_text"],
                "reference_output_token_ids": expected_ids,
                "llama_cpp_output_text": message.get("content"),
                "llama_cpp_reasoning_text": message.get("reasoning_content"),
                "llama_cpp_output_token_ids": actual_ids,
                "llama_cpp_output_tokens": [step["token"] for step in content_logprobs],
                "finish_reason": choice["finish_reason"],
                "exact_generation": exact_generation,
                "matching_token_count": sum(token_matches),
                "compared_token_count": max(len(expected_ids), len(actual_ids)),
                "step_comparisons": step_comparisons,
                "llama_cpp_top_logprobs": content_logprobs,
            }
        )

    result = {
        "status": "captured_not_acceptance_thresholded",
        "model_label": args.model_label,
        "gguf_sha256": args.gguf_sha256,
        "llama_cpp_commit": args.llama_cpp_commit,
        "reference": {
            "path": str(args.reference),
            "checkpoint_revision": reference["checkpoint"]["revision"],
            "runtime": "vLLM direct compressed checkpoint",
        },
        "execution": {
            "batch_size": 1,
            "temperature": 0.0,
            "seed": reference["execution"]["seed"],
            "top_logprobs": reference["execution"]["top_logprobs"],
            "prompt_cache_reuse": False,
        },
        "summary": {
            "prompt_count": len(captures),
            "exact_generation_count": exact_generation_count,
            "matching_token_count": agreed_tokens,
            "compared_token_count": compared_tokens,
            "token_agreement": agreed_tokens / compared_tokens,
        },
        "prompts": captures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {exact_generation_count}/{len(captures)} exact prompts, "
        f"{agreed_tokens}/{compared_tokens} matching tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
