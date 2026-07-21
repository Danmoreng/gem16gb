#!/usr/bin/env python3
"""Generate deterministic tokenizer and generation fixtures with vLLM.

This tool deliberately imports the reference runtime only after parsing its
arguments. Run it from the pinned, external reference environment documented
in docs/CORRECTNESS.md; it is not a runtime dependency of g4.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any


PROMPTS = (
    {
        "id": "exact_blue_no_thinking",
        "messages": [{"role": "user", "content": "Reply with exactly the word blue."}],
        "enable_thinking": False,
    },
    {
        "id": "sky_sentence_no_thinking",
        "messages": [
            {"role": "user", "content": "Explain why the sky is blue in one sentence."}
        ],
        "enable_thinking": False,
    },
    {
        "id": "integer_product_thinking",
        "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
        "enable_thinking": True,
    },
)


class GoldenError(RuntimeError):
    pass


@dataclass(frozen=True)
class LogprobEntry:
    token_id: int
    logprob: float
    rank: int | None
    decoded_token: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("models/gemma4-12b-nvfp4.lock.json")
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise GoldenError(f"required package is not installed: {name}") from error
    return versions


def prompt_token_ids(tokenizer: Any, prompt: dict[str, Any]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        prompt["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=prompt["enable_thinking"],
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise GoldenError(f"tokenizer returned invalid input IDs for {prompt['id']}")
    return encoded


def serialize_logprobs(logprobs: Any) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for position in logprobs or []:
        entries = [
            LogprobEntry(
                token_id=int(token_id),
                logprob=float(value.logprob),
                rank=None if value.rank is None else int(value.rank),
                decoded_token=value.decoded_token,
            )
            for token_id, value in position.items()
        ]
        entries.sort(key=lambda item: (item.rank is None, item.rank or sys.maxsize, item.token_id))
        result.append([asdict(entry) for entry in entries])
    return result


def main() -> int:
    args = parse_args()
    if args.max_model_len < 1 or args.max_tokens < 1:
        print("error: token limits must be positive", file=sys.stderr)
        return 2
    if shutil.which("ninja") is None:
        print(
            "error: ninja is not on PATH; activate the reference environment before running",
            file=sys.stderr,
        )
        return 2

    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        from transformers import AutoTokenizer
        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise GoldenError("CUDA is not available to the reference runtime")
        model = args.model.resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        tokenized = [(prompt, prompt_token_ids(tokenizer, prompt)) for prompt in PROMPTS]

        llm = LLM(
            model=str(model),
            tokenizer=str(model),
            max_model_len=args.max_model_len,
            max_logprobs=20,
            gpu_memory_utilization=0.90,
            cpu_offload_gb=0,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            seed=0,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            logprobs=20,
            seed=0,
        )

        fixtures: list[dict[str, Any]] = []
        for prompt, input_ids in tokenized:
            request = llm.generate(
                [{"prompt_token_ids": input_ids}], sampling, use_tqdm=False
            )[0]
            completion = request.outputs[0]
            fixtures.append(
                {
                    **prompt,
                    "rendered_prompt": tokenizer.decode(input_ids, skip_special_tokens=False),
                    "prompt_token_ids": input_ids,
                    "output_token_ids": list(completion.token_ids),
                    "output_text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "cumulative_logprob": completion.cumulative_logprob,
                    "top_logprobs": serialize_logprobs(completion.logprobs),
                }
            )

        device = torch.cuda.get_device_properties(0)
        document = {
            "schema_version": 1,
            "checkpoint": {
                "repository": lock.get("repository"),
                "revision": lock.get("revision"),
                "local_directory_name": model.name,
            },
            "reference_runtime": {
                "python": platform.python_version(),
                "packages": package_versions(
                    (
                        "vllm",
                        "torch",
                        "transformers",
                        "compressed-tensors",
                        "flashinfer-python",
                        "nvidia-cutlass-dsl",
                    )
                ),
                "torch_cuda": torch.version.cuda,
                "device_name": device.name,
                "device_compute_capability": list(torch.cuda.get_device_capability(0)),
                "device_total_memory_bytes": device.total_memory,
            },
            "execution": {
                "text_only": True,
                "batch_size": 1,
                "cpu_offload_gb": 0,
                "max_model_len": args.max_model_len,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "seed": 0,
                "top_logprobs": 20,
                "enforce_eager": True,
                "prefix_caching": False,
                "chunked_prefill": True,
            },
            "prompts": fixtures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output} with {len(fixtures)} reference generations")
        return 0
    except (GoldenError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
