#!/usr/bin/env python3
"""Summarize raw llama-bench JSON without selecting favorable runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--hardware-class", required=True)
    return parser.parse_args()


def stats(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples)
    degrees_freedom = len(samples) - 1
    critical = T_CRITICAL_95.get(degrees_freedom, 1.960)
    half_width = critical * standard_deviation / math.sqrt(len(samples))
    return {
        "sample_count": len(samples),
        "mean": mean,
        "median": statistics.median(samples),
        "standard_deviation": standard_deviation,
        "minimum": min(samples),
        "maximum": max(samples),
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    entries: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"error: expected a JSON array in {path}")
        entries.extend(payload)

    summaries = []
    for entry in entries:
        throughput = [float(value) for value in entry["samples_ts"]]
        mode = "prefill" if entry["n_prompt"] else "decode"
        item: dict[str, Any] = {
            "mode": mode,
            "prompt_tokens": int(entry["n_prompt"]),
            "generated_tokens": int(entry["n_gen"]),
            "existing_context_tokens": int(entry["n_depth"]),
            "throughput_tokens_per_second": stats(throughput),
        }
        if mode == "decode":
            per_token_ms = [
                float(value) / int(entry["n_gen"]) / 1_000_000.0
                for value in entry["samples_ns"]
            ]
            item["aggregate_time_per_token_ms"] = stats(per_token_ms)
            item["latency_limitation"] = (
                "llama-bench reports aggregate generation time, not an inter-token latency distribution"
            )
        summaries.append(item)

    summaries.sort(
        key=lambda item: (
            0 if item["mode"] == "prefill" else 1,
            item["prompt_tokens"] or item["existing_context_tokens"],
        )
    )
    first = entries[0]
    result = {
        "status": "development_characterization_not_accepted_baseline",
        "label": args.label,
        "hardware_class": args.hardware_class,
        "source_files": [str(path) for path in args.inputs],
        "engine": {
            "build_commit": first["build_commit"],
            "build_number": first["build_number"],
            "backend": first["backends"],
            "model_sha256": "e4910e01c4275e58acbf2c38c4d4fb81acf61bb8aa04eed121eb5ac942705e8a",
            "kv_type_k": first["type_k"],
            "kv_type_v": first["type_v"],
            "flash_attention": bool(first["flash_attn"]),
            "gpu_layers": first["n_gpu_layers"],
            "cpu_moe_layers": first["n_cpu_moe"],
            "batch_size": first["n_batch"],
            "micro_batch_size": first["n_ubatch"],
            "threads": first["n_threads"],
        },
        "methodology": {
            "conditioning_repetitions": 3,
            "measured_repetitions": len(first["samples_ts"]),
            "primary_statistic": "median",
            "confidence_interval": "two-sided 95% Student t interval around the arithmetic mean",
            "prompt_cache_reuse": False,
        },
        "results": summaries,
        "open_gates": [
            "quality acceptance threshold",
            "profiler-level native NVFP4 invocation trace",
            "per-token latency distribution",
            "power, clock, and thermal time series",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(summaries)} benchmark configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
