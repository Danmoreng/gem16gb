#!/usr/bin/env python3
"""Export a deterministic llama.cpp GGUF tensor inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--llama-cpp-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--converter-commit", required=True)
    parser.add_argument("--converter-patch", type=Path)
    parser.add_argument("--expect-tensors", type=int)
    parser.add_argument("--expect-nvfp4", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    gguf_python = args.llama_cpp_source / "gguf-py"
    if not gguf_python.is_dir():
        raise SystemExit(f"error: llama.cpp gguf-py not found at {gguf_python}")
    if not args.gguf.is_file():
        raise SystemExit(f"error: GGUF not found: {args.gguf}")
    sys.path.insert(0, str(gguf_python))

    from gguf import GGUFReader  # pylint: disable=import-outside-toplevel

    reader = GGUFReader(args.gguf, "r")
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "byte_count": 0, "element_count": 0}
    )
    tensors = []
    for index, tensor in enumerate(reader.tensors):
        tensor_type = tensor.tensor_type.name
        counts[tensor_type]["tensor_count"] += 1
        counts[tensor_type]["byte_count"] += int(tensor.n_bytes)
        counts[tensor_type]["element_count"] += int(tensor.n_elements)
        tensors.append(
            {
                "index": index,
                "name": tensor.name,
                "type": tensor_type,
                "shape": [int(dim) for dim in tensor.shape.tolist()],
                "element_count": int(tensor.n_elements),
                "byte_count": int(tensor.n_bytes),
                "data_offset": int(tensor.data_offset),
                "metadata_offset": int(tensor.field.offset),
            }
        )

    selected_metadata = {}
    for key in (
        "general.architecture",
        "general.file_type",
        "general.quantization_version",
        "gemma4.block_count",
        "gemma4.context_length",
        "gemma4.embedding_length",
        "gemma4.feed_forward_length",
        "gemma4.attention.head_count",
        "gemma4.attention.head_count_kv",
        "gemma4.attention.sliding_window",
        "gemma4.final_logit_softcapping",
    ):
        field = reader.fields.get(key)
        if field is not None:
            selected_metadata[key] = json_value(field.contents())

    result = {
        "status": "generated_unvalidated_quality",
        "label": args.label,
        "source_revision": args.source_revision,
        "converter_commit": args.converter_commit,
        "converter_patch": (
            {
                "path": str(args.converter_patch),
                "sha256": sha256_file(args.converter_patch),
            }
            if args.converter_patch
            else None
        ),
        "gguf": {
            "path": str(args.gguf),
            "size_bytes": args.gguf.stat().st_size,
            "sha256": sha256_file(args.gguf),
            "version": int(reader.fields["GGUF.version"].contents()),
            "alignment": int(reader.alignment),
            "tensor_count": len(tensors),
            "metadata_count": len(reader.fields) - 3,
        },
        "selected_metadata": selected_metadata,
        "totals_by_type": dict(sorted(counts.items())),
        "tensors": tensors,
    }

    if args.expect_tensors is not None and len(tensors) != args.expect_tensors:
        raise SystemExit(
            f"error: expected {args.expect_tensors} tensors, found {len(tensors)}"
        )
    nvfp4_count = counts["NVFP4"]["tensor_count"]
    if args.expect_nvfp4 is not None and nvfp4_count != args.expect_nvfp4:
        raise SystemExit(
            f"error: expected {args.expect_nvfp4} NVFP4 tensors, found {nvfp4_count}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {len(tensors)} tensors, "
        f"{nvfp4_count} NVFP4, {args.gguf.stat().st_size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
