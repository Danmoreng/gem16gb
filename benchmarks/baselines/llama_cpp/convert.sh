#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 MODEL_DIR OUTPUT_GGUF [converter options...]" >&2
  exit 2
fi

model_dir="$1"
output_gguf="$2"
shift 2

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
expected_commit="$(tr -d '[:space:]' < "${script_dir}/commit.txt")"
source_dir="${LLAMA_CPP_SOURCE:-${repo_root}/third_party/cache/llama.cpp}"
python_bin="${LLAMA_CPP_CONVERT_PYTHON:-${repo_root}/third_party/cache/llama-convert-venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "error: converter Python not found at ${python_bin}" >&2
  echo "install the pinned requirements/requirements-convert_hf_to_gguf.txt first" >&2
  exit 2
fi
if [[ "$(git -C "${source_dir}" rev-parse HEAD)" != "${expected_commit}" ]]; then
  echo "error: llama.cpp source is not at pinned commit ${expected_commit}" >&2
  exit 2
fi
if [[ ! -f "${model_dir}/config.json" ]]; then
  echo "error: model directory has no config.json: ${model_dir}" >&2
  exit 2
fi

exec "${python_bin}" "${source_dir}/convert_hf_to_gguf.py" \
  --outtype auto \
  --outfile "${output_gguf}" \
  "$@" \
  "${model_dir}"
