#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
expected_commit="$(tr -d '[:space:]' < "${script_dir}/commit.txt")"
source_dir="${LLAMA_CPP_SOURCE:-${repo_root}/third_party/cache/llama.cpp}"
target_dir="${LLAMA_CPP_PATCHED_SOURCE:-${repo_root}/third_party/cache/llama.cpp-mixed}"
patch_file="${script_dir}/patches/0001-support-mixed-fp8-nvfp4-compressed-tensors.patch"

if [[ ! -d "${source_dir}/.git" ]]; then
  echo "error: pinned clean llama.cpp source not found at ${source_dir}" >&2
  exit 2
fi
if [[ "$(git -C "${source_dir}" rev-parse HEAD)" != "${expected_commit}" ]]; then
  echo "error: source is not at pinned commit ${expected_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${source_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "error: source worktree must be clean" >&2
  exit 2
fi
if [[ -e "${target_dir}" ]]; then
  echo "error: patched target already exists: ${target_dir}" >&2
  exit 2
fi

git clone --shared "${source_dir}" "${target_dir}"
git -C "${target_dir}" checkout --detach "${expected_commit}"
git -C "${target_dir}" apply --unidiff-zero --check "${patch_file}"
git -C "${target_dir}" apply --unidiff-zero "${patch_file}"

echo "prepared patched converter source at ${target_dir}"
