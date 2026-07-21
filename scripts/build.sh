#!/usr/bin/env bash
set -euo pipefail

# Structure adapted from the neighboring qwen35x build helper (MIT, inspected
# 2026-07-21); no qwen35x source code is linked or vendored.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
preset="host-debug"
run_tests=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) preset="blackwell-release" ;;
    --host) preset="host-debug" ;;
    --sanitize) preset="host-sanitize" ;;
    --test) run_tests=true ;;
    --help|-h)
      echo "Usage: scripts/build.sh [--host|--cuda|--sanitize] [--test]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done

cmake --preset "$preset" -S "$repo_root"
cmake --build --preset "$preset" --parallel
if [[ "$run_tests" == true ]]; then
  ctest --preset "$preset"
fi

