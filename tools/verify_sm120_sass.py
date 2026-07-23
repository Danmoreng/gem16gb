#!/usr/bin/env python3
"""Verify that a CUDA binary contains the native SM120 NVFP4 MMA instruction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


EXPECTED = "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"


def find_cuobjdump() -> str:
    executable = "cuobjdump.exe" if os.name == "nt" else "cuobjdump"
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = Path(cuda_path) / "bin" / executable
        if candidate.is_file():
            return str(candidate)
    discovered = shutil.which(executable)
    if discovered:
        return discovered
    raise FileNotFoundError("cuobjdump was not found in CUDA_PATH/bin or PATH")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"CUDA binary does not exist: {args.binary}")

    result = subprocess.run(
        [find_cuobjdump(), "--dump-sass", str(args.binary)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    count = result.stdout.count(EXPECTED)
    if count == 0:
        print(f"missing expected instruction: {EXPECTED}", file=sys.stderr)
        return 1
    print(f"verified {EXPECTED}: occurrences={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
