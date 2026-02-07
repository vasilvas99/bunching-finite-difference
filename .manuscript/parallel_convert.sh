#!/usr/bin/env bash

set -euo pipefail

# -------------------------
# Usage:
#   ./run_parallel_convert.sh INPUT_DIR [OUTPUT_DIR] PYTHON_SCRIPT
#
# Examples:
#   ./run_parallel_convert.sh input_dir orb2ch.py
#   ./run_parallel_convert.sh input_dir output_dir orb2ch.py
# -------------------------

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_DIR [OUTPUT_DIR] PYTHON_SCRIPT"
  exit 1
fi

# Resolve INPUT_DIR
INPUT_DIR="$(realpath "$1")"

# Handle optional OUTPUT_DIR
if [[ $# -eq 2 ]]; then
  OUTPUT_DIR="${INPUT_DIR}_npz"
  PYTHON_SCRIPT="$2"
else
  OUTPUT_DIR="$(realpath "$2")"
  PYTHON_SCRIPT="$3"
fi

# Resolve script path
PYTHON_SCRIPT="$(realpath "$PYTHON_SCRIPT")"

# Validation
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: INPUT_DIR does not exist: $INPUT_DIR"
  exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "ERROR: PYTHON_SCRIPT does not exist: $PYTHON_SCRIPT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export INPUT_DIR OUTPUT_DIR PYTHON_SCRIPT

find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | \
parallel -j $(nproc) '
  n=$(basename {})
  uv run python3 "$PYTHON_SCRIPT" \
    "$INPUT_DIR/$n/" \
    "$OUTPUT_DIR/$n.npz" \
  && \
  uv run python3 -m viz2d "$OUTPUT_DIR/$n.npz"
'
