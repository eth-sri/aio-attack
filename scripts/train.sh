#!/usr/bin/env bash
# Train an attacked model from a JSON config.
#
#   scripts/train.sh <config.json> [--flag value ...]
#
# Trailing --flag value pairs override the matching config entry, e.g.
#   scripts/train.sh configs/Qwen2.5-7B-Instruct/content_injection/mcd_new.json --output_dir outputs/run1/model
#
# Single-GPU by default; select the device with CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?usage: scripts/train.sh <config.json> [--flag value ...]}"
shift

PYTHONPATH=src exec "${PYTHON:-python}" src/main.py "$CONFIG" "$@"
