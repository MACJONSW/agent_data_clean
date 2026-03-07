#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/dev.yaml}"

python "${ROOT_DIR}/run_agent_posttrain_pipeline.py" --config "${CONFIG_PATH}"
