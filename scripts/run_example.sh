#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${ROOT_DIR}/run_agent_posttrain_pipeline.py" --config "${ROOT_DIR}/examples/agent_posttrain_pipeline/config.yaml"
