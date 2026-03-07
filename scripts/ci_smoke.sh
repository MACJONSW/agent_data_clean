#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m unittest discover -s "${ROOT_DIR}/tests/agent_posttrain_pipeline" -p 'test_*.py'
python "${ROOT_DIR}/scripts/validate_config.py" --config "${ROOT_DIR}/configs/dev.yaml"
