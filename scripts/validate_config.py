from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_posttrain_pipeline.config import build_pipeline_config


def _load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            import json as _json

            return _json.load(handle)
        import yaml

        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate agent post-training pipeline config")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()
    cfg = build_pipeline_config(_load_config(args.config))
    print(json.dumps({"valid": True, "work_dir": cfg.runtime.work_dir, "source_count": len(cfg.sources)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
