from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            return json.load(handle)
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guidance
            raise ModuleNotFoundError("Loading YAML config requires PyYAML. Install it or use JSON config.") from exc
        return yaml.safe_load(handle)


def main() -> None:
    from .pipeline import AgentDataPipeline

    parser = argparse.ArgumentParser(description="Agent post-training data pipeline using local Data-Juicer")
    parser.add_argument("--config", required=True, help="YAML/JSON config path")
    args = parser.parse_args()
    summary = AgentDataPipeline(_load_config(args.config)).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
