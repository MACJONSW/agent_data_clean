from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = ["pipeline_metrics.json", "pipeline_metrics.prom"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate monitoring outputs")
    parser.add_argument("--dir", required=True, help="Monitoring output directory")
    args = parser.parse_args()
    output_dir = Path(args.dir)
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing monitoring artifacts: {', '.join(missing)}")
    print(json.dumps({"valid": True, "dir": str(output_dir), "files": REQUIRED_FILES}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
