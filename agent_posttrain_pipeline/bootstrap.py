from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_JUICER_DIR = ROOT_DIR / "data-juicer"


def ensure_data_juicer_on_path() -> None:
    """Import local Data-Juicer checkout without modifying its source tree."""
    path_str = str(DATA_JUICER_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
