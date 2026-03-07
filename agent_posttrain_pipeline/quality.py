from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence

from .dj_bridge import Fields, StatsKeys, TokenNumFilter
from .config import QualityConfig
from .schema import STANDARD_MESSAGES_KEY, STANDARD_QUALITY_KEY, STANDARD_TEXT_KEY

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    if ratio <= 0:
        return float(values[0])
    if ratio >= 1:
        return float(values[-1])
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * ratio
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class QualityAnalyzer:
    def __init__(self, config: QualityConfig):
        self.config = config

    def enrich(self, dataset, *, num_proc: int = 1):
        if self.config.hf_tokenizer:
            dataset = self._enrich_with_data_juicer_tokenizer(dataset, num_proc=num_proc)
        else:
            dataset = self._enrich_with_fallback_tokenizer(dataset, num_proc=num_proc)
        return dataset.map(self._attach_turn_metrics, num_proc=num_proc, desc="Attaching turn-level quality metrics")

    def _enrich_with_data_juicer_tokenizer(self, dataset, *, num_proc: int = 1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(Fields.stats, [{}] * len(dataset))
        op = TokenNumFilter(
            hf_tokenizer=self.config.hf_tokenizer,
            min_num=0,
            max_num=10**12,
            text_key=STANDARD_TEXT_KEY,
        )

        def _annotate(sample: Dict[str, Any]) -> Dict[str, Any]:
            token_count = int((sample.get(Fields.stats) or {}).get(StatsKeys.num_token, 0))
            quality = dict(sample.get(STANDARD_QUALITY_KEY) or {})
            quality["token_count"] = token_count
            sample[STANDARD_QUALITY_KEY] = quality
            return sample

        dataset = dataset.map(op.compute_stats, num_proc=num_proc, desc="Computing token stats with Data-Juicer TokenNumFilter")
        return dataset.map(_annotate, num_proc=num_proc, desc="Syncing token stats into quality signals")

    def _enrich_with_fallback_tokenizer(self, dataset, *, num_proc: int = 1):
        def _annotate(sample: Dict[str, Any]) -> Dict[str, Any]:
            quality = dict(sample.get(STANDARD_QUALITY_KEY) or {})
            quality["token_count"] = estimate_token_count(sample.get(STANDARD_TEXT_KEY, ""))
            sample[STANDARD_QUALITY_KEY] = quality
            stats = dict(sample.get(Fields.stats) or {})
            stats[StatsKeys.num_token] = quality["token_count"]
            sample[Fields.stats] = stats
            return sample

        return dataset.map(_annotate, num_proc=num_proc, desc="Computing token stats with fallback tokenizer")

    @staticmethod
    def _attach_turn_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
        quality = dict(sample.get(STANDARD_QUALITY_KEY) or {})
        quality["message_count"] = len(sample.get(STANDARD_MESSAGES_KEY) or [])
        sample[STANDARD_QUALITY_KEY] = quality
        return sample

    def summarize(self, dataset) -> Dict[str, Any]:
        values = [row[STANDARD_QUALITY_KEY]["token_count"] for row in dataset if row.get(STANDARD_QUALITY_KEY)]
        return {
            "sample_count": len(values),
            "token_count": {
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": (sum(values) / len(values)) if values else 0,
                "percentiles": {str(p): percentile(values, p) for p in self.config.percentiles},
            },
        }

    def filter(self, dataset, *, num_proc: int = 1):
        cfg = self.config

        def _keep(sample: Dict[str, Any]) -> bool:
            quality = sample.get(STANDARD_QUALITY_KEY) or {}
            token_count = int(quality.get("token_count", 0))
            message_count = int(quality.get("message_count", 0))
            assistant_turns = int(quality.get("assistant_turn_count", 0))
            if cfg.min_tokens is not None and token_count < cfg.min_tokens:
                return False
            if cfg.max_tokens is not None and token_count > cfg.max_tokens:
                return False
            if message_count < cfg.min_turns:
                return False
            if cfg.max_turns is not None and message_count > cfg.max_turns:
                return False
            if assistant_turns < cfg.min_assistant_turns:
                return False
            return True

        return dataset.filter(_keep, num_proc=num_proc, desc="Applying quality filters")

    @staticmethod
    def sample_metric_values(dataset, key: str, limit: int) -> List[float]:
        rows = dataset.select(range(min(len(dataset), limit))) if len(dataset) else dataset
        values: List[float] = []
        for row in rows:
            quality = row.get(STANDARD_QUALITY_KEY) or {}
            if key in quality:
                values.append(float(quality[key]))
        return values
