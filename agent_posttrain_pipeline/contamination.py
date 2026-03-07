from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Sequence, Set, Tuple

from .config import ContaminationConfig
from .schema import STANDARD_QUALITY_KEY, STANDARD_TEXT_KEY

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


class NGramContaminationFilter:
    def __init__(self, config: ContaminationConfig):
        self.config = config
        self.reference_ngrams: Set[bytes] = set()

    def fit(self, texts: Sequence[str]) -> None:
        for text in texts:
            self.reference_ngrams.update(self._hash_ngrams(text))

    def transform(self, dataset, *, num_proc: int = 1):
        if not self.reference_ngrams:
            return dataset

        def _annotate(sample: Dict[str, object]) -> Dict[str, object]:
            ratio, matched, total = self.score_text(str(sample.get(STANDARD_TEXT_KEY, "")))
            quality = dict(sample.get(STANDARD_QUALITY_KEY) or {})
            quality["contamination_ratio"] = ratio
            quality["contamination_matched_ngrams"] = matched
            quality["contamination_total_ngrams"] = total
            sample[STANDARD_QUALITY_KEY] = quality
            return sample

        dataset = dataset.map(_annotate, num_proc=num_proc, desc="Scoring n-gram contamination")
        return dataset.filter(self._keep_sample, num_proc=num_proc, desc="Filtering contaminated samples")

    def score_text(self, text: str) -> Tuple[float, int, int]:
        hashed = self._hash_ngrams(text)
        total = len(hashed)
        if total == 0:
            return 0.0, 0, 0
        matched = sum(1 for item in hashed if item in self.reference_ngrams)
        return matched / total, matched, total

    def _keep_sample(self, sample: Dict[str, object]) -> bool:
        quality = sample.get(STANDARD_QUALITY_KEY) or {}
        ratio = float(quality.get("contamination_ratio", 0.0))
        matched = int(quality.get("contamination_matched_ngrams", 0))
        if matched < self.config.min_matched_ngrams:
            return True
        return ratio <= self.config.max_contamination_ratio

    def _hash_ngrams(self, text: str) -> Set[bytes]:
        tokens = self._tokenize(text)
        size = self.config.ngram_size
        if len(tokens) < size:
            return set()
        return {
            hashlib.blake2b(" ".join(tokens[i : i + size]).encode("utf-8"), digest_size=8).digest()
            for i in range(len(tokens) - size + 1)
        }

    def _tokenize(self, text: str) -> List[str]:
        normalized = text or ""
        if self.config.normalize_case:
            normalized = normalized.lower()
        if self.config.strip_punctuation:
            normalized = _PUNCT_RE.sub(" ", normalized)
        return _WORD_RE.findall(normalized)
