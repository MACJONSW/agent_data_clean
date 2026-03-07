import unittest

from agent_posttrain_pipeline.config import ContaminationConfig
from agent_posttrain_pipeline.contamination import NGramContaminationFilter


class ContaminationFilterTest(unittest.TestCase):
    def test_ngram_filter_detects_overlap(self):
        config = ContaminationConfig(enabled=True, ngram_size=3, max_contamination_ratio=0.2, min_matched_ngrams=1)
        contamination_filter = NGramContaminationFilter(config)
        contamination_filter.fit(["agent tool use planning execution verification"])
        ratio, matched, total = contamination_filter.score_text("agent tool use planning execution verification")
        self.assertGreater(ratio, 0.9)
        self.assertGreater(matched, 0)
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
