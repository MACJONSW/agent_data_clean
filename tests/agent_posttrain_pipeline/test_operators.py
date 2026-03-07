import unittest

from agent_posttrain_pipeline.config import ContaminationConfig, DataSourceConfig, OperatorInjectionConfig
from agent_posttrain_pipeline.operators import NGramContaminationOperator, OperatorContext, build_extra_operators


class FakeDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def map(self, function, num_proc=1, desc=""):
        return FakeDataset([function(dict(row)) for row in self.rows])

    def filter(self, function, num_proc=1, desc=""):
        return FakeDataset([dict(row) for row in self.rows if function(dict(row))])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rows[key]
        return [row[key] for row in self.rows]


class DummyDatasetOperator:
    def __init__(self, keep_substring: str = "abc"):
        self.keep_substring = keep_substring

    def run(self, dataset):
        return dataset.filter(
            lambda sample: self.keep_substring in sample["text"],
            desc="Filtering rows with dummy operator",
        )


class PipelineOperatorsTest(unittest.TestCase):
    def test_ngram_operator_filters_contaminated_rows(self):
        dataset = FakeDataset(
            [
                {"text": "agent tool use planning execution verification"},
                {"text": "fresh sample without overlap"},
            ]
        )
        reference_dataset = FakeDataset(
            [{"text": "agent tool use planning execution verification"}]
        )
        config = ContaminationConfig(
            enabled=True,
            reference_sources=[DataSourceConfig(path="reference.jsonl")],
            ngram_size=3,
            max_contamination_ratio=0.3,
            min_matched_ngrams=1,
        )

        result = NGramContaminationOperator(
            config,
            source_loader=lambda sources, num_proc=1: reference_dataset,
        ).apply(dataset, context=OperatorContext(num_proc=1))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "fresh sample without overlap")

    def test_imported_operator_can_be_applied_from_class_path(self):
        operators = build_extra_operators(
            [
                OperatorInjectionConfig(
                    hook="post_normalized",
                    class_path=f"{__name__}.DummyDatasetOperator",
                    init_kwargs={"keep_substring": "abc"},
                    name="dummy_gate",
                )
            ]
        )
        dataset = FakeDataset(
            [
                {"text": "abc123"},
                {"text": "zzz"},
            ]
        )

        result = operators[0].apply(dataset, context=OperatorContext(num_proc=1))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "abc123")


if __name__ == "__main__":
    unittest.main()
