import unittest

from agent_posttrain_pipeline.adapters import AgentSampleAdapter
from agent_posttrain_pipeline.config import DataSourceConfig, SourceAdapterConfig


class AgentSampleAdapterTest(unittest.TestCase):
    def test_prompt_response_adapter(self):
        adapter = AgentSampleAdapter(SourceAdapterConfig(id_key="id", prompt_key="prompt", response_key="response", keep_fields=["task"], source_name="alpaca_like"))
        sample = adapter.adapt_record({"id": "demo-1", "prompt": "Summarize the error.", "response": "The agent retried after a timeout.", "task": "debug"}, row_idx=0, source=DataSourceConfig(path="demo.jsonl"))
        self.assertEqual(sample["sample_id"], "demo-1")
        self.assertEqual(sample["messages"][0]["role"], "user")
        self.assertEqual(sample["messages"][1]["role"], "assistant")
        self.assertEqual(sample["metadata"]["task"], "debug")


if __name__ == "__main__":
    unittest.main()
