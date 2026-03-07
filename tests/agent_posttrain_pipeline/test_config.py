import unittest

from agent_posttrain_pipeline.config import build_pipeline_config


class AgentPipelineConfigTest(unittest.TestCase):
    def test_nested_config_is_coerced(self):
        cfg = build_pipeline_config(
            {
                "sources": [{"path": "demo.jsonl", "adapter": {"messages_key": "messages"}}],
                "runtime": {"work_dir": "/tmp/agent_posttrain_test", "num_proc": 2},
                "monitoring": {"enabled": True, "labels": {"env": "test"}},
                "operators": [
                    {
                        "hook": "post_quality",
                        "class_path": "data_juicer.ops.filter.alphanumeric_filter.AlphanumericFilter",
                        "init_kwargs": {"text_key": "text", "min_ratio": 0.1},
                    }
                ],
            }
        )
        self.assertEqual(cfg.runtime.work_dir, "/tmp/agent_posttrain_test")
        self.assertEqual(cfg.runtime.num_proc, 2)
        self.assertEqual(cfg.sources[0].adapter.messages_key, "messages")
        self.assertEqual(cfg.monitoring.labels["env"], "test")
        self.assertEqual(cfg.operators[0].hook, "post_quality")
        self.assertEqual(cfg.operators[0].init_kwargs["min_ratio"], 0.1)


if __name__ == "__main__":
    unittest.main()
