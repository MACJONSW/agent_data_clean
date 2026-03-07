import unittest

from agent_posttrain_pipeline.inference import parse_judge_response, render_prompt


class InferenceUtilsTest(unittest.TestCase):
    def test_render_prompt_uses_context(self):
        prompt = render_prompt("Task: {text} / {metadata_json}", {"text": "hello", "metadata": {"source": "demo"}})
        self.assertIn("hello", prompt)
        self.assertIn("demo", prompt)

    def test_parse_judge_response_json(self):
        payload = parse_judge_response('{"score": 8.5, "reason": "good traceability"}', score_scale=10.0)
        self.assertEqual(payload["score"], 8.5)
        self.assertAlmostEqual(payload["normalized_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
