import tempfile
import unittest
from pathlib import Path

from monitoring.pipeline_monitor import PipelineMonitor


class PipelineMonitorTest(unittest.TestCase):
    def test_monitor_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PipelineMonitor(output_dir=tmp_dir, job_name="agent_posttrain_pipeline")
            monitor.record_stage("normalized", 10, 0.5, note="ok")
            result = monitor.finalize(success=True, total_duration_seconds=1.2, final_rows=10)
            self.assertTrue((Path(tmp_dir) / "pipeline_metrics.json").exists())
            self.assertTrue((Path(tmp_dir) / "pipeline_metrics.prom").exists())
            self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
