from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class PipelineMonitor:
    def __init__(
        self,
        *,
        output_dir: str,
        job_name: str,
        labels: Optional[Dict[str, str]] = None,
        emit_json: bool = True,
        emit_prometheus_text: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.job_name = job_name
        self.labels = labels or {}
        self.emit_json = emit_json
        self.emit_prometheus_text = emit_prometheus_text
        self.stage_records: List[Dict[str, object]] = []
        self.run_started_unix = int(time.time())

    def record_stage(self, stage: str, rows: int, duration_seconds: float, *, note: str = "") -> None:
        self.stage_records.append(
            {
                "stage": stage,
                "rows": rows,
                "duration_seconds": round(duration_seconds, 6),
                "note": note,
                "recorded_at_unix": int(time.time()),
            }
        )

    def finalize(
        self,
        *,
        success: bool,
        total_duration_seconds: float,
        final_rows: int = 0,
        error_message: Optional[str] = None,
    ) -> Dict[str, object]:
        payload = {
            "job_name": self.job_name,
            "labels": self.labels,
            "run_started_unix": self.run_started_unix,
            "run_finished_unix": int(time.time()),
            "success": success,
            "total_duration_seconds": round(total_duration_seconds, 6),
            "final_rows": final_rows,
            "error_message": error_message,
            "stages": self.stage_records,
        }
        if self.emit_json:
            (self.output_dir / "pipeline_metrics.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if self.emit_prometheus_text:
            (self.output_dir / "pipeline_metrics.prom").write_text(
                self._render_prometheus(payload),
                encoding="utf-8",
            )
        return {
            "output_dir": str(self.output_dir),
            "metrics_json": str(self.output_dir / "pipeline_metrics.json") if self.emit_json else None,
            "metrics_prom": str(self.output_dir / "pipeline_metrics.prom") if self.emit_prometheus_text else None,
            "success": success,
        }

    def _render_prometheus(self, payload: Dict[str, object]) -> str:
        lines = [
            "# HELP agent_pipeline_run_success Whether the pipeline run succeeded.",
            "# TYPE agent_pipeline_run_success gauge",
            f"agent_pipeline_run_success{self._label_str()} {1 if payload['success'] else 0}",
            "# HELP agent_pipeline_total_duration_seconds Total pipeline runtime in seconds.",
            "# TYPE agent_pipeline_total_duration_seconds gauge",
            f"agent_pipeline_total_duration_seconds{self._label_str()} {payload['total_duration_seconds']}",
            "# HELP agent_pipeline_final_rows Final cleaned row count.",
            "# TYPE agent_pipeline_final_rows gauge",
            f"agent_pipeline_final_rows{self._label_str()} {payload['final_rows']}",
            "# HELP agent_pipeline_stage_rows Output rows by stage.",
            "# TYPE agent_pipeline_stage_rows gauge",
            "# HELP agent_pipeline_stage_duration_seconds Stage processing duration in seconds.",
            "# TYPE agent_pipeline_stage_duration_seconds gauge",
        ]
        for stage in self.stage_records:
            stage_labels = {**self.labels, "stage": str(stage["stage"])}
            label_str = self._label_str(stage_labels)
            lines.append(f"agent_pipeline_stage_rows{label_str} {stage['rows']}")
            lines.append(f"agent_pipeline_stage_duration_seconds{label_str} {stage['duration_seconds']}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _escape_label_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _label_str(self, extra_labels: Optional[Dict[str, str]] = None) -> str:
        labels = dict(self.labels)
        if extra_labels:
            labels.update(extra_labels)
        labels.setdefault("job", self.job_name)
        if not labels:
            return ""
        rendered = ",".join(f'{key}="{self._escape_label_value(str(val))}"' for key, val in sorted(labels.items()))
        return "{" + rendered + "}"
