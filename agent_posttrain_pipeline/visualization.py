from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Sequence


class PipelineVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, summary: Dict[str, Any], *, token_values: Sequence[float], judge_values: Sequence[float]) -> Dict[str, str]:
        svg_path = self.output_dir / "pipeline.svg"
        html_path = self.output_dir / "report.html"
        mermaid_path = self.output_dir / "pipeline.mmd"
        json_path = self.output_dir / "summary.json"
        stages = summary.get("stages", [])
        svg_path.write_text(self._render_pipeline_svg(stages), encoding="utf-8")
        mermaid_path.write_text(self._render_mermaid(stages), encoding="utf-8")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        html_path.write_text(self._render_html(stages, token_values, judge_values, svg_path.read_text(encoding="utf-8")), encoding="utf-8")
        return {"svg": str(svg_path), "html": str(html_path), "mermaid": str(mermaid_path), "summary_json": str(json_path)}

    def _render_pipeline_svg(self, stages):
        width = 320 * max(len(stages), 1)
        box_w = 220
        box_h = 90
        y = 60
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220" viewBox="0 0 {width} 220">',
            '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#19323c"/></marker></defs>',
            '<rect width="100%" height="100%" fill="#f3efe5" rx="18"/>',
        ]
        for idx, stage in enumerate(stages):
            x = 40 + idx * 300
            parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="18" fill="#fffdf8" stroke="#19323c" stroke-width="2"/>')
            parts.append(f'<text x="{x + 16}" y="{y + 28}" font-size="18" font-family="monospace" fill="#19323c">{html.escape(str(stage["stage"]))}</text>')
            parts.append(f'<text x="{x + 16}" y="{y + 56}" font-size="24" font-family="monospace" fill="#0f6b6f">rows: {stage["rows"]}</text>')
            note = html.escape(str(stage.get("note", "")))
            if note:
                parts.append(f'<text x="{x + 16}" y="{y + 78}" font-size="12" font-family="monospace" fill="#5a5a5a">{note}</text>')
            if idx < len(stages) - 1:
                parts.append(f'<line x1="{x + box_w}" y1="{y + box_h / 2}" x2="{x + 280}" y2="{y + box_h / 2}" stroke="#19323c" stroke-width="3" marker-end="url(#arrow)"/>')
        parts.append("</svg>")
        return "".join(parts)

    def _render_mermaid(self, stages):
        lines = ["flowchart LR"]
        for idx, stage in enumerate(stages):
            lines.append(f'    stage_{idx}["{stage["stage"]}\\nrows={stage["rows"]}"]')
            if idx > 0:
                lines.append(f"    stage_{idx - 1} --> stage_{idx}")
        return "\n".join(lines)

    def _render_html(self, stages, token_values, judge_values, svg):
        return "".join([
            "<html><head><meta charset='utf-8'><title>Agent Pipeline Report</title>",
            "<style>body{font-family:ui-monospace,Monaco,monospace;background:#f3efe5;color:#19323c;padding:24px;}section{background:#fffdf8;border:1px solid #d8d2c4;border-radius:18px;padding:20px;margin-bottom:18px;}table{border-collapse:collapse;width:100%;}th,td{padding:8px 10px;border-bottom:1px solid #ece6d8;text-align:left;}</style></head><body>",
            "<h1>Agent Post-Training Data Pipeline</h1>",
            "<section><h2>Pipeline Graph</h2>",
            svg,
            "</section>",
            "<section><h2>Stage Summary</h2>",
            self._render_table(stages),
            "</section>",
            f"<section><h2>Token Samples</h2><pre>{html.escape(json.dumps(list(token_values[:50]), ensure_ascii=False, indent=2))}</pre></section>",
            f"<section><h2>Judge Samples</h2><pre>{html.escape(json.dumps(list(judge_values[:50]), ensure_ascii=False, indent=2))}</pre></section>",
            "</body></html>",
        ])

    @staticmethod
    def _render_table(stages):
        rows = ["<table><thead><tr><th>stage</th><th>rows</th><th>note</th></tr></thead><tbody>"]
        for stage in stages:
            rows.append(f'<tr><td>{html.escape(str(stage["stage"]))}</td><td>{stage["rows"]}</td><td>{html.escape(str(stage.get("note", "")))}</td></tr>')
        rows.append("</tbody></table>")
        return "".join(rows)
