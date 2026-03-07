from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

from .config import AgentPipelineConfig, build_pipeline_config
from .dj_bridge import DocumentMinhashDeduplicator, Exporter
from .inference import BatchLLMProcessor, apply_ray_vllm_stage, build_backend
from .loaders import load_and_normalize_sources
from .operators import (
    DataJuicerOperatorAdapter,
    NGramContaminationOperator,
    OperatorContext,
    build_extra_operators,
)
from .quality import QualityAnalyzer
from .schema import STANDARD_JUDGE_KEY
from .visualization import PipelineVisualizer
from monitoring.pipeline_monitor import PipelineMonitor


@dataclass
class StageRecord:
    stage: str
    rows: int
    note: str = ""
    duration_seconds: float = 0.0


class AgentDataPipeline:
    def __init__(self, config: AgentPipelineConfig | Dict[str, Any]):
        self.config = build_pipeline_config(config)
        self.work_dir = Path(self.config.runtime.work_dir)
        self.stage_records: List[StageRecord] = []
        self.quality_summary: Dict[str, Any] = {}
        self.visualization_files: Dict[str, str] = {}
        self.shard_manifests: Dict[str, str] = {}
        self.operator_context = OperatorContext(num_proc=self.config.runtime.num_proc)
        self.extra_operators = build_extra_operators(self.config.operators)
        self.monitor = PipelineMonitor(
            output_dir=self.config.monitoring.output_dir or str(self.work_dir / "monitoring"),
            job_name=self.config.monitoring.job_name,
            labels=self.config.monitoring.labels,
            emit_json=self.config.monitoring.emit_json,
            emit_prometheus_text=self.config.monitoring.emit_prometheus_text,
        )

    def run(self) -> Dict[str, Any]:
        run_started_at = time.perf_counter()
        try:
            dataset = self._run_dataset_stage(
                "normalized",
                lambda: load_and_normalize_sources(self.config.sources, num_proc=self.config.runtime.num_proc),
                note="sources standardized to agent schema",
            )
            if self.config.sharding.enabled:
                self.shard_manifests["normalized"] = self._materialize_shards(dataset, phase="normalized")
            dataset = self._apply_extra_operators(dataset, hook="post_normalized")
            if self.config.dedup.enabled:
                dataset = self._run_dataset_stage(
                    "deduplicated",
                    lambda: self._apply_near_dedup(dataset),
                    note="Data-Juicer document MinHash near-dedup",
                )
            dataset = self._apply_extra_operators(dataset, hook="post_dedup")
            if self.config.contamination.enabled:
                dataset = self._run_dataset_stage(
                    "decontaminated",
                    lambda: self._apply_contamination_filter(dataset),
                    note="n-gram contamination filter",
                )
            dataset = self._apply_extra_operators(dataset, hook="post_contamination")
            if self.config.quality.enabled:
                dataset = self._run_dataset_stage(
                    "quality_filtered",
                    lambda: self._apply_quality_filters(dataset),
                    note="token and turn-based filters",
                )
            dataset = self._apply_extra_operators(dataset, hook="post_quality")
            if self.config.augmentation.enabled:
                dataset = self._run_dataset_stage(
                    "augmented",
                    lambda: self._apply_augmentation(dataset),
                    note="model-generated augmentation",
                )
            dataset = self._apply_extra_operators(dataset, hook="post_augmentation")
            if self.config.judge.enabled:
                dataset = self._run_dataset_stage(
                    "judge_filtered",
                    lambda: self._apply_judge(dataset),
                    note="judge score write-back and filter",
                )
            dataset = self._apply_extra_operators(dataset, hook="post_judge")
            dataset = self._apply_extra_operators(dataset, hook="pre_export")
            if self.config.sharding.enabled:
                self.shard_manifests["cleaned"] = self._materialize_shards(dataset, phase="cleaned")
            export_started_at = time.perf_counter()
            self._export_dataset(dataset)
            self.monitor.record_stage("exported", len(dataset), time.perf_counter() - export_started_at, note="exported cleaned dataset")
            summary = self._build_summary(dataset)
            summary["monitoring"] = self.monitor.finalize(
                success=True,
                total_duration_seconds=time.perf_counter() - run_started_at,
                final_rows=len(dataset),
            )
            self._write_summary(summary)
            if self.config.visualization.enabled:
                visualizer = PipelineVisualizer(
                    self.config.visualization.output_dir or str(self.work_dir / "visualizations")
                )
                token_values = QualityAnalyzer.sample_metric_values(
                    dataset,
                    "token_count",
                    self.config.quality.sample_for_report,
                )
                judge_values = []
                if self.config.judge.enabled:
                    rows = (
                        dataset.select(range(min(len(dataset), self.config.visualization.include_samples)))
                        if len(dataset)
                        else dataset
                    )
                    for row in rows:
                        judge = row.get(STANDARD_JUDGE_KEY) or {}
                        if self.config.judge.score_key in judge:
                            judge_values.append(float(judge[self.config.judge.score_key]))
                self.visualization_files = visualizer.render(summary, token_values=token_values, judge_values=judge_values)
                summary["visualizations"] = self.visualization_files
                self._write_summary(summary)
            return summary
        except Exception as exc:
            self.monitor.finalize(
                success=False,
                total_duration_seconds=time.perf_counter() - run_started_at,
                error_message=str(exc),
            )
            raise

    def _apply_near_dedup(self, dataset):
        cfg = self.config.dedup
        return DataJuicerOperatorAdapter(
            name="document_minhash_deduplicator",
            hook="post_normalized",
            note="Data-Juicer document MinHash near-dedup",
            operator=DocumentMinhashDeduplicator(
                text_key="text",
                tokenization=cfg.tokenization,
                window_size=cfg.window_size,
                lowercase=cfg.lowercase,
                ignore_pattern=cfg.ignore_pattern,
                tokenizer_model=cfg.tokenizer_model,
                num_permutations=cfg.num_permutations,
                jaccard_threshold=cfg.jaccard_threshold,
            ),
        ).apply(dataset, context=self.operator_context)

    def _apply_contamination_filter(self, dataset):
        return NGramContaminationOperator(self.config.contamination).apply(dataset, context=self.operator_context)

    def _apply_quality_filters(self, dataset):
        analyzer = QualityAnalyzer(self.config.quality)
        dataset = analyzer.enrich(dataset, num_proc=self.config.runtime.num_proc)
        self.quality_summary = analyzer.summarize(dataset)
        return analyzer.filter(dataset, num_proc=self.config.runtime.num_proc)

    def _apply_augmentation(self, dataset):
        cfg = self.config.augmentation
        if cfg.backend == "ray_vllm":
            return apply_ray_vllm_stage(dataset, cfg)
        return BatchLLMProcessor(build_backend(cfg)).apply_generation_stage(dataset, cfg)

    def _apply_judge(self, dataset):
        cfg = self.config.judge
        if cfg.backend == "ray_vllm":
            return apply_ray_vllm_stage(dataset, cfg)
        return BatchLLMProcessor(build_backend(cfg)).apply_judge_stage(dataset, cfg)

    def _apply_extra_operators(self, dataset, *, hook: str):
        for operator in self.extra_operators:
            if operator.hook != hook:
                continue
            current_dataset = dataset
            dataset = self._run_dataset_stage(
                f"operator_{operator.name}",
                lambda current=current_dataset, op=operator: op.apply(current, context=self.operator_context),
                note=operator.note or f"injected operator at {hook}",
            )
        return dataset

    def _materialize_shards(self, dataset, *, phase: str) -> str:
        shard_dir = Path(self.config.sharding.output_dir or self.work_dir / "shards") / phase
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_rows = max(1, self.config.sharding.rows_per_shard)
        shard_count = max(1, ceil(len(dataset) / shard_rows))
        manifest = []
        for index in range(shard_count):
            shard = dataset.shard(num_shards=shard_count, index=index, contiguous=True)
            shard_path = shard_dir / f"shard-{index:05d}.{self.config.sharding.export_type}"
            Exporter(export_path=str(shard_path), export_type=self.config.sharding.export_type, export_stats=False, keep_stats_in_res_ds=False, keep_hashes_in_res_ds=False).export(shard)
            manifest.append({"shard_id": index, "rows": len(shard), "path": str(shard_path)})
        manifest_path = shard_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(manifest_path)

    def _export_dataset(self, dataset) -> None:
        Exporter(export_path=self.config.export.path, export_type=self.config.export.export_type, export_shard_size=self.config.export.export_shard_size, keep_stats_in_res_ds=self.config.export.keep_stats_in_result, keep_hashes_in_res_ds=self.config.export.keep_hashes_in_result, export_stats=self.config.export.export_stats, num_proc=self.config.runtime.num_proc).export(dataset)

    def _run_dataset_stage(self, stage: str, fn, *, note: str = ""):
        started_at = time.perf_counter()
        dataset = fn()
        duration = time.perf_counter() - started_at
        self._record_stage(stage, dataset, note=note, duration_seconds=duration)
        self.monitor.record_stage(stage, len(dataset), duration, note=note)
        return dataset

    def _record_stage(self, stage: str, dataset, note: str = "", duration_seconds: float = 0.0) -> None:
        self.stage_records.append(
            StageRecord(stage=stage, rows=len(dataset), note=note, duration_seconds=duration_seconds)
        )

    def _build_summary(self, dataset) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "work_dir": str(self.work_dir),
            "export_path": self.config.export.path,
            "row_count": len(dataset),
            "stages": [record.__dict__ for record in self.stage_records],
            "quality": self.quality_summary,
            "shards": self.shard_manifests,
        }
        if self.config.judge.enabled:
            scores = []
            for row in dataset:
                judge = row.get(STANDARD_JUDGE_KEY) or {}
                if self.config.judge.score_key in judge:
                    scores.append(float(judge[self.config.judge.score_key]))
            if scores:
                summary["judge"] = {"sample_count": len(scores), "min": min(scores), "max": max(scores), "mean": sum(scores) / len(scores)}
        return summary

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        (self.work_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
