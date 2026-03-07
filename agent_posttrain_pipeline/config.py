from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


@dataclass
class SourceAdapterConfig:
    source_name: Optional[str] = None
    id_key: Optional[str] = None
    messages_key: Optional[str] = None
    message_role_key: str = "role"
    message_content_key: str = "content"
    system_prompt_key: Optional[str] = None
    prompt_key: Optional[str] = None
    response_key: Optional[str] = None
    instruction_key: Optional[str] = None
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    text_key: Optional[str] = None
    metadata_key: Optional[str] = None
    keep_fields: List[str] = field(default_factory=list)
    drop_empty_messages: bool = True


@dataclass
class DataSourceConfig:
    path: str
    format: Optional[str] = None
    adapter: SourceAdapterConfig = field(default_factory=SourceAdapterConfig)
    load_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardingConfig:
    enabled: bool = True
    rows_per_shard: int = 50000
    export_type: str = "parquet"
    output_dir: Optional[str] = None


@dataclass
class DedupConfig:
    enabled: bool = True
    tokenization: str = "space"
    window_size: int = 5
    lowercase: bool = True
    ignore_pattern: Optional[str] = None
    tokenizer_model: Optional[str] = None
    num_permutations: int = 256
    jaccard_threshold: float = 0.8


@dataclass
class ContaminationConfig:
    enabled: bool = False
    reference_sources: List[DataSourceConfig] = field(default_factory=list)
    ngram_size: int = 13
    max_contamination_ratio: float = 0.2
    min_matched_ngrams: int = 1
    normalize_case: bool = True
    strip_punctuation: bool = True


@dataclass
class QualityConfig:
    enabled: bool = True
    hf_tokenizer: Optional[str] = None
    min_tokens: Optional[int] = 16
    max_tokens: Optional[int] = 8192
    min_turns: int = 1
    max_turns: Optional[int] = None
    min_assistant_turns: int = 1
    percentiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    sample_for_report: int = 1000


@dataclass
class AugmentationConfig:
    enabled: bool = False
    backend: str = "async_http"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    callable_path: Optional[str] = None
    is_hf_model: bool = True
    prompt_template: str = "{text}"
    system_prompt: Optional[str] = None
    batch_size: int = 32
    max_concurrency: int = 16
    timeout: float = 120.0
    max_retries: int = 3
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    output_field: str = "augmentation"
    output_text_key: str = "generated_answer"
    keep_prompt: bool = False


@dataclass
class JudgeConfig:
    enabled: bool = False
    backend: str = "async_http"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    callable_path: Optional[str] = None
    is_hf_model: bool = True
    prompt_template: str = (
        "You are a strict judge for agent-training data. Score the sample from 0 to 10.\n"
        "Return JSON with keys score and reason.\n\n{text}"
    )
    system_prompt: Optional[str] = None
    batch_size: int = 16
    max_concurrency: int = 8
    timeout: float = 120.0
    max_retries: int = 3
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    output_field: str = "judge"
    score_key: str = "score"
    reason_key: str = "reason"
    raw_response_key: str = "raw_response"
    score_scale: float = 10.0
    min_score: Optional[float] = None
    keep_prompt: bool = False


@dataclass
class ExportConfig:
    path: str = "outputs/agent_posttrain_pipeline/cleaned_agent_data.parquet"
    export_type: Optional[str] = None
    export_shard_size: int = 0
    export_stats: bool = True
    keep_stats_in_result: bool = False
    keep_hashes_in_result: bool = False


@dataclass
class VisualizationConfig:
    enabled: bool = True
    output_dir: Optional[str] = None
    include_samples: int = 200


@dataclass
class MonitoringConfig:
    enabled: bool = True
    output_dir: Optional[str] = None
    job_name: str = "agent_posttrain_pipeline"
    emit_json: bool = True
    emit_prometheus_text: bool = True
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperatorInjectionConfig:
    enabled: bool = True
    hook: str = "post_normalized"
    class_path: str = ""
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    run_kwargs: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    note: str = ""


@dataclass
class RuntimeConfig:
    num_proc: int = 1
    work_dir: str = "outputs/agent_posttrain_pipeline"


@dataclass
class AgentPipelineConfig:
    sources: List[DataSourceConfig]
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    operators: List[OperatorInjectionConfig] = field(default_factory=list)

    def ensure_directories(self) -> None:
        Path(self.runtime.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.export.path).parent.mkdir(parents=True, exist_ok=True)
        if self.sharding.enabled:
            Path(self.sharding.output_dir or Path(self.runtime.work_dir) / "shards").mkdir(parents=True, exist_ok=True)
        if self.visualization.enabled:
            Path(self.visualization.output_dir or Path(self.runtime.work_dir) / "visualizations").mkdir(
                parents=True,
                exist_ok=True,
            )
        if self.monitoring.enabled:
            Path(self.monitoring.output_dir or Path(self.runtime.work_dir) / "monitoring").mkdir(
                parents=True,
                exist_ok=True,
            )


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _coerce_value(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)

    if is_dataclass(tp):
        return _coerce_dataclass(tp, value)
    if origin in (list, List, Sequence):
        inner = get_args(tp)[0] if get_args(tp) else Any
        return [_coerce_value(inner, item) for item in value]
    if origin in (dict, Dict, Mapping):
        _, val_type = get_args(tp) if get_args(tp) else (Any, Any)
        return {key: _coerce_value(val_type, val) for key, val in value.items()}
    return value


def _coerce_dataclass(cls: Type[T], value: Any) -> T:
    if isinstance(value, cls):
        return value
    if value is None:
        return cls()  # type: ignore[misc]
    if not isinstance(value, Mapping):
        return value  # type: ignore[return-value]

    kwargs: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field_info in fields(cls):
        if field_info.name not in value:
            if field_info.default is not MISSING or field_info.default_factory is not MISSING:
                continue
            continue
        kwargs[field_info.name] = _coerce_value(type_hints.get(field_info.name, field_info.type), value[field_info.name])
    return cls(**kwargs)


def build_pipeline_config(config_like: Union[AgentPipelineConfig, Mapping[str, Any]]) -> AgentPipelineConfig:
    cfg = config_like if isinstance(config_like, AgentPipelineConfig) else _coerce_dataclass(AgentPipelineConfig, config_like)
    if not cfg.sources:
        raise ValueError("At least one data source is required.")
    cfg.ensure_directories()
    return cfg
