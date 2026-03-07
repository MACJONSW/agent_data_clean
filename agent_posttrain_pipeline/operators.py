from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Mapping, Sequence

from .bootstrap import ensure_data_juicer_on_path
from .config import ContaminationConfig, OperatorInjectionConfig
from .contamination import NGramContaminationFilter

SUPPORTED_OPERATOR_HOOKS = {
    "post_normalized",
    "post_dedup",
    "post_contamination",
    "post_quality",
    "post_augmentation",
    "post_judge",
    "pre_export",
}


@dataclass(frozen=True)
class OperatorContext:
    num_proc: int = 1


@dataclass
class DatasetOperator:
    name: str
    hook: str
    note: str = ""

    def apply(self, dataset, *, context: OperatorContext):
        raise NotImplementedError


class NGramContaminationOperator(DatasetOperator):
    def __init__(self, config: ContaminationConfig, source_loader=None):
        super().__init__(
            name="ngram_contamination",
            hook="post_dedup",
            note="n-gram contamination filter",
        )
        self.config = config
        self.source_loader = source_loader

    def apply(self, dataset, *, context: OperatorContext):
        if not self.config.reference_sources:
            raise ValueError("Contamination filtering requires reference_sources.")
        source_loader = self.source_loader
        if source_loader is None:
            from .loaders import load_and_normalize_sources

            source_loader = load_and_normalize_sources
        texts = []
        for source in self.config.reference_sources:
            ref_dataset = source_loader([source], num_proc=context.num_proc)
            texts.extend(ref_dataset["text"])
        contamination_filter = NGramContaminationFilter(self.config)
        contamination_filter.fit(texts)
        return contamination_filter.transform(dataset, num_proc=context.num_proc)


class DataJuicerOperatorAdapter(DatasetOperator):
    def __init__(
        self,
        *,
        name: str,
        hook: str,
        operator: Any,
        note: str = "",
        run_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__(name=name, hook=hook, note=note)
        self.operator = operator
        self.run_kwargs = dict(run_kwargs or {})

    def apply(self, dataset, *, context: OperatorContext):
        run = getattr(self.operator, "run", None)
        if callable(run):
            return run(dataset, **self._filter_run_kwargs(run, context))
        if callable(self.operator):
            return self.operator(dataset)
        raise TypeError(f"Configured operator [{self.name}] is neither callable nor exposes run(dataset).")

    def _filter_run_kwargs(self, run, context: OperatorContext) -> Dict[str, Any]:
        signature = inspect.signature(run)
        kwargs = dict(self.run_kwargs)
        if "num_proc" in signature.parameters and "num_proc" not in kwargs:
            kwargs["num_proc"] = context.num_proc
        return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_extra_operators(configs: Sequence[OperatorInjectionConfig]) -> list[DatasetOperator]:
    operators: list[DatasetOperator] = []
    for cfg in configs:
        if not cfg.enabled:
            continue
        if cfg.hook not in SUPPORTED_OPERATOR_HOOKS:
            supported = ", ".join(sorted(SUPPORTED_OPERATOR_HOOKS))
            raise ValueError(f"Unsupported operator hook [{cfg.hook}]. Supported hooks: {supported}")
        if not cfg.class_path:
            raise ValueError("Operator injection requires class_path.")
        operator_cls = _import_from_path(cfg.class_path)
        operator = operator_cls(**(cfg.init_kwargs or {}))
        name = cfg.name or operator_cls.__name__
        note = cfg.note or f"injected operator from {cfg.class_path}"
        operators.append(
            DataJuicerOperatorAdapter(
                name=name,
                hook=cfg.hook,
                operator=operator,
                note=note,
                run_kwargs=cfg.run_kwargs,
            )
        )
    return operators


def _import_from_path(class_path: str):
    ensure_data_juicer_on_path()
    module_path, _, attr = class_path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid operator class path [{class_path}]")
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Failed to import [{class_path}]. Install the required runtime dependencies for this operator first."
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Operator class [{class_path}] not found.") from exc
