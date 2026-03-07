from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from .config import AugmentationConfig, JudgeConfig


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


@dataclass
class GenerationResult:
    text: Optional[str]
    raw: Any = None
    error: Optional[str] = None


class AsyncBackend:
    async def generate_many(self, prompts: Sequence[str], system_prompt: Optional[str] = None) -> List[GenerationResult]:
        raise NotImplementedError


class CallableBackend(AsyncBackend):
    def __init__(self, callable_path: str):
        module_name, attr_name = callable_path.split(":", 1)
        module = importlib.import_module(module_name)
        self.func = getattr(module, attr_name)

    async def generate_many(self, prompts: Sequence[str], system_prompt: Optional[str] = None) -> List[GenerationResult]:
        results: List[GenerationResult] = []
        for prompt in prompts:
            try:
                if inspect.iscoroutinefunction(self.func):
                    value = await self.func(prompt=prompt, system_prompt=system_prompt)
                else:
                    value = self.func(prompt=prompt, system_prompt=system_prompt)
                if isinstance(value, Mapping):
                    results.append(GenerationResult(text=value.get("text"), raw=value))
                else:
                    results.append(GenerationResult(text=str(value), raw=value))
            except Exception as exc:  # noqa: BLE001
                results.append(GenerationResult(text=None, error=str(exc)))
        return results


class OpenAICompatibleBackend(AsyncBackend):
    def __init__(self, *, model: str, base_url: str, api_key: str, timeout: float, max_concurrency: int, max_retries: int, sampling_params: Optional[Dict[str, Any]] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.sampling_params = sampling_params or {}

    async def generate_many(self, prompts: Sequence[str], system_prompt: Optional[str] = None) -> List[GenerationResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._generate_one(client, prompt=prompt, system_prompt=system_prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)

    async def _generate_one(self, client: httpx.AsyncClient, *, prompt: str, system_prompt: Optional[str]) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        payload = {"model": self.model, "messages": messages, **self.sampling_params}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()
                    raw = response.json()
                    return GenerationResult(text=raw["choices"][0]["message"]["content"], raw=raw)
                except Exception as exc:  # noqa: BLE001
                    if attempt == self.max_retries:
                        return GenerationResult(text=None, error=str(exc))
                    await asyncio.sleep(min(2**attempt, 10))
        return GenerationResult(text=None, error="unknown generation failure")


class BatchLLMProcessor:
    def __init__(self, backend: AsyncBackend):
        self.backend = backend

    def apply_generation_stage(self, dataset, config: AugmentationConfig):
        def _run_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            rows = _batch_to_rows(batch)
            prompts = [render_prompt(config.prompt_template, row) for row in rows]
            responses = asyncio.run(self.backend.generate_many(prompts, system_prompt=config.system_prompt))
            merged = []
            for row, prompt, response in zip(rows, prompts, responses):
                payload = dict(row.get(config.output_field) or {})
                payload[config.output_text_key] = response.text
                payload["backend"] = config.backend
                payload["model"] = config.model
                if response.error:
                    payload["error"] = response.error
                if config.keep_prompt:
                    payload["prompt"] = prompt
                merged.append(payload)
            return {config.output_field: merged}

        return dataset.map(_run_batch, batched=True, batch_size=max(1, config.batch_size), desc="Running augmentation model stage")

    def apply_judge_stage(self, dataset, config: JudgeConfig):
        def _run_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            rows = _batch_to_rows(batch)
            prompts = [render_prompt(config.prompt_template, row) for row in rows]
            responses = asyncio.run(self.backend.generate_many(prompts, system_prompt=config.system_prompt))
            merged = []
            for row, prompt, response in zip(rows, prompts, responses):
                parsed = parse_judge_response(response.text, score_scale=config.score_scale)
                payload = dict(row.get(config.output_field) or {})
                payload[config.score_key] = parsed["score"]
                payload[config.reason_key] = parsed["reason"]
                payload["normalized_score"] = parsed["normalized_score"]
                payload[config.raw_response_key] = response.text
                payload["backend"] = config.backend
                payload["model"] = config.model
                if response.error:
                    payload["error"] = response.error
                if config.keep_prompt:
                    payload["prompt"] = prompt
                merged.append(payload)
            return {config.output_field: merged}

        dataset = dataset.map(_run_batch, batched=True, batch_size=max(1, config.batch_size), desc="Running judge model stage")
        if config.min_score is not None:
            dataset = dataset.filter(lambda sample: float((sample.get(config.output_field) or {}).get(config.score_key, 0.0)) >= config.min_score, desc="Filtering by judge score")
        return dataset


def flatten_context(row: Mapping[str, Any]) -> Dict[str, Any]:
    context = dict(row)
    for key in ("metadata", "quality_signals", "augmentation", "judge"):
        context[f"{key}_json"] = json.dumps(row.get(key) or {}, ensure_ascii=False, sort_keys=True)
    return context


def render_prompt(template: str, row: Mapping[str, Any]) -> str:
    return template.format_map(SafeFormatDict(flatten_context(row))).strip()


def build_backend(config: Any) -> AsyncBackend:
    if config.backend == "callable":
        if not config.callable_path:
            raise ValueError("callable backend requires callable_path")
        return CallableBackend(config.callable_path)
    if config.backend == "async_http":
        api_key = config.api_key or os.environ.get(config.api_key_env)
        if not config.base_url:
            raise ValueError("async_http backend requires base_url")
        if not api_key:
            raise ValueError(f"Missing API key. Set {config.api_key_env} or api_key.")
        if not config.model:
            raise ValueError("async_http backend requires model")
        return OpenAICompatibleBackend(model=config.model, base_url=config.base_url, api_key=api_key, timeout=config.timeout, max_concurrency=config.max_concurrency, max_retries=config.max_retries, sampling_params=config.sampling_params)
    raise ValueError(f"Unsupported backend: {config.backend}")


def apply_ray_vllm_stage(dataset, config: Any):
    from .dj_bridge import LLMRayVLLMEnginePipeline

    prompt_column = "__agent_pipeline_prompt__"
    response_column = "__agent_pipeline_response__"

    def _render(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[prompt_column] = render_prompt(config.prompt_template, sample)
        return sample

    dataset = dataset.map(_render, desc="Rendering prompts for ray vLLM stage")
    op = LLMRayVLLMEnginePipeline(
        api_or_hf_model=config.model,
        is_hf_model=getattr(config, "is_hf_model", True),
        query_key=prompt_column,
        response_key=response_column,
        system_prompt=config.system_prompt,
        sampling_params=config.sampling_params,
        api_url=getattr(config, "base_url", None),
        api_key=getattr(config, "api_key", None) or os.environ.get(getattr(config, "api_key_env", "OPENAI_API_KEY")),
        batch_size=max(1, config.batch_size),
        num_proc=max(1, getattr(config, "max_concurrency", 1)),
    )
    dataset = op.run(dataset)
    if isinstance(config, AugmentationConfig):
        def _merge(sample: Dict[str, Any]) -> Dict[str, Any]:
            payload = dict(sample.get(config.output_field) or {})
            payload[config.output_text_key] = sample.get(response_column)
            payload["backend"] = config.backend
            payload["model"] = config.model
            if config.keep_prompt:
                payload["prompt"] = sample.get(prompt_column)
            sample[config.output_field] = payload
            return sample
    else:
        def _merge(sample: Dict[str, Any]) -> Dict[str, Any]:
            parsed = parse_judge_response(sample.get(response_column), score_scale=config.score_scale)
            payload = dict(sample.get(config.output_field) or {})
            payload[config.score_key] = parsed["score"]
            payload[config.reason_key] = parsed["reason"]
            payload["normalized_score"] = parsed["normalized_score"]
            payload[config.raw_response_key] = sample.get(response_column)
            payload["backend"] = config.backend
            payload["model"] = config.model
            if config.keep_prompt:
                payload["prompt"] = sample.get(prompt_column)
            sample[config.output_field] = payload
            return sample
    dataset = dataset.map(_merge, desc="Merging ray vLLM outputs")
    dataset = dataset.remove_columns([prompt_column, response_column])
    if isinstance(config, JudgeConfig) and config.min_score is not None:
        dataset = dataset.filter(lambda sample: float((sample.get(config.output_field) or {}).get(config.score_key, 0.0)) >= config.min_score, desc="Filtering by judge score")
    return dataset


_SCORE_RE = re.compile(r"(?P<score>\d+(?:\.\d+)?)")


def parse_judge_response(text: Optional[str], *, score_scale: float = 10.0) -> Dict[str, Any]:
    if not text:
        return {"score": 0.0, "reason": "", "normalized_score": 0.0}
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        score = float(payload.get("score", 0.0))
        reason = str(payload.get("reason", ""))
        return {"score": score, "reason": reason, "normalized_score": min(score / score_scale, 1.0)}
    except Exception:  # noqa: BLE001
        pass
    match = _SCORE_RE.search(stripped)
    score = float(match.group("score")) if match else 0.0
    reason = stripped if not match else stripped[match.end():].strip(" :-\n")
    return {"score": score, "reason": reason, "normalized_score": min(score / score_scale, 1.0)}


def _batch_to_rows(batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    columns = list(batch.keys())
    if not columns:
        return []
    row_count = len(batch[columns[0]])
    return [{column: batch[column][idx] for column in columns} for idx in range(row_count)]
