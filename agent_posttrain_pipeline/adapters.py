from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .config import DataSourceConfig, SourceAdapterConfig
from .schema import (
    STANDARD_ANSWER_KEY,
    STANDARD_AUGMENTATION_KEY,
    STANDARD_ID_KEY,
    STANDARD_JUDGE_KEY,
    STANDARD_METADATA_KEY,
    STANDARD_MESSAGES_KEY,
    STANDARD_QUALITY_KEY,
    STANDARD_SOURCE_KEY,
    STANDARD_SYSTEM_PROMPT_KEY,
    STANDARD_TEXT_KEY,
    build_default_quality_signals,
    conversation_to_text,
    extract_last_assistant_message,
    make_stable_sample_id,
    normalize_message,
    normalize_whitespace,
)


class AgentSampleAdapter:
    def __init__(self, config: SourceAdapterConfig):
        self.config = config

    def adapt_record(self, record: Mapping[str, Any], row_idx: int, source: DataSourceConfig) -> Dict[str, Any]:
        source_name = self.config.source_name or source.adapter.source_name or source.path.rsplit("/", 1)[-1]
        messages = self._extract_messages(record)
        system_prompt = self._extract_system_prompt(record)
        text = self._extract_text(record, messages=messages, system_prompt=system_prompt)
        sample_id = self._extract_id(record, row_idx=row_idx, source_name=source_name, text=text)
        answer = self._extract_answer(record, messages)
        metadata = self._extract_metadata(record, source_name=source_name, source_path=source.path)
        return {
            STANDARD_ID_KEY: sample_id,
            STANDARD_TEXT_KEY: text,
            STANDARD_MESSAGES_KEY: messages,
            STANDARD_SYSTEM_PROMPT_KEY: system_prompt,
            STANDARD_ANSWER_KEY: answer,
            STANDARD_SOURCE_KEY: source_name,
            STANDARD_METADATA_KEY: metadata,
            STANDARD_QUALITY_KEY: build_default_quality_signals(text, messages),
            STANDARD_AUGMENTATION_KEY: {},
            STANDARD_JUDGE_KEY: {},
        }

    def _extract_messages(self, record: Mapping[str, Any]) -> List[Dict[str, Any]]:
        cfg = self.config
        messages_value = record.get(cfg.messages_key) if cfg.messages_key else None
        messages: List[Dict[str, Any]] = []
        if isinstance(messages_value, list):
            for item in messages_value:
                normalized = normalize_message(item, role_key=cfg.message_role_key, content_key=cfg.message_content_key)
                if normalized is None and not cfg.drop_empty_messages:
                    normalized = {"role": "user", "content": ""}
                if normalized is not None:
                    messages.append(normalized)
            if messages:
                return messages

        prompt = self._first_non_empty(record, [cfg.prompt_key, cfg.instruction_key, cfg.text_key])
        extra_input = self._first_non_empty(record, [cfg.input_key])
        response = self._first_non_empty(record, [cfg.response_key, cfg.output_key])
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if extra_input:
            if messages:
                messages[-1]["content"] = normalize_whitespace(f"{messages[-1]['content']}\n{extra_input}")
            else:
                messages.append({"role": "user", "content": extra_input})
        if response:
            messages.append({"role": "assistant", "content": response})
        if not messages and cfg.text_key and record.get(cfg.text_key):
            messages.append({"role": "user", "content": normalize_whitespace(str(record[cfg.text_key]))})
        return messages

    def _extract_system_prompt(self, record: Mapping[str, Any]) -> Optional[str]:
        if not self.config.system_prompt_key:
            return None
        value = record.get(self.config.system_prompt_key)
        if value is None:
            return None
        normalized = normalize_whitespace(str(value))
        return normalized or None

    def _extract_text(self, record: Mapping[str, Any], *, messages: List[Mapping[str, Any]], system_prompt: Optional[str]) -> str:
        if self.config.text_key and record.get(self.config.text_key):
            return normalize_whitespace(str(record[self.config.text_key]))
        return conversation_to_text(messages, system_prompt=system_prompt)

    def _extract_id(self, record: Mapping[str, Any], row_idx: int, source_name: str, text: str) -> str:
        if self.config.id_key and record.get(self.config.id_key) is not None:
            return str(record[self.config.id_key])
        return make_stable_sample_id(source_name, row_idx, text)

    def _extract_answer(self, record: Mapping[str, Any], messages: List[Mapping[str, Any]]) -> Optional[str]:
        response = self._first_non_empty(record, [self.config.response_key, self.config.output_key])
        return response or extract_last_assistant_message(messages)

    def _extract_metadata(self, record: Mapping[str, Any], source_name: str, source_path: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"source_name": source_name, "source_path": source_path, "adapter": "AgentSampleAdapter"}
        if self.config.metadata_key and isinstance(record.get(self.config.metadata_key), MutableMapping):
            metadata.update(record[self.config.metadata_key])
        for field_name in self.config.keep_fields:
            if field_name in record:
                metadata[field_name] = record[field_name]
        return metadata

    @staticmethod
    def _first_non_empty(record: Mapping[str, Any], keys: List[Optional[str]]) -> Optional[str]:
        for key in keys:
            if not key:
                continue
            value = record.get(key)
            if value is None:
                continue
            normalized = normalize_whitespace(str(value))
            if normalized:
                return normalized
        return None
