from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

STANDARD_ID_KEY = "sample_id"
STANDARD_TEXT_KEY = "text"
STANDARD_MESSAGES_KEY = "messages"
STANDARD_SYSTEM_PROMPT_KEY = "system_prompt"
STANDARD_ANSWER_KEY = "answer"
STANDARD_SOURCE_KEY = "source"
STANDARD_METADATA_KEY = "metadata"
STANDARD_QUALITY_KEY = "quality_signals"
STANDARD_AUGMENTATION_KEY = "augmentation"
STANDARD_JUDGE_KEY = "judge"

_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def make_stable_sample_id(source_name: str, row_idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_name}-{row_idx:08d}-{digest}"


def normalize_message(message: Any, role_key: str = "role", content_key: str = "content") -> Optional[Dict[str, Any]]:
    if message is None:
        return None
    if isinstance(message, str):
        content = normalize_whitespace(message)
        return {"role": "user", "content": content} if content else None
    if not isinstance(message, Mapping):
        content = normalize_whitespace(str(message))
        return {"role": "user", "content": content} if content else None

    role = normalize_whitespace(str(message.get(role_key, "user") or "user")).lower()
    content = normalize_whitespace(str(message.get(content_key, "") or ""))
    if not content:
        return None
    normalized = {"role": role, "content": content}
    if "tool_calls" in message and message["tool_calls"] is not None:
        normalized["tool_calls"] = message["tool_calls"]
    return normalized


def conversation_to_text(messages: Iterable[Mapping[str, Any]], system_prompt: Optional[str] = None) -> str:
    chunks: List[str] = []
    if system_prompt:
        sys_text = normalize_whitespace(system_prompt)
        if sys_text:
            chunks.append(f"system: {sys_text}")
    for message in messages:
        role = normalize_whitespace(str(message.get("role", "user") or "user")).lower()
        content = normalize_whitespace(str(message.get("content", "") or ""))
        if not content:
            continue
        chunks.append(f"{role}: {content}")
        if message.get("tool_calls"):
            chunks.append(f"tool_calls: {json.dumps(message['tool_calls'], ensure_ascii=False, sort_keys=True)}")
    return "\n\n".join(chunks).strip()


def extract_last_assistant_message(messages: Iterable[Mapping[str, Any]]) -> Optional[str]:
    for message in reversed(list(messages)):
        if str(message.get("role", "")).lower() == "assistant":
            content = normalize_whitespace(str(message.get("content", "") or ""))
            if content:
                return content
    return None


def build_default_quality_signals(text: str, messages: List[Mapping[str, Any]]) -> Dict[str, Any]:
    assistant_turn_count = sum(1 for item in messages if str(item.get("role", "")).lower() == "assistant")
    user_turn_count = sum(1 for item in messages if str(item.get("role", "")).lower() == "user")
    return {
        "char_count": len(text),
        "message_count": len(messages),
        "assistant_turn_count": assistant_turn_count,
        "user_turn_count": user_turn_count,
    }
