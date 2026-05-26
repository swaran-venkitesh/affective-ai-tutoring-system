from __future__ import annotations

import os
import re
import time
import threading
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Sequence, Any

import requests

DEFAULT_BASE_URL = os.getenv("TUTOR_QWEN_BASE_URL", "http://172.16.13.91:8092/v1")
DEFAULT_MODEL_ID = os.getenv("TUTOR_QWEN_MODEL_ID", "qwen3.5-9b")
DEFAULT_CONTEXT_WINDOW = int(os.getenv("QWEN_CONTEXT_WINDOW", "10000"))
DEFAULT_TIMEOUT_SEC = float(os.getenv("QWEN_TIMEOUT_SEC", "20"))
DEFAULT_RETRIES = int(os.getenv("QWEN_RETRIES", "0"))

# Default system prompt (not critical for changes)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

@dataclass
class QwenConfig:
    base_url: str = DEFAULT_BASE_URL
    model_id: str = DEFAULT_MODEL_ID
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    default_request_kwargs: Dict[str, Any] = field(default_factory=dict)

    temperature: float = 0.6
    max_tokens: int = 900
    context_window: int = DEFAULT_CONTEXT_WINDOW
    context_margin_tokens: int = 256
    min_output_tokens: int = 128

    # reliability settings for API requests
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    retries: int = DEFAULT_RETRIES
    retry_backoff_sec: float = 0.6


class QwenChat:
    def __init__(self, config: Optional[QwenConfig] = None):
        self.cfg = config or QwenConfig()
        self.base_url = self.cfg.base_url.rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.session = requests.Session()

        self._lock = threading.Lock()

        # Initialize history ONLY for stream_tokens(), if streaming is needed
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_prompt}
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        chars = len(text)
        words = len(text.split())
        return max(chars // 4, int(words * 1.35))

    def _clip_prompt_text(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        if self._estimate_tokens(text) <= max_tokens:
            return text

        max_chars = max(160, max_tokens * 4)
        marker_positions = [
            text.rfind(marker)
            for marker in ("STUDENT_MESSAGE:", "QUESTION:", "Q:", "USER:", "ACTIVE MODE:")
        ]
        marker_pos = max(marker_positions) if marker_positions else -1
        if marker_pos >= 0:
            tail = text[marker_pos:].strip()
            if len(tail) >= max_chars:
                return tail[-max_chars:]
            head_budget = max_chars - len(tail) - 16
            head = text[:max(0, head_budget)].rstrip()
            if head:
                return head + "\n...\n" + tail
            return tail

        head_chars = int(max_chars * 0.35)
        tail_chars = max_chars - head_chars - 8
        return text[:head_chars].rstrip() + "\n...\n" + text[-tail_chars:].lstrip()

    def _fit_to_context(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        budget = max(256, int(self.cfg.context_window) - int(self.cfg.context_margin_tokens))
        sys_tokens = self._estimate_tokens(system_prompt)
        user_tokens = self._estimate_tokens(user_prompt)
        output_tokens = int(max_tokens)

        if sys_tokens + user_tokens + output_tokens <= budget:
            return user_prompt, output_tokens

        output_tokens = min(output_tokens, max(int(self.cfg.min_output_tokens), budget - sys_tokens - user_tokens))

        if sys_tokens + user_tokens + output_tokens > budget:
            allowed_user_tokens = max(128, budget - sys_tokens - output_tokens)
            user_prompt = self._clip_prompt_text(user_prompt, allowed_user_tokens)
            user_tokens = self._estimate_tokens(user_prompt)

        if sys_tokens + user_tokens + output_tokens > budget:
            output_tokens = max(int(self.cfg.min_output_tokens), budget - sys_tokens - user_tokens)

        if output_tokens < int(self.cfg.min_output_tokens):
            allowed_user_tokens = max(96, budget - sys_tokens - int(self.cfg.min_output_tokens))
            user_prompt = self._clip_prompt_text(user_prompt, allowed_user_tokens)
            output_tokens = int(self.cfg.min_output_tokens)

        return user_prompt, max(16, output_tokens)

    def _is_context_overflow(self, err: Exception) -> bool:
        msg = str(err)
        return (
            "context length is only" in msg.lower()
            or "maximum input length" in msg.lower()
            or "parameter=input_tokens" in msg.lower()
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_chat_completion(
        self,
        payload: Dict[str, Any],
        *,
        stream: bool = False,
    ) -> requests.Response:
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.cfg.timeout_sec,
            stream=stream,
        )
        response.raise_for_status()
        return response

    def complete_once(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Stateless single-shot completion (NO history).
        This method is called when you want to get a single completion from the model.
        """
        temp = self.cfg.temperature if temperature is None else float(temperature)
        mtok = self.cfg.max_tokens if max_tokens is None else int(max_tokens)
        merged_kwargs: Dict[str, Any] = dict(self.cfg.default_request_kwargs or {})
        if extra_kwargs:
            merged_kwargs.update(extra_kwargs)
        user_prompt, mtok = self._fit_to_context(system_prompt, user_prompt, mtok)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.retries + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": self.cfg.model_id,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": mtok,
                    **merged_kwargs,
                }
                if stop:
                    payload["stop"] = list(stop)
                resp = self._request_chat_completion(payload)
                data = resp.json()
                out = (
                    (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
                    or ""
                )
                return out.strip()
            except Exception as e:
                last_err = e
                if self._is_context_overflow(e):
                    mtok = max(int(self.cfg.min_output_tokens), mtok // 2)
                    user_prompt, mtok = self._fit_to_context(system_prompt, user_prompt, mtok)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                if attempt >= self.cfg.retries:
                    break
                time.sleep(self.cfg.retry_backoff_sec * (attempt + 1))

        # If it failed after retries, raise (server.py catches & logs)
        raise RuntimeError(f"Qwen complete_once failed after retries: {last_err}")

    def stream_tokens(self, user_text: str) -> Iterator[str]:
        """Optional legacy streaming with history (not used by your state machine)."""
        with self._lock:
            self.history.append({"role": "user", "content": user_text})
            messages = list(self.history)

        assistant_text = ""
        payload = {
            "model": self.cfg.model_id,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            **dict(self.cfg.default_request_kwargs or {}),
        }
        with self._request_chat_completion(payload, stream=True) as response:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (((event.get("choices") or [{}])[0]).get("delta") or {})
                token = delta.get("content")
                if token:
                    assistant_text += token
                    yield token

        with self._lock:
            self.history.append({"role": "assistant", "content": assistant_text})
