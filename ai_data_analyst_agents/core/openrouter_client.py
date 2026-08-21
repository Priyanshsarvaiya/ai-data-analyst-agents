from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

import httpx
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class OpenRouterEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_SITE_URL: Optional[str] = None
    OPENROUTER_APP_NAME: Optional[str] = None

class OpenRouterError(RuntimeError):
    """Raised when the provider cannot return a usable model response."""


class OpenRouterClient:
    """
    Minimal OpenRouter chat-completions client.
    Endpoint: https://openrouter.ai/api/v1/chat/completions
    """

    def __init__(self, timeout_s: int = 60, max_attempts: int = 4):
        env = OpenRouterEnv()
        if not env.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY missing. Put it in .env")
        self.api_key = env.OPENROUTER_API_KEY
        self.site_url = env.OPENROUTER_SITE_URL
        self.app_name = env.OPENROUTER_APP_NAME
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout_s = timeout_s
        self.max_attempts = max(1, min(int(max_attempts), 8))
        self.last_usage: Dict[str, Any] = {}
        self.last_finish_reason: str | None = None

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        if not str(model).strip():
            raise ValueError("model must be non-empty")
        if not messages:
            raise ValueError("messages must contain at least one item")
        if int(max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data: Dict[str, Any] | None = None
        retryable_status = {429, 500, 502, 503, 504}
        last_err: Exception | None = None

        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                    r.raise_for_status()
                    parsed = r.json()
                    data = parsed if isinstance(parsed, dict) else {}
                    break
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status in retryable_status and attempt < self.max_attempts:
                        sleep_s = min(2.0 * attempt, 8.0)
                        time.sleep(sleep_s)
                        last_err = e
                        continue
                    if status in retryable_status:
                        raise OpenRouterError(
                            f"OpenRouter request failed after {self.max_attempts} attempts (HTTP {status})."
                        ) from e
                    raise
                except httpx.HTTPError as e:
                    if attempt < self.max_attempts:
                        sleep_s = min(2.0 * attempt, 8.0)
                        time.sleep(sleep_s)
                        last_err = e
                        continue
                    raise OpenRouterError(
                        f"OpenRouter network request failed after {self.max_attempts} attempts."
                    ) from e

        if data is None:
            raise OpenRouterError("OpenRouter returned no response data.") from last_err
        usage = data.get("usage")
        self.last_usage = dict(usage) if isinstance(usage, dict) else {}
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            raise OpenRouterError("OpenRouter returned no completion choices.")

        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        self.last_finish_reason = (
            str(choices[0].get("finish_reason"))
            if isinstance(choices[0], dict) and choices[0].get("finish_reason") is not None
            else None
        )
        content = msg.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if isinstance(part, dict):
                    txt = part.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            joined = "\n".join(p for p in parts if p and p.strip()).strip()
            if joined:
                return joined

        if isinstance(content, dict):
            txt = content.get("text")
            if isinstance(txt, str):
                return txt.strip()

        if "reasoning" in msg and isinstance(msg["reasoning"], str):
            return msg["reasoning"].strip()

        if "text" in choices[0] and isinstance(choices[0]["text"], str):
            return choices[0]["text"].strip()

        raise OpenRouterError("OpenRouter returned a completion without usable text content.")
