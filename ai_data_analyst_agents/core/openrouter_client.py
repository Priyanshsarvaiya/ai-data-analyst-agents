from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from pydantic_settings import BaseSettings


class OpenRouterEnv(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_SITE_URL: Optional[str] = None
    OPENROUTER_APP_NAME: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


class OpenRouterClient:
    """
    Minimal OpenRouter chat-completions client.
    Endpoint: https://openrouter.ai/api/v1/chat/completions
    """

    def __init__(self, timeout_s: int = 60):
        env = OpenRouterEnv()
        if not env.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY missing. Put it in .env")
        self.api_key = env.OPENROUTER_API_KEY
        self.site_url = env.OPENROUTER_SITE_URL
        self.app_name = env.OPENROUTER_APP_NAME
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout_s = timeout_s

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
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

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        return data["choices"][0]["message"]["content"]