"""OpenRouter chat client. Free-tier models only; usage tracked per call.

cost_usd records what the call *would* cost at list price — for :free models
that is $0, logged alongside the tokens actually consumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from footnote.config import Secrets

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRIES = 4


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int


class OpenRouterClient:
    def __init__(self, secrets: Secrets | None = None):
        self.secrets = secrets or Secrets()

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1200,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.secrets.openrouter_api_key}"}

        delay = 3.0
        for attempt in range(RETRIES):
            t0 = time.perf_counter()
            resp = httpx.post(API_URL, json=payload, headers=headers, timeout=180)
            latency = int((time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:  # OpenRouter tunnels provider errors in 200s
                    if attempt < RETRIES - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    raise RuntimeError(f"OpenRouter error: {data['error']}")
                usage = data.get("usage", {})
                choice = data["choices"][0]
                return LLMResponse(
                    text=choice["message"]["content"] or "",
                    model=data.get("model", model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost_usd=float(usage.get("cost", 0.0)),
                    latency_ms=latency,
                )
            if resp.status_code in (408, 429, 500, 502, 503) and attempt < RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("unreachable")  # pragma: no cover
