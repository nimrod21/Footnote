"""Multi-provider chat client over OpenAI-compatible APIs. Free tiers only.

Models are qualified as "provider::model", e.g. "groq::llama-3.3-70b-versatile"
or "openrouter::openai/gpt-oss-20b:free". Free tiers rate-limit unpredictably
and cap daily usage, so resilience = a fallback chain across *providers*, not
just across models on one provider (OpenRouter's 50-requests/day account cap
proved that the hard way).

cost_usd records what the call *would* cost at list price — for free tiers
that is $0, logged alongside the tokens actually consumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from footnote.config import Secrets

RETRIES = 4

# provider -> (chat completions endpoint, Secrets attribute holding the key)
PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "openrouter_api_key"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "groq_api_key"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini_api_key"),
    "cerebras": ("https://api.cerebras.ai/v1/chat/completions", "cerebras_api_key"),
}


def split_model(qualified: str) -> tuple[str, str]:
    """'groq::llama-3.3-70b-versatile' -> ('groq', 'llama-3.3-70b-versatile')."""
    provider, sep, model = qualified.partition("::")
    if not sep:
        return "openrouter", qualified  # unqualified = openrouter
    return provider, model


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
        reasoning_effort: str | None = None,  # cap thinking on reasoning models
    ) -> LLMResponse:
        provider, model_id = split_model(model)
        api_url, key_attr = PROVIDERS[provider]
        api_key = getattr(self.secrets, key_attr)
        if not api_key:
            raise RuntimeError(f"no API key configured for provider {provider}")
        payload: dict = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort and provider == "openrouter":
            payload["reasoning"] = {"effort": reasoning_effort}  # OpenRouter extension
        headers = {"Authorization": f"Bearer {api_key}"}

        delay = 3.0
        for attempt in range(RETRIES):
            t0 = time.perf_counter()
            resp = httpx.post(api_url, json=payload, headers=headers, timeout=180)
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
                content = choice["message"]["content"] or ""
                if not content.strip() and attempt < RETRIES - 1:
                    # reasoning models can burn the whole budget thinking
                    payload["max_tokens"] = max_tokens * 2
                    time.sleep(2)
                    continue
                return LLMResponse(
                    text=content,
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

    def chat_with_fallback(self, models: list[str], messages: list[dict], **kw) -> LLMResponse:
        """Try each model in order — free tiers rate-limit unpredictably, and the
        zero-cost constraint says degrade to another free model, never upgrade."""
        last: RuntimeError | None = None
        for model in models:
            try:
                resp = self.chat(model, messages, **kw)
                if resp.text.strip():
                    return resp
                last = RuntimeError(f"{model} returned empty content")
            except RuntimeError as e:
                last = e
        raise last or RuntimeError("no models given")
