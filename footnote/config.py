"""Single typed config object driving ingestion, retrieval, and answering.

Every eval run records the exact RunConfig that produced it, so results are
reproducible and ablations are just config diffs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """API keys and endpoints. Loaded from .env — never part of RunConfig."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jina_api_key: str = ""
    qdrant_url: str = ""  # empty -> embedded local mode
    qdrant_api_key: str = ""
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    cerebras_api_key: str = ""
    hf_token: str = ""


class RunConfig(BaseModel):
    """Everything that can move an eval number lives here."""

    # corpus
    corpus: list[str] = ["gdpr", "ai_act"]
    chunk_strategy: Literal["provision", "article", "window"] = "provision"
    window_size: int = 512  # tokens, window strategy only
    window_overlap: int = 64

    # embeddings
    embedding_provider: Literal["jina", "cohere", "local"] = "jina"
    embedding_model: str = "jina-embeddings-v3"

    # retrieval
    dense_enabled: bool = True
    sparse_enabled: bool = True
    top_k_dense: int = 50
    top_k_sparse: int = 50
    fusion_k: int = 60  # RRF constant
    rerank_enabled: bool = True
    rerank_model: str = "jina-reranker-v2-base-multilingual"
    rerank_top_n: int = 10

    # answering
    # "provider::model" — the chain crosses PROVIDERS, not just models, because
    # free tiers cap per account per day (OpenRouter: 50 requests/day).
    generation_model: str = "groq::llama-3.3-70b-versatile"
    fallback_models: list[str] = [
        "gemini::gemini-3.1-flash-lite-preview",
        "openrouter::openai/gpt-oss-20b:free",  # original bake-off winner: 7/8 JSON
    ]
    judge_model: str = "gemini::gemini-3.1-flash-lite-preview"  # different vendor than generator
    prompt_version: str = "v1"
    confidence_threshold: float = 0.0  # calibrated in M6

    # agent
    agent_enabled: bool = False
    max_hops: int = 4
    max_cost_usd: float = 0.05  # hard cap per question, even at $0 list price

    def config_hash(self) -> str:
        """Stable hash recorded in every trace row and results file."""
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
