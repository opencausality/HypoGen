"""Application configuration via Pydantic settings.

Reads from environment variables and ``.env`` files.  Provider resolution
follows the priority chain: explicit config → env var → Ollama default.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("hypogen")

# ── Supported LLM providers ───────────────────────────────────────────────────


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    MISTRAL = "mistral"
    TOGETHER_AI = "together_ai"


# ── Default model per provider ────────────────────────────────────────────────

DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "llama3.1",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
    LLMProvider.GROQ: "llama-3.1-70b-versatile",
    LLMProvider.MISTRAL: "mistral-large-latest",
    LLMProvider.TOGETHER_AI: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
}

# ── Settings ──────────────────────────────────────────────────────────────────


class HypoGenSettings(BaseSettings):
    """Central configuration for HypoGen.

    Values are loaded in this priority order:
    1. Explicit constructor arguments
    2. Environment variables (prefixed ``HYPOGEN_``)
    3. ``.env`` file in the working directory
    4. Built-in defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="HYPOGEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM settings ─────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="LLM provider to use for causal extraction and hypothesis generation.",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model name. Defaults to provider-specific recommendation.",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low values yield consistent extractions.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retry attempts for failed LLM calls.",
    )

    # ── Extraction & graph settings ───────────────────────────────────────
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for including a causal claim in the knowledge graph.",
    )
    min_papers_for_edge: int = Field(
        default=1,
        ge=1,
        description="Minimum number of papers that must support an edge for it to be included.",
    )

    # ── Hypothesis generation settings ───────────────────────────────────
    max_hypotheses: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of hypotheses to generate per analysis run.",
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── API server ───────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1", description="API listen address.")
    api_port: int = Field(default=8000, description="API listen port.")

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log level: {v}")
        return v

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def resolved_model(self) -> str:
        """Return the model name, falling back to the provider default."""
        if self.llm_model:
            return self.llm_model
        return DEFAULT_MODELS[self.llm_provider]

    @property
    def litellm_model(self) -> str:
        """Return the model string expected by LiteLLM.

        Ollama models must be prefixed with ``ollama/``.
        """
        model = self.resolved_model
        if self.llm_provider == LLMProvider.OLLAMA and not model.startswith("ollama/"):
            return f"ollama/{model}"
        return model


# ── Singleton accessor ────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> HypoGenSettings:
    """Return the cached application settings singleton."""
    return HypoGenSettings()


def configure_logging(settings: HypoGenSettings | None = None) -> None:
    """Set up Python logging based on application settings."""
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-16s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "litellm", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("HypoGen logging configured at %s level", settings.log_level)
