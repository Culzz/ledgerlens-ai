"""Application settings, read from the environment.

Every setting has a default, so the project runs with no ``.env`` at all. The
default LLM provider is ``fake``, which means an unconfigured checkout still
works end to end and can never spend money. Turning on a real provider is a
change to configuration, not to code.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://ledgerlens:ledgerlens@localhost:5432/ledgerlens"

    # --- LLM provider ---
    # "fake" runs offline. Anything else must speak the OpenAI chat-completions
    # shape; see app.services.llm.factory for the known names.
    llm_provider: str = "fake"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""
    llm_request_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_min_seconds_between_calls: float = 0.0
    llm_max_output_tokens: int | None = None
    # Free tiers cost nothing, so the cost column reads 0 by policy rather than
    # by price table. Set false to estimate against list prices instead.
    llm_free_tier: bool = True
    llm_monthly_budget_usd: float | None = 25.0
    llm_budget_ledger_path: str = "./data/llm_budget.json"

    # --- Embeddings (local, no API cost) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Uploads ---
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 25
    allowed_upload_extensions: str = ".pdf"

    # --- Evaluation ---
    finqa_data_dir: str = "./data/finqa"
    eval_seed: int = 42

    @field_validator("llm_monthly_budget_usd", "llm_max_output_tokens", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: Any) -> Any:
        # An empty env var should mean "not set", not a parse error.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def upload_extensions(self) -> tuple[str, ...]:
        return tuple(
            part.strip() for part in self.allowed_upload_extensions.split(",") if part.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
