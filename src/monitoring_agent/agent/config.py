"""Environment-backed configuration for the single monitoring agent."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from monitoring_agent.paths import DEFAULT_CHECKPOINT_DB, PROJECT_ROOT


class AgentSettings(BaseSettings):
    """Configuration loaded from LLM_* variables and GROQ_API_KEY."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["groq"] = "groq"
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=1, ge=0, le=5)
    max_revision_attempts: int = Field(default=1, ge=0, le=2)
    checkpoint_backend: Literal["memory", "sqlite"] = Field(
        default="memory",
        validation_alias="AGENT_CHECKPOINT_BACKEND",
    )
    checkpoint_db: Path = Field(
        default=DEFAULT_CHECKPOINT_DB,
        validation_alias="AGENT_CHECKPOINT_DB",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GROQ_API_KEY",
        repr=False,
    )

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_zero(cls, value: float) -> float:
        """Keep the MVP deterministic at the provider boundary."""
        if value != 0:
            raise ValueError("LLM_TEMPERATURE must be 0 for the MVP.")
        return value

    def require_groq_api_key(self) -> str:
        """Return the live key only at client construction time."""
        if self.groq_api_key is None or not self.groq_api_key.get_secret_value().strip():
            raise ValueError(
                "GROQ_API_KEY is not configured. Set it in the environment or use "
                "--use-fake-llm for an offline demonstration."
            )
        return self.groq_api_key.get_secret_value()
