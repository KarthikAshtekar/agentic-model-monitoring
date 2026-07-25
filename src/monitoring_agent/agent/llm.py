"""Groq structured-output adapter and dependency-injection contracts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.prompts import (
    RECOMMENDATION_SYSTEM_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
)
from monitoring_agent.agent.schemas import AgentRecommendation, AgentTriage

StructuredSchema = TypeVar("StructuredSchema", bound=BaseModel)


@dataclass(slots=True)
class StructuredCallResult:
    """Normalised provider result without retaining hidden model content."""

    parsed: BaseModel | None
    metadata: dict[str, Any]
    error: dict[str, str] | None


class StructuredMonitoringLLM(Protocol):
    """Minimal graph dependency that live and fake clients implement."""

    provider_name: str
    model_name: str
    is_fake: bool

    def triage(self, payload: dict[str, Any]) -> StructuredCallResult:
        """Return a strict triage object or a captured failure."""

    def recommend(self, payload: dict[str, Any]) -> StructuredCallResult:
        """Return a strict recommendation object or a captured failure."""


def create_groq_llm(settings: AgentSettings) -> Any:
    """Construct the configured live ChatGroq model after checking the key."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.model,
        api_key=settings.require_groq_api_key(),
        temperature=settings.temperature,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _token_usage(raw: Any) -> dict[str, Any] | None:
    usage = getattr(raw, "usage_metadata", None)
    if usage:
        return dict(usage)
    response_metadata = getattr(raw, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    return dict(token_usage) if isinstance(token_usage, dict) else None


class GroqStructuredMonitoringLLM:
    """Live Groq adapter using strict provider-native JSON Schema outputs."""

    provider_name = "groq"
    is_fake = False

    def __init__(self, settings: AgentSettings) -> None:
        self.model_name = settings.model
        model = create_groq_llm(settings)
        self._triage = model.with_structured_output(
            AgentTriage,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        self._recommendation = model.with_structured_output(
            AgentRecommendation,
            method="json_schema",
            strict=True,
            include_raw=True,
        )

    def _invoke(
        self,
        runnable: Any,
        schema_name: str,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> StructuredCallResult:
        started = time.perf_counter()
        try:
            response = runnable.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=json.dumps(payload, sort_keys=True, default=str)
                    ),
                ]
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            raw = response.get("raw")
            parsed = response.get("parsed")
            parsing_error = response.get("parsing_error")
            metadata = {
                "provider": self.provider_name,
                "model": self.model_name,
                "schema": schema_name,
                "latency_ms": latency_ms,
                "token_usage": _token_usage(raw),
                "parse_success": parsed is not None and parsing_error is None,
                "is_fake": False,
            }
            if parsing_error is not None or parsed is None:
                error = parsing_error or ValueError("Provider returned no parsed output.")
                return StructuredCallResult(
                    parsed=None,
                    metadata=metadata,
                    error={
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
            return StructuredCallResult(parsed=parsed, metadata=metadata, error=None)
        except Exception as exc:  # provider failures must enter deterministic fallback
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return StructuredCallResult(
                parsed=None,
                metadata={
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "schema": schema_name,
                    "latency_ms": latency_ms,
                    "token_usage": None,
                    "parse_success": False,
                    "is_fake": False,
                },
                error={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )

    def triage(self, payload: dict[str, Any]) -> StructuredCallResult:
        """Invoke the strict triage schema."""
        return self._invoke(
            self._triage,
            "AgentTriage",
            TRIAGE_SYSTEM_PROMPT,
            payload,
        )

    def recommend(self, payload: dict[str, Any]) -> StructuredCallResult:
        """Invoke the strict recommendation schema."""
        return self._invoke(
            self._recommendation,
            "AgentRecommendation",
            RECOMMENDATION_SYSTEM_PROMPT,
            payload,
        )
