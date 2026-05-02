from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_MODEL = "claude-3-5-haiku-latest"
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MAX_TOKENS = 350


class SupportAgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    anthropic_api_key: str | None = None
    model: str = DEFAULT_MODEL
    confidence_threshold: float = Field(default=DEFAULT_CONFIDENCE_THRESHOLD)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)

    @field_validator("anthropic_api_key")
    @classmethod
    def normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized_value = value.strip()
        return normalized_value or None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value:
            raise ValueError("SUPPORT_AGENT_MODEL must not be empty.")
        return value

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, value: float) -> float:
        normalized_value = float(value)
        if not 0.0 <= normalized_value <= 1.0:
            raise ValueError(
                "SUPPORT_AGENT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0."
            )
        return normalized_value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        normalized_value = int(value)
        if normalized_value < 1:
            raise ValueError("SUPPORT_AGENT_MAX_TOKENS must be greater than 0.")
        return normalized_value

    @classmethod
    def from_env(cls) -> SupportAgentSettings:
        return cls(
            anthropic_api_key=_read_optional_string_setting("ANTHROPIC_API_KEY"),
            model=_read_string_setting("SUPPORT_AGENT_MODEL", DEFAULT_MODEL),
            confidence_threshold=_read_float_setting(
                "SUPPORT_AGENT_CONFIDENCE_THRESHOLD",
                DEFAULT_CONFIDENCE_THRESHOLD,
            ),
            max_tokens=_read_int_setting(
                "SUPPORT_AGENT_MAX_TOKENS",
                DEFAULT_MAX_TOKENS,
            ),
        )


def _read_optional_string_setting(name: str) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    return raw_value


def _read_string_setting(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value


def _read_float_setting(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number between 0.0 and 1.0; got {raw_value!r}."
        ) from exc


def _read_int_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a positive integer; got {raw_value!r}."
        ) from exc
