from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_MODEL = "claude-3-5-haiku-latest"
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MAX_TOKENS = 350


class ModelTriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    queue: str
    priority: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=280)


class TriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    queue: str
    priority: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=280)
    used_fallback: bool = False


class SupportTriageClassifier:
    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        confidence_threshold: float | None = None,
        max_tokens: int | None = None,
        allowed_queues: Sequence[str] | None = None,
        allowed_priorities: Sequence[str] | None = None,
        fallback_queue: str = "manual_review",
        fallback_priority: str = "medium",
    ) -> None:
        self.allowed_queues = tuple(
            allowed_queues
            or ("billing", "technical", "account", "bug", "general", "manual_review")
        )
        self.allowed_priorities = tuple(
            allowed_priorities or ("low", "medium", "high", "urgent")
        )
        self.fallback_queue = fallback_queue
        self.fallback_priority = fallback_priority
        self.model = model or os.getenv("SUPPORT_AGENT_MODEL", DEFAULT_MODEL)
        self.confidence_threshold = self._resolve_confidence_threshold(
            confidence_threshold
        )
        self.max_tokens = self._resolve_max_tokens(max_tokens)
        self.client = client if client is not None else self._build_client()

    def classify_ticket(self, ticket: Mapping[str, Any]) -> TriageDecision:
        if self.client is None:
            return self._fallback(
                "The classifier is unavailable because the Anthropic API key is not configured."
            )

        prompt = self._build_prompt(ticket)
        try:
            raw_response = self._call_model(prompt)
            decision = self._parse_model_response(raw_response)
            return self._finalize_decision(decision)
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
            return self._fallback(
                "The classifier returned an invalid triage result and the ticket was routed to manual_review."
            )
        except Exception:
            return self._fallback(
                "The classifier was unavailable during triage and the ticket was routed to manual_review."
            )

    def _build_client(self) -> Anthropic | None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return Anthropic(api_key=api_key)

    @staticmethod
    def _resolve_confidence_threshold(confidence_threshold: float | None) -> float:
        if confidence_threshold is None:
            confidence_threshold = SupportTriageClassifier._read_float_setting(
                "SUPPORT_AGENT_CONFIDENCE_THRESHOLD",
                DEFAULT_CONFIDENCE_THRESHOLD,
            )
        return SupportTriageClassifier._validate_confidence_threshold(
            confidence_threshold
        )

    @staticmethod
    def _resolve_max_tokens(max_tokens: int | None) -> int:
        if max_tokens is None:
            max_tokens = SupportTriageClassifier._read_int_setting(
                "SUPPORT_AGENT_MAX_TOKENS",
                DEFAULT_MAX_TOKENS,
            )
        return SupportTriageClassifier._validate_max_tokens(max_tokens)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _validate_confidence_threshold(value: float) -> float:
        normalized_value = float(value)
        if not 0.0 <= normalized_value <= 1.0:
            raise ValueError(
                "SUPPORT_AGENT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0."
            )
        return normalized_value

    @staticmethod
    def _validate_max_tokens(value: int) -> int:
        normalized_value = int(value)
        if normalized_value < 1:
            raise ValueError("SUPPORT_AGENT_MAX_TOKENS must be greater than 0.")
        return normalized_value

    def _build_prompt(self, ticket: Mapping[str, Any]) -> str:
        ticket_payload = json.dumps(ticket, indent=2, sort_keys=True)
        return f"""
You are an internal support triage assistant.
Classify the ticket into exactly one queue and one priority.
Use the ticket text plus metadata to determine urgency. Enterprise customers and
phone or API incidents can justify a higher priority when the issue is severe.

Rules:
- Choose queue from: {", ".join(self.allowed_queues)}
- Choose priority from: {", ".join(self.allowed_priorities)}
- Keep rationale to one sentence and under 280 characters.
- Confidence must be a number between 0 and 1.
- If the issue is ambiguous, choose manual_review.
- Respond with JSON only. Do not wrap it in markdown.

Return this exact JSON shape:
{{
  "queue": "billing",
  "priority": "medium",
  "confidence": 0.84,
  "rationale": "A short explanation of the decision."
}}

Ticket:
{ticket_payload}
""".strip()

    def _call_model(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system="Return only valid JSON that matches the requested schema.",
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_text(response)

    def _extract_text(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if content is None:
            raise ValueError("Model response did not contain content.")

        text_blocks: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_blocks.append(block.text)

        if not text_blocks:
            raise ValueError("Model response did not contain any text blocks.")

        return "".join(text_blocks)

    def _parse_model_response(self, raw_response: str) -> ModelTriageDecision:
        json_payload = self._extract_json_object(raw_response)
        parsed_response = json.loads(json_payload)
        return ModelTriageDecision.model_validate(parsed_response)

    def _extract_json_object(self, raw_response: str) -> str:
        stripped_response = raw_response.strip()
        if stripped_response.startswith("```"):
            lines = stripped_response.splitlines()
            stripped_response = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        start_index = stripped_response.find("{")
        end_index = stripped_response.rfind("}")
        if start_index == -1 or end_index == -1:
            raise ValueError("Model response did not contain JSON.")

        return stripped_response[start_index : end_index + 1]

    def _finalize_decision(self, decision: ModelTriageDecision) -> TriageDecision:
        queue = decision.queue.lower()
        priority = decision.priority.lower()

        if queue not in self.allowed_queues or priority not in self.allowed_priorities:
            return self._fallback(
                "The classifier returned an unsupported queue or priority and the ticket was routed to manual_review."
            )

        normalized_rationale = self._normalize_rationale(decision.rationale)
        if decision.confidence < self.confidence_threshold:
            return self._fallback(
                (
                    f"Model confidence was {decision.confidence:.2f}, so the ticket "
                    "was routed to manual_review."
                ),
                confidence=decision.confidence,
            )

        return TriageDecision(
            queue=queue,
            priority=priority,
            confidence=decision.confidence,
            rationale=normalized_rationale,
            used_fallback=False,
        )

    def _fallback(self, rationale: str, *, confidence: float = 0.0) -> TriageDecision:
        return TriageDecision(
            queue=self.fallback_queue,
            priority=self.fallback_priority,
            confidence=max(0.0, min(confidence, 1.0)),
            rationale=self._normalize_rationale(rationale),
            used_fallback=True,
        )

    def _normalize_rationale(self, rationale: str) -> str:
        collapsed = " ".join(rationale.split())
        if not collapsed:
            collapsed = "The ticket was routed to manual_review."
        if len(collapsed) <= 280:
            return collapsed
        return f"{collapsed[:277].rstrip()}..."
