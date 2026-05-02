from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from settings import SupportAgentSettings

logger = logging.getLogger(__name__)

FALLBACK_REASON_INVALID_MODEL_RESPONSE = "invalid_model_response"
FALLBACK_REASON_LOW_CONFIDENCE = "low_confidence"
FALLBACK_REASON_MISSING_API_KEY = "missing_api_key"
FALLBACK_REASON_MODEL_UNAVAILABLE = "model_unavailable"
FALLBACK_REASON_UNSUPPORTED_CLASSIFICATION = "unsupported_classification"


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
        settings: SupportAgentSettings | None = None,
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
        self.settings = self._resolve_settings(
            settings=settings,
            model=model,
            confidence_threshold=confidence_threshold,
            max_tokens=max_tokens,
        )
        self.model = self.settings.model
        self.confidence_threshold = self.settings.confidence_threshold
        self.max_tokens = self.settings.max_tokens
        self.client = client if client is not None else self._build_client()

    @property
    def can_classify_live(self) -> bool:
        return self.client is not None

    def classify_ticket(self, ticket: Mapping[str, Any]) -> TriageDecision:
        if self.client is None:
            return self._fallback(
                FALLBACK_REASON_MISSING_API_KEY,
                "The classifier is unavailable because the Anthropic API key is not configured."
            )

        prompt = self._build_prompt(ticket)
        try:
            raw_response = self._call_model(prompt)
            decision = self._parse_model_response(raw_response)
            return self._finalize_decision(decision)
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._fallback(
                FALLBACK_REASON_INVALID_MODEL_RESPONSE,
                "The classifier returned an invalid triage result and the ticket was routed to manual_review."
                ,
                error=exc,
            )
        except Exception as exc:
            return self._fallback(
                FALLBACK_REASON_MODEL_UNAVAILABLE,
                "The classifier was unavailable during triage and the ticket was routed to manual_review."
                ,
                error=exc,
            )

    def _build_client(self) -> Anthropic | None:
        api_key = self.settings.anthropic_api_key
        if not api_key:
            return None
        return Anthropic(api_key=api_key)

    @staticmethod
    def _resolve_settings(
        *,
        settings: SupportAgentSettings | None,
        model: str | None,
        confidence_threshold: float | None,
        max_tokens: int | None,
    ) -> SupportAgentSettings:
        resolved_settings = settings or SupportAgentSettings.from_env()
        overrides: dict[str, str | float | int] = {}
        if model is not None:
            overrides["model"] = model
        if confidence_threshold is not None:
            overrides["confidence_threshold"] = confidence_threshold
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        if not overrides:
            return resolved_settings
        return SupportAgentSettings.model_validate(
            resolved_settings.model_dump() | overrides
        )

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
                FALLBACK_REASON_UNSUPPORTED_CLASSIFICATION,
                "The classifier returned an unsupported queue or priority and the ticket was routed to manual_review."
            )

        normalized_rationale = self._normalize_rationale(decision.rationale)
        if decision.confidence < self.confidence_threshold:
            return self._fallback(
                FALLBACK_REASON_LOW_CONFIDENCE,
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

    def _fallback(
        self,
        reason: str,
        rationale: str,
        *,
        confidence: float = 0.0,
        error: Exception | None = None,
    ) -> TriageDecision:
        normalized_confidence = max(0.0, min(confidence, 1.0))
        self._log_fallback(reason, confidence=normalized_confidence, error=error)
        return TriageDecision(
            queue=self.fallback_queue,
            priority=self.fallback_priority,
            confidence=normalized_confidence,
            rationale=self._normalize_rationale(rationale),
            used_fallback=True,
        )

    def _log_fallback(
        self,
        reason: str,
        *,
        confidence: float,
        error: Exception | None,
    ) -> None:
        if error is None:
            logger.warning(
                "support_agent.triage_fallback reason=%s fallback_queue=%s fallback_priority=%s confidence=%.2f model=%s",
                reason,
                self.fallback_queue,
                self.fallback_priority,
                confidence,
                self.model,
            )
            return

        logger.warning(
            "support_agent.triage_fallback reason=%s fallback_queue=%s fallback_priority=%s confidence=%.2f model=%s error=%s",
            reason,
            self.fallback_queue,
            self.fallback_priority,
            confidence,
            self.model,
            type(error).__name__,
        )

    def _normalize_rationale(self, rationale: str) -> str:
        collapsed = " ".join(rationale.split())
        if not collapsed:
            collapsed = "The ticket was routed to manual_review."
        if len(collapsed) <= 280:
            return collapsed
        return f"{collapsed[:277].rstrip()}..."
