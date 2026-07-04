from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Iterable


class SupportAgentMetrics:
    def __init__(self, *, known_fallback_reasons: Iterable[str] = ()) -> None:
        self._lock = Lock()
        self._fallback_counts = Counter({reason: 0 for reason in known_fallback_reasons})
        self._decision_counts: Counter[tuple[str, str, str]] = Counter()

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self._fallback_counts[reason] += 1

    def record_decision(
        self,
        *,
        queue: str,
        priority: str,
        used_fallback: bool,
    ) -> None:
        normalized_key = (queue, priority, str(used_fallback).lower())
        with self._lock:
            self._decision_counts[normalized_key] += 1

    def render(self, *, classifier_live: bool) -> str:
        with self._lock:
            fallback_counts = dict(sorted(self._fallback_counts.items()))
            decision_counts = dict(sorted(self._decision_counts.items()))

        lines = [
            "# HELP support_agent_classifier_live Whether live model-backed triage is available.",
            "# TYPE support_agent_classifier_live gauge",
            f"support_agent_classifier_live {1 if classifier_live else 0}",
            "# HELP support_agent_triage_decisions_total Total number of triage decisions returned.",
            "# TYPE support_agent_triage_decisions_total counter",
            f"support_agent_triage_decisions_total {sum(decision_counts.values())}",
            "# HELP support_agent_triage_decision_total Total number of triage decisions by queue, priority, and fallback status.",
            "# TYPE support_agent_triage_decision_total counter",
            "# HELP support_agent_triage_fallback_events_total Total number of fallback decisions.",
            "# TYPE support_agent_triage_fallback_events_total counter",
            f"support_agent_triage_fallback_events_total {sum(fallback_counts.values())}",
            "# HELP support_agent_triage_fallback_total Total number of fallback decisions by reason.",
            "# TYPE support_agent_triage_fallback_total counter",
        ]

        for (queue, priority, used_fallback), count in decision_counts.items():
            lines.append(
                f'support_agent_triage_decision_total{{queue="{queue}",priority="{priority}",used_fallback="{used_fallback}"}} {count}'
            )

        for reason, count in fallback_counts.items():
            lines.append(
                f'support_agent_triage_fallback_total{{reason="{reason}"}} {count}'
            )

        return "\n".join(lines) + "\n"
