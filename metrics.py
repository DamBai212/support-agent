from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Iterable


class SupportAgentMetrics:
    def __init__(self, *, known_fallback_reasons: Iterable[str] = ()) -> None:
        self._lock = Lock()
        self._fallback_counts = Counter({reason: 0 for reason in known_fallback_reasons})

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self._fallback_counts[reason] += 1

    def render(self, *, classifier_live: bool) -> str:
        with self._lock:
            fallback_counts = dict(sorted(self._fallback_counts.items()))

        lines = [
            "# HELP support_agent_classifier_live Whether live model-backed triage is available.",
            "# TYPE support_agent_classifier_live gauge",
            f"support_agent_classifier_live {1 if classifier_live else 0}",
            "# HELP support_agent_triage_fallback_events_total Total number of fallback decisions.",
            "# TYPE support_agent_triage_fallback_events_total counter",
            f"support_agent_triage_fallback_events_total {sum(fallback_counts.values())}",
            "# HELP support_agent_triage_fallback_total Total number of fallback decisions by reason.",
            "# TYPE support_agent_triage_fallback_total counter",
        ]

        for reason, count in fallback_counts.items():
            lines.append(
                f'support_agent_triage_fallback_total{{reason="{reason}"}} {count}'
            )

        return "\n".join(lines) + "\n"
