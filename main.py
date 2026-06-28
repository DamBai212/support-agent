from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, Response, status
from fastapi.responses import PlainTextResponse

from classifier import FALLBACK_REASONS, SupportTriageClassifier
from metrics import SupportAgentMetrics
from router import (
    ALLOWED_PRIORITIES,
    ALLOWED_QUEUES,
    SupportPriority,
    SupportQueue,
    router as triage_router,
)
from settings import SupportAgentSettings


def build_health_payload() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Support agent is running",
    }


def build_ready_payload(classifier: SupportTriageClassifier) -> dict[str, str]:
    if classifier.can_classify_live:
        return {
            "status": "ok",
            "message": "Support agent is ready to classify live traffic",
            "classifier_status": "live",
        }

    return {
        "status": "degraded",
        "message": "Support agent is running in fallback-only mode",
        "classifier_status": "fallback_only",
    }


def create_metrics() -> SupportAgentMetrics:
    return SupportAgentMetrics(known_fallback_reasons=FALLBACK_REASONS)


def create_classifier(
    metrics: SupportAgentMetrics | None = None,
) -> SupportTriageClassifier:
    settings = SupportAgentSettings.from_env()
    return SupportTriageClassifier(
        metrics=metrics,
        settings=settings,
        allowed_queues=ALLOWED_QUEUES,
        allowed_priorities=ALLOWED_PRIORITIES,
        fallback_queue=SupportQueue.MANUAL_REVIEW.value,
        fallback_priority=SupportPriority.MEDIUM.value,
    )


def create_app(
    *,
    classifier: SupportTriageClassifier | None = None,
    metrics: SupportAgentMetrics | None = None,
) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Support Agent")
    app.state.metrics = metrics or create_metrics()
    app.state.triage_classifier = classifier or create_classifier(app.state.metrics)
    if classifier is not None and hasattr(classifier, "attach_metrics"):
        classifier.attach_metrics(app.state.metrics)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return build_health_payload()

    @app.get("/ready")
    def readiness_check(response: Response) -> dict[str, str]:
        payload = build_ready_payload(app.state.triage_classifier)
        if payload["status"] != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return payload

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return app.state.metrics.render(
            classifier_live=app.state.triage_classifier.can_classify_live
        )

    app.include_router(triage_router)
    return app


app = create_app()
