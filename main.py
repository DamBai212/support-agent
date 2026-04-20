from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from classifier import SupportTriageClassifier
from router import (
    ALLOWED_PRIORITIES,
    ALLOWED_QUEUES,
    SupportPriority,
    SupportQueue,
    router as triage_router,
)


def create_classifier() -> SupportTriageClassifier:
    return SupportTriageClassifier(
        allowed_queues=ALLOWED_QUEUES,
        allowed_priorities=ALLOWED_PRIORITIES,
        fallback_queue=SupportQueue.MANUAL_REVIEW.value,
        fallback_priority=SupportPriority.MEDIUM.value,
    )


def create_app(
    *, classifier: SupportTriageClassifier | None = None
) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Support Agent")
    app.state.triage_classifier = classifier or create_classifier()

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok", "message": "Support agent is running"}

    app.include_router(triage_router)
    return app


app = create_app()
