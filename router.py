from __future__ import annotations

from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from classifier import SupportTriageClassifier


class SupportQueue(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    BUG = "bug"
    GENERAL = "general"
    MANUAL_REVIEW = "manual_review"


class SupportPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CustomerTier(str, Enum):
    FREE = "free"
    STANDARD = "standard"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class SupportChannel(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    PHONE = "phone"
    WEB = "web"
    API = "api"
    SOCIAL = "social"


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=10, max_length=4000)
    customer_tier: CustomerTier | None = None
    channel: SupportChannel | None = None
    account_id: str | None = Field(default=None, min_length=1, max_length=100)
    language: str | None = Field(default=None, min_length=2, max_length=20)
    context: dict[str, str] = Field(default_factory=dict)


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: SupportQueue
    priority: SupportPriority
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=280)
    used_fallback: bool


ALLOWED_QUEUES = tuple(queue.value for queue in SupportQueue)
ALLOWED_PRIORITIES = tuple(priority.value for priority in SupportPriority)

classifier_service = SupportTriageClassifier(
    allowed_queues=ALLOWED_QUEUES,
    allowed_priorities=ALLOWED_PRIORITIES,
    fallback_queue=SupportQueue.MANUAL_REVIEW.value,
    fallback_priority=SupportPriority.MEDIUM.value,
)


def get_classifier() -> SupportTriageClassifier:
    return classifier_service


ClassifierDependency = Annotated[SupportTriageClassifier, Depends(get_classifier)]

router = APIRouter()


@router.post("/triage", response_model=TriageResponse, tags=["triage"])
def triage_ticket(
    ticket: TriageRequest,
    triage_classifier: ClassifierDependency,
) -> TriageResponse:
    decision = triage_classifier.classify_ticket(
        ticket.model_dump(mode="json", exclude_none=True)
    )
    return TriageResponse.model_validate(decision.model_dump())
