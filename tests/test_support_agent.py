from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from classifier import SupportTriageClassifier, TriageDecision
from main import app, create_app
from router import get_classifier


class FakeMessagesAPI:
    def __init__(self, responder):
        self.responder = responder
        self.last_prompt: str | None = None

    def create(self, *, model, max_tokens, temperature, system, messages):
        del model, max_tokens, temperature, system
        self.last_prompt = messages[0]["content"]
        response_text = self.responder(self.last_prompt)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=response_text)]
        )


class FakeAnthropicClient:
    def __init__(self, responder):
        self.messages = FakeMessagesAPI(responder)


class ClassifierStub:
    def __init__(self, decision: TriageDecision):
        self.decision = decision
        self.last_ticket = None

    def classify_ticket(self, ticket):
        self.last_ticket = ticket
        return self.decision


class SupportTriageClassifierTests(unittest.TestCase):
    def test_classifies_clear_billing_ticket(self):
        client = FakeAnthropicClient(
            lambda prompt: json.dumps(
                {
                    "queue": "billing",
                    "priority": "high",
                    "confidence": 0.93,
                    "rationale": "The ticket is about duplicate billing charges.",
                }
            )
        )
        classifier = SupportTriageClassifier(client=client, confidence_threshold=0.55)

        decision = classifier.classify_ticket(
            {
                "subject": "Charged twice for March invoice",
                "body": "I was billed two times for the same plan this month.",
                "customer_tier": "standard",
                "channel": "email",
            }
        )

        self.assertEqual(decision.queue, "billing")
        self.assertEqual(decision.priority, "high")
        self.assertEqual(decision.confidence, 0.93)
        self.assertFalse(decision.used_fallback)

    def test_metadata_is_present_in_prompt_for_priority_decisions(self):
        def responder(prompt: str) -> str:
            self.assertIn('"customer_tier": "enterprise"', prompt)
            self.assertIn('"channel": "phone"', prompt)
            return json.dumps(
                {
                    "queue": "technical",
                    "priority": "urgent",
                    "confidence": 0.89,
                    "rationale": "An enterprise customer reported a severe outage over the phone.",
                }
            )

        client = FakeAnthropicClient(responder)
        classifier = SupportTriageClassifier(client=client, confidence_threshold=0.55)

        decision = classifier.classify_ticket(
            {
                "subject": "API outage",
                "body": "Our production API calls are failing for all customers.",
                "customer_tier": "enterprise",
                "channel": "phone",
            }
        )

        self.assertEqual(decision.priority, "urgent")
        self.assertFalse(decision.used_fallback)

    def test_low_confidence_ticket_falls_back_to_manual_review(self):
        client = FakeAnthropicClient(
            lambda prompt: json.dumps(
                {
                    "queue": "general",
                    "priority": "low",
                    "confidence": 0.28,
                    "rationale": "The issue is unclear from the ticket content.",
                }
            )
        )
        classifier = SupportTriageClassifier(client=client, confidence_threshold=0.55)

        decision = classifier.classify_ticket(
            {
                "subject": "Question",
                "body": "Something happened and I am not sure what to ask for.",
            }
        )

        self.assertEqual(decision.queue, "manual_review")
        self.assertEqual(decision.priority, "medium")
        self.assertTrue(decision.used_fallback)
        self.assertAlmostEqual(decision.confidence, 0.28)

    def test_invalid_model_output_falls_back_to_manual_review(self):
        client = FakeAnthropicClient(lambda prompt: "this is not valid json")
        classifier = SupportTriageClassifier(client=client, confidence_threshold=0.55)

        decision = classifier.classify_ticket(
            {
                "subject": "Need help",
                "body": "The app is acting strangely and I need support.",
            }
        )

        self.assertEqual(decision.queue, "manual_review")
        self.assertEqual(decision.priority, "medium")
        self.assertTrue(decision.used_fallback)

    def test_missing_api_key_falls_back_to_manual_review(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            classifier = SupportTriageClassifier(
                confidence_threshold=0.55,
                max_tokens=350,
            )

        decision = classifier.classify_ticket(
            {
                "subject": "Cannot reset password",
                "body": "Password reset emails are not arriving for my account.",
            }
        )

        self.assertEqual(decision.queue, "manual_review")
        self.assertEqual(decision.priority, "medium")
        self.assertTrue(decision.used_fallback)
        self.assertIn("API key is not configured", decision.rationale)

    def test_unsupported_model_values_fall_back_to_manual_review(self):
        for invalid_payload in (
            {
                "queue": "sales",
                "priority": "high",
                "confidence": 0.92,
                "rationale": "This should not use an unsupported queue.",
            },
            {
                "queue": "billing",
                "priority": "critical",
                "confidence": 0.92,
                "rationale": "This should not use an unsupported priority.",
            },
        ):
            with self.subTest(invalid_payload=invalid_payload):
                client = FakeAnthropicClient(
                    lambda prompt, invalid_payload=invalid_payload: json.dumps(
                        invalid_payload
                    )
                )
                classifier = SupportTriageClassifier(
                    client=client,
                    confidence_threshold=0.55,
                )

                decision = classifier.classify_ticket(
                    {
                        "subject": "Unexpected response",
                        "body": "The app produced an unexpected classification.",
                    }
                )

                self.assertEqual(decision.queue, "manual_review")
                self.assertEqual(decision.priority, "medium")
                self.assertTrue(decision.used_fallback)

    def test_invalid_confidence_threshold_env_raises_clear_error(self):
        with patch.dict(
            os.environ,
            {"SUPPORT_AGENT_CONFIDENCE_THRESHOLD": "not-a-number"},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "SUPPORT_AGENT_CONFIDENCE_THRESHOLD",
            ):
                SupportTriageClassifier()

        with patch.dict(
            os.environ,
            {"SUPPORT_AGENT_CONFIDENCE_THRESHOLD": "1.5"},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "SUPPORT_AGENT_CONFIDENCE_THRESHOLD",
            ):
                SupportTriageClassifier()

    def test_invalid_max_tokens_env_raises_clear_error(self):
        with patch.dict(
            os.environ,
            {"SUPPORT_AGENT_MAX_TOKENS": "zero"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "SUPPORT_AGENT_MAX_TOKENS"):
                SupportTriageClassifier()

        with patch.dict(
            os.environ,
            {"SUPPORT_AGENT_MAX_TOKENS": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "SUPPORT_AGENT_MAX_TOKENS"):
                SupportTriageClassifier()


class SupportAgentApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_health_endpoint_returns_success(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "message": "Support agent is running"},
        )

    def test_triage_endpoint_returns_structured_response(self):
        stub = ClassifierStub(
            TriageDecision(
                queue="billing",
                priority="high",
                confidence=0.91,
                rationale="The ticket describes a clear invoice issue for a premium customer.",
                used_fallback=False,
            )
        )
        app.dependency_overrides[get_classifier] = lambda: stub

        response = self.client.post(
            "/triage",
            json={
                "subject": "Invoice mismatch",
                "body": "My invoice shows a plan I never purchased.",
                "customer_tier": "enterprise",
                "channel": "email",
                "account_id": "acct_123",
                "context": {"region": "eu-west-2"},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["queue"], "billing")
        self.assertEqual(response.json()["priority"], "high")
        self.assertEqual(stub.last_ticket["customer_tier"], "enterprise")
        self.assertEqual(stub.last_ticket["context"]["region"], "eu-west-2")

    def test_triage_endpoint_requires_body(self):
        response = self.client.post(
            "/triage",
            json={"subject": "Missing body"},
        )

        self.assertEqual(response.status_code, 422)

    def test_triage_endpoint_rejects_invalid_metadata_shape(self):
        response = self.client.post(
            "/triage",
            json={
                "subject": "Bad metadata",
                "body": "This ticket body is long enough to pass validation.",
                "context": ["not", "a", "dict"],
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_create_app_wires_the_app_owned_classifier(self):
        stub = ClassifierStub(
            TriageDecision(
                queue="technical",
                priority="urgent",
                confidence=0.97,
                rationale="The stubbed classifier was used for this outage.",
                used_fallback=False,
            )
        )
        test_app = create_app(classifier=stub)
        client = TestClient(test_app)

        response = client.post(
            "/triage",
            json={
                "subject": "Service outage",
                "body": "The production service is unavailable for every request.",
                "channel": "api",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["queue"], "technical")
        self.assertEqual(stub.last_ticket["channel"], "api")
