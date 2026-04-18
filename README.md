# Support Agent

`support-agent` is an LLM-powered internal tooling agent that classifies incoming support requests, generates structured responses, and routes work by intent. It is built to demonstrate real AI integration in an internal operations workflow, with production-minded validation, fallback handling, and error recovery around every model decision.

It accepts ticket text plus selected metadata, sends that context to an Anthropic model, and returns a structured routing decision with a queue, priority, confidence score, short rationale, and fallback flag.

## Tech Used

- Python
- LLM APIs
- Prompt engineering
- Structured output handling
- REST APIs
- Workflow orchestration

## What It Does

- Exposes a health endpoint for service checks
- Exposes a synchronous `POST /triage` endpoint for support ticket classification
- Validates request and response payloads with Pydantic
- Uses an in-code queue and priority taxonomy for consistent routing
- Falls back to `manual_review` when the model is unavailable, uncertain, or returns invalid output
- Runs unit tests in GitHub Actions on pushes and pull requests to `main`

## Why This Project Matters

- Demonstrates a prompt-engineered classification pipeline with structured outputs, error recovery, and feedback-oriented fallback behavior
- Shows deliberate use of LLM inference where judgment and ambiguity handling add value, while keeping deterministic validation and routing safeguards in code
- Automates a previously manual triage workflow, which is the kind of internal tooling that helps teams move faster with more confidence

## Queue And Priority Taxonomy

Supported queues:

- `billing`
- `technical`
- `account`
- `bug`
- `general`
- `manual_review`

Supported priorities:

- `low`
- `medium`
- `high`
- `urgent`

## API Endpoints

### `GET /health`

Returns a simple health payload:

```json
{
  "status": "ok",
  "message": "Support agent is running"
}
```

### `POST /triage`

Accepts a support ticket and returns a structured triage decision.

Example request:

```json
{
  "subject": "Production API outage",
  "body": "Our production integration is failing for every request and customers cannot log in.",
  "customer_tier": "enterprise",
  "channel": "phone",
  "account_id": "acct_12345",
  "language": "en",
  "context": {
    "region": "eu-west-2",
    "product_area": "auth"
  }
}
```

Example response:

```json
{
  "queue": "technical",
  "priority": "urgent",
  "confidence": 0.91,
  "rationale": "An enterprise customer reported a severe production outage affecting authentication.",
  "used_fallback": false
}
```

Request fields:

- `subject`: required string, 3 to 200 characters
- `body`: required string, 10 to 4000 characters
- `customer_tier`: optional enum of `free`, `standard`, `pro`, `business`, `enterprise`
- `channel`: optional enum of `email`, `chat`, `phone`, `web`, `api`, `social`
- `account_id`: optional string
- `language`: optional string
- `context`: optional string-to-string object for extra metadata

Response fields:

- `queue`: chosen support queue
- `priority`: chosen urgency level
- `confidence`: float from `0.0` to `1.0`
- `rationale`: short human-readable explanation
- `used_fallback`: `true` when the result came from safe fallback handling instead of a trusted model decision

## Fallback Behavior

The service is designed to fail safely.

- If `ANTHROPIC_API_KEY` is missing, requests still succeed but return a fallback decision
- If the model returns malformed JSON, unsupported values, or a low-confidence result, the service routes to `manual_review`
- The current fallback priority is `medium`
- The confidence threshold defaults to `0.55`

## Local Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

The app loads a local `.env` file automatically through `python-dotenv`.

Example `.env`:

```bash
ANTHROPIC_API_KEY=your_key_here
SUPPORT_AGENT_MODEL=claude-3-5-haiku-latest
SUPPORT_AGENT_CONFIDENCE_THRESHOLD=0.55
SUPPORT_AGENT_MAX_TOKENS=350
```

Environment variables:

- `ANTHROPIC_API_KEY`: Anthropic API key used for live classification
- `SUPPORT_AGENT_MODEL`: model name to call, defaults to `claude-3-5-haiku-latest`
- `SUPPORT_AGENT_CONFIDENCE_THRESHOLD`: minimum confidence required to trust a model result
- `SUPPORT_AGENT_MAX_TOKENS`: max response tokens for the model call

### 4. Start the API

```bash
uvicorn main:app --reload
```

Once the server is running:

- API base URL: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

## Example `curl` Commands

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Triage request:

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Invoice does not match our plan",
    "body": "We were charged for seats we do not have on our account.",
    "customer_tier": "business",
    "channel": "email",
    "context": {
      "invoice_month": "2026-04"
    }
  }'
```

## Running Tests

Run the unit suite locally with:

```bash
python -m unittest discover -s tests -v
```

The test suite covers:

- health endpoint behavior
- successful triage responses
- request validation failures
- low-confidence fallback
- invalid model output fallback
- metadata flowing into prompt construction

## CI

GitHub Actions is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

On every push or pull request to `main`, the workflow:

- checks out the repository
- installs dependencies from `requirements.txt`
- runs `python -m unittest discover -s tests -v`

## Project Layout

```text
.
|-- .github/workflows/ci.yml
|-- classifier.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- router.py
`-- tests/
    `-- test_support_agent.py
```

## Implementation Notes

- `main.py` creates the FastAPI app and registers the triage router
- `router.py` defines the request and response models plus the `/triage` endpoint
- `classifier.py` handles prompt construction, Anthropic API calls, JSON parsing, validation, and fallback logic
- The service is stateless and does not store tickets or decisions
