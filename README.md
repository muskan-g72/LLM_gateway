```bash
docker compose up --build
```

# Minimal LLM Gateway

A small FastAPI service that authenticates virtual keys, atomically enforces request budgets, records provider token usage in SQLite, and falls back from Groq to Gemini.

## Setup

```bash
cp .env.example .env
# Put a Groq key and a Google AI Studio key in .env.
docker compose up --build
```

The service listens on `http://localhost:8000`. SQLite data survives container restarts in the `gateway-data` volume. To reset all seeded-key usage, run `docker compose down -v` before starting again.

## Contract examples

```bash
curl -i http://localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer vk_open' \
  -H 'Content-Type: application/json' \
  -d '{"model":"any-string","messages":[{"role":"user","content":"hello"}]}'

curl 'http://localhost:8000/usage?key=vk_open'
```

Success has exactly the required shape:

```json
{"content":"...","usage":{"prompt_tokens":12,"completion_tokens":34}}
```

Missing or unknown keys return `401`; admitted requests after a key reaches its budget return `429`. The caller's `model` value is accepted but intentionally not sent upstream: gateway-owned `GROQ_MODEL` and `GEMINI_MODEL` make the fixed `any-string` contract work across providers.

## Seeded budgets

Budgets use **requests**, not currency or tokens.

| key | budget | expected behavior |
| --- | ---: | --- |
| `vk_open` | 50 | enough for about 50 requests |
| `vk_tiny` | 2 | two `200`s, then `429` |
| `vk_edge` | 1 | one request can be admitted |

`GET /usage` reports `spend == requests` and `remaining == budget - spend`; token counters are the upstream provider's measured values.

## Forced fallback

Start a fresh container with primary failures forced:

```bash
FORCE_PRIMARY_FAIL=1 docker compose up --build --force-recreate
```

Every Groq attempt is skipped and the same request is sent once to Gemini. If both providers fail, the gateway releases the reservation and returns `502`; there are no retries.

## Security and operational notes

- Provider credentials are read only from environment variables; `.env` is ignored by Git and excluded from the Docker build context.
- Upstream error bodies, URLs, headers, and exception details are never returned to callers.
- A `BEGIN IMMEDIATE` SQLite transaction serializes budget reservations. With a healthy provider, ten simultaneous requests against fresh `vk_edge` produce one `200` and nine `429`s.
-  Successful usage and budgets persist across restarts. One application worker keeps this take-home operationally simple; budget correctness does not depend on that choice because SQLite serializes reservations across threads and processes sharing the database file.