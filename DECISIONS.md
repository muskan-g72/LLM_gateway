# Decisions

1. **Stack, and why.** FastAPI keeps the HTTP contract explicit, SQLite supplies durable atomic transactions without infrastructure, and direct HTTP calls keep the provider layer visible.

2. **Request lifecycle.**

```text
client
  -> parse Bearer virtual key
  -> BEGIN IMMEDIATE; atomically reserve one request
  -> 401 unknown / 429 exhausted
  -> call Groq (or Gemini after primary failure)
  -> persist provider tokens and usage event
  -> return the fixed response shape
```

3. **Why enforce here?** Callers may be buggy, compromised, or simply unaware of shared spend. The gateway is the last common point before cost is created, so one rule protects every caller. Relying on callers would duplicate policy and turn budget enforcement into an honor system.

4. **Concurrency prediction.** Exactly **one** of ten parallel requests to a fresh `vk_edge` returns `200`; the other nine return `429`. `BEGIN IMMEDIATE` serializes the check-and-increment, so no second request can observe the old balance. If both provider attempts fail, the gateway releases its reservation, but already-rejected parallel calls are not retried. I seed `vk_tiny` at 2 because I read “exhausted within 3 identical requests” as two successes followed by a third-attempt `429`.

5. **Fallback policy.** Any primary `ProviderError`, including `FORCE_PRIMARY_FAIL=1`, causes one Gemini attempt so a transient or provider-specific failure remains invisible to the caller. If Gemini also fails, return `502` and release the request reservation because retries would add latency, cost, and ambiguity outside this brief.

6. **What I cut.** I cut streaming, retries, caching, a model router, admin APIs, migrations, and a UI because none improves the fixed grading path; SQLite event rows are sufficient for an auditable minimal log. The caller's `model` is validated but not forwarded because Groq and Gemini use different identifiers, so provider models are gateway-owned configuration.

7. **Least-certain decision.** Request-count budgets are deterministic and make admission explainable. Token budgets would align limits more closely with real provider cost. However, token cost is unknown until after generation unless I reserve an estimate, which either overshoots or rejects usable capacity. For this fixed exercise I chose concurrency correctness and transparent math over more realistic billing.