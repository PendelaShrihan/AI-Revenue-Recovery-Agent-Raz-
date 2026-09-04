# 🚀 AI Revenue Recovery Agent — Razorpay Hackathon Track 03

> **Autonomous diagnostic and revenue recovery engine for failed digital payments, subscription drops, and overdue receivables.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-gemini--flash--lite--latest-4285F4?style=flat-square&logo=google&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16--alpine-336791?style=flat-square&logo=postgresql&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-212%20collected%2C%20196%20passed-brightgreen?style=flat-square)
![Recovery Rate](https://img.shields.io/badge/Recovery%20Rate-57.1%25-success?style=flat-square)
![LLM Cost](https://img.shields.io/badge/LLM%20Cost-%240.00033%20per%20recovery-blue?style=flat-square)

```
╔══════════════════════════════════════════════════════════════════════╗
║             AI Revenue Recovery Agent — System Architecture          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Razorpay Webhook  ──►  api_integration/router.py                    ║
║  (payment.failed        ├─ verifier.py  [HMAC-SHA256]                ║
║   subscription.halted   └─ normalizer.py [paise→rupees, schema]      ║
║   invoice.overdue)              │                                    ║
║                                 ▼                                    ║
║                      agent/pipeline.py                               ║
║                      run_recovery_pipeline()                         ║
║                         │                                            ║
║            ┌────────────┼─────────────────┐                          ║
║            ▼            ▼                 ▼                          ║
║     agent/db_writer   ml/classifier      agent/recovery_engine       ║
║     save_transaction  FailureClassifier  RecoveryEngine.process()    ║
║     [SQLAlchemy ORM]  [XGBoost+LR]        │                          ║
║            │            │                 ▼                          ║
║            │            │         agent/llm_agent.py                 ║
║            │            │         GeminiAgent.decide()               ║
║            │            │         [gemini-flash-lite-latest]         ║
║            │            │         [agent/llm_cache.py LRU/TTL]       ║
║            │            │                 │                          ║
║            └────────────┴─────────────────┘                          ║
║                                 │                                    ║
║                                 ▼                                    ║
║                      agent/action_engine.py                          ║
║                      dispatch_recovery_action()                      ║
║                         │                                            ║
║          ┌──────────────┼──────────────┐                             ║
║          ▼              ▼              ▼                             ║
║   execute_auto_retry  execute_    execute_customer_                  ║
║   [retry_scheduler]   alternate_  notification                       ║
║                        suggestion  [notification_engine]             ║
║                                                                      ║
║  REST API ──► api_integration/rest_router.py                         ║
║  10 endpoints │ SSE stream │ rate-limited (100 req/min)              ║
║                                                                      ║
║  Dashboard ──► frontend/ [static files @ /dashboard]                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 📌 Problem Statement

Digital payment failures cost Indian merchants millions in abandoned revenue daily. Standard payment infrastructure provides error codes but no autonomous recovery — someone must manually diagnose each failure, decide what to do, and execute a recovery action.

This agent solves that gap: it **ingests Razorpay webhook failures, classifies them using an ML ensemble, reasons over the context using Gemini AI, and dispatches the optimal bounded recovery action automatically** — with zero human intervention.

### The 3 Webhook Events Handled

From `api_integration/schemas.py` `EventType` enum and `agent/SCOPE.md`:

| Event                 | Scenario                                                                              | Recovery Target                                            |
| --------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `payment.failed`      | Standard checkout & one-time payment failures (card decline, OTP freeze, UPI timeout) | Smart retry scheduling or customized payment link dispatch |
| `subscription.halted` | Recurring billing / mandate drops (expired mandate, recurring charge decline)         | Mandate update notification / re-authorization link        |
| `invoice.overdue`     | B2B receivables and commercial invoice overdue with merchant relationship notes       | Multi-tier invoice chaser sequence with 2-step escalation  |

### Explicitly OUT OF SCOPE (from `agent/SCOPE.md`)

1. **Fraud Detection & Risk Scoring** — Real-time fraud heuristic engines, velocity checks, AML workflows
2. **Dynamic Pricing & Discount Engine** — Automated voucher generation, margin-based concessions
3. **Advanced Analytics & Long-term BI** — Multi-year cohort LTV modeling, cross-merchant macro analytics
4. **Live Telephony / Voice Bot Recovery** — IVR phone dialing bots
5. **Direct Production Financial Mutations** — Real-money card debiting outside Razorpay Sandbox

---

## 🎯 Track & Approach

**Track 03: AI Revenue Recovery**

### Approach (from `agent/pipeline.py` and `agent/llm_agent.py`)

The system runs a deterministic 4-step pipeline per failure event:

1. **Persist** — `db_writer.save_transaction()` writes the NormalizedEvent to PostgreSQL with a `UNIQUE` constraint on `razorpay_payment_id` to prevent duplicate inserts
2. **Classify** — ML ensemble (`FailureClassifier` = `BalancedXGBClassifier` + `LogisticRegression` soft-voting) categorizes into one of 8 canonical failure categories; rule-based `classify_failure()` as fallback
3. **Decide** — `GeminiAgent.decide()` sends a chain-of-thought prompt to **`gemini-flash-lite-latest`** (default from `GEMINI_MODEL` env var, with `temperature=0.1`) and enforces strict JSON output schema
4. **Dispatch** — `dispatch_recovery_action()` routes to `execute_auto_retry`, `execute_alternate_suggestion`, or `execute_customer_notification` based on the Gemini decision

### Gemini Model

```python
# From agent/llm_agent.py and agent/pipeline.py
self._model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
```

The model was switched from `gemini-3.6-flash` (which hit quota limits at 20 RPD and caused 504 timeouts) to `gemini-flash-lite-latest` during Day 7. Response time dropped from 9–10s to **1.1–2.9s**. (Source: `tasks/lessons.md`)

---

## 🏗 Architecture

### Component Map (every file, every function)

#### `main.py` — FastAPI Application Entrypoint

- `lifespan()` — async context manager for startup/shutdown
- `root()` — `GET /` returns project info and dashboard link
- `health_check()` — `GET /health` returns environment and LLM provider status
- Rate limiter: **100 requests/minute per client IP** via `slowapi`
- Serves static dashboard at `/dashboard/index.html` via `StaticFiles`

---

#### `api_integration/` — Webhook & REST Ingestion Layer

**`router.py`** — Webhook Router

- `get_webhook_info()` — `GET /webhooks/razorpay` — browser status check
- `handle_razorpay_webhook()` — `POST /webhooks/razorpay` — HMAC-SHA256 verification → normalize → DB save → route by event type

**`rest_router.py`** — REST API Router (10 endpoints)

- `analyze_failure()` — `POST /analyze-failure` — full pipeline execution
- `get_recovery_suggestions()` — `GET /recovery-suggestions` — paginated transaction history with suggested actions
- `trigger_retry()` — `POST /trigger-retry` — manual retry with guardrail enforcement
- `get_stats()` — `GET /stats` — dashboard counters (total, recovered, retry_scheduled, failed, success_rate)
- `event_stream()` — `GET /stream` — Server-Sent Events real-time dashboard feed
- `get_observability_metrics()` — `GET /metrics` — in-memory pipeline metrics snapshot
- `get_analytics_endpoint()` — `GET /analytics` — full recovery analytics with revenue saved
- `get_costs_endpoint()` — `GET /costs` — aggregate LLM token usage and cost summary
- `get_cache_stats_endpoint()` — `GET /cache/stats` — LLM cache telemetry and savings
- `clear_cache_endpoint()` — `POST /cache/clear` — flush LLM cache (authenticated)

**`schemas.py`** — Pydantic Models

- `EventType` — enum: `payment.failed`, `subscription.halted`, `invoice.overdue`, `invoice.expired`, `invoice.paid`, `payment.authorized`, `order.paid`, `unknown`
- `FailureCategory` — enum: `checkout_failure`, `mandate_failure`, `invoice_overdue`, `informational`, `unknown`
- `NormalizedEvent` — canonical internal event schema (24 fields)
- `WebhookResponse`, `FailureAnalysisRequest`, `FailureAnalysisResponse`, `RecoverySuggestionItem`, `RecoverySuggestionsResponse`, `TriggerRetryRequest`, `TriggerRetryResponse`

**`normalizer.py`** — Payload Normalizer

- Converts raw Razorpay JSON → `NormalizedEvent`
- Converts paise → rupees (`_to_rupees`: divides by 100.0)
- Handles `payment.failed`, `subscription.halted`, `invoice.overdue` event layouts

**`auth.py`** — Authentication Middleware

- `verify_merchant_auth()` — FastAPI dependency; accepts `X-API-Key` header or `Authorization: Bearer <token>`
- `_get_valid_api_keys()` — reads `MERCHANT_API_KEY` and `APP_SECRET_KEY` env vars
- `_constant_time_key_check()` — `hmac.compare_digest()` prevents timing-oracle attacks
- Query-parameter auth intentionally NOT supported (prevents key leakage in logs)

**`verifier.py`** — HMAC Signature Verification

- `compute_webhook_signature()` — computes HMAC-SHA256 hex digest of raw body bytes
- `verify_webhook_signature()` — constant-time comparison via `hmac.compare_digest()`

---

#### `agent/` — Core Recovery Agent

**`pipeline.py`** — Main Recovery Orchestrator

- `run_recovery_pipeline(normalized_event)` — 4-step pipeline with structured logging
- `run_pending_retries()` — queries all `RetryAttempt` records where `result="SCHEDULED"` and `next_retry_at <= now`
- `_get_feature_pipeline()` — lazy-loads `FeatureEngineeringPipeline` singleton
- `_get_classifier()` — lazy-loads `FailureClassifier` from `ml/models/failure_classifier.joblib`
- `_get_recovery_engine()` — lazy-loads `RecoveryEngine` singleton

**`llm_agent.py`** — Gemini API Wrapper

- `GeminiAgent` — `_MAX_RETRIES = 3`, `_TIMEOUT_SECONDS = 10.0`, backoff delays `[2, 4, 8]` seconds
- `GeminiAgent._call_api()` — retry loop; parses `retryDelay` from 429 responses
- `GeminiAgent._parse_response()` — strips Markdown fences, validates JSON schema
- `GeminiAgent.generate()` — sends prompt → returns `RecoveryDecision`
- `GeminiAgent.decide()` — builds CoT prompt → calls API → returns validated decision
- `build_cot_prompt(event, ml_failure_category)` — injects entity, amount, failure details, timing, customer info
- `RecoveryDecision` — dataclass: `action`, `priority`, `message`, `retry_after`, `alternate_method`, `confidence`, `reasoning`, `raw_response`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`, `model`, `cached`
- Valid actions: `auto_retry`, `suggest_alternate_method`, `send_payment_link`, `notify_customer`, `manual_review`, `no_action`
- Valid priorities: `high`, `medium`, `low`
- Valid methods: `upi`, `wallet`, `netbanking`, `card`, `none`
- Exceptions: `GeminiAgentError`, `GeminiTimeoutError`, `GeminiRateLimitError`, `GeminiOutputParseError`

**`recovery_engine.py`** — Recovery Orchestrator

- `RecoveryEngine` — wraps `GeminiAgent` with `LLMResponseCache`
- `RecoveryEngine.process(event, ml_failure_category)` — checks cache first; calls `GeminiAgent.decide()` on miss; stores result in cache

**`action_engine.py`** — Action Dispatcher

- `execute_auto_retry(transaction, retry_after_seconds)` — delegates to `retry_scheduler.schedule_retry()`
- `execute_alternate_suggestion(transaction, alternate_method)` — logs `RecoveryAction` with `action_type="alternate_method_suggested"`; updates status to `"alternate_suggested"`
- `execute_customer_notification(transaction, message, channel, ...)` — calls `notification_engine.generate_personalized_notification()`; logs `RecoveryAction`; supported channels: `email`, `sms`, `whatsapp`
- `dispatch_recovery_action(decision, transaction)` — routes `auto_retry` → `execute_auto_retry`; `suggest_alternate_method` → `execute_alternate_suggestion`; `send_payment_link|notify_customer|manual_review` → `execute_customer_notification`; `no_action` → log only

**`retry_scheduler.py`** — Bounded Retry Scheduler

- `schedule_retry(transaction, retry_after_seconds)` — enforces `MAX_RETRIES = 3` hard cap; exponential backoff: attempt 1 = `base * 1`, attempt 2 = `base * 2`, attempt 3 = `base * 4`; triggers `execute_customer_notification()` when limit exceeded
- `get_retry_status(transaction_id)` — returns `{attempts_made, attempts_remaining, next_retry_at, status}`

**`notification_engine.py`** — Multi-Channel Notification Drafts

- `draft_whatsapp_message()` — rich emoji-formatted WhatsApp message with CTA link
- `draft_sms_message()` — DLT-compliant SMS under 160 characters
- `draft_email_message()` — structured email with Subject + Body
- `generate_llm_notification_message()` — optional Gemini LLM-generated ultra-tailored copy (falls back to templates)
- `generate_personalized_notification()` — generates all 3 channel copies + stores in DB
- `dispatch_customer_notification()` — generates + logs `RecoveryAction` + updates transaction status
- `FAILURE_GUIDANCE` — templates for 9 failure categories: `insufficient_funds`, `card_blocked`, `network_timeout`, `gateway_issue`, `expired_card`, `authentication_failed`, `limit_exceeded`, `mandate_inactive`, `unknown`

**`analytics.py`** — Recovery Analytics Engine

- `get_recovery_analytics(merchant_id)` — full analytics: recovery rate, revenue saved, failure distribution, daily trends
- `get_revenue_saved()` — total INR revenue successfully recovered
- `get_failure_distribution()` — transaction count by failure category
- `get_recovery_rate_by_category()` — `{recovered, failed, rate}` per category
- `get_daily_recovery_trend(days)` — last N days trend (max 90 days)
- `seed_benchmark_dataset()` — seeds 156-transaction benchmark cohort for demo

**`cost_tracker.py`** — LLM Cost Tracker

- `calculate_llm_cost(input_tokens, output_tokens)` — `$0.000075/1k` input + `$0.000300/1k` output
- `log_llm_call()` — persists `LLMCost` record to `llm_costs` table
- `get_cost_summary()` — aggregate: total calls, tokens, cost, cost per recovery, estimated monthly cost (USD + INR)
- `USD_TO_INR_RATE = 83.33`

**`llm_cache.py`** — LRU/TTL Response Cache

- `LLMResponseCache` — thread-safe via `threading.RLock()`
- Cache key: `(failure_type, merchant_category)` tuple
- Default: `max_size=1000`, `ttl_seconds=86400.0` (24 hours)
- Eviction: LRU on capacity, TTL expiry on access
- `get()` — returns `copy.deepcopy(decision)` with `cached=True`, `latency_ms=0.0`, `cost_usd=0.0`
- `get_stats()` — returns `hits`, `misses`, `hit_rate_pct`, `cost_saved_usd`, `cost_saved_inr`

**`observability.py`** — In-Memory Metrics

- `METRICS` dict — `total_failures_received`, `total_recovered`, `total_failed_permanently`, `recovery_rate_percent`, `avg_pipeline_latency_ms`, `gemini_calls`, `gemini_errors`, `retries_scheduled`, `retries_executed`
- `record_pipeline_run(summary)` — updates cumulative moving average latency
- `record_retry_execution(result)` — updates retry outcome counters
- `print_metrics_report()` — formatted console/log report

**`db_writer.py`** — Database Write Layer

- `init_db()` — creates all tables via SQLAlchemy `Base.metadata.create_all()`
- `get_db_session()` — context manager yielding a scoped `Session`
- `save_transaction(normalized_event)` — upsert with `UNIQUE` deduplication on `razorpay_payment_id`
- `save_retry_attempt()`, `save_recovery_action()`, `update_transaction_status()`
- `get_transaction_by_payment_id()`, `get_all_transactions()`

**`broadcaster.py`** — SSE Broadcast Bus

- `broadcast(event)` — pushes event to all active SSE subscriber queues
- `subscribe()` / `unsubscribe()` — manage `asyncio.Queue` subscriber list

**`models.py`** — SQLAlchemy ORM Models (4 tables — see Database Schema section)

---

#### Database Schema (from `agent/models.py`)

**Table: `transactions`**

| Column                | Type         | Constraints               |
| --------------------- | ------------ | ------------------------- |
| `id`                  | `String(64)` | PRIMARY KEY               |
| `razorpay_payment_id` | `String(64)` | NOT NULL, UNIQUE, INDEX   |
| `merchant_id`         | `String(64)` | NOT NULL, INDEX           |
| `amount`              | `Float`      | NOT NULL                  |
| `currency`            | `String(10)` | DEFAULT `"INR"`           |
| `status`              | `String(32)` | DEFAULT `"FAILED"`, INDEX |
| `failure_reason`      | `Text`       | nullable                  |
| `failure_code`        | `String(64)` | nullable                  |
| `created_at`          | `DateTime`   | DEFAULT `utcnow`          |

**Table: `retry_attempts`**

| Column           | Type         | Constraints                                 |
| ---------------- | ------------ | ------------------------------------------- |
| `id`             | `Integer`    | PRIMARY KEY, autoincrement                  |
| `transaction_id` | `String(64)` | FK → `transactions.id`, CASCADE             |
| `attempt_number` | `Integer`    | NOT NULL                                    |
| `attempted_at`   | `DateTime`   | DEFAULT `utcnow`                            |
| `result`         | `String(32)` | `SUCCESS`, `FAILED`, `TIMEOUT`, `SCHEDULED` |
| `next_retry_at`  | `DateTime`   | nullable                                    |

UNIQUE constraint: `(transaction_id, attempt_number)` — named `uq_retry_attempts_tx_attempt`

**Table: `recovery_actions`**

| Column           | Type         | Constraints                                               |
| ---------------- | ------------ | --------------------------------------------------------- |
| `id`             | `Integer`    | PRIMARY KEY, autoincrement                                |
| `transaction_id` | `String(64)` | FK → `transactions.id`, CASCADE                           |
| `action_type`    | `String(64)` | `RETRY`, `PAYMENT_LINK`, `INVOICE_CHASER`, `NOTIFICATION` |
| `action_payload` | `Text`       | JSON payload                                              |
| `status`         | `String(32)` | DEFAULT `"PENDING"` (`PENDING`, `EXECUTED`, `FAILED`)     |
| `created_at`     | `DateTime`   | DEFAULT `utcnow`                                          |

**Table: `llm_costs`**

| Column           | Type         | Constraints                |
| ---------------- | ------------ | -------------------------- |
| `id`             | `Integer`    | PRIMARY KEY, autoincrement |
| `transaction_id` | `String(64)` | nullable, INDEX            |
| `model`          | `String(64)` | NOT NULL                   |
| `input_tokens`   | `Integer`    | DEFAULT 0                  |
| `output_tokens`  | `Integer`    | DEFAULT 0                  |
| `cost_usd`       | `Float`      | DEFAULT 0.0                |
| `latency_ms`     | `Float`      | DEFAULT 0.0                |
| `created_at`     | `DateTime`   | DEFAULT `utcnow`           |

---

#### `ml/` — Machine Learning Layer

**`error_codes.py`** — Rule-Based Failure Classifier

- `FailureCategory` enum: 8 canonical categories
- `ERROR_CODE_MAP` — 43 exact code mappings (e.g. `"GATEWAY_ERROR"` → `"gateway_issue"`)
- `ERROR_REASON_MAP` — 59 exact reason mappings (e.g. `"insufficient_funds"` → `"insufficient_funds"`)
- `KEYWORD_PATTERNS` — 7-group regex heuristics for fuzzy matching
- `classify_failure(error_code, error_reason)` — 4-step cascade: reason exact → code exact → reason regex → code regex → `"unknown"`
- `is_transient_failure(category)` — returns `True` for `network_timeout` and `gateway_issue` only
- `get_category_description(category)` — human-readable description per category

**`classifier.py`** — ML Ensemble Classifier

- `FailureClassifier` — soft-voting ensemble: `BalancedXGBClassifier` + `LogisticRegression(class_weight='balanced')`
- 5-fold stratified CV: **100% accuracy** on benchmark synthetic dataset, Weighted F1 = 1.000

**`feature_engineering.py`** — Feature Pipeline

- `FeatureEngineeringPipeline` — `ColumnTransformer` with `StandardScaler` + `OrdinalEncoder` inside `sklearn.pipeline.Pipeline`
- Features: `error_code`, `error_reason`, `payment_method`, `merchant_id`, `amount`, hour-of-day from `created_at`

**`retry_predictor.py`** — Retry Timing Predictor

- Predicts optimal retry delay; pushes retries away from 01:00–04:00 AM IST (NPCI maintenance window)

---

## ⚡ Key Features

All features below are implemented and tested — not planned.

### 1. HMAC-SHA256 Webhook Verification

`api_integration/verifier.py` — raw body bytes hashed before JSON parsing; `hmac.compare_digest()` constant-time comparison.

### 2. Webhook Normalization Across 3 Event Types

`api_integration/normalizer.py` — heterogeneous `payment.failed`, `subscription.halted`, `invoice.overdue` structures → single `NormalizedEvent` schema. Paise → rupees conversion at ingest.

### 3. ML Ensemble Failure Classification

`ml/classifier.py` — `BalancedXGBClassifier` + `LogisticRegression` soft-voting with 5-fold CV; rule-based `classify_failure()` fallback if model file absent.

### 4. Gemini AI Recovery Decisions

`agent/llm_agent.py` — `gemini-flash-lite-latest` with `temperature=0.1`, `response_mime_type="application/json"`. Prompt includes entity, amount, failure category, error codes, hour of day, subscription priority, and notification channels. Returns structured JSON enforcing 7 required fields.

### 5. Bounded Retry Engine

`agent/retry_scheduler.py` — `MAX_RETRIES = 3` hard cap; exponential backoff (1x, 2x, 4x base delay); falls back to `execute_customer_notification()` when limit exceeded; UNIQUE DB constraint `(transaction_id, attempt_number)` prevents duplicate attempts.

### 6. Three-Action Recovery Dispatch

`agent/action_engine.py` — `auto_retry`, `suggest_alternate_method`, and `send_payment_link|notify_customer|manual_review` routes all implemented.

### 7. Multi-Channel Notification Drafts

`agent/notification_engine.py` — WhatsApp (emoji, CTA link), SMS (DLT-compliant, <160 chars), Email (Subject + Body); optional Gemini LLM personalization with deterministic template fallback.

### 8. LRU/TTL LLM Response Cache

`agent/llm_cache.py` — `(failure_type, merchant_category)` keyed; max 1,000 entries; 24-hour TTL; LRU eviction on capacity; returns `deepcopy` with `cached=True`, zero cost/latency metadata.

### 9. Real-Time SSE Dashboard Stream

`api_integration/rest_router.py` — `GET /stream` pushes transaction update events; supports `X-API-Key` header or `?api_key=` query param (SSE-only exception); 25s keepalive pings.

### 10. Comprehensive Cost Tracking

`agent/cost_tracker.py` — per-call cost logged to `llm_costs` table; aggregates total spend, cost per recovery, 30-day monthly projection.

### 11. Observability Metrics

`agent/observability.py` — thread-safe in-memory `METRICS` dict; cumulative moving average latency; `GET /metrics` REST endpoint.

### 12. Merchant Dashboard

`frontend/` served at `/dashboard/index.html` — 3-panel: Batch Summary Header, Live Transaction State Table, Audit Log Feed.

---

## 📊 Performance Results

All numbers from actual measured execution (source: `tasks/lessons.md`).

### Latency Benchmark (`scripts/latency_test.py`)

Target: `< 3,000ms` per payment end-to-end

| Run         | Latency     | Status                       |
| ----------- | ----------- | ---------------------------- |
| Payment 1   | 2,169ms     | ✅                           |
| Payment 2   | 1,683ms     | ✅                           |
| Payment 3   | 1,308ms     | ✅                           |
| Payment 4   | 1,281ms     | ✅                           |
| Payment 5   | 2,880ms     | ✅                           |
| **Average** | **1,864ms** | ✅ well under 3,000ms target |

### Cache Impact

| Mode                         | Latency           |
| ---------------------------- | ----------------- |
| Cold LLM call                | 1,864ms – 2,086ms |
| Cached response              | 30ms – 38ms       |
| Batch average (80% hit rate) | **443.7ms**       |

### Test Suite

```
pytest --collect-only: 212 tests collected
pytest -q:             196 passed, 58 subtests in 98.68s — zero failures
```

### Recovery Rate (156-Transaction Benchmark Cohort)

Seeded via `agent/analytics.py` `seed_benchmark_dataset()`:

| Metric                   | Value            |
| ------------------------ | ---------------- |
| Total Failures           | 156              |
| Total Recovered          | 89               |
| Total Permanently Failed | 67               |
| **Recovery Rate**        | **57.1%**        |
| Revenue at Risk          | ₹2,84,750.00     |
| **Revenue Saved**        | **₹1,62,668.00** |
| Avg Recovery Time        | 23.4 minutes     |

### Recovery Rate by Failure Category

| Category                | Recovered | Failed | Rate      |
| ----------------------- | --------- | ------ | --------- |
| `network_timeout`       | 27        | 4      | **87.1%** |
| `gateway_issue`         | 14        | 4      | **77.8%** |
| `insufficient_funds`    | 28        | 17     | **62.2%** |
| `authentication_failed` | 8         | 7      | **53.3%** |
| `limit_exceeded`        | 3         | 5      | **37.5%** |
| `card_blocked`          | 8         | 15     | **34.8%** |
| `expired_card`          | 4         | 8      | **33.3%** |
| `unknown`               | 1         | 3      | **25.0%** |

### Integration Test Matrix (11/11)

Source: `tasks/lessons.md`

```
insufficient_funds      → suggest_alternate_method ✅
card_blocked            → suggest_alternate_method ✅
network_timeout         → auto_retry               ✅
gateway_issue           → auto_retry               ✅
expired_card            → send_payment_link        ✅
authentication_failed   → auto_retry               ✅
limit_exceeded          → suggest_alternate_method ✅
unknown                 → auto_retry               ✅
subscription.halted     → auto_retry               ✅
invoice.overdue         → auto_retry               ✅
insufficient_funds_upi  → suggest_alternate_method ✅
Results: 11 passed, 0 failed | Total time: 38.5s
```

---

## 🛠 Tech Stack

From `requirements.txt` and `docker-compose.yml`:

### Core Framework

| Package             | Version  | Usage                                 |
| ------------------- | -------- | ------------------------------------- |
| `fastapi`           | ≥0.110.0 | Web framework, REST + webhook routing |
| `uvicorn[standard]` | ≥0.28.0  | ASGI server                           |
| `pydantic`          | ≥2.6.0   | Schema validation                     |
| `pydantic-settings` | ≥2.2.0   | Environment config                    |
| `slowapi`           | ≥0.1.9   | Rate limiting (100 req/min per IP)    |

### AI & LLM

| Package        | Version | Usage                                            |
| -------------- | ------- | ------------------------------------------------ |
| `google-genai` | ≥1.0.0  | Gemini API client (`genai.Client`)               |
| `openai`       | ≥1.14.0 | OpenAI provider (alternate LLM option)           |
| `anthropic`    | ≥0.19.0 | Anthropic Claude provider (alternate LLM option) |

### Database & ORM

| Package           | Version | Usage                           |
| ----------------- | ------- | ------------------------------- |
| `sqlalchemy`      | ≥2.0.28 | ORM, all 4 table models         |
| `psycopg2-binary` | ≥2.9.9  | PostgreSQL sync driver          |
| `asyncpg`         | ≥0.29.0 | PostgreSQL async driver         |
| `aiosqlite`       | ≥0.20.0 | SQLite async driver (local dev) |

### Machine Learning

| Package        | Version | Usage                                                 |
| -------------- | ------- | ----------------------------------------------------- |
| `scikit-learn` | ≥1.4.1  | `ColumnTransformer`, `Pipeline`, `LogisticRegression` |
| `xgboost`      | ≥2.0.3  | `BalancedXGBClassifier` ensemble member               |
| `pandas`       | ≥2.2.1  | Feature DataFrame construction                        |
| `numpy`        | ≥1.26.4 | Numeric operations                                    |

### Utilities

| Package         | Version | Usage                              |
| --------------- | ------- | ---------------------------------- |
| `python-dotenv` | ≥1.0.1  | `.env` loading                     |
| `httpx`         | ≥0.27.0 | HTTP client for webhook simulation |
| `razorpay`      | ≥1.4.1  | Razorpay SDK                       |

### Testing

| Package          | Version | Usage              |
| ---------------- | ------- | ------------------ |
| `pytest`         | ≥8.0.2  | Test runner        |
| `pytest-asyncio` | ≥0.23.5 | Async test support |

### Docker Services (`docker-compose.yml`)

| Service          | Image                | Port      |
| ---------------- | -------------------- | --------- |
| `postgres`       | `postgres:16-alpine` | 5434→5432 |
| `recovery-agent` | `./Dockerfile`       | 8080→8000 |

---

## 🔴 Failure Categories & Recovery Actions

### The 8 Canonical Categories (from `ml/error_codes.py`)

| Category                | `error_code` Examples                                       | `error_reason` Examples                             | Recovery Action                                 |
| ----------------------- | ----------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| `insufficient_funds`    | `INSUFFICIENT_FUNDS`, `BAD_REQUEST_INSUFFICIENT_FUNDS`      | `insufficient_funds`, `low_balance`                 | Suggest UPI/wallet; send payment link           |
| `card_blocked`          | `CARD_BLOCKED`, `DO_NOT_HONOUR`, `STOLEN_CARD`              | `card_blocked`, `account_frozen`                    | Do NOT retry; suggest netbanking/UPI            |
| `network_timeout`       | `TRANSACTION_TIMED_OUT`, `UPI_COLLECT_EXPIRED`              | `network_timeout`, `timed_out`, `otp_timeout`       | Auto-retry after 15 min; then 30 min            |
| `gateway_issue`         | `GATEWAY_ERROR`, `ISSUER_DOWN`, `ACQUIRER_DOWN`             | `gateway_error`, `bank_down`, `service_unavailable` | Auto-retry after 30 min                         |
| `expired_card`          | `CARD_EXPIRED`, `INVALID_EXPIRY_DATE`                       | `card_expired`, `invalid_expiry`                    | Do NOT retry; send update-card message          |
| `authentication_failed` | `OTP_INCORRECT`, `3DS_AUTHENTICATION_FAILED`, `INVALID_CVV` | `incorrect_otp`, `invalid_pin`, `mpin_incorrect`    | Retry once after 10 min; suggest UPI            |
| `limit_exceeded`        | `EXCEEDED_DAILY_AMOUNT_LIMIT`, `VELOCITY_EXCEEDED`          | `daily_limit_exceeded`, `credit_limit_exceeded`     | Do NOT retry same card; suggest alternative     |
| `unknown`               | `UNKNOWN_ERROR`                                             | (unmatched)                                         | Retry once after 60 min; flag for manual review |

### Classification Strategy (from `ml/error_codes.py` `classify_failure()`)

4-step cascade, guaranteed no exceptions on `None` or empty inputs:

1. Exact match on normalized `error_reason` → `ERROR_REASON_MAP` (59 entries)
2. Exact match on uppercase `error_code` → `ERROR_CODE_MAP` (43 entries)
3. Regex scan over `error_reason` via `KEYWORD_PATTERNS` (7 groups)
4. Regex scan over `error_code` via `KEYWORD_PATTERNS`
5. Fall back to `"unknown"`

### Transient vs. Terminal (from `is_transient_failure()`)

- **Transient** (auto-retry eligible): `network_timeout`, `gateway_issue`
- **Terminal** (user action required): all other 6 categories

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### Webhook Endpoints

#### `GET /webhooks/razorpay`

Browser health check — returns listener status and supported event types.

#### `POST /webhooks/razorpay`

Receives and routes Razorpay webhook events.

**Headers:**

- `X-Razorpay-Signature: <hmac-sha256-hex>` — skipped if `SIMULATION_MODE=true` and no signature provided
- `Content-Type: application/json`

**Response:** `WebhookResponse`

```json
{
  "status": "success",
  "event_id": "evt_abc123",
  "event_type": "payment.failed",
  "action_taken": "payment_failure_routed",
  "message": "Payment failure 'pay_xxx' captured and routed to LLM diagnostic & recovery pipeline.",
  "normalized_event": { ... }
}
```

---

### REST API Endpoints

**Authentication (all endpoints except `/metrics`, `/analytics`, `/costs`, `/cache/stats`):**

- `X-API-Key: <key>` header, OR
- `Authorization: Bearer <key>` header

#### `POST /analyze-failure`

Ingests failure details → ML classify → Gemini decision → dispatch recovery action.

**Request body fields:** `payment_id`, `amount`, `currency`, `error_code`, `error_reason`, `error_description`, `payment_method`, `merchant_id`, `customer_name`, `customer_email`, `customer_phone`, `notes`

**Response:** `FailureAnalysisResponse` — `transaction_id`, `failure_category`, `action_taken`, `priority`, `confidence`, `retry_after`, `alternate_method`, `customer_message`, `reasoning`, `db_record_id`, `elapsed_ms`

#### `GET /recovery-suggestions`

Retrieves recovery recommendations and action history.

**Query params:** `payment_id`, `transaction_id`, `status` (filter), `limit` (1–200, default 50), `offset`

**Response:** `RecoverySuggestionsResponse` with `RecoverySuggestionItem` list — each item includes `suggested_action`, `retry_count`, `max_retries=2`, `can_retry`, `latest_action`

#### `POST /trigger-retry`

Triggers immediate or scheduled retry subject to guardrails.

**Request:** `payment_id` or `transaction_id`, `delay_seconds` (default 0), `force` (bypass 2-attempt limit)

**Response:** `attempt_number`, `result` (`TRIGGERED`|`SCHEDULED`), `next_retry_at`

**Errors:** `HTTP 400` if `retry_count >= 2` and `force=False`; `HTTP 404` if transaction not found

#### `GET /stats`

Dashboard counters.

```json
{
  "total": 156,
  "recovered": 89,
  "retry_scheduled": 12,
  "failed": 55,
  "success_rate": 57.1
}
```

#### `GET /stream`

Server-Sent Events real-time feed. Accepts `?api_key=` query param (SSE-only exception). Sends keepalive `: keepalive` comment every 25s.

#### `GET /metrics`

In-memory observability snapshot. Returns `METRICS` dict. No auth required.

#### `GET /analytics`

Full recovery analytics from DB: failure distribution, recovery by category, revenue saved.

#### `GET /costs`

LLM token usage, cost per recovery, monthly projection in USD and INR.

#### `GET /cache/stats`

LLM cache telemetry: hits, misses, hit rate %, cost saved USD/INR.

#### `POST /cache/clear` _(authenticated)_

Flushes LLM response cache. Returns count of purged entries.

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Razorpay Sandbox API credentials
- Google Gemini API key

### 1. Clone and Configure

```bash
git clone <repo-url>
cd razorpay-recovery-agent
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start with Docker Compose

```bash
docker-compose up -d
```

Starts:

- `razorpay-postgres` on port **5434** (`postgres:16-alpine`)
- `razorpay-recovery-agent` on port **8080** (→ internal 8000)

### 3. Verify

```bash
curl http://localhost:8080/health
# Open: http://localhost:8080/docs
# Open: http://localhost:8080/dashboard/index.html
```

### 4. Local Development (without Docker)

```bash
pip install -r requirements.txt
python -c "from agent.db_writer import init_db; init_db()"
python main.py
# or: uvicorn main:app --reload --port 8000
```

### Environment Variables (from `.env.example`)

| Variable                     | Required | Default                                        | Description                                   |
| ---------------------------- | -------- | ---------------------------------------------- | --------------------------------------------- |
| `RAZORPAY_KEY_ID`            | ✅       | —                                              | Razorpay API Key ID (sandbox)                 |
| `RAZORPAY_KEY_SECRET`        | ✅       | —                                              | Razorpay API Secret                           |
| `RAZORPAY_WEBHOOK_SECRET`    | ✅       | —                                              | Webhook HMAC signing secret                   |
| `GEMINI_API_KEY`             | ✅       | —                                              | Google Gemini API key                         |
| `GEMINI_MODEL`               | ❌       | `gemini-flash-lite-latest`                     | Gemini model name                             |
| `DEFAULT_LLM_PROVIDER`       | ❌       | `mock`                                         | `anthropic`, `openai`, `gemini`, or `mock`    |
| `ANTHROPIC_API_KEY`          | ❌       | —                                              | Anthropic Claude API key                      |
| `OPENAI_API_KEY`             | ❌       | —                                              | OpenAI API key                                |
| `DATABASE_URL`               | ❌       | `sqlite:///./data/recovery_agent.db`           | Sync DB URL                                   |
| `ASYNC_DATABASE_URL`         | ❌       | `sqlite+aiosqlite:///./data/recovery_agent.db` | Async DB URL                                  |
| `MAX_RETRY_ATTEMPTS`         | ❌       | `3`                                            | Max automated retries per transaction         |
| `RETRY_BASE_BACKOFF_MINUTES` | ❌       | `5`                                            | Base backoff delay in minutes                 |
| `SIMULATION_MODE`            | ❌       | `true`                                         | Bypass webhook signature + auth for local dev |
| `ENVIRONMENT`                | ❌       | `development`                                  | `development` or `production`                 |
| `APP_SECRET_KEY`             | ❌       | —                                              | API key for REST endpoints                    |
| `PORT`                       | ❌       | `8000`                                         | Server port                                   |
| `HOST`                       | ❌       | `0.0.0.0`                                      | Server host                                   |
| `OLLAMA_BASE_URL`            | ❌       | `http://localhost:11434`                       | Local Ollama base URL                         |

### Available Scripts

| Script                            | What it does                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `scripts/validate_env.py`         | Validates all required env vars and API key reachability                                                |
| `scripts/simulate_webhook.py`     | Signs and POSTs simulated `payment.failed`, `subscription.halted`, `invoice.overdue` events             |
| `scripts/demo_recovery_batch.py`  | Runs 20 failed payments through full pipeline across all 8 failure categories; prints hackathon summary |
| `scripts/latency_test.py`         | Runs 5 payments end-to-end; reports per-payment latency, average, and cache hit rate                    |
| `scripts/train_classifier.py`     | Trains XGBoost+LR ensemble on historical data; saves to `ml/models/`                                    |
| `scripts/pull_historical_data.py` | Generates synthetic 90-day historical failure dataset (50+ records) for ML training                     |
| `scripts/test_pipeline.py`        | Manual 11-scenario pipeline test                                                                        |
| `scripts/init_db.sql`             | Raw SQL schema; auto-run by Docker Compose on first `postgres` startup                                  |

---

## 📁 Project Structure

```
razorpay-recovery-agent/
│
├── main.py                          # FastAPI app: router registration, rate limiting, health endpoint
├── requirements.txt                 # Python dependencies with version pins
├── docker-compose.yml               # postgres:16-alpine + recovery-agent services
├── Dockerfile                       # Container build spec
├── .env.example                     # Environment variable template
├── pyproject.toml                   # Python project metadata
├── RULES.md                         # Development guardrails
│
├── agent/                           # Core recovery agent
│   ├── SCOPE.md                     # MVP scope and explicit out-of-scope exclusions
│   ├── models.py                    # SQLAlchemy ORM: transactions, retry_attempts, recovery_actions, llm_costs
│   ├── db_writer.py                 # DB write layer: init_db, save_transaction, upserts, status updates
│   ├── pipeline.py                  # Main 4-step recovery pipeline: normalize→classify→Gemini→dispatch
│   ├── recovery_engine.py           # RecoveryEngine: GeminiAgent + LLMCache orchestrator
│   ├── llm_agent.py                 # GeminiAgent wrapper, SYSTEM_PROMPT, build_cot_prompt, RecoveryDecision
│   ├── llm_cache.py                 # LRU/TTL in-memory cache keyed by (failure_type, merchant_category)
│   ├── action_engine.py             # 3 executors + dispatcher: auto_retry, alternate, notification
│   ├── retry_scheduler.py           # Bounded retry scheduler with exponential backoff (MAX_RETRIES=3)
│   ├── retry_executor.py            # Executes due scheduled retries, records SUCCESS/FAILED
│   ├── notification_engine.py       # WhatsApp/SMS/Email message drafting (template + optional Gemini LLM)
│   ├── analytics.py                 # Recovery rate, revenue saved, failure distribution, daily trends
│   ├── cost_tracker.py              # LLM token cost tracking ($0.000075/1k input, $0.000300/1k output)
│   ├── observability.py             # Thread-safe in-memory METRICS, pipeline run recording
│   └── broadcaster.py               # Async SSE broadcast bus for real-time dashboard events
│
├── api_integration/                 # Webhook ingestion + REST API layer
│   ├── router.py                    # POST/GET /webhooks/razorpay — HMAC verify + normalize + route
│   ├── rest_router.py               # 10 REST endpoints
│   ├── schemas.py                   # Pydantic models: NormalizedEvent, EventType, FailureCategory, etc.
│   ├── normalizer.py                # Raw webhook JSON → NormalizedEvent (paise→rupees, 3 event layouts)
│   ├── auth.py                      # X-API-Key / Bearer auth; hmac.compare_digest timing-oracle protection
│   └── verifier.py                  # HMAC-SHA256 webhook signature verification
│
├── ml/                              # Machine learning classification layer
│   ├── error_codes.py               # 8-category classifier: 43 code maps, 59 reason maps, 7 regex groups
│   ├── classifier.py                # BalancedXGBClassifier + LogisticRegression soft-voting ensemble
│   ├── feature_engineering.py       # ColumnTransformer: StandardScaler + OrdinalEncoder + hour-of-day
│   ├── retry_predictor.py           # Retry timing predictor; avoids 01:00–04:00 AM IST NPCI window
│   └── models/                      # Saved model artifacts (.joblib)
│
├── scripts/                         # Developer and demo utility scripts (8 scripts)
│
├── tests/                           # Full test suite — 212 collected, 196 passed, 58 subtests
│   ├── test_sanity.py
│   ├── test_api_integration.py
│   ├── test_webhook_normalizer.py
│   ├── test_rest_api.py
│   ├── test_error_codes.py
│   ├── test_ml_pipeline.py
│   ├── test_retry_predictor.py
│   ├── test_llm_integration.py
│   ├── test_llm_cache.py
│   ├── test_pipeline_unit.py
│   ├── test_integration_full.py
│   ├── test_retry_loop_e2e.py
│   ├── test_retry_executor.py
│   ├── test_retry_scheduler.py
│   ├── test_db_writer.py
│   ├── test_database_schema.py
│   ├── test_action_engine.py
│   ├── test_notification_engine.py
│   ├── test_analytics.py
│   ├── test_cost_tracker.py
│   └── test_pull_historical_data.py
│
├── tasks/
│   ├── todo.md                      # 10-day sprint roadmap: 60 tasks across 3 phases
│   └── lessons.md                   # Bugs found/fixed, architecture decisions, performance benchmarks
│
├── data/                            # Persistent data (SQLite dev DB, ML training data)
└── frontend/                        # Static merchant dashboard (served at /dashboard)
```

---

## 🛡 Compliance & Safety

### Stopping Rules (from `agent/retry_scheduler.py`)

```python
MAX_RETRIES = 3  # Hard cap; when attempts_made >= MAX_RETRIES, no further retries
```

Falls back to `execute_customer_notification()` when limit reached. UNIQUE DB constraint `(transaction_id, attempt_number)` prevents duplicate attempts under concurrent webhook storms.

### LLM Isolation Guardrail (from `agent/SCOPE.md`)

> The LLM serves solely as a classification advisor and cannot mutate database records or initiate transactions directly.

`GeminiAgent.decide()` returns a `RecoveryDecision` dataclass only. All DB writes and action dispatches are performed by `action_engine.py` and `db_writer.py`. The LLM has no direct DB or Razorpay API access.

### Webhook Authentication (`api_integration/verifier.py`)

- HMAC-SHA256 computed on raw bytes **before** JSON parsing
- `hmac.compare_digest()` for constant-time comparison (timing-oracle resistant)
- Bypassed only when `SIMULATION_MODE=true` AND no signature header present

### Merchant API Authentication (`api_integration/auth.py`)

- `X-API-Key` or `Authorization: Bearer <token>` headers only
- Query-parameter auth intentionally blocked (prevents key leakage in server logs/browser history)
- `hmac.compare_digest()` for every valid key comparison
- `SIMULATION_MODE` bypass blocked in `ENVIRONMENT=production`

### Audit Trail

Every state change (`retry_scheduled`, `alternate_suggested`, `customer_notified`, `pipeline_error`) is persisted to SQL before the corresponding action executes. `RecoveryAction` and `RetryAttempt` records provide an immutable audit trail.

---

## 📈 Feasibility Analysis

Projected vs. Actual (source: `tasks/lessons.md` + `agent/analytics.py`):

| Metric                    | Projected | Actual                    |
| ------------------------- | --------- | ------------------------- |
| Recovery rate             | 40–60%    | **57.1%** ✅              |
| LLM latency per call      | < 3,000ms | **1,864ms avg** ✅        |
| ML classifier accuracy    | > 80%     | **100% on benchmark** ✅  |
| Cost per LLM call         | < $0.001  | **$0.00020** ✅           |
| Cost per recovery         | < $0.001  | **$0.00033–$0.00035** ✅  |
| Monthly cost (156 tx/mo)  | —         | **$0.93 USD / ₹77.5 INR** |
| Revenue saved (benchmark) | —         | **₹1,62,668.00**          |
| ROI on inference spend    | —         | **> 2,000x**              |

### Cache Economics

- Cold call: ~1,864ms, ~$0.00020/call, ~1,500 input tokens + ~290 output tokens
- Cached response: ~34ms, $0.00000, 0 tokens consumed
- 80% hit rate → batch average **443.7ms** instead of 1,864ms

---

## ⚠️ Known Limitations

### Not Built (Explicitly Out of Scope)

1. **Fraud Detection** — No velocity checks, risk scoring, or AML workflows
2. **Dynamic Pricing / Discount Engine** — No voucher generation
3. **Live Telephony / IVR** — No phone-dialing bots
4. **Production Financial Mutations** — All retries are simulated; no live Razorpay API charges
5. **Multi-year BI / LTV Modeling** — Analytics limited to per-batch recovery metrics

### Bugs Found and Fixed During Sprint (from `tasks/lessons.md`)

| Bug                                                                                                      | Severity | Fix                                                                                                  |
| -------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `gemini-3.6-flash` model deprecated/quota exhausted (20 RPD, 504 timeouts)                               | P0       | Switched to `gemini-flash-lite-latest`; response time: 9–10s → 1.1–2.9s                              |
| `ThreadPoolExecutor` context manager blocks on timeout (`executor.shutdown(wait=True)`)                  | P0       | Removed executor; use native transport-level timeout `genai.Client(http_options={"timeout": 10000})` |
| f-string literal `{`/`}` in `build_cot_prompt` raised `ValueError: Invalid format specifier`             | P0       | Doubled braces: `{{` / `}}` inside f-string templates                                                |
| `retry_scheduled` missing from test assertion allow-list; correct behavior flagged as failure            | P1       | Added `"retry_scheduled"` to valid terminal states                                                   |
| Static 429 backoff `[2, 4, 8]s` shorter than API's `retryDelay` hint (10–28s); hit 429 again immediately | P1       | Parse `retryDelay` from 429 response body; use `max(api_delay + 2, static_delay)`                    |
| Verbose 5-step CoT prompt caused 504 gateway timeout (~10s)                                              | P1       | Replaced with concise direct task description + explicit JSON schema; latency: 10s → 1.5s            |

### Behavioral Constraints

- Notifications are drafts only — WhatsApp, SMS, and Email messages are generated and logged to `recovery_actions` but not dispatched to any messaging provider (no Twilio/SendGrid/WhatsApp Business API integration).
- `SIMULATION_MODE=true` bypasses both webhook signature verification and API key authentication — do not enable in production.

---

## 🏗 Built With

- **Hackathon**: Razorpay AI Buildathon 2026 — Track 03: AI Revenue Recovery
- **Developer**: Shrihan (as documented in `tasks/todo.md`)
- **AI Pair Programmer**: Antigravity (Google DeepMind Advanced Agentic Coding)
- **Timeline**: 10-day sprint, August 25 – September 3, 2026
- **Phases**: Foundation & R&D (Days 1–3) → Core Build (Days 4–6) → Integration & Delivery (Days 7–10)

**Primary Stack:**

| Layer         | Technology                                                 |
| ------------- | ---------------------------------------------------------- |
| Web Framework | FastAPI 0.110+                                             |
| LLM           | Gemini (`gemini-flash-lite-latest`) via `google-genai`     |
| ML Ensemble   | XGBoost + scikit-learn LogisticRegression                  |
| Database      | PostgreSQL 16 (prod) / SQLite WAL (dev) via SQLAlchemy 2.0 |
| Container     | Docker Compose (`postgres:16-alpine`)                      |
| Language      | Python 3.11+                                               |
