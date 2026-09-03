# 10-Day Sprint Roadmap: AI Revenue Recovery Agent

**Project**: AI Revenue Recovery Agent — Razorpay Hackathon (Track 03)  
**Developer**: Shrihan  
**Sprint Window**: Tuesday, August 25 – Thursday, September 3, 2026  
**Total Tasks**: 60 tasks across 3 phases  

---

## 📊 Phase Overview

| Phase | Days | Focus | Status |
|---|---|---|---|
| **Phase 1: Foundation & R&D** | Days 1–3 (Aug 25–27) | Architecture, Data Pipeline, Webhooks & Baseline ML Model | 🟡 In Progress |
| **Phase 2: Core Build** | Days 4–6 (Aug 28–30) | LLM Agent, Bounded State Machine, REST API & Smart Retry | ⚪ Pending |
| **Phase 3: Integration & Delivery** | Days 7–10 (Aug 31–Sep 3) | Testing, Dashboard, Cost Tracking, Polish & Submission | ⚪ Pending |

---

## 📅 Daily Milestones & Checkpoints

### 🔹 Day 1 — Architecture & Environment Setup (Tuesday, Aug 25, 2026)
**Goal**: Establish zero-ambiguity foundation, create repository architecture, and provision dev environment.
- [x] **Morning (Research)**: Read full R&D document end-to-end; highlight unknowns, guardrails, and stopping rules.
- [x] **Morning (Build)**: Set up project repository structure: `/agent`, `/api_integration`, `/ml`, `/tests`, `/scripts`.
- [x] **Morning (Build)**: Provision development environment: Python 3.10+, Docker (`Dockerfile`, `docker-compose.yml`), `requirements.txt`, `.env.example`.
- [x] **Afternoon (Review)**: Define MVP scope: Revenue Recovery Agent only (Fraud & Dynamic Pricing deferred to future).
- [x] **Afternoon (Review)**: Draft `tasks/todo.md` with all 10-day milestones and daily checkpoints.
- [x] **Afternoon (Build)**: Configure API keys / sandbox credentials for Razorpay and LLM providers & validate via script.

---

### 🔹 Day 2 — Data Pipeline & Razorpay Integration (Wednesday, Aug 26, 2026)
**Goal**: Build real-time webhook listeners, data ingestion layer, and local SQL transaction storage.
- [x] **Morning (Build)**: Implement Razorpay webhook listener — capture `payment.failed`, `order.paid`, `payment.authorized`, `subscription.halted`, `invoice.overdue`.
- [x] **Morning (Build)**: Build data ingestion layer to normalize raw webhook payloads into standard internal schema.
- [x] **Morning (Build)**: Set up local DB (SQLite with WAL mode / PostgreSQL) for transactions, failure reasons, and audit logs (`agent/db_writer.py`).
- [x] **Afternoon (Research)**: Synthesize / pull 90-day historical failure data (50+ records) for ML training & validation (`scripts/pull_historical_data.py`).
- [x] **Afternoon (Research)**: Map all Razorpay error codes (`BAD_REQUEST_ERROR`, `GATEWAY_ERROR`, `insufficient_funds`, `card_blocked`, `network_timeout`) (`ml/error_codes.py`).
- [x] **Afternoon (Test)**: Write unit tests for webhook parsing and data normalization across all 3 event streams.

---

### 🔹 Day 3 — ML Model & Failure Classification (Thursday, Aug 27, 2026)
**Goal**: Train failure classifier and retry-timing predictor to optimize recovery success rates.
- [x] **Morning (Build)**: Build failure classification model — train on error codes, merchant category, amount, and time-of-day features.
- [x] **Morning (Build)**: Create feature engineering pipeline: encode categorical vars, normalize amounts, extract hour-of-day.
- [x] **Afternoon (Test)**: Train initial model on historical data; evaluate precision & recall per failure class (Target: >80% accuracy).
- [x] **Afternoon (Build)**: Build retry-timing predictor: given a failure type, output optimal retry delay in minutes.
- [x] **Afternoon (Review)**: Log model metrics and data quality findings to `tasks/lessons.md`.

---

### 🔹 Day 4 — LLM Core Agent & Prompt Engineering (Friday, Aug 28, 2026)
**Goal**: Build autonomous LLM diagnostic engine with strict JSON schema validation and recovery reasoning.
- [ ] **Morning (Build)**: Build AI agent wrapper around Claude Sonnet / GPT-4 / Gemini / Ollama with payment failure system prompt.
- [ ] **Morning (Build)**: Implement Chain-of-Thought (CoT) prompt: inject failure context + unstructured notes $\rightarrow$ structured recovery recommendation.
- [ ] **Morning (Build)**: Build structured output parser: enforce strict JSON `{ "diagnosis": "...", "recovery_action": "...", "confidence": 0.95 }`.
- [ ] **Afternoon (Build)**: Implement 3 core recovery actions: Auto-retry, Alternate payment method suggestion (Hinglish copy), B2B Invoice reminder sequence.
- [ ] **Afternoon (Integrate)**: Wire ML classifier output $\rightarrow$ LLM agent input $\rightarrow$ action dispatcher into a single unified pipeline.
- [ ] **Afternoon (Test)**: Manual test: simulate 5 failure scenarios end-to-end and verify recovery actions.

---

### 🔹 Day 5 — API Layer & Merchant Dashboard (Saturday, Aug 29, 2026)
**Goal**: Expose REST API endpoints and build the live 3-panel "Measured Money Recovered" dashboard UI.
- [ ] **Morning (Build)**: Build REST API: `POST /analyze-failure`, `GET /recovery-suggestions`, `POST /trigger-retry`, `POST /batch-ingest`.
- [ ] **Morning (Build)**: Add authentication middleware (API key / merchant token access).
- [ ] **Afternoon (Build)**: Build 3-panel Merchant Dashboard UI (Batch Summary Header, Transaction State Table, Audit Log Feed).
- [ ] **Afternoon (Integrate)**: Implement real-time status updates: Webhook in $\rightarrow$ Agent process $\rightarrow$ Dashboard update via SSE / polling.
- [ ] **Afternoon (Test)**: API integration tests with mock Razorpay payloads.

---

### 🔹 Day 6 — Smart Retry Engine & Notification System (Sunday, Aug 30, 2026)
**Goal**: Build intelligent retry scheduler with exponential backoff and personalized customer notification engine.
- [ ] **Morning (Build)**: Build intelligent retry scheduler using ML timing predictor with exponential backoff (hard-capped at 2 retries).
- [ ] **Morning (Build)**: Implement retry executor: call Razorpay API to re-attempt payment and record result.
- [ ] **Afternoon (Build)**: Build customer communication module: draft WhatsApp / SMS / Email recovery copy via LLM (including Hinglish).
- [ ] **Afternoon (Build)**: Add notification personalization: merchant name, amount, dynamic payment link, alternate method.
- [ ] **Afternoon (Test)**: Test retry loop end-to-end in sandbox: failed payment $\rightarrow$ retry after $N$ mins $\rightarrow$ success/failure logged.

---

### 🔹 Day 7 — Integration Testing & Error Handling (Monday, Aug 31, 2026)
**Goal**: Harden edge-case handling, verify zero infinite loops, and ensure latency $<3$s.
- [ ] **Morning (Test)**: Run full integration test suite: all 10+ failure scenarios from the R&D doc's test matrix.
- [ ] **Morning (Build)**: Add robust error handling everywhere: retry exhaustion, LLM timeouts, Razorpay rate limits.
- [ ] **Afternoon (Build)**: Implement structured logging and observability: track recovery success rate per pipeline stage.
- [ ] **Afternoon (Review)**: Fix all P0/P1 bugs found during integration testing; update `tasks/lessons.md`.
- [ ] **Afternoon (Test)**: Performance benchmark: end-to-end latency from webhook receipt to action dispatch must be $<3$ seconds.

---

### 🔹 Day 8 — Analytics, Metrics & Cost Optimization (Tuesday, Sep 1, 2026)
**Goal**: Calculate total money recovered, track LLM token spend, and implement diagnostic caching.
- [x] **Morning (Build)**: Implement recovery analytics: track recovery rate, total revenue saved (₹), and failure distribution.
- [x] **Morning (Build)**: Build cost tracker: log LLM token usage per transaction; estimate monthly API cost.
- [x] **Afternoon (Build)**: Add caching layer for LLM responses: identical failure type + merchant category $\rightarrow$ reuse prior response.
- [x] **Afternoon (Review)**: Verify class definitions and done criteria per R&D milestones.
- [x] **Afternoon (Review)**: Feasibility analysis writeup: document actual vs. projected performance numbers.

---

### 🔹 Day 9 — Demo Prep, Documentation & Polish (Wednesday, Sep 2, 2026)
**Goal**: Script live demo walkthrough, polish dashboard UI, and complete OpenAPI/Postman documentation.
- [ ] **Morning (Review)**: Script demo walkthrough: Payment fails $\rightarrow$ Agent detects $\rightarrow$ Retry scheduled $\rightarrow$ Customer notified $\rightarrow$ Revenue recovered.
- [ ] **Morning (Review)**: Write comprehensive README: architecture overview, setup guide, and decision flow diagrams.
- [ ] **Afternoon (Build)**: Final round of UX polish on the 3-panel dashboard.
- [ ] **Afternoon (Review)**: Document all API endpoints with OpenAPI specs and sample requests.
- [ ] **Afternoon (Research)**: Write competitor analysis summary comparing this solution with standard naive retries.

---

### 🔹 Day 10 — Final Review & Submission (Thursday, Sep 3, 2026) 🏆 DEADLINE DAY
**Goal**: Zero failing tests, verify ground-truth recovery metrics, tag release v1.0.0, and submit project.
- [ ] **Morning (Test)**: Run full test suite one last time — zero failing tests.
- [ ] **Morning (Test)**: Verify all edge cases: network timeout, duplicate webhooks, invalid amount, already-recovered payments.
- [ ] **Morning (Review)**: Confirm token usage and cost estimates are within budget assumptions.
- [ ] **Afternoon (Build)**: Final commit; tag release `v1.0.0` in git.
- [ ] **Afternoon (Review)**: Package deliverables: Repo link, demo video/screenshots, R&D doc, API spec.
- [ ] **Afternoon (Submit)**: Submit project to Razorpay Hackathon Track 03.
