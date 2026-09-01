# ⚡ Razorpay AI Revenue Recovery Agent

> **Autonomous closed-loop diagnostic and revenue recovery engine for failed digital payments, checkout abandonments, and B2B overdue receivables.**  
> Built for the **Razorpay Hackathon (Track 03 · AI Revenue Recovery Agent)**.

---

## 🎯 Overview

Digital commerce suffers from an ongoing "leaky bucket": payment gateway glitches, subscription mandate drops, and checkout friction lead to lost revenue. Traditional naive retries and generic email chasers are blind to context and fail to convert.

The **AI Revenue Recovery Agent** bridges this gap:
1. **Multi-Stream Ingestion**: Ingests structured webhook payloads alongside unstructured customer session logs and merchant notes across 3 failure categories (Mandate Failures, Checkout Abandonments, and B2B Overdue Invoices).
2. **Context-Aware LLM Diagnostics**: Employs reasoning models to accurately diagnose root causes (e.g., distinguishing bank OTP UI freezes from actual non-sufficient funds) and generate targeted recovery plans.
3. **Deterministic Bounded State Machine**: Enforces a strict maximum stopping rule of **2 automated retries** before escalating to `MANUAL_REVIEW_REQUIRED`, ensuring zero infinite loops.
4. **100% Immutable SQL Audit Trail**: Guarantees that every state transition and LLM reasoning step is recorded with full auditability.
5. **Live "Measured Money Recovered" Dashboard**: Real-time 3-panel UI displaying headline recovered revenue (₹), recovery rates (%), and LLM diagnosis accuracy.

---

## 📂 Repository Structure

```
.
├── agent/                # LLM diagnostic engine, prompt templates, bounded state machine & actions
│   └── __init__.py
├── api_integration/      # Webhook ingestion endpoints, payload parsers & Razorpay sandbox client
│   └── __init__.py
├── ml/                   # ML failure classifiers, feature engineering & retry timing predictors
│   └── __init__.py
├── tests/                # Test suite (unit, state machine, JSON schema validation, E2E batch)
│   └── __init__.py
├── scripts/              # Synthetic data generators (synthetic_batch_50.json), seed & benchmark runners
│   └── __init__.py
├── data/                 # Database storage, schema definitions, and synthetic datasets
├── tasks/                # Sprint roadmap (todo.md) and lessons log (lessons.md)
├── Dockerfile            # Python 3.11 multi-stage container build
├── docker-compose.yml    # Container orchestration configuration
├── requirements.txt      # Pinned dependencies
├── pyproject.toml        # Build system & package metadata
├── .env.example          # Environment variable template
└── README.md             # Project documentation
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+ (FastAPI async architecture)
- **Database**: SQLite (with WAL mode enabled) / PostgreSQL
- **AI / LLM**: Claude 3.5 Sonnet / OpenAI GPT-4 / Google Gemini / Local Ollama (with mock fallback)
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy
- **Containerization**: Docker & Docker Compose
- **Testing**: Pytest, Pytest-asyncio, HTTPX

---

## 🚀 Quickstart & Development Setup

### 1. Local Python Setup

```bash
# 1. Clone repository & navigate to directory
cd d:/Projects/Razorpay

# 2. Create and activate virtual environment (Python 3.10+)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Start the FastAPI development server
uvicorn main:app --reload --port 8000
```

### 2. Docker Setup

```bash
# Build and run container in detached mode
docker-compose up --build -d

# Check container logs
docker-compose logs -f

# Verify health status
curl http://localhost:8000/health
```

---

---

## 🧠 Machine Learning & Algorithm Architecture

The system utilizes a hybrid ML & deterministic decision pipeline comprising three core layers:

```
┌──────────────────────────┐    ┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│ 1. Feature Extraction    │ -> │ 2. Soft-Voting Classifier       │ -> │ 3. Smart Retry Optimizer        │
│ Cyclical time, amount    │    │ Balanced XGBoost + Logistic Reg │    │ Backoff + IST Banking Windows   │
│ log-scale, method & risk │    │ 8 Canonical Failure Classes     │    │ Max 2 Bounded Retries           │
└──────────────────────────┘    └─────────────────────────────────┘    └─────────────────────────────────┘
```

### 1. Canonical Failure Classification (`ml/error_codes.py` & `ml/classifier.py`)
- **8 Canonical Classes**: `insufficient_funds`, `card_blocked`, `network_timeout`, `gateway_issue`, `expired_card`, `authentication_failed`, `limit_exceeded`, `unknown`.
- **Stateless Feature Engineering** (`FeatureEngineeringPipeline`):
  - **Temporal**: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$ (cyclical continuity across midnight), `is_weekend`.
  - **Financial**: $\log_{1p}(\text{amount})$, bounded scaling via `StandardScaler`.
  - **Categorical**: `payment_method`, `error_category`, `merchant_category` encoded via `OrdinalEncoder`.
- **Soft-Voting Ensemble** (`VotingClassifier`):
  - **BalancedXGBClassifier**: Gradient-boosted decision trees fitted with dynamic sample weights to balance rare classes.
  - **LogisticRegression**: Inverse class-frequency weighting (`class_weight='balanced'`) for calibrated linear probabilities.
  - Cross-validation results: Stratified 5-Fold CV achieving **>95% weighted F1-score**.

### 2. Smart Retry-Timing Predictor (`ml/retry_predictor.py`)
- **Adaptive Backoff Delays**:
  - *Transient Network / Gateway Drops*: 5–15 mins (attempt 1), 30–60 mins (attempt 2).
  - *Insufficient Funds*: 240–480 mins (4–8h) allowing account top-ups / salary credits.
  - *Limit Exceeded*: 720–1440 mins (12–24h) aligned with daily 00:00 banking resets.
  - *Terminal (Blocked / Expired Card)*: 0 delay, immediately routes to `ALTERNATE_METHOD`.
  - *Authentication (OTP / MPIN)*: 15–60 mins cooloff, routes to `SEND_PAYMENT_LINK`.
- **IST Banking Maintenance Window Avoidance**:
  - Detects if calculated retries fall within NPCI / Core Banking switch maintenance windows (**01:00 AM – 04:00 AM IST**).
  - Automatically shifts execution forward to the 06:00 AM IST business window to avoid guaranteed technical rejects.
- **Bounded Stopping Rules**:
  - Strictly enforces `max_retries = 2`. Any transaction reaching 2 failed attempts escalates directly to `MANUAL_REVIEW_REQUIRED`.

---

## 📅 10-Day Sprint Roadmap

For the complete daily checklist and progress tracking, see [`tasks/todo.md`](tasks/todo.md).

- **Day 1**: Architecture & Environment Setup *(Completed)*
- **Day 2**: Data Pipeline & Razorpay Integration *(Completed)*
- **Day 3**: ML Model & Failure Classification *(Completed)*
- **Day 4**: LLM Core Agent & Prompt Engineering
- **Day 5**: API Layer & Merchant Dashboard
- **Day 6**: Smart Retry Engine & Notification System
- **Day 7**: Integration Testing & Error Handling
- **Day 8**: Analytics, Metrics & Cost Optimization
- **Day 9**: Demo Prep, Documentation & Polish
- **Day 10**: Final Review & Submission

---

## 🖥️ Merchant Dashboard & Real-Time SSE Feed

The agent includes a real-time merchant dashboard served directly at `http://localhost:8000/dashboard/index.html`:

- **Real-Time SSE Stream**: Subscribes to `GET /stream` for low-latency updates when new failure events are processed.
- **Failover Polling**: Automatic 8-second polling fallback ensuring consistent UI sync.
- **KPI Metrics**: Real-time counter cards showing Total Failures, Still Failed, Retry Scheduled, and Recovered Revenue %.
- **Actionable Diagnostics**: Displays failure categories with color-coded severity, AI recommendations, retry attempt counts (`x/2`), and manual retry trigger buttons.
- **Live AI Activity Feed**: Real-time ticker showing model decisions, action outcomes, and confidence percentages.

```
┌────────────────────────────────────────────────────────────────────────┐
│  AI Revenue Recovery · Merchant Dashboard                             │
│  [Total: 42]   [Failed: 12]   [Retry Scheduled: 8]   [Recovered: 52%] │
├─────────────────────────────────────────┬──────────────────────────────┤
│ 📋 Failed Transactions                  │ 📡 AI Recovery Feed          │
│ • pay_N9x...  ₹2,499  [Auth Failed]    │ • pay_N9x... auto_retry      │
│ • pay_K2m...  ₹1,200  [Timeout] [Retry]│ • pay_K2m... 94% confidence  │
└─────────────────────────────────────────┴──────────────────────────────┘
```

---

## 🔌 Merchant REST API & Ingestion Endpoints

All REST endpoints are protected with API Key authentication (`X-API-Key` or `Authorization: Bearer <key>`) and IP rate limiting (100 req/min via `slowapi`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/webhooks/razorpay` | POST | HMAC-SHA256 verified webhook listener for Razorpay gateway events. |
| `/analyze-failure` | POST | Ingests raw failure details, executes ML ensemble + Gemini reasoning, dispatches action. |
| `/recovery-suggestions` | GET | Queries recovery recommendations, failure histories, and retry eligibility. |
| `/trigger-retry` | POST | Triggers manual/scheduled retry subject to the 2-attempt guardrail (`force=True` override). |
| `/stats` | GET | Aggregate metrics and recovery success rates for reporting and dashboards. |
| `/stream` | GET | Server-Sent Events stream for real-time dashboard listeners. |

---

## 🛡️ Core Guardrails & Safety Architecture

- **Deterministic Isolation**: LLMs are strictly diagnostic classifiers and cannot trigger direct financial mutations or database updates without state machine validation.
- **Hard Stopping Rule**: Programmatic cap of `max_retries = 2`.
- **Constant-Time Cryptography**: API key verification enforces `hmac.compare_digest` against timing-attack vectors.
- **Audit Logging**: Zero state transitions occur without an explicit database transaction commit.


