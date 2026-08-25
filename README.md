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
├── api-integration/      # Webhook ingestion endpoints, payload parsers & Razorpay sandbox client
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

## 📅 10-Day Sprint Roadmap

For the complete daily checklist and progress tracking, see [`tasks/todo.md`](tasks/todo.md).

- **Day 1**: Architecture & Environment Setup *(Current)*
- **Day 2**: Data Pipeline & Razorpay Integration
- **Day 3**: ML Model & Failure Classification
- **Day 4**: LLM Core Agent & Prompt Engineering
- **Day 5**: API Layer & Merchant Dashboard
- **Day 6**: Smart Retry Engine & Notification System
- **Day 7**: Integration Testing & Error Handling
- **Day 8**: Analytics, Metrics & Cost Optimization
- **Day 9**: Demo Prep, Documentation & Polish
- **Day 10**: Final Review & Submission

---

## 🛡️ Core Guardrails & Safety Architecture

- **Deterministic Isolation**: LLMs are strictly diagnostic classifiers and cannot trigger direct financial mutations or database updates without state machine validation.
- **Hard Stopping Rule**: Programmatic cap of `max_retries = 2`.
- **Audit Logging**: Zero state transitions occur without an explicit database transaction commit.
