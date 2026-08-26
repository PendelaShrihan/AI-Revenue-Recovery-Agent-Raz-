# Sprint Lessons & Architecture Log: AI Revenue Recovery Agent

## 📌 Day 1: Architecture & Environment Setup
- **Architectural Boundary**: LLM is strictly isolated as a diagnostic engine and is programmatically prevented from mutating the database or triggering financial mutations directly.
- **Stopping Rule**: Max 2 retry attempts hard-coded into state transitions to eliminate infinite loop risks.
- **Multi-Event Ingestion**: Ingests across 3 core streams (Mandate Failures, Checkout Abandonments with friction notes, and B2B Overdue Invoices).
- **Environment**: Python 3.11 with FastAPI async core + SQLite WAL mode (upgrade path to PostgreSQL).

## 📌 Day 2: Webhooks & Ingestion Normalizer
- **Package Naming Convention**: Renamed `api-integration/` to PEP 8-compliant `api_integration/` to permit standard Python module importing.
- **Raw Request Hashing**: Webhook HMAC-SHA256 signature verification MUST operate on raw `request.body()` bytes prior to JSON parsing; re-encoding or formatting alters hashes and breaks cryptographic validation.
- **Paise to Standard Currency**: Razorpay sends amounts in integer paise; all ingestion schemas must divide by 100.0 (`_to_rupees`) while preserving decimal precision for financial auditing.
- **Heterogeneous Event Mapping**: `subscription.halted` and `invoice.overdue` have different entity layouts compared to standard `payment.failed` events; normalizing to a single `NormalizedEvent` guarantees downstream isolation.

