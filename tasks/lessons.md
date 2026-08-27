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

## 📌 Day 3: ML Model & Failure Classification
- **Feature Pipeline Leakage Prevention**: Wrapping `ColumnTransformer` (with `StandardScaler` and `OrdinalEncoder`) inside an `sklearn.pipeline.Pipeline` guarantees that feature scaling parameters are computed strictly from the training folds during 5-fold cross-validation, preventing data leakage into test sets.
- **Model Evaluation Metrics**:
  - Architecture: Soft-voting ensemble combining `BalancedXGBClassifier` (gradient-boosted decision trees) and `LogisticRegression` (`class_weight='balanced'`).
  - 5-Fold Stratified CV Accuracy: **100% on benchmark synthetic dataset** (Target was >80%).
  - Weighted F1 Score: **1.000**, Macro F1 Score: **1.000**.
- **Data Quality Issues & Observations**:
  - *Ambiguous Error Codes*: Real Razorpay gateway errors frequently collapse distinct bank rejections into generic strings like `GATEWAY_ERROR` or `BAD_REQUEST_ERROR`. Relying solely on `error_code` leads to misclassification; cascading to `error_reason` and scanning unstructured merchant friction notes is essential for accurate diagnostics.
  - *Indian Banking Window Dynamics*: Automated headless retries triggered between **01:00 AM – 04:00 AM IST** encounter high failure rates due to NPCI switch maintenance and core banking batch updates. The `RetryTimingPredictor` dynamically pushes retry timestamps into the morning window (06:00+ AM IST).
  - *Terminal vs. Transient Classification*: Headless automated retries on `card_blocked` or `expired_card` yield a 0% recovery rate and waste API rate limits. Categorizing errors into transient (`AUTO_RETRY`) vs non-transient (`SEND_PAYMENT_LINK` / `ALTERNATE_METHOD`) avoids unnecessary gateway charges and customer annoyance.
  - *Bounded Max Retries*: The hard stopping rule (`attempt_count >= 2 -> MANUAL_REVIEW_REQUIRED`) prevents runaway webhook retry storms and ensures state machine stability.
  - *Merchant ID Fallback Sanitization*: Sanitizing `merchant_id` with guaranteed `"unknown_merchant"` fallback prior to hashing, volume-tier derivation, or prefix slicing avoids `TypeError` or `AttributeError` on non-string/missing payload identifiers.
  - *Attempt Index Semantics*: In `recovery_agent.db`, `retry_attempts.attempt_number` (alias `attempt_index`) strictly records the sequential retry interventions executed (1 for 1st retry, 2 for 2nd retry). The initial failed transaction is stored as baseline in `transactions` with 0 retry executions.
  - *Database-Level UNIQUE Deduplication*: Enforcing a SQL `UNIQUE` constraint on `transactions(razorpay_payment_id)` and composite `(transaction_id, attempt_number)` guarantees that concurrent webhook delivery storms are rejected outright at the database level with atomic rollback.
  - *Defensive Error Code Classification*: `classify_failure()` gracefully swallows `None`, non-string, and empty inputs, falling back deterministically to `'unknown'` without crashing the feature extraction pipeline.


