# 🎯 AI Revenue Recovery Agent — MVP Scope Definition

**Project**: AI Revenue Recovery Agent (Razorpay Hackathon Track 03)  
**Document**: `agent/SCOPE.md`  
**Status**: APPROVED MVP DEFINITION  
**Sprint Phase**: Phase 1 (Foundation & Setup)

---

## 📌 Executive Summary

The MVP focuses strictly on building an autonomous diagnostic and revenue recovery engine for failed digital payments, subscription drops, and overdue receivables. It captures failure events, diagnoses root causes via LLM context parsing, executes bounded retry/recovery actions, and logs 100% of state transitions to an audit trail.

---

## ✅ IN SCOPE (MVP Focus)

1. **Payment Failure Detection & Ingestion**:
   - Webhook reception and normalization for digital payment errors, subscription drop-offs, and B2B receivables.
   - Handling structured failure codes (`BAD_REQUEST_ERROR`, `GATEWAY_ERROR`, `insufficient_funds`, `card_blocked`, `network_timeout`).
   - Parsing unstructured contextual notes (e.g., checkout friction text, OTP freeze reports, merchant dispute logs).

2. **LLM-Powered Failure Classification**:
   - Extraction and reasoning over unstructured session data to distinguish transient gateway issues, user friction, and balance problems.
   - Enforcing strict JSON structured output (`diagnosis`, `recovery_action`, `confidence`).
   - Fallback to deterministic rules when LLM confidence is low or malformed.

3. **Intelligent Bounded Retry Engine**:
   - Predictive retry delay calculation based on failure category and time-of-day.
   - Hard stopping rule: **Maximum 2 automated retries per transaction** before state escalation.
   - Elimination of infinite retry loops.

4. **Customer Recovery Notification Drafts**:
   - Context-aware recovery messages for WhatsApp / SMS / Email.
   - Localized Hinglish messaging templates for high-friction checkout drop-offs.
   - Dynamic payment link inclusion.

5. **Merchant Dashboard (3-Panel Basic UI)**:
   - **Panel 1**: Batch Summary Header (Total ₹ recovered, Recovery Rate %, Diagnosis Accuracy %).
   - **Panel 2**: Live Transaction State Table.
   - **Panel 3**: Chronological SQL Audit Log Feed.

6. **REST API Layer**:
   - Endpoints for `POST /analyze-failure`, `GET /recovery-suggestions`, `POST /trigger-retry`, and `POST /batch-ingest`.
   - Health check and environment verification endpoints.

---

## 🚫 OUT OF SCOPE (Deferred to Future Phases)

The following modules are explicitly excluded from the MVP sprint to prevent scope creep and ensure core execution excellence:

1. **Fraud Detection & Risk Scoring Module**:
   - Real-time fraud heuristic engines, velocity checks, and anti-money laundering (AML) workflows.
2. **Dynamic Pricing & Discount Engine**:
   - Automated discount voucher generation, margin-based price concessions, or dynamic checkout markdown algorithms.
3. **Advanced Analytics & Long-term BI**:
   - Multi-year cohort lifetime value (LTV) modeling, cross-merchant macro analytics, and custom data warehouse export connectors.
4. **Live Telephony / Voice Bot Recovery**:
   - Interactive voice response (IVR) phone dialing bots (deferred to future roadmap).
5. **Direct Production Financial Mutations**:
   - Real-money card debiting outside of the Razorpay Sandbox environment.

---

## ⚡ Handled Razorpay Webhook Events

The AI Revenue Recovery Agent processes the following 3 core Razorpay webhook events:

| # | Razorpay Webhook Event | Scenario & Purpose | Target Recovery Action |
|---|---|---|---|
| 1 | `payment.failed` | Standard checkout & one-time payment failures (card decline, network timeout, UPI OTP drop). | Smart retry scheduling or customized payment link dispatch (with Hinglish copy). |
| 2 | `subscription.halted` | Recurring billing / mandate drops (e.g. expired mandate, recurring charge decline). | Mandate update notification / re-authorization link generation. |
| 3 | `invoice.overdue` | B2B receivables and commercial invoice overdue events with merchant relationship notes. | Multi-tier invoice chaser sequence with 2-step escalation. |

---

## 🛡️ Non-Negotiable Operational Guardrails

- **Strict Retry Cap**: `max_retries = 2`.
- **State Machine Isolation**: The LLM serves solely as a classification advisor and cannot mutate database records or initiate transactions directly.
- **Audit Logging**: Every single state change must be persisted in SQL before executing the corresponding action.
