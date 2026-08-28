"""
Razorpay AI Revenue Recovery Agent - Agent Module
Contains LLM diagnostic reasoning, database write layer, and recovery actions.
"""

from agent.models import (
    Base,
    Transaction,
    RetryAttempt,
    RecoveryAction
)

from agent.db_writer import (
    get_db_engine,
    get_db_session,
    init_db,
    save_transaction,
    save_recovery_action,
    update_transaction_status,
    save_retry_attempt,
    get_transaction_by_payment_id,
    get_all_transactions
)

from agent.llm_agent import (
    GeminiAgent,
    GeminiAgentError,
    GeminiOutputParseError,
    RecoveryDecision,
    SYSTEM_PROMPT,
    build_cot_prompt,
)

from agent.recovery_engine import (
    RecoveryEngine,
    get_recovery_decision,
)

__version__ = "0.1.0"

__all__ = [
    # ORM models
    "Base",
    "Transaction",
    "RetryAttempt",
    "RecoveryAction",
    # DB layer
    "get_db_engine",
    "get_db_session",
    "init_db",
    "save_transaction",
    "save_recovery_action",
    "update_transaction_status",
    "save_retry_attempt",
    "get_transaction_by_payment_id",
    "get_all_transactions",
    # LLM wrapper
    "GeminiAgent",
    "GeminiAgentError",
    "GeminiOutputParseError",
    "RecoveryDecision",
    "SYSTEM_PROMPT",
    "build_cot_prompt",
    # Recovery engine
    "RecoveryEngine",
    "get_recovery_decision",
]
