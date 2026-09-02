"""
Razorpay AI Revenue Recovery Agent - Agent Module
Contains LLM diagnostic reasoning, database write layer, recovery actions, and pipeline.
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

from agent.action_engine import (
    execute_auto_retry,
    execute_alternate_suggestion,
    execute_customer_notification,
    dispatch_recovery_action,
)

from agent.retry_scheduler import (
    schedule_retry,
    get_retry_status,
)

from agent.retry_executor import (
    execute_retry,
)

from agent.notification_engine import (
    draft_whatsapp_message,
    draft_sms_message,
    draft_email_message,
    generate_personalized_notification,
    dispatch_customer_notification,
    FAILURE_GUIDANCE,
)

from agent.pipeline import (
    run_recovery_pipeline,
    run_pending_retries,
)

from agent.broadcaster import (
    broadcast,
    subscribe,
    unsubscribe,
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
    # Action engine
    "execute_auto_retry",
    "execute_alternate_suggestion",
    "execute_customer_notification",
    "dispatch_recovery_action",
    # Retry Scheduler & Executor
    "schedule_retry",
    "get_retry_status",
    "execute_retry",
    # Notification & Communication Engine
    "draft_whatsapp_message",
    "draft_sms_message",
    "draft_email_message",
    "generate_personalized_notification",
    "dispatch_customer_notification",
    "FAILURE_GUIDANCE",
    # Full pipeline
    "run_recovery_pipeline",
    "run_pending_retries",
]
