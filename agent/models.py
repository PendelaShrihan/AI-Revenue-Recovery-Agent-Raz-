"""
SQLAlchemy ORM Models for AI Revenue Recovery Agent.
Maps to tables: transactions, retry_attempts, and recovery_actions.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    Numeric,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Transaction(Base):
    """
    Transactions table model.
    Stores core transaction state, failure reasons, and recovery progress.
    Enforces UNIQUE constraint on razorpay_payment_id to prevent duplicate inserts at DB level.
    """
    __tablename__ = "transactions"

    id = Column(String(64), primary_key=True, index=True)
    razorpay_payment_id = Column(String(64), nullable=False, unique=True, index=True)
    merchant_id = Column(String(64), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="INR", nullable=False)
    status = Column(String(32), default="FAILED", nullable=False, index=True)
    failure_reason = Column(Text, nullable=True)
    failure_code = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    retry_attempts = relationship(
        "RetryAttempt",
        back_populates="transaction",
        cascade="all, delete-orphan",
        order_by="RetryAttempt.attempt_number"
    )
    recovery_actions = relationship(
        "RecoveryAction",
        back_populates="transaction",
        cascade="all, delete-orphan",
        order_by="RecoveryAction.created_at"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "razorpay_payment_id": self.razorpay_payment_id,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "failure_code": self.failure_code,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "retry_count": len(self.retry_attempts) if self.retry_attempts else 0
        }

    def __repr__(self) -> str:
        return f"<Transaction(id='{self.id}', payment_id='{self.razorpay_payment_id}', status='{self.status}', amount={self.amount})>"


class RetryAttempt(Base):
    """
    Retry Attempts table model.
    Tracks each automated recovery retry execution and outcomes (enforces max 2 attempts).

    NOTE ON ATTEMPT INDEXING:
    `attempt_number` (and its alias property `attempt_index`) strictly counts
    the sequential retry attempts executed by the recovery agent (1 for 1st retry,
    2 for 2nd retry). It does NOT include the initial failure transaction
    (which is stored in the parent `transactions` table).
    """
    __tablename__ = "retry_attempts"
    __table_args__ = (
        UniqueConstraint("transaction_id", "attempt_number", name="uq_retry_attempts_tx_attempt"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, index=True)
    attempt_number = Column(Integer, nullable=False)
    attempted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    result = Column(String(32), nullable=False)  # SUCCESS, FAILED, TIMEOUT
    next_retry_at = Column(DateTime, nullable=True)

    # Relationship
    transaction = relationship("Transaction", back_populates="retry_attempts")

    @property
    def attempt_index(self) -> int:
        """Alias for attempt_number: tracks retry attempts performed (1 or 2)."""
        return self.attempt_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "attempt_number": self.attempt_number,
            "attempt_index": self.attempt_index,
            "attempted_at": self.attempted_at.isoformat() if self.attempted_at else None,
            "result": self.result,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None
        }

    def __repr__(self) -> str:
        return f"<RetryAttempt(id={self.id}, tx='{self.transaction_id}', attempt={self.attempt_number}, result='{self.result}')>"


class RecoveryAction(Base):
    """
    Recovery Actions table model.
    Stores actions taken by the recovery engine (smart retries, payment links, customer notifications).
    """
    __tablename__ = "recovery_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, index=True)
    action_type = Column(String(64), nullable=False)  # RETRY, PAYMENT_LINK, INVOICE_CHASER, NOTIFICATION
    action_payload = Column(Text, nullable=True)  # JSON or text payload
    status = Column(String(32), default="PENDING", nullable=False)  # PENDING, EXECUTED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    transaction = relationship("Transaction", back_populates="recovery_actions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "action_type": self.action_type,
            "action_payload": self.action_payload,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f"<RecoveryAction(id={self.id}, tx='{self.transaction_id}', type='{self.action_type}', status='{self.status}')>"


class LLMCost(Base):
    """
    LLM Costs table model.
    Tracks token usage, execution latency, and approximate cost per LLM invocation.
    """
    __tablename__ = "llm_costs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), nullable=True, index=True)
    model = Column(String(64), nullable=False)
    input_tokens = Column(Integer, default=0, nullable=False)
    output_tokens = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<LLMCost(id={self.id}, tx='{self.transaction_id}', model='{self.model}', cost=${self.cost_usd:.6f})>"
