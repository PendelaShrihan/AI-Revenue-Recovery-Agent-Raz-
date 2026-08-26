"""
Pydantic Schemas for Razorpay Webhook Ingestion & Payload Normalization.
Defines the canonical NormalizedEvent schema across all payment failure streams.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class EventType(str, Enum):
    """Supported Razorpay Webhook Event Types."""
    PAYMENT_FAILED = "payment.failed"
    SUBSCRIPTION_HALTED = "subscription.halted"
    INVOICE_OVERDUE = "invoice.overdue"
    INVOICE_EXPIRED = "invoice.expired"
    INVOICE_PAID = "invoice.paid"
    PAYMENT_AUTHORIZED = "payment.authorized"
    ORDER_PAID = "order.paid"
    UNKNOWN = "unknown"


class FailureCategory(str, Enum):
    """High-level classification of failure stream."""
    CHECKOUT_FAILURE = "checkout_failure"      # One-time checkout drops (card decline, OTP freeze, UPI timeout)
    MANDATE_FAILURE = "mandate_failure"        # Recurring billing / e-mandate drops
    INVOICE_OVERDUE = "invoice_overdue"        # B2B delayed receivables / overdue commercial invoice
    INFORMATIONAL = "informational"            # Non-failure events (e.g. order.paid, payment.authorized)
    UNKNOWN = "unknown"


class NormalizedEvent(BaseModel):
    """
    Unified, clean internal event representation for all Razorpay webhook events.
    Decouples downstream ML, LLM diagnostics, and recovery engines from raw gateway structures.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    event_id: str = Field(..., description="Unique event identifier or synthetic hash")
    event_type: str = Field(..., description="Original Razorpay webhook event name")
    failure_category: FailureCategory = Field(..., description="Internal failure categorization")
    entity_type: str = Field(..., description="Primary entity type: payment, subscription, invoice")
    entity_id: str = Field(..., description="Primary identifier (e.g. pay_xxx, sub_xxx, inv_xxx)")
    merchant_id: str = Field(default="unknown", description="Razorpay Account ID / Merchant ID")
    amount: float = Field(..., description="Normalized monetary amount in standard currency units (e.g. INR)")
    currency: str = Field(default="INR", description="Three-letter ISO currency code")
    status: str = Field(..., description="Normalized lifecycle status (e.g. FAILED, HALTED, OVERDUE)")
    
    # Associated Identifiers
    payment_id: Optional[str] = Field(default=None, description="Razorpay payment ID if present")
    order_id: Optional[str] = Field(default=None, description="Razorpay order ID if present")
    subscription_id: Optional[str] = Field(default=None, description="Razorpay subscription ID if present")
    invoice_id: Optional[str] = Field(default=None, description="Razorpay invoice ID if present")
    
    # Customer Metadata
    customer_id: Optional[str] = Field(default=None, description="Razorpay customer ID")
    customer_name: Optional[str] = Field(default=None, description="Customer full name")
    customer_email: Optional[str] = Field(default=None, description="Customer email address")
    customer_phone: Optional[str] = Field(default=None, description="Customer contact phone number")
    
    # Payment Method & Technical Diagnostics
    payment_method: Optional[str] = Field(default=None, description="Payment method: card, upi, netbanking, etc.")
    error_code: Optional[str] = Field(default=None, description="Standardized error code (e.g. BAD_REQUEST_ERROR)")
    error_description: Optional[str] = Field(default=None, description="Detailed gateway error description")
    error_source: Optional[str] = Field(default=None, description="Error source: customer, bank, gateway, business")
    error_step: Optional[str] = Field(default=None, description="Step where failure occurred: payment_authentication, etc.")
    error_reason: Optional[str] = Field(default=None, description="Granular error reason: insufficient_funds, etc.")
    
    # Unstructured Context & Raw Artifacts
    notes: Dict[str, Any] = Field(default_factory=dict, description="Unstructured merchant, checkout friction, or PO notes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of event occurrence")
    raw_payload: Dict[str, Any] = Field(default_factory=dict, description="Raw original webhook JSON payload for audit trail")

    def to_transaction_dict(self) -> Dict[str, Any]:
        """Maps normalized event into a dictionary compatible with the Transaction ORM model."""
        return {
            "id": f"tx_{self.entity_id}",
            "razorpay_payment_id": self.payment_id or self.entity_id,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status,
            "failure_reason": self.error_description or self.error_reason or f"Event {self.event_type} received",
            "failure_code": self.error_code or self.error_reason or "UNKNOWN_ERROR",
            "created_at": self.created_at
        }


class WebhookResponse(BaseModel):
    """Response returned to Razorpay or client upon receiving a webhook."""
    status: str = Field(default="success", description="Status indicator: success, error, ignored")
    event_id: Optional[str] = Field(default=None, description="Event ID processed")
    event_type: Optional[str] = Field(default=None, description="Event type processed")
    action_taken: str = Field(..., description="Action taken by the webhook listener")
    message: str = Field(..., description="Human-readable processing summary")
    normalized_event: Optional[NormalizedEvent] = Field(default=None, description="Normalized event payload summary")
