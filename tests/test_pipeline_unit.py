# tests/test_pipeline_unit.py
"""Unit tests for Recovery Pipeline (agent/pipeline.py).

Tests:
    - run_recovery_pipeline() end-to-end integration flow
"""

from unittest.mock import MagicMock, patch
import pytest

from agent.pipeline import run_recovery_pipeline
from agent.db_writer import init_db
from agent.llm_agent import RecoveryDecision
from api_integration.schemas import NormalizedEvent, FailureCategory


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """Use temporary SQLite DB for pipeline testing."""
    db_file = tmp_path / "test_pipeline.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    yield engine


@patch("agent.pipeline._get_recovery_engine")
def test_run_recovery_pipeline_success(mock_get_engine):
    # Mock Gemini decision
    mock_engine = MagicMock()
    mock_decision = RecoveryDecision(
        action="suggest_alternate_method",
        priority="high",
        message="Please pay via UPI.",
        retry_after=0,
        alternate_method="upi",
        confidence=0.95,
        reasoning="Insufficient funds detected.",
    )
    mock_engine.process.return_value = mock_decision
    mock_get_engine.return_value = mock_engine

    event = NormalizedEvent(
        event_id="evt_pipe_unit_001",
        event_type="payment.failed",
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id="pay_PIPE_UNIT_001",
        merchant_id="merchant_pipe_01",
        amount=2499.0,
        currency="INR",
        status="FAILED",
        error_code="BAD_REQUEST_ERROR",
        error_reason="insufficient_funds",
        error_description="Insufficient balance",
    )

    summary = run_recovery_pipeline(event)

    assert summary["transaction_id"] == "tx_pay_PIPE_UNIT_001"
    assert summary["failure_category"] == "insufficient_funds"
    assert summary["action_taken"] == "suggest_alternate_method"
    assert summary["alternate_method"] == "upi"
    assert summary["status"] == "recovery_initiated"
    assert summary["db_record_id"] is not None
    assert summary["confidence"] == 0.95
