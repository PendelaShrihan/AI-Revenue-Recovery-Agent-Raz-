# tests/test_cost_tracker.py
"""Tests for LLM Cost Tracker and GET /costs REST endpoint."""

import pytest
from fastapi.testclient import TestClient

from agent.cost_tracker import (
    COST_PER_1K_INPUT_TOKENS,
    COST_PER_1K_OUTPUT_TOKENS,
    calculate_llm_cost,
    log_llm_call,
    get_cost_summary,
)
from agent.db_writer import init_db, get_db_session
from agent.models import LLMCost
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database tables are initialized before each test."""
    init_db()


def test_calculate_llm_cost_math():
    """Verify formula for calculating LLM cost from token counts."""
    # 1,000 input tokens = $0.000075
    # 1,000 output tokens = $0.000300
    cost = calculate_llm_cost(1000, 1000)
    expected = 0.000075 + 0.000300
    assert abs(cost - expected) < 1e-6

    # 1,500 input + 300 output tokens
    cost_sample = calculate_llm_cost(1500, 300)
    expected_sample = (1.5 * 0.000075) + (0.3 * 0.000300)
    assert abs(cost_sample - expected_sample) < 1e-6


def test_log_llm_call_persists_to_db():
    """Verify log_llm_call writes record to llm_costs table and returns breakdown."""
    test_tx_id = "tx_cost_test_01"
    breakdown = log_llm_call(
        transaction_id=test_tx_id,
        input_tokens=1200,
        output_tokens=250,
        model="gemini-flash-lite-latest",
        latency_ms=1450.5,
    )

    assert breakdown["transaction_id"] == test_tx_id
    assert breakdown["input_tokens"] == 1200
    assert breakdown["output_tokens"] == 250
    assert breakdown["latency_ms"] == 1450.5
    assert breakdown["cost_usd"] > 0.0
    assert "created_at" in breakdown

    # Verify query from database
    with get_db_session() as s:
        record = s.query(LLMCost).filter_by(transaction_id=test_tx_id).first()
        assert record is not None
        assert record.input_tokens == 1200
        assert record.output_tokens == 250
        assert record.model == "gemini-flash-lite-latest"


def test_get_cost_summary_structure():
    """Verify get_cost_summary returns all required fields and respects constraints."""
    summary = get_cost_summary()

    required_keys = [
        "total_llm_calls",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
        "cost_per_recovery_usd",
        "estimated_monthly_cost_usd",
        "estimated_monthly_cost_inr",
        "model_used",
        "avg_latency_ms",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key in cost summary: {key}"

    # Definition of Done: cost per recovery under $0.001
    assert summary["cost_per_recovery_usd"] < 0.001
    assert summary["estimated_monthly_cost_usd"] >= 0.0
    assert summary["estimated_monthly_cost_inr"] >= 0.0


def test_get_costs_rest_endpoint():
    """Verify GET /costs and GET /api/costs return HTTP 200 and valid JSON."""
    for path in ["/costs", "/api/costs"]:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_llm_calls" in data
        assert "cost_per_recovery_usd" in data
        assert data["cost_per_recovery_usd"] < 0.001
        assert "estimated_monthly_cost_usd" in data
        assert "estimated_monthly_cost_inr" in data
