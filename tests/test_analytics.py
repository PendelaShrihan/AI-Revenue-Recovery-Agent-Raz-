# tests/test_analytics.py
"""Tests for Recovery Analytics Engine and GET /analytics REST endpoint."""

import pytest
from fastapi.testclient import TestClient

from agent.analytics import (
    CANONICAL_CATEGORIES,
    get_recovery_analytics,
    get_revenue_saved,
    get_failure_distribution,
    get_recovery_rate_by_category,
    get_daily_recovery_trend,
)
from agent.db_writer import init_db
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database tables are initialized before each test."""
    init_db()


def test_get_recovery_analytics_schema():
    """Verify get_recovery_analytics returns full required JSON schema."""
    analytics = get_recovery_analytics()

    expected_keys = [
        "total_failures_received",
        "total_recovered",
        "total_permanently_failed",
        "recovery_rate_percent",
        "total_revenue_at_risk",
        "total_revenue_saved",
        "revenue_saved_percent",
        "failure_distribution",
        "recovery_by_category",
        "avg_recovery_time_minutes",
        "generated_at",
    ]
    for key in expected_keys:
        assert key in analytics, f"Missing key in analytics response: {key}"

    # Definition of Done: Recovery rate target 40-60%
    assert 40.0 <= analytics["recovery_rate_percent"] <= 60.0
    assert analytics["total_revenue_saved"] > 0.0
    assert analytics["total_revenue_at_risk"] >= analytics["total_revenue_saved"]

    # Verify all 8 canonical failure categories exist in distribution
    dist = analytics["failure_distribution"]
    assert len(dist) == 8
    for cat in CANONICAL_CATEGORIES:
        assert cat in dist, f"Missing category '{cat}' in failure_distribution"

    # Verify recovery by category structure
    by_cat = analytics["recovery_by_category"]
    assert len(by_cat) == 8
    for cat in CANONICAL_CATEGORIES:
        assert cat in by_cat, f"Missing category '{cat}' in recovery_by_category"
        assert "recovered" in by_cat[cat]
        assert "failed" in by_cat[cat]
        assert "rate" in by_cat[cat]


def test_analytics_helper_functions():
    """Verify individual analytics helper query functions."""
    # Revenue saved
    saved = get_revenue_saved()
    assert isinstance(saved, (int, float))
    assert saved > 0

    # Failure distribution
    dist = get_failure_distribution()
    assert isinstance(dist, dict)
    assert len(dist) == 8
    for cat in CANONICAL_CATEGORIES:
        assert cat in dist

    # Recovery rate by category
    rates = get_recovery_rate_by_category()
    assert isinstance(rates, dict)
    assert len(rates) == 8
    for cat in CANONICAL_CATEGORIES:
        assert cat in rates
        assert 0.0 <= rates[cat]["rate"] <= 100.0

    # Daily trend
    trend = get_daily_recovery_trend(days=7)
    assert isinstance(trend, list)
    assert len(trend) == 7
    for day in trend:
        assert "date" in day
        assert "total_failures" in day
        assert "recovered_count" in day
        assert "revenue_saved" in day
        assert "recovery_rate_percent" in day


def test_get_analytics_rest_endpoint():
    """Verify GET /analytics and GET /api/analytics endpoints return HTTP 200."""
    for path in ["/analytics", "/api/analytics"]:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()

        assert "recovery_rate_percent" in data
        assert 40.0 <= data["recovery_rate_percent"] <= 60.0
        assert "total_revenue_saved" in data
        assert "failure_distribution" in data
        assert "recovery_by_category" in data
        assert len(data["failure_distribution"]) == 8
