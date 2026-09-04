# tests/test_llm_cache.py
"""Unit tests for the LLM Response Caching Layer.

Verifies:
- Key normalization (failure_type + merchant_category)
- Cache hit, miss, and LRU eviction
- TTL expiration behavior
- Metric tracking (hits, misses, hit_rate_pct, cost_saved)
- Integration with RecoveryEngine (zero API calls on identical scenarios)
- REST API cache endpoints (/cache/stats and /cache/clear)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.llm_agent import RecoveryDecision
from agent.llm_cache import LLMResponseCache, get_llm_cache
from agent.recovery_engine import RecoveryEngine
from api_integration.schemas import EventType, FailureCategory, NormalizedEvent
from main import app


@pytest.fixture
def clean_cache() -> LLMResponseCache:
    """Provides a fresh, isolated cache instance."""
    cache = LLMResponseCache(max_size=10, ttl_seconds=3600.0)
    cache.clear()
    return cache


@pytest.fixture
def sample_decision() -> RecoveryDecision:
    """Fixture for a standard RecoveryDecision."""
    return RecoveryDecision(
        action="auto_retry",
        priority="medium",
        message="Payment failed due to temporary network timeout. Automatically retrying.",
        retry_after=900,
        alternate_method="upi",
        confidence=0.92,
        reasoning="Network timeouts are transient; retry is optimal.",
        raw_response='{"action":"auto_retry"}',
        latency_ms=1420.5,
        cost_usd=0.00021,
        model="gemini-flash-lite-latest",
        cached=False,
    )


def test_key_normalization():
    """Key normalization strips whitespace and coerces lowercase."""
    assert LLMResponseCache.normalize_key(" Insufficient_Funds ", " Ecommerce ") == ("insufficient_funds", "ecommerce")
    assert LLMResponseCache.normalize_key(None, None) == ("unknown", "unknown")
    assert LLMResponseCache.normalize_key("", "   ") == ("unknown", "unknown")


def test_cache_miss_and_hit(clean_cache: LLMResponseCache, sample_decision: RecoveryDecision):
    """Setting an entry allows subsequent retrieval with cached metadata."""
    # 1. Miss on empty cache
    res = clean_cache.get("insufficient_funds", "ecommerce")
    assert res is None
    stats = clean_cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0

    # 2. Set decision
    clean_cache.set("insufficient_funds", "ecommerce", sample_decision)
    assert clean_cache.get_stats()["total_entries"] == 1

    # 3. Hit on matching key
    cached = clean_cache.get("insufficient_funds", "ecommerce")
    assert cached is not None
    assert cached.cached is True
    assert cached.latency_ms == 0.0
    assert cached.cost_usd == 0.0
    assert cached.action == sample_decision.action
    assert cached.message == sample_decision.message
    assert cached.retry_after == sample_decision.retry_after
    assert cached.alternate_method == sample_decision.alternate_method

    # 4. Telemetry verification
    stats = clean_cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0
    assert stats["cost_saved_usd"] > 0.0


def test_cache_key_specificity(clean_cache: LLMResponseCache, sample_decision: RecoveryDecision):
    """Cache returns None if failure_type OR merchant_category differs."""
    clean_cache.set("network_timeout", "electronics", sample_decision)

    # Different merchant category -> miss
    assert clean_cache.get("network_timeout", "fashion") is None

    # Different failure type -> miss
    assert clean_cache.get("insufficient_funds", "electronics") is None

    # Exact match -> hit
    hit = clean_cache.get("network_timeout", "electronics")
    assert hit is not None
    assert hit.action == "auto_retry"


def test_cache_invalidation(clean_cache: LLMResponseCache, sample_decision: RecoveryDecision):
    """Invalidate by failure type, merchant category, or full purge."""
    clean_cache.set("card_blocked", "ecommerce", sample_decision)
    clean_cache.set("card_blocked", "saas", sample_decision)
    clean_cache.set("expired_card", "ecommerce", sample_decision)
    assert clean_cache.get_stats()["total_entries"] == 3

    # Invalidate by merchant category
    purged = clean_cache.invalidate(merchant_category="saas")
    assert purged == 1
    assert clean_cache.get("card_blocked", "saas") is None
    assert clean_cache.get("card_blocked", "ecommerce") is not None

    # Invalidate by failure type
    purged = clean_cache.invalidate(failure_type="card_blocked")
    assert purged == 1
    assert clean_cache.get("card_blocked", "ecommerce") is None

    # Invalidate all remaining
    clean_cache.clear()
    assert clean_cache.get_stats()["total_entries"] == 0


def test_lru_capacity_eviction(sample_decision: RecoveryDecision):
    """Least recently accessed item is evicted when max_size is exceeded."""
    cache = LLMResponseCache(max_size=2, ttl_seconds=3600.0)
    cache.set("cat_1", "merch_1", sample_decision)
    time.sleep(0.01)
    cache.set("cat_2", "merch_2", sample_decision)

    # Access cat_1 so cat_2 becomes the least recently used
    time.sleep(0.01)
    assert cache.get("cat_1", "merch_1") is not None

    # Adding a 3rd item should evict cat_2
    time.sleep(0.01)
    cache.set("cat_3", "merch_3", sample_decision)

    assert cache.get("cat_1", "merch_1") is not None
    assert cache.get("cat_3", "merch_3") is not None
    assert cache.get("cat_2", "merch_2") is None  # Evicted!


def test_ttl_expiration(sample_decision: RecoveryDecision):
    """Entries expire once age exceeds ttl_seconds."""
    cache = LLMResponseCache(max_size=10, ttl_seconds=0.05)  # 50ms TTL
    cache.set("timeout", "saas", sample_decision)

    # Immediately accessible
    assert cache.get("timeout", "saas") is not None

    # Wait for TTL to lapse
    time.sleep(0.08)
    assert cache.get("timeout", "saas") is None


def test_concurrent_access_thread_safety(sample_decision: RecoveryDecision):
    """Cache handles high concurrency without race conditions."""
    cache = LLMResponseCache(max_size=100, ttl_seconds=60.0)

    def worker(idx: int):
        f_type = f"failure_{idx % 5}"
        m_cat = f"merch_{idx % 3}"
        cache.set(f_type, m_cat, sample_decision)
        val = cache.get(f_type, m_cat)
        assert val is not None
        stats = cache.get_stats()
        assert stats["total_entries"] <= 100

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(50)]
        for f in futures:
            f.result()

    assert cache.get_stats()["hits"] >= 50


def test_recovery_engine_caching_integration(clean_cache: LLMResponseCache, sample_decision: RecoveryDecision):
    """RecoveryEngine reuses cached decision and skips Gemini call on repeat scenarios."""
    engine = RecoveryEngine(cache=clean_cache)

    # Mock the GeminiAgent inside RecoveryEngine
    mock_agent = MagicMock()
    mock_agent.generate.return_value = sample_decision
    engine._agent = mock_agent

    event = NormalizedEvent(
        event_id="evt_test_cache_001",
        event_type=EventType.PAYMENT_FAILED.value,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id="pay_cache_001",
        merchant_id="acc_test_merch_01",
        amount=1500.0,
        currency="INR",
        status="FAILED",
        error_reason="insufficient_funds",
        notes={"merchant_category": "ecommerce"},
    )

    # First run: Cache MISS -> calls Gemini agent
    dec1 = engine.process(event, ml_failure_category="insufficient_funds")
    assert mock_agent.generate.call_count == 1
    assert dec1.action == "auto_retry"
    assert dec1.cached is False

    # Second run with same failure_type and merchant_category:
    # Cache HIT -> Gemini agent NOT called again!
    event_repeat = event.model_copy(update={"event_id": "evt_test_cache_002", "entity_id": "pay_cache_002"})
    dec2 = engine.process(event_repeat, ml_failure_category="insufficient_funds")
    assert mock_agent.generate.call_count == 1  # STILL 1, no second call!
    assert dec2.cached is True
    assert dec2.latency_ms == 0.0
    assert dec2.cost_usd == 0.0
    assert dec2.action == dec1.action

    # Third run with DIFFERENT merchant category:
    # Cache MISS -> calls Gemini agent
    event_diff = event.model_copy(update={"notes": {"merchant_category": "healthcare"}})
    dec3 = engine.process(event_diff, ml_failure_category="insufficient_funds")
    assert mock_agent.generate.call_count == 2  # Incremented to 2!


def test_rest_api_cache_endpoints(monkeypatch):
    """Verify GET /cache/stats and POST /cache/clear endpoints."""
    monkeypatch.setenv("MERCHANT_API_KEY", "test_merchant_key_12345")
    monkeypatch.setenv("SIMULATION_MODE", "false")
    client = TestClient(app)

    # Populate global cache with a sample entry
    cache = get_llm_cache()
    cache.clear()
    cache.set(
        "gateway_issue",
        "fintech",
        RecoveryDecision(
            action="auto_retry",
            priority="high",
            message="Gateway issue detected. Scheduled retry.",
            retry_after=600,
            alternate_method="none",
            confidence=0.95,
            reasoning="Gateway transient fault.",
        ),
    )

    # 1. GET /cache/stats
    res = client.get("/cache/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_entries"] >= 1
    assert "hits" in data
    assert "misses" in data
    assert "cost_saved_usd" in data

    # 2. POST /cache/clear (requires auth)
    # With invalid key -> 401
    res_unauth = client.post("/cache/clear", headers={"X-API-Key": "invalid_key_12345"})
    assert res_unauth.status_code == 401

    # With valid auth -> 200
    res_auth = client.post(
        "/cache/clear",
        headers={"X-API-Key": "test_merchant_key_12345"},
    )
    assert res_auth.status_code == 200
    assert res_auth.json()["status"] == "success"

    # Verify cache is now empty
    stats_after = client.get("/cache/stats").json()
    assert stats_after["total_entries"] == 0
