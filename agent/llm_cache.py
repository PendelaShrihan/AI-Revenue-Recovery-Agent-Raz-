# agent/llm_cache.py
"""In-memory thread-safe caching layer for LLM responses.

Key Formulation:
    Cache Key = (failure_type, merchant_category)
    - failure_type: Canonical error reason/category (e.g. 'insufficient_funds', 'network_timeout')
    - merchant_category: Industry vertical or merchant volume tier (e.g. 'ecommerce', 'low_volume')

When a failure event matches an existing cache entry:
    - The previous RecoveryDecision is reused immediately.
    - Zero API calls are made to Gemini, saving latency (~1800ms -> <1ms) and token spend ($0.00).
    - Cache statistics (hits, misses, hit_rate_pct, cost_saved_usd) are tracked.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from agent.llm_agent import RecoveryDecision

_logger = logging.getLogger(__name__)

# Default estimated savings per cached invocation ($0.00020 USD ~ ₹0.017 INR)
DEFAULT_COST_SAVED_PER_HIT = 0.00020


@dataclass
class CacheEntry:
    """Single cached recovery decision record with metadata."""

    decision: RecoveryDecision
    failure_type: str
    merchant_category: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    original_latency_ms: float = 0.0
    original_cost_usd: float = 0.0


class LLMResponseCache:
    """Thread-safe LRU/TTL caching layer for LLM recovery decisions.

    Stores RecoveryDecision instances indexed by `(failure_type, merchant_category)`.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 86400.0,  # 24-hour default TTL
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[Tuple[str, str], CacheEntry] = {}
        self._lock = threading.RLock()

        # Operational metrics
        self._hits: int = 0
        self._misses: int = 0
        self._total_cost_saved_usd: float = 0.0

    # ------------------------------------------------------------------
    # Normalization & Key Formulation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_key(failure_type: Optional[str], merchant_category: Optional[str]) -> Tuple[str, str]:
        """Normalize failure_type and merchant_category to consistent trimmed lowercase."""
        norm_failure = str(failure_type or "").strip().lower() or "unknown"
        norm_merchant = str(merchant_category or "").strip().lower() or "unknown"
        return (norm_failure, norm_merchant)

    # ------------------------------------------------------------------
    # Cache Operations
    # ------------------------------------------------------------------

    def get(
        self,
        failure_type: Optional[str],
        merchant_category: Optional[str],
    ) -> Optional[RecoveryDecision]:
        """Retrieve a cached RecoveryDecision if present and unexpired.

        Returns a deep copy of the decision with `cached=True`, `latency_ms=0.0`,
        and `cost_usd=0.0`.
        """
        key = self.normalize_key(failure_type, merchant_category)
        now = time.time()

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                _logger.debug("[LLMCache] Cache MISS for key=%s", key)
                return None

            # Check expiration if TTL is set
            if self.ttl_seconds > 0 and (now - entry.created_at) > self.ttl_seconds:
                _logger.debug("[LLMCache] Expired entry for key=%s (age=%.1fs > ttl=%.1fs)", key, now - entry.created_at, self.ttl_seconds)
                del self._cache[key]
                self._misses += 1
                return None

            # Cache hit!
            entry.hit_count += 1
            entry.last_accessed = now
            self._hits += 1

            saved_cost = entry.original_cost_usd or DEFAULT_COST_SAVED_PER_HIT
            self._total_cost_saved_usd += saved_cost

            _logger.info(
                "[LLMCache] Cache HIT for key=%s | hits=%d | saved ~$%.5f USD | original_latency=%.1fms",
                key, entry.hit_count, saved_cost, entry.original_latency_ms,
            )

            # Return a detached copy marked as cached with zero new latency/cost
            cached_decision = copy.deepcopy(entry.decision)
            cached_decision.cached = True
            cached_decision.latency_ms = 0.0
            cached_decision.cost_usd = 0.0
            cached_decision.input_tokens = 0
            cached_decision.output_tokens = 0
            return cached_decision

    def set(
        self,
        failure_type: Optional[str],
        merchant_category: Optional[str],
        decision: RecoveryDecision,
    ) -> None:
        """Store a RecoveryDecision in the cache for the given key tuple."""
        key = self.normalize_key(failure_type, merchant_category)
        now = time.time()

        with self._lock:
            # Enforce capacity if limit reached
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict the least recently accessed entry
                lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
                del self._cache[lru_key]
                _logger.debug("[LLMCache] Evicted LRU entry for key=%s", lru_key)

            entry = CacheEntry(
                decision=copy.deepcopy(decision),
                failure_type=key[0],
                merchant_category=key[1],
                created_at=now,
                last_accessed=now,
                hit_count=0,
                original_latency_ms=getattr(decision, "latency_ms", 0.0),
                original_cost_usd=getattr(decision, "cost_usd", 0.0) or DEFAULT_COST_SAVED_PER_HIT,
            )
            self._cache[key] = entry
            _logger.debug(
                "[LLMCache] Stored cache entry for key=%s action='%s' (size=%d/%d)",
                key, decision.action, len(self._cache), self.max_size,
            )

    def clear(self) -> None:
        """Flush all cache entries and reset counters."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._total_cost_saved_usd = 0.0
            _logger.info("[LLMCache] Cleared %d cache entries", count)

    def invalidate(
        self,
        failure_type: Optional[str] = None,
        merchant_category: Optional[str] = None,
    ) -> int:
        """Invalidate entries matching specified criteria. Returns count of invalidated items."""
        with self._lock:
            if failure_type is None and merchant_category is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            keys_to_remove = []
            norm_failure = str(failure_type).strip().lower() if failure_type else None
            norm_merchant = str(merchant_category).strip().lower() if merchant_category else None

            for (f_type, m_cat) in self._cache.keys():
                match_failure = (norm_failure is None) or (f_type == norm_failure)
                match_merchant = (norm_merchant is None) or (m_cat == norm_merchant)
                if match_failure and match_merchant:
                    keys_to_remove.append((f_type, m_cat))

            for k in keys_to_remove:
                del self._cache[k]

            _logger.info("[LLMCache] Invalidated %d entries matching (failure=%s, merchant=%s)", len(keys_to_remove), norm_failure, norm_merchant)
            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Return operational cache telemetry and financial savings metrics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate_pct = round((self._hits / total_requests * 100.0), 2) if total_requests > 0 else 0.0

            entries_summary = []
            for (f_type, m_cat), entry in self._cache.items():
                entries_summary.append({
                    "failure_type": f_type,
                    "merchant_category": m_cat,
                    "action": entry.decision.action,
                    "hit_count": entry.hit_count,
                    "age_seconds": round(time.time() - entry.created_at, 1),
                })

            return {
                "total_entries": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "total_lookups": total_requests,
                "hit_rate_pct": hit_rate_pct,
                "cost_saved_usd": round(self._total_cost_saved_usd, 6),
                "cost_saved_inr": round(self._total_cost_saved_usd * 83.33, 4),
                "entries": entries_summary,
            }


# ---------------------------------------------------------------------------
# Global Singleton Accessor
# ---------------------------------------------------------------------------

_GLOBAL_CACHE: Optional[LLMResponseCache] = None
_CACHE_INIT_LOCK = threading.Lock()


def get_llm_cache() -> LLMResponseCache:
    """Return the global LLMResponseCache singleton instance."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        with _CACHE_INIT_LOCK:
            if _GLOBAL_CACHE is None:
                _GLOBAL_CACHE = LLMResponseCache()
    return _GLOBAL_CACHE
