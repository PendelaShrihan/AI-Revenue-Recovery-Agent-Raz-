"""
Simple in-memory SSE broadcaster for real-time dashboard updates.

Any coroutine can call `broadcast(event_dict)` after completing pipeline work.
All active SSE connections receive the event within milliseconds.
"""

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)

# Global list of asyncio Queues — one per live SSE connection
_subscribers: List[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber. Returns its dedicated queue."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(q)
    logger.debug("[Broadcaster] SSE client connected (total=%d)", len(_subscribers))
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove a subscriber queue on client disconnect."""
    try:
        _subscribers.remove(q)
        logger.debug("[Broadcaster] SSE client disconnected (total=%d)", len(_subscribers))
    except ValueError:
        pass


async def broadcast(event: dict) -> None:
    """Push *event* to all live SSE subscriber queues (non-blocking)."""
    if not _subscribers:
        return
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("[Broadcaster] Slow SSE client — queue full, event dropped.")


__all__ = ["subscribe", "unsubscribe", "broadcast"]
