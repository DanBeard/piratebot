"""In-memory message bus for the prop mesh.

Supports topic subscriptions with glob patterns and routes messages to
local handlers or other transports.
"""

from __future__ import annotations

import fnmatch
import logging
from collections import defaultdict
from typing import Any, Callable, Optional

from props.lib.message import Message

logger = logging.getLogger("prop_bus")

Handler = Callable[[Message], Any]


class MessageBus:
    """Central pubsub bus used by all broker transports."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, set[Handler]] = defaultdict(set)
        self._interceptors: list[Callable[[Message], Optional[Message]]] = []

    def subscribe(self, topic_pattern: str, handler: Handler) -> None:
        """Subscribe to a topic or glob pattern."""
        self._subscriptions[topic_pattern].add(handler)

    def unsubscribe(self, topic_pattern: str, handler: Handler) -> None:
        self._subscriptions[topic_pattern].discard(handler)
        if not self._subscriptions[topic_pattern]:
            self._subscriptions.pop(topic_pattern, None)

    def add_interceptor(
        self, interceptor: Callable[[Message], Optional[Message]]
    ) -> None:
        """Add a function that can transform or drop messages before routing."""
        self._interceptors.append(interceptor)

    def publish(self, msg: Message) -> None:
        """Publish a message to all matching subscribers."""
        for interceptor in self._interceptors:
            msg = interceptor(msg)
            if msg is None:
                return

        matched = False
        for pattern, handlers in list(self._subscriptions.items()):
            if _match_topic(pattern, msg.topic):
                matched = True
                for handler in handlers:
                    try:
                        handler(msg)
                    except Exception:
                        logger.exception("Handler failed for %s", msg.topic)
        if not matched and logger.isEnabledFor(logging.DEBUG):
            logger.debug("No subscriber for topic %s from %s", msg.topic, msg.source)

    def topic_has_subscribers(self, topic: str) -> bool:
        for pattern in self._subscriptions:
            if _match_topic(pattern, topic):
                return True
        return False


def _match_topic(pattern: str, topic: str) -> bool:
    """Match a topic against a pattern supporting * and ** globs."""
    if pattern == topic:
        return True
    if pattern == "*":
        return True
    # Convert ** into a regex wildcard and * into single-segment wildcard.
    if "**" in pattern or "*" in pattern:
        return fnmatch.fnmatch(topic, pattern)
    return False
