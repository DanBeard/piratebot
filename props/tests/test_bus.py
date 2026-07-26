"""Tests for the prop mesh message bus."""

from __future__ import annotations

from props.lib.bus import MessageBus
from props.lib.message import Message


def test_exact_topic_routing():
    bus = MessageBus()
    received = []
    bus.subscribe("effects.thunder.clap", lambda m: received.append(m.topic))
    bus.publish(Message(topic="effects.thunder.clap", source="x"))
    bus.publish(Message(topic="effects.smoke.burst", source="x"))
    assert received == ["effects.thunder.clap"]


def test_wildcard_routing():
    bus = MessageBus()
    received = []
    bus.subscribe("effects.*", lambda m: received.append(m.topic))
    bus.publish(Message(topic="effects.thunder.clap", source="x"))
    bus.publish(Message(topic="scene.idle", source="x"))
    assert set(received) == {"effects.thunder.clap"}


def test_interceptor_can_drop_message():
    bus = MessageBus()
    received = []
    bus.add_interceptor(lambda m: None if m.topic == "ignore" else m)
    bus.subscribe("*", lambda m: received.append(m.topic))
    bus.publish(Message(topic="ignore", source="x"))
    bus.publish(Message(topic="keep", source="x"))
    assert received == ["keep"]


def test_no_subscriber_does_not_raise():
    bus = MessageBus()
    bus.publish(Message(topic="orphan", source="x"))
