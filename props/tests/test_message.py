"""Tests for the prop mesh message envelope and serialization."""

from __future__ import annotations

import pytest

from props.lib.message import MAGIC, Message, Timing, Meta


@pytest.fixture
def sample_message():
    return Message(
        topic="effects.thunder.clap",
        source="broker",
        target=None,
        payload={"duration_ms": 800, "flash_count": 3},
        timing=Timing(delay_ms=150),
        meta=Meta(seq=1, session="halloween-2026-test"),
        timestamp=1_700_000_000.0,
    )


def test_json_roundtrip(sample_message):
    raw = sample_message.to_json()
    msg = Message.from_json(raw)
    assert msg.topic == sample_message.topic
    assert msg.source == sample_message.source
    assert msg.payload == sample_message.payload
    assert msg.timing.delay_ms == 150
    assert msg.meta.session == "halloween-2026-test"


def test_cbor_roundtrip(sample_message):
    raw = sample_message.to_cbor()
    msg = Message.from_cbor(raw)
    assert msg.topic == sample_message.topic
    assert msg.payload == sample_message.payload
    assert msg.timing.delay_ms == 150


def test_framed_roundtrip_json(sample_message):
    raw = sample_message.to_framed(codec="json")
    assert raw[:2] == MAGIC
    msg = Message.from_framed(raw)
    assert msg.topic == sample_message.topic
    assert msg.payload == sample_message.payload


def test_framed_roundtrip_cbor(sample_message):
    raw = sample_message.to_framed(codec="cbor")
    assert raw[:2] == MAGIC
    msg = Message.from_framed(raw)
    assert msg.topic == sample_message.topic
    assert msg.payload == sample_message.payload


def test_target_is_optional():
    msg = Message(topic="x", source="y")
    d = msg.to_dict()
    assert "target" not in d
    msg2 = Message.from_dict(d)
    assert msg2.target is None
