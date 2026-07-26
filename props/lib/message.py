"""Prop mesh message envelope with JSON and CBOR serialization.

The binary framing reserves a flags byte with space for future flood-mesh
(TTL) and codec negotiation bits without changing the header shape.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import cbor2
except ImportError:  # pragma: no cover
    cbor2 = None  # type: ignore

MAGIC = b"\x50\x42"  # 'P' 'B' for PirateBot
VERSION = 0x01

# Flags bits (reserved for future use)
FLAG_FLOOD_MESH = 0x01  # enable up-to-N-hop relaying (future)
FLAG_TLV_BINARY = 0x02  # payload is custom packed, not CBOR (future)


@dataclass
class Timing:
    delay_ms: Optional[int] = None
    at_ts: Optional[float] = None
    expire_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Timing":
        if not data:
            return cls()
        return cls(
            delay_ms=data.get("delay_ms"),
            at_ts=data.get("at_ts"),
            expire_ms=data.get("expire_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.delay_ms is not None:
            out["delay_ms"] = self.delay_ms
        if self.at_ts is not None:
            out["at_ts"] = self.at_ts
        if self.expire_ms is not None:
            out["expire_ms"] = self.expire_ms
        return out


@dataclass
class Meta:
    seq: int = 0
    session: str = "halloween-2026"
    codecs: list[str] = field(default_factory=lambda: ["json", "cbor"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Meta":
        if not data:
            return cls()
        return cls(
            seq=data.get("seq", 0),
            session=data.get("session", "halloween-2026"),
            codecs=data.get("codecs", ["json", "cbor"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seq": self.seq,
            "session": self.session,
        }
        if self.codecs:
            out["codecs"] = self.codecs
        return out


@dataclass
class Message:
    topic: str
    source: str
    target: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    timing: Timing = field(default_factory=Timing)
    meta: Meta = field(default_factory=Meta)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            topic=data["topic"],
            source=data["source"],
            target=data.get("target"),
            payload=data.get("payload", {}),
            timing=Timing.from_dict(data.get("timing", {})),
            meta=Meta.from_dict(data.get("meta", {})),
            timestamp=data.get("timestamp", time.time()),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "topic": self.topic,
            "source": self.source,
            "payload": self.payload,
            "timing": self.timing.to_dict(),
            "meta": self.meta.to_dict(),
            "timestamp": self.timestamp,
        }
        if self.target is not None:
            out["target"] = self.target
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Message":
        return cls.from_dict(json.loads(raw))

    def to_cbor(self) -> bytes:
        if cbor2 is None:
            raise ImportError("cbor2 is required for CBOR encoding")
        return cbor2.dumps(self.to_dict())

    @classmethod
    def from_cbor(cls, raw: bytes) -> "Message":
        if cbor2 is None:
            raise ImportError("cbor2 is required for CBOR decoding")
        return cls.from_dict(cbor2.loads(raw))

    def to_framed(self, codec: str = "cbor", flags: int = 0) -> bytes:
        """Pack message into a framed binary envelope.

        Format:
            [magic:2][version:1][flags:1][codec:1][payload_len:2][payload:N]
        codec: 0x01=json, 0x02=cbor, 0x03... reserved
        """
        if codec == "json":
            payload = self.to_json().encode("utf-8")
            codec_byte = 0x01
        elif codec == "cbor":
            payload = self.to_cbor()
            codec_byte = 0x02
        else:
            raise ValueError(f"Unsupported framed codec: {codec}")

        if len(payload) > 0xFFFF:
            raise ValueError("Payload too large for framed format")

        header = MAGIC + bytes([VERSION, flags, codec_byte])
        length = len(payload).to_bytes(2, "big")
        return header + length + payload

    @classmethod
    def from_framed(cls, raw: bytes) -> "Message":
        """Parse a framed binary envelope and return the message."""
        if len(raw) < 7:
            raise ValueError("Framed message too short")
        if raw[:2] != MAGIC:
            raise ValueError("Invalid magic bytes")
        version = raw[2]
        if version != VERSION:
            raise ValueError(f"Unsupported framed version: {version}")
        # flags = raw[3]  # reserved for future flood-mesh / TTL handling
        codec_byte = raw[4]
        length = int.from_bytes(raw[5:7], "big")
        payload = raw[7 : 7 + length]

        if codec_byte == 0x01:
            return cls.from_json(payload)
        if codec_byte == 0x02:
            return cls.from_cbor(payload)
        raise ValueError(f"Unsupported codec byte: {codec_byte}")

    def __repr__(self) -> str:
        return (
            f"Message(topic={self.topic!r}, source={self.source!r}, "
            f"target={self.target!r})"
        )
