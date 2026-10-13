"""Opt-in, loopback-only visual classification for Vera's decoded frames.

This is an *untrusted* semantic observer, not an authorization surface. There
is no automatic model selection, cloud endpoint, screenshot persistence, or
controller capability. The selected Ollama model must actually support images.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import math
import re
import struct
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener
from urllib.error import URLError
import zlib

from .frames import RGBFrame
from .live import VideoFrame

DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/generate"


class _DenyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Never follow an Ollama redirect to another host, and do not use OS HTTP proxies.
_LOCAL_OPENER = build_opener(ProxyHandler({}), _DenyRedirect())
MAX_VISION_PIXELS = 640 * 360
SCENES = frozenset({"gameplay", "desktop", "menu", "outdoor", "indoor", "unknown"})
VISIBLE = frozenset({"person", "animal", "vehicle", "structure", "screen", "terrain", "text", "other"})
_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "enum": sorted(SCENES)},
        "visible": {"type": "array", "items": {"type": "string", "enum": sorted(VISIBLE)}, "maxItems": 6},
        "summary": {"type": "string", "maxLength": 200},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["scene", "visible", "summary", "confidence"],
    "additionalProperties": False,
}
PROMPT = (
    "You are describing one video frame as untrusted visual evidence for Vera. "
    "Describe only clearly visible content. Return JSON with exactly the schema fields "
    "scene, visible, summary, confidence. Use 'unknown' when uncertain. "
    "Do not write actions, commands, world coordinates, paths, safety judgments, or controller instructions. "
    "A frame cannot prove motion or collision clearance."
)


def _chunk(tag: bytes, value: bytes) -> bytes:
    return (struct.pack(">I", len(value)) + tag + value
            + struct.pack(">I", binascii.crc32(tag + value) & 0xFFFFFFFF))


def encode_rgb_png(frame: RGBFrame) -> bytes:
    """Standard-library, bounded PNG encoding for local Ollama image input."""
    if frame.width * frame.height > MAX_VISION_PIXELS:
        raise ValueError("semantic image exceeds pixel budget")
    stride = frame.width * 3
    filtered = b"".join(
        b"\x00" + frame.rgb[y * stride : (y + 1) * stride]
        for y in range(frame.height)
    )
    header = struct.pack(">IIBBBBB", frame.width, frame.height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(filtered))
            + _chunk(b"IEND", b""))


def _checked_endpoint(endpoint: str) -> str:
    if type(endpoint) is not str or len(endpoint) > 256:
        raise ValueError("local vision endpoint invalid")
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme == "http"
                 and parsed.hostname in {"127.0.0.1", "::1"}
                 and parsed.path == "/api/generate"
                 and parsed.port is not None
                 and 1 <= parsed.port <= 65535
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("local vision endpoint must be a numeric loopback Ollama generate URL")
    return endpoint


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    status: str
    scene: str
    visible: tuple[str, ...]
    summary: str
    model_reported_confidence: float | None
    model: str
    reason: str
    action_authority: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "VERA_VISION_LOCAL_SEMANTIC_V1",
            "status": self.status,
            "scene": self.scene,
            "visible": list(self.visible),
            "summary": self.summary,
            "model_reported_confidence_uncalibrated": self.model_reported_confidence,
            "model": self.model,
            "provider": "OLLAMA_LOOPBACK",
            "reason": self.reason,
            "provenance": "UNTRUSTED_MODEL_INFERENCE",
            "action_authority": False,
        }


class LocalOllamaObserver:
    """Explicit, bounded image-model inference. Defaults never call this class."""

    def __init__(self, *, model: str, endpoint: str = DEFAULT_ENDPOINT,
                 timeout_seconds: float = 20,
                 opener: Callable[..., object] | None = None) -> None:
        if (type(model) is not str or not _MODEL_NAME.fullmatch(model)
                or model.lower().endswith(":cloud")
                or any(part in ("", ".", "..") for part in model.split("/"))):
            raise ValueError("explicit local image-capable model required")
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or not 1 <= timeout_seconds <= 60):
            raise ValueError("local observer timeout invalid")
        self.model = model
        self.endpoint = _checked_endpoint(endpoint)
        self.timeout = float(timeout_seconds)
        self.opener = opener if opener is not None else _LOCAL_OPENER.open

    def _parse(self, payload: object) -> SemanticObservation:
        if type(payload) is not dict or payload.get("done") is not True:
            raise ValueError("incomplete Ollama response")
        raw = payload.get("response")
        if type(raw) is not str or len(raw) > 4096:
            raise ValueError("invalid model response")
        data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid number")))
        if type(data) is not dict or set(data) != {"scene", "visible", "summary", "confidence"}:
            raise ValueError("semantic output schema invalid")
        scene, visible, summary, confidence = (data[k] for k in ("scene", "visible", "summary", "confidence"))
        if type(scene) is not str or scene not in SCENES:
            raise ValueError("scene invalid")
        if (type(visible) is not list or len(visible) > 6
                or any(type(item) is not str or item not in VISIBLE for item in visible)
                or len(set(visible)) != len(visible)):
            raise ValueError("visible categories invalid")
        if (type(summary) is not str or len(summary) > 200
                or any(ord(char) < 32 for char in summary)):
            raise ValueError("summary invalid")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError("confidence invalid")
        return SemanticObservation(
            "MODEL_INFERENCE", scene, tuple(visible), summary, float(confidence),
            self.model, "MODEL_REPORTED_UNCALIBRATED",
        )

    def observe(self, frame: RGBFrame) -> SemanticObservation:
        png = encode_rgb_png(frame)
        request_body = {
            "model": self.model,
            "prompt": PROMPT,
            "images": [base64.b64encode(png).decode("ascii")],
            "format": SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 256},
            "keep_alive": "30s",
        }
        request = Request(
            self.endpoint,
            data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                body = response.read(16_385)
            if len(body) > 16_384:
                raise ValueError("model response exceeds size budget")
            return self._parse(json.loads(
                body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid number"))
            ))
        except (OSError, ValueError, TypeError, UnicodeError, URLError):
            # No model text, prompt, image, URL or exception message leaks.
            return SemanticObservation(
                "UNAVAILABLE", "unknown", (), "", None, self.model, "LOCAL_MODEL_UNAVAILABLE_OR_INVALID",
            )


class LocalSemanticSampler:
    """Rate-limits frame-to-model calls; default feedback loop stays pixel-only."""

    def __init__(self, observer: LocalOllamaObserver, *, interval_seconds: float = 5) -> None:
        if (type(interval_seconds) not in (int, float) or not math.isfinite(interval_seconds)
                or not 0.25 <= interval_seconds <= 3600):
            raise ValueError("semantic sampling interval invalid")
        self.observer = observer
        self.interval = float(interval_seconds)
        self._last_at: float | None = None

    def observe_if_due(self, packet: VideoFrame) -> SemanticObservation | None:
        stamp = packet.decoded_at_monotonic
        if not math.isfinite(stamp):
            return None
        if self._last_at is not None:
            if stamp < self._last_at:
                # Regressed observation time is not license to replay stale imagery.
                return None
            if stamp - self._last_at < self.interval:
                return None
        self._last_at = stamp
        return self.observer.observe(packet.frame)
