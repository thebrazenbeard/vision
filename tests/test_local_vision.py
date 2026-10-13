"""Local vision is Vera's opt-in inference layer, not an action route."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import math
import struct
import zlib

import pytest

from vera_vision import RGBFrame
from vera_vision.cli import parser
from vera_vision.live import VideoFrame
from vera_vision.local_vision import (
    LocalOllamaObserver,
    LocalSemanticSampler,
    encode_rgb_png,
)


def frame(color: tuple[int, int, int] = (255, 0, 0)) -> RGBFrame:
    return RGBFrame(16, 16, bytes(color) * 256)


class FakeResponse:
    def __init__(self, body):
        self.raw = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, n):
        return self.raw[:n]


def test_local_vision_encodes_truecolor_png_losslessly():
    image = encode_rgb_png(frame())
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    length = struct.unpack(">I", image[8:12])[0]
    assert length == 13
    assert image[12:16] == b"IHDR"
    width, height, depth, color = struct.unpack(">IIBB", image[16:26])
    assert (width, height, depth, color) == (16, 16, 8, 2)
    off = 8 + 12 + 13
    chunk_len = struct.unpack(">I", image[off:off + 4])[0]
    assert image[off + 4:off + 8] == b"IDAT"
    data = zlib.decompress(image[off + 8:off + 8 + chunk_len])
    assert data[:4] == bytes((0, 255, 0, 0))


def test_loopback_observer_sends_one_explicit_image_and_parses_categories():
    calls = []
    def opener(request, timeout):
        body = json.loads(request.data)
        calls.append((request.full_url, request.get_method(), body, timeout))
        data = {
            "done": True,
            "response": json.dumps({
                "scene": "gameplay",
                "visible": ["terrain", "animal"],
                "summary": "An animal near uneven ground.",
                "confidence": 0.4,
            }),
        }
        return FakeResponse(json.dumps(data).encode())

    observer = LocalOllamaObserver(model="local-vision:7b", opener=opener)
    result = observer.observe(frame())
    assert result.status == "MODEL_INFERENCE"
    assert result.scene == "gameplay"
    assert result.visible == ("terrain", "animal")
    assert result.action_authority is False
    assert result.to_dict()["provenance"] == "UNTRUSTED_MODEL_INFERENCE"
    url, method, payload, timeout = calls[0]
    assert url == "http://127.0.0.1:11434/api/generate"
    assert method == "POST"
    assert payload["model"] == "local-vision:7b"
    assert payload["stream"] is False
    assert len(payload["images"]) == 1
    assert base64.b64decode(payload["images"][0]).startswith(b"\x89PNG")
    assert "controller" in payload["prompt"]
    assert timeout <= 60


@pytest.mark.parametrize("endpoint", [
    "https://example.com/api/generate",
    "http://localhost:11434/api/generate",
    "http://user:password@127.0.0.1:11434/api/generate",
    "http://127.0.0.1:11434/api/generate?remote=true",
    "http://127.0.0.1:11434/api/chat",
    "http://127.0.0.1:11434",
    "http://127.0.0.2:11434/api/generate",
])
def test_external_or_ambiguous_model_endpoint_refused(endpoint):
    with pytest.raises(ValueError):
        LocalOllamaObserver(model="local-vision", endpoint=endpoint)


@pytest.mark.parametrize("model", ["", "../model", "vision:cloud", " model", "model\nother"])
def test_model_must_be_explicit_and_not_a_cloud_label(model):
    with pytest.raises(ValueError):
        LocalOllamaObserver(model=model)


@pytest.mark.parametrize("model_data", [
    {"scene": "gameplay", "visible": ["animal"], "summary": "press controller A", "confidence": float("nan")},
    {"scene": "admin", "visible": [], "summary": "", "confidence": 0.7},
    {"scene": "gameplay", "visible": ["execute"], "summary": "", "confidence": 0.7},
    {"scene": "gameplay", "visible": [], "summary": "evil\nhidden", "confidence": 0.7},
    {"scene": "gameplay", "visible": [], "summary": "", "confidence": 0.7, "action": "PRESS_A"},
])
def test_untrusted_model_schema_rejected_without_raw_text(model_data):
    def opener(request, timeout):
        return FakeResponse(json.dumps({"done": True, "response": json.dumps(model_data)}).encode())
    obs = LocalOllamaObserver(model="vision-local", opener=opener).observe(frame())
    assert obs.status == "UNAVAILABLE"
    assert obs.scene == "unknown"
    assert obs.summary == ""
    assert not obs.visible
    assert obs.action_authority is False
    assert "PRESS_A" not in json.dumps(obs.to_dict())


def test_incomplete_response_never_becomes_factual_observation():
    obs = LocalOllamaObserver(
        model="vision-local",
        opener=lambda request, timeout: FakeResponse(
            b'{"done":false,"response":"{\\"scene\\":\\"gameplay\\"}"}'
        ),
    ).observe(frame())
    assert obs.status == "UNAVAILABLE"


def test_sampler_calls_model_only_when_due_and_rejects_clock_regression():
    calls = []
    class Observer:
        def observe(self, image):
            calls.append(image)
            return "observation"
    sampler = LocalSemanticSampler(Observer(), interval_seconds=2)
    def packet(seq, tick):
        return VideoFrame(seq, tick, frame(), 0)
    assert sampler.observe_if_due(packet(1, 1.0)) == "observation"
    assert sampler.observe_if_due(packet(2, 1.5)) is None
    assert sampler.observe_if_due(packet(3, 3.0)) == "observation"
    assert sampler.observe_if_due(packet(4, 2.0)) is None
    assert len(calls) == 2


def test_cli_exposes_explicit_local_model_only_not_enabled_by_default():
    normal = parser().parse_args(["watch", "--kind", "file", "--source", "video.mp4"])
    assert normal.local_vision_model is None
    selected = parser().parse_args(["watch", "--kind", "file", "--source", "video.mp4",
                                    "--local-vision-model", "gemma4:4b",
                                    "--local-vision-interval", "3"])
    assert selected.local_vision_model == "gemma4:4b"
    assert selected.local_vision_interval == 3


def test_default_loopback_transport_refuses_redirects_and_http_proxies():
    from vera_vision.local_vision import _DenyRedirect
    from urllib.request import ProxyHandler
    observer = LocalOllamaObserver(model="image-model")
    handlers = observer.opener.__self__.handlers
    assert any(isinstance(h, _DenyRedirect) for h in handlers)
    # urllib can omit a ProxyHandler with no configured proxies entirely.
    assert not any(isinstance(h, ProxyHandler) and bool(h.proxies) for h in handlers)
    redirect = next(h for h in handlers if isinstance(h, _DenyRedirect))
    assert redirect.redirect_request(None, None, 302, "redirect", {}, "https://example.com") is None
