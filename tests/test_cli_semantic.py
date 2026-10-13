"""CLI-level evidence that Vera observes video; no implicit model call exists."""
from __future__ import annotations

import json

from vera_vision import RGBFrame
from vera_vision import cli
from vera_vision.live import VideoFrame
from vera_vision.local_vision import SemanticObservation


def test_opt_in_local_model_emits_bounded_merged_session_events(monkeypatch, capsys):
    seen = []
    class Reader:
        def __init__(self, source, *, binary):
            self.source = source
        def frames(self, **kwargs):
            for index, timestamp, color in ((1, 1.0, (0, 0, 0)), (2, 1.2, (255, 0, 0))):
                yield VideoFrame(index, timestamp, RGBFrame(16, 16, bytes(color) * 256), 0)
    class Model:
        def __init__(self, *, model):
            seen.append(("created", model))
        def observe(self, frame):
            seen.append(("observed", frame.width))
            return SemanticObservation(
                "MODEL_INFERENCE", "gameplay", ("terrain",),
                "A landscape is visible.", 0.4, "local-fake",
                "MODEL_REPORTED_UNCALIBRATED",
            )

    monkeypatch.setattr(cli, "FFmpegFrames", Reader)
    monkeypatch.setattr(cli, "LocalOllamaObserver", Model)
    result = cli.main([
        "watch", "--kind", "file", "--source", "recording.mp4",
        "--width", "16", "--height", "16", "--frames", "2", "--seconds", "3",
        "--local-vision-model", "local-fake", "--local-vision-interval", "5",
    ])
    assert result == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    temporal = [e for e in events if e.get("kind") == "FRAME_OBSERVED"]
    semantic = [e for e in events if e.get("kind") == "SEMANTIC_OBSERVATION"]
    assert len(temporal) == 2
    assert len(semantic) == 1  # Rate limit prevents repeated model calls
    assert len({e["session_id"] for e in temporal + semantic}) == 1
    assert semantic[0]["sequence"] == temporal[0]["sequence"]
    assert semantic[0]["observer"] == "VERA"
    assert semantic[0]["semantic"]["status"] == "MODEL_INFERENCE"
    assert semantic[0]["semantic"]["action_authority"] is False
    assert all(e.get("image_pixels_included") is False for e in semantic + temporal)
    assert seen == [("created", "local-fake"), ("observed", 16)]
    assert events[-1]["controller_effects"] == 0


def test_default_cli_never_initializes_model(monkeypatch, capsys):
    class Reader:
        def __init__(self, source, *, binary):
            pass
        def frames(self, **kwargs):
            yield VideoFrame(1, 1.0, RGBFrame(16, 16, bytes(768)), 0)
    def unexpected_model(**kwargs):
        raise AssertionError("Vision unexpectedly requested model inference")
    monkeypatch.setattr(cli, "FFmpegFrames", Reader)
    monkeypatch.setattr(cli, "LocalOllamaObserver", unexpected_model)
    assert cli.main(["watch", "--kind", "file", "--source", "recording.mp4", "--width", "16", "--height", "16"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e.get("kind") for e in events if "kind" in e] == ["FRAME_OBSERVED"]
