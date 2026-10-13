"""The observer is Vera: ARK is only a selectable video source."""
from __future__ import annotations

from io import BytesIO, StringIO
import json
import subprocess
import sys

import pytest

from vera_vision import RGBFrame
from vera_vision.live import FFmpegFrames, VideoFrame, VideoSource, VideoSourceError, ffmpeg_arguments
from vera_vision.pipeline import VisionPipeline


def test_ark_window_is_a_source_not_an_observer():
    source = VideoSource("window", "Chiaki - PS4", 16, 16, 2)
    args = ffmpeg_arguments(source, platform="win32")
    assert "title=Chiaki - PS4" in args
    assert "gdigrab" in args
    assert args[-1] == "pipe:1"
    assert "-vf" in args
    with pytest.raises(ValueError, match="Windows"):
        ffmpeg_arguments(source, platform="linux")


@pytest.mark.parametrize("url", (
    "rtsp://localhost:8554/ark", "rtmp://localhost/live/ark",
    "https://example.invalid/feed.m3u8", "srt://localhost:8890?streamid=ark",
))
def test_approved_network_video_sources(url):
    source = VideoSource("stream", url)
    args = ffmpeg_arguments(source, platform="linux")
    assert args[args.index("-i") + 1] == url
    assert "-an" in args


@pytest.mark.parametrize("locator", (
    "file:///etc/passwd", "ftp://example.com/s", "rtsp://user:password@host/live",
    "rtsp://host/path\nnext", "rtsp://", "",
))
def test_invalid_source_fails_closed(locator):
    with pytest.raises(ValueError):
        VideoSource("stream", locator)


def test_process_is_argument_vector_not_shell_and_rgb_frames_are_decoded():
    rgb1 = bytes((11, 12, 13)) * (16 * 16)
    rgb2 = bytes((19, 20, 21)) * (16 * 16)
    created = []
    class Process:
        def __init__(self):
            self.stdout = BytesIO(rgb1 + rgb2)
            self.terminated = False
        def poll(self): return 0
        def wait(self, timeout=None): return 0
        def terminate(self): self.terminated = True
        def kill(self): self.terminated = True

    def spawn(args, **options):
        created.append((args, options))
        return Process()

    source = VideoSource("file", "my game recording.mp4", 16, 16, 2)
    reader = FFmpegFrames(source, process_factory=spawn)
    frames = list(reader.frames(max_frames=10, max_seconds=3))
    assert len(frames) == 2
    assert frames[0].frame == RGBFrame(16, 16, rgb1)
    assert frames[1].frame == RGBFrame(16, 16, rgb2)
    assert frames[1].sequence == 2
    assert created[0][1]["shell"] is False
    assert created[0][1]["stderr"] == subprocess.DEVNULL
    assert reader._process is None


def test_vera_receives_ordered_observations_and_no_raw_pixels_in_events():
    source = VideoSource("file", "recording.mp4", 16, 16, 2)
    a = RGBFrame(16, 16, bytes((255, 0, 0)) * 256)
    b = RGBFrame(16, 16, bytes((0, 255, 0)) * 256)
    frames = [VideoFrame(1, 10.0, a, 0), VideoFrame(3, 11.0, b, 1)]
    output = StringIO()
    delivered = []
    count = VisionPipeline().run(
        frames, session_id="vera-observer", source_kind=source.kind,
        output=output, on_frame=lambda packet, obs: delivered.append((packet, obs)),
    )
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert count == 2
    assert [e["sequence"] for e in events] == [1, 3]
    assert events[1]["dropped_frames"] == 1
    assert events[0]["observer"] == "VERA"
    assert events[0]["session_id"] == "vera-observer"
    assert events[1]["decoded_at_monotonic"] == 11.0
    assert events[1]["timestamp_basis"] == "OBSERVER_HOST_DECODE_MONOTONIC"
    assert events[1]["temporal"]["status"] == "PIXEL_CHANGE_DETECTED"
    assert not events[1]["action_authority"]
    assert not events[1]["image_pixels_included"]
    assert len(delivered) == 2
    assert delivered[-1][0].frame == b
    assert "\"rgb\":" not in output.getvalue()  # No raw pixel payloads.


def test_reader_rejects_nonfinite_bounds_and_cannot_launch_unsafe_binary():
    reader = FFmpegFrames(VideoSource("file", "a.mp4", 16, 16, 2))
    with pytest.raises(ValueError):
        list(reader.frames(max_frames=10, max_seconds=float("nan")))
    with pytest.raises(ValueError):
        VideoSource("file", "a.mp4", 0, 16, 2)


def test_empty_decode_sanitizes_provider_response():
    class Process:
        stdout = BytesIO()
        def poll(self): return 1
        def terminate(self): raise AssertionError("terminated dead process")
        def wait(self, timeout=None): return 1
    reader = FFmpegFrames(
        VideoSource("file", "secret-video.mp4", 16, 16, 2),
        process_factory=lambda *a, **kw: Process(),
    )
    with pytest.raises(VideoSourceError, match="VIDEO_STREAM_EMPTY_OR_UNDECODABLE") as exc:
        list(reader.frames(max_frames=2, max_seconds=2))
    assert "secret-video" not in str(exc.value)
