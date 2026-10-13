"""Failure-mode tests for live video observation, not gameplay control."""
from __future__ import annotations

from io import BytesIO
import subprocess
import threading

import pytest

from vera_vision import RGBFrame
from vera_vision.live import FFmpegFrames, VideoFrame, VideoSource, VideoSourceError
from vera_vision.pipeline import VisionPipeline


class FakeProcess:
    def __init__(self, data: bytes):
        self.stdout = BytesIO(data)
        self.terminated = False

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def reader(source: VideoSource, process: FakeProcess) -> FFmpegFrames:
    return FFmpegFrames(source, process_factory=lambda *a, **k: process)


def test_network_stream_early_eof_does_not_claim_healthy_completion():
    frame = bytes((0, 0, 0)) * 256
    child = FakeProcess(frame)
    with pytest.raises(VideoSourceError, match="LIVE_VIDEO_SOURCE_ENDED"):
        list(reader(VideoSource("stream", "rtsp://localhost:8554/ark", 16, 16), child)
             .frames(max_frames=10, max_seconds=2))
    assert len(frame) == 16 * 16 * 3


def test_file_eof_is_allowed_after_frames():
    frame = bytes((0, 0, 0)) * 256
    frames = list(reader(VideoSource("file", "sample.mp4", 16, 16), FakeProcess(frame))
                  .frames(max_frames=10, max_seconds=2))
    assert len(frames) == 1


def test_silent_blocking_source_times_out_and_terminates_child():
    unblock = threading.Event()

    class HangingPipe:
        def read(self, n: int) -> bytes:
            unblock.wait(timeout=3)
            return b""

    class HangingProcess(FakeProcess):
        def __init__(self):
            super().__init__(b"")
            self.stdout = HangingPipe()
            self.alive = True
        def poll(self):
            return None if self.alive else 1
        def terminate(self):
            self.terminated = True
            self.alive = False
            unblock.set()

    child = HangingProcess()
    with pytest.raises(VideoSourceError, match="LIVE_VIDEO_NO_FRAMES_TIMEOUT"):
        list(reader(VideoSource("stream", "rtsp://localhost:8554/ark", 16, 16), child)
             .frames(max_frames=10, max_seconds=1, max_idle_seconds=0.15))
    assert child.terminated


def test_invalid_idle_budget_fails_before_starting_process():
    child = FakeProcess(b"")
    for value in (float("nan"), float("inf"), 0, True):
        with pytest.raises(ValueError):
            list(reader(VideoSource("file", "sample.mp4"), child).frames(max_idle_seconds=value))


def test_pipeline_closes_owned_frame_source_after_callback_error():
    class FrameIterator:
        def __init__(self):
            self.closed = False
            self.used = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.used:
                raise StopIteration
            self.used = True
            return VideoFrame(1, 10.0, RGBFrame(16, 16, bytes(768)), 0)

        def close(self):
            self.closed = True

    frames = FrameIterator()
    def fail(*args):
        raise RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        VisionPipeline().run(frames, session_id="vera", source_kind="file", on_frame=fail)
    assert frames.closed


def test_pipeline_does_not_claim_source_time_from_decoder():
    event = VisionPipeline().observe(
        VideoFrame(1, 42.5, RGBFrame(16, 16, bytes(768)), 0),
        session_id="vera", source_kind="stream",
    ).to_dict()
    assert event["decoded_at_monotonic"] == 42.5
    assert event["timestamp_basis"] == "OBSERVER_HOST_DECODE_MONOTONIC"
    assert "captured_at_monotonic" not in event
    assert event["source_kind"] == "stream"
    assert event["action_authority"] is False
