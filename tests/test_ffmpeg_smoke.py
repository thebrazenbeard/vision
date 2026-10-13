"""Exercise a real FFmpeg decode pipeline on synthetic video when FFmpeg is installed.

No network, webcam, ARK, Chiaki, or operator display is accessed.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from vera_vision.live import FFmpegFrames, VideoSource
from vera_vision.pipeline import VisionPipeline


def test_ffmpeg_synthetic_file_ingestion(tmp_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg is not installed on this runner")
    recording = tmp_path / "synthetic.avi"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=5",
        "-frames:v", "8", "-an", "-c:v", "mpeg4", "-q:v", "3",
        "-y", str(recording),
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            timeout=20, check=False)
    if result.returncode != 0:
        pytest.skip("installed FFmpeg cannot encode the controlled synthetic fixture")
    source = VideoSource("file", str(recording), width=32, height=32, fps=2)
    reader = FFmpegFrames(source, binary=ffmpeg)
    packets = list(reader.frames(max_frames=12, max_seconds=8, max_idle_seconds=5))
    assert packets
    assert all(p.frame.width == 32 and p.frame.height == 32 for p in packets)
    assert all(len(p.frame.rgb) == 32 * 32 * 3 for p in packets)
    assert all(p.sequence > 0 for p in packets)
    assert reader._process is None
    observed = [
        VisionPipeline().observe(p, session_id="synthetic-fixture", source_kind="file").to_dict()
        for p in packets
    ]
    assert all(e["observer"] == "VERA" and e["action_authority"] is False for e in observed)
    assert all(e["timestamp_basis"] == "OBSERVER_HOST_DECODE_MONOTONIC" for e in observed)
    assert all(e["image_pixels_included"] is False for e in observed)
