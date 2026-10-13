"""Continuous, bounded FFmpeg video ingestion for Vera Vision.

A window title, file, or explicitly selected media URL is decoded to RGB24.
Pixels stay in memory; no external model call or controller input is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import queue
import math
import subprocess
import sys
import threading
import time
from typing import Callable, Iterator
from urllib.parse import urlsplit

from .frames import RGBFrame

ALLOWED_STREAM_PROTOCOLS = frozenset({"rtsp", "rtmp", "rtmps", "srt", "http", "https"})
_MAX_FRAME_BYTES = 640 * 360 * 3


@dataclass(frozen=True, slots=True)
class VideoSource:
    """Explicit video source. A title selects an exact Windows window, not a substring."""

    kind: str
    locator: str
    width: int = 320
    height: int = 180
    fps: int = 2

    def __post_init__(self) -> None:
        if self.kind not in ("window", "stream", "file"):
            raise ValueError("video source kind must be window, stream, or file")
        if type(self.locator) is not str or not 1 <= len(self.locator) <= 2048:
            raise ValueError("video source locator is invalid")
        if any(c in self.locator for c in "\r\n\0"):
            raise ValueError("video locator contains unsafe control characters")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("frame dimensions must be integers")
        if not (16 <= self.width <= 640 and 16 <= self.height <= 360):
            raise ValueError("decoded frame dimensions exceed realtime budget")
        if self.width * self.height * 3 > _MAX_FRAME_BYTES:
            raise ValueError("decoded frame budget exceeded")
        if type(self.fps) is not int or not 1 <= self.fps <= 15:
            raise ValueError("fps must be between 1 and 15")
        if self.kind == "stream":
            parsed = urlsplit(self.locator)
            if (parsed.scheme.lower() not in ALLOWED_STREAM_PROTOCOLS
                    or not parsed.netloc or parsed.username is not None
                    or parsed.password is not None):
                raise ValueError("stream URL must use an allowed protocol without inline credentials")
        if self.kind == "window" and not 2 <= len(self.locator) <= 256:
            raise ValueError("window title length invalid")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded frame. The timestamp is when the observer decoded it, not when the source recorded it."""

    sequence: int
    decoded_at_monotonic: float
    frame: RGBFrame
    dropped_since_previous: int


class VideoSourceError(RuntimeError):
    """Sanitized failure: no URL, username, token, or process stderr in error."""


def ffmpeg_arguments(source: VideoSource, *, binary: str = "ffmpeg",
                     platform: str | None = None) -> list[str]:
    """Build a command argument vector. Never launch through a shell."""
    if type(binary) is not str or not binary.strip():
        raise ValueError("FFmpeg binary is required")
    args = [binary, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if source.kind == "window":
        if (platform or sys.platform) != "win32":
            raise ValueError("window capture requires Windows gdigrab")
        args += ["-f", "gdigrab", "-draw_mouse", "0", "-framerate",
                 str(source.fps), "-i", "title=" + source.locator]
    else:
        args += ["-i", source.locator]
    # Decoder output is fixed-size, irrespective of remote video dimensions.
    # Preserve aspect ratio and pad rather than stretch to a false geometry.
    video_filter = (
        f"fps={source.fps},"
        f"scale={source.width}:{source.height}:force_original_aspect_ratio=decrease,"
        f"pad={source.width}:{source.height}:(ow-iw)/2:(oh-ih)/2"
    )
    args += ["-map", "0:v:0", "-an", "-sn", "-dn",
             "-vf", video_filter, "-pix_fmt", "rgb24",
             "-f", "rawvideo", "-y", "pipe:1"]
    return args


def _read_frame(stream: object, n: int) -> bytes | None:
    """Read one frame even if pipe reads return short chunks."""
    output = bytearray()
    while len(output) < n:
        chunk = stream.read(n - len(output))  # type: ignore[attr-defined]
        if not chunk:
            return None
        output.extend(chunk)
    return bytes(output)


class FFmpegFrames:
    """Read latest frames with a bounded queue, dropping old frames under load.

    The reader thread is daemonized and stopped by terminating its child process.
    The producer never waits on consumer inference. This is not a synchronized
    audio/video recorder; it is a best-effort, frame-based perception source.
    """

    def __init__(self, source: VideoSource, *, binary: str = "ffmpeg",
                 process_factory: Callable[..., object] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.source = source
        self.binary = binary
        self.process_factory = process_factory
        self.clock = clock
        self._process: object | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()

    def _stop(self) -> None:
        self._closed.set()
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:  # type: ignore[attr-defined]
                process.terminate()  # type: ignore[attr-defined]
                try:
                    process.wait(timeout=1)  # type: ignore[attr-defined]
                except subprocess.TimeoutExpired:
                    process.kill()  # type: ignore[attr-defined]
                    process.wait(timeout=1)  # type: ignore[attr-defined]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._process = None
        self._thread = None

    def close(self) -> None:
        self._stop()

    def frames(self, *, max_frames: int = 100, max_seconds: float = 60,
               max_idle_seconds: float = 10.0) -> Iterator[VideoFrame]:
        """Return up to max_frames or max_seconds; explicit finite bounds mandatory."""
        if type(max_frames) is not int or not 1 <= max_frames <= 100_000:
            raise ValueError("max_frames out of bounds")
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0.1 <= max_seconds <= 3600:
            raise ValueError("max_seconds out of bounds")
        if type(max_idle_seconds) not in (int, float) or not math.isfinite(max_idle_seconds) or not 0.05 <= max_idle_seconds <= 60:
            raise ValueError("max_idle_seconds out of bounds")
        if self._process is not None:
            raise RuntimeError("reader already active")
        frame_bytes = self.source.width * self.source.height * 3
        command = ffmpeg_arguments(self.source, binary=self.binary)
        try:
            process = self.process_factory(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                shell=False, bufsize=0,
            )
        except OSError as exc:
            raise VideoSourceError("FFMPEG_START_FAILED") from None
        self._process = process
        self._closed.clear()
        buffer: queue.Queue[tuple[int, float, bytes]] = queue.Queue(maxsize=2)
        completed = threading.Event()
        errors: list[str] = []

        def read_loop() -> None:
            sequence = 0
            try:
                while not self._closed.is_set():
                    chunk = _read_frame(process.stdout, frame_bytes)  # type: ignore[attr-defined]
                    if chunk is None:
                        break
                    sequence += 1
                    packet = (sequence, self.clock(), chunk)
                    try:
                        buffer.put_nowait(packet)
                    except queue.Full:
                        try:
                            buffer.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            buffer.put_nowait(packet)
                        except queue.Full:
                            pass
            except (OSError, ValueError, TypeError):
                errors.append("FRAME_PIPE_READ_FAILED")
            finally:
                completed.set()

        self._thread = threading.Thread(target=read_loop, name="vera-vision-frame-reader", daemon=True)
        self._thread.start()
        start = self.clock()
        last_sequence = 0
        delivered = 0
        last_frame_time = start
        try:
            while delivered < max_frames:
                remaining = max_seconds - (self.clock() - start)
                if remaining <= 0:
                    break
                try:
                    sequence, captured_at, data = buffer.get(timeout=min(remaining, 0.1))
                except queue.Empty:
                    if completed.is_set() and buffer.empty():
                        break
                    if self.clock() - last_frame_time >= max_idle_seconds:
                        raise VideoSourceError(
                            "LIVE_VIDEO_NO_FRAMES_TIMEOUT" if delivered == 0
                            else "LIVE_VIDEO_FRAME_IDLE_TIMEOUT"
                        )
                    continue
                dropped = max(0, sequence - last_sequence - 1)
                last_sequence = sequence
                delivered += 1
                last_frame_time = self.clock()
                yield VideoFrame(sequence, captured_at,
                                 RGBFrame(self.source.width, self.source.height, data),
                                 dropped)
            if errors:
                raise VideoSourceError(errors[0])
            if delivered == 0:
                if completed.is_set():
                    raise VideoSourceError("VIDEO_STREAM_EMPTY_OR_UNDECODABLE")
                raise VideoSourceError("LIVE_VIDEO_NO_FRAMES_TIMEOUT")
            if (self.source.kind in ("stream", "window")
                    and completed.is_set() and delivered < max_frames):
                raise VideoSourceError("LIVE_VIDEO_SOURCE_ENDED")
        finally:
            self._stop()
