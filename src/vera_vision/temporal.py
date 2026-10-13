"""Sampled spatial image change; never identity, motion vectors or collision authority."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import math
import time
from typing import Callable

from .frames import RGBFrame

SAMPLE_SIDE = 16
CHANNEL_THRESHOLD = 12
REGION_SAMPLE_MIN = 4
REGION_NAMES = ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT")


@dataclass(frozen=True, slots=True)
class FrameSummary:
    session_id: str
    step: int
    monotonic_seconds: float
    sha256_prefix: str
    width: int
    height: int
    samples: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    status: str
    elapsed_ms: int | None
    change_fraction: float | None
    mean_max_channel_delta: float | None
    changed_regions: tuple[str, ...]
    sample_count: int
    provenance: str = "SAMPLED_RGB_PIXELS"
    semantic_state: str = "UNKNOWN"
    action_authority: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "VERA_VISION_TEMPORAL_OBSERVATION_V1",
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "pixel_change_fraction": self.change_fraction,
            "pixel_change_mean_max_channel_delta": self.mean_max_channel_delta,
            "changed_regions": list(self.changed_regions),
            "sample_count": self.sample_count,
            "provenance": self.provenance,
            "semantic_state": self.semantic_state,
            "action_authority": self.action_authority,
        }


def sample_rgb(frame: RGBFrame) -> tuple[int, ...]:
    """Return 256 RGB triplets; keep neither the original frame nor alpha."""
    positions_x = [min(frame.width - 1, (2 * i + 1) * frame.width // (2 * SAMPLE_SIDE)) for i in range(SAMPLE_SIDE)]
    positions_y = [min(frame.height - 1, (2 * i + 1) * frame.height // (2 * SAMPLE_SIDE)) for i in range(SAMPLE_SIDE)]
    samples = bytearray(SAMPLE_SIDE * SAMPLE_SIDE * 3)
    at = 0
    for y in positions_y:
        for x in positions_x:
            offset = (y * frame.width + x) * 3
            samples[at:at + 3] = frame.rgb[offset:offset + 3]
            at += 3
    return tuple(samples)


class TemporalAnalyzer:
    def __init__(self, *, history_size: int = 8, max_gap_seconds: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if type(history_size) is not int or not 2 <= history_size <= 32:
            raise ValueError("history_size must be in [2, 32]")
        if type(max_gap_seconds) not in (float, int) or not math.isfinite(max_gap_seconds) or not 0.01 <= max_gap_seconds <= 3600:
            raise ValueError("invalid max_gap_seconds")
        self._clock = clock
        self.max_gap_seconds = float(max_gap_seconds)
        self._history: deque[FrameSummary] = deque(maxlen=history_size)

    @property
    def history(self) -> tuple[FrameSummary, ...]:
        return tuple(self._history)

    def observe(self, frame: RGBFrame, *, session_id: str, step: int,
                monotonic_seconds: float | None = None) -> TemporalObservation:
        if type(session_id) is not str or not session_id or len(session_id) > 128:
            raise ValueError("session_id invalid")
        if type(step) is not int or step < 0:
            raise ValueError("step invalid")
        now = float(self._clock() if monotonic_seconds is None else monotonic_seconds)
        if not math.isfinite(now):
            raise ValueError("clock invalid")
        samples = sample_rgb(frame)
        record = FrameSummary(session_id, step, now, sha256(frame.rgb).hexdigest()[:16],
                              frame.width, frame.height, samples)
        prev = self._history[-1] if self._history else None
        if prev is None or prev.session_id != session_id:
            self._history.clear()
            status = "FIRST_FRAME"
        elif (prev.width, prev.height) != (frame.width, frame.height):
            self._history.clear()
            status = "DIMENSIONS_CHANGED"
        elif step <= prev.step:
            self._history.clear()
            status = "STEP_REGRESSION"
        elif now < prev.monotonic_seconds:
            self._history.clear()
            status = "CLOCK_REGRESSION"
        elif now == prev.monotonic_seconds:
            status = "SAME_TICK"
        elif now - prev.monotonic_seconds > self.max_gap_seconds:
            self._history.clear()
            status = "STALE_GAP"
        else:
            diff = [max(abs(samples[3 * i + channel] - prev.samples[3 * i + channel])
                        for channel in range(3)) for i in range(SAMPLE_SIDE * SAMPLE_SIDE)]
            changed = [d > CHANNEL_THRESHOLD for d in diff]
            regions = []
            half = SAMPLE_SIDE // 2
            for title, dy, dx in (("TOP_LEFT", 0, 0), ("TOP_RIGHT", 0, 1),
                                  ("BOTTOM_LEFT", 1, 0), ("BOTTOM_RIGHT", 1, 1)):
                count = sum(changed[y * SAMPLE_SIDE + x]
                            for y in range(dy * half, (dy + 1) * half)
                            for x in range(dx * half, (dx + 1) * half))
                if count >= REGION_SAMPLE_MIN:
                    regions.append(title)
            status = "PIXEL_CHANGE_DETECTED" if any(changed) else "STATIC_SAMPLE"
            self._history.append(record)
            return TemporalObservation(status, round((now - prev.monotonic_seconds) * 1000),
                                       sum(changed) / len(changed), sum(diff) / len(diff),
                                       tuple(regions), SAMPLE_SIDE * SAMPLE_SIDE)
        self._history.append(record)
        return TemporalObservation(status, 0 if status == "SAME_TICK" else None,
                                   None, None, (), SAMPLE_SIDE * SAMPLE_SIDE)
