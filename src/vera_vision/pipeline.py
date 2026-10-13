"""Continuous perception events from a live video source.

Only structured observations leave this module. A consumer can opt into raw RGB
frames through the explicit in-process callback; no pixels are logged by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterable, TextIO

from .live import VideoFrame
from .temporal import TemporalAnalyzer, TemporalObservation


@dataclass(frozen=True, slots=True)
class VisionEvent:
    """One bounded, evidence-labeled observation from a decoded video frame."""

    session_id: str
    decoded_at_monotonic: float
    sequence: int
    dropped_frames: int
    source_kind: str
    temporal: TemporalObservation

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "VERA_VISION_LIVE_EVENT_V1",
            "kind": "FRAME_OBSERVED",
            "observer": "VERA",
            "session_id": self.session_id,
            "decoded_at_monotonic": self.decoded_at_monotonic,
            "timestamp_basis": "OBSERVER_HOST_DECODE_MONOTONIC",
            "source_kind": self.source_kind,
            "sequence": self.sequence,
            "dropped_frames": self.dropped_frames,
            "temporal": self.temporal.to_dict(),
            "image_pixels_included": False,
            "action_authority": False,
        }


class VisionPipeline:
    """Subscribe to frames and produce chronological, typed visual evidence.

    on_frame is an explicit, in-process boundary for future OCR, detector,
    visual reasoning or object-tracking adapters. It receives decoded pixels and
    can raise to stop the session; no network provider is configured here.
    """

    def __init__(self, *, analyzer: TemporalAnalyzer | None = None) -> None:
        self.analyzer = analyzer or TemporalAnalyzer()

    def observe(self, packet: VideoFrame, *, session_id: str, source_kind: str,
                on_frame: Callable[[VideoFrame, TemporalObservation], None] | None = None) -> VisionEvent:
        observation = self.analyzer.observe(
            packet.frame,
            session_id=session_id,
            step=packet.sequence,
            monotonic_seconds=packet.decoded_at_monotonic,
        )
        if on_frame is not None:
            on_frame(packet, observation)
        return VisionEvent(session_id, packet.decoded_at_monotonic, packet.sequence, packet.dropped_since_previous, source_kind, observation)

    def run(self, frames: Iterable[VideoFrame], *, session_id: str, source_kind: str,
            output: TextIO | None = None,
            on_frame: Callable[[VideoFrame, TemporalObservation], None] | None = None) -> int:
        total = 0
        try:
            for packet in frames:
                event = self.observe(packet, session_id=session_id,
                                     source_kind=source_kind, on_frame=on_frame)
                if output is not None:
                    output.write(json.dumps(event.to_dict(), separators=(",", ":"), allow_nan=False) + "\n")
                    output.flush()
                total += 1
            return total
        finally:
            closer = getattr(frames, "close", None)
            if callable(closer):
                closer()
