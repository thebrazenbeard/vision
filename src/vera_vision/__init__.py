"""Vera Vision: reusable visual-evidence primitives without controller authority."""
from .frames import RGBFrame
from .png import decode_rgba_png
from .live import FFmpegFrames, VideoFrame, VideoSource
from .pipeline import VisionPipeline, VisionEvent
from .temporal import TemporalAnalyzer, TemporalObservation, FrameSummary, sample_rgb

__all__ = ["RGBFrame", "VideoSource", "VideoFrame", "FFmpegFrames", "VisionPipeline", "VisionEvent", "decode_rgba_png", "TemporalAnalyzer", "TemporalObservation", "FrameSummary", "sample_rgb"]
