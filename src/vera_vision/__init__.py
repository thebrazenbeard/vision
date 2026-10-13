"""Vera Vision: reusable visual-evidence primitives without controller authority."""
from .frames import RGBFrame
from .png import decode_rgba_png
from .temporal import TemporalAnalyzer, TemporalObservation, FrameSummary, sample_rgb

__all__ = ["RGBFrame", "decode_rgba_png", "TemporalAnalyzer", "TemporalObservation", "FrameSummary", "sample_rgb"]
