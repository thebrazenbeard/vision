"""Validated RGB24 inputs; capture transport and credentials are deliberately absent."""
from __future__ import annotations

from dataclasses import dataclass

MAX_PIXELS = 16_000_000
MAX_DIMENSION = 16_384


@dataclass(frozen=True, slots=True)
class RGBFrame:
    """RGB byte buffer in top-left raster order, no alpha channel."""

    width: int
    height: int
    rgb: bytes

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("frame dimensions must be integers")
        if not (1 <= self.width <= MAX_DIMENSION and 1 <= self.height <= MAX_DIMENSION):
            raise ValueError("invalid frame dimensions")
        if self.width * self.height > MAX_PIXELS:
            raise ValueError("frame pixel budget exceeded")
        if type(self.rgb) is not bytes or len(self.rgb) != self.width * self.height * 3:
            raise ValueError("frame must contain exactly RGB24 bytes")
