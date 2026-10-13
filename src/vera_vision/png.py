"""Strict RGBA8/filter-0 PNG input adapter for controlled screenshot encoders.

This is *not* a general PNG decoder. Decode errors are explicit, not inferred as
an unchanged image. The compressed and decompressed byte counts are bounded.
"""
from __future__ import annotations

import binascii
import struct
import zlib

from .frames import MAX_DIMENSION, MAX_PIXELS, RGBFrame

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ENCODED_BYTES = 64_000_000


def decode_rgba_png(payload: bytes) -> RGBFrame:
    if type(payload) is not bytes or not (57 <= len(payload) <= MAX_ENCODED_BYTES):
        raise ValueError("PNG length invalid")
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError("PNG signature invalid")
    pos = len(PNG_SIGNATURE)
    dims: tuple[int, int] | None = None
    image_data = bytearray()
    seen_end = False
    seen_data = False
    while pos + 12 <= len(payload):
        n = struct.unpack_from(">I", payload, pos)[0]
        if n + 12 > len(payload) - pos:
            raise ValueError("PNG chunk truncated")
        tag = payload[pos + 4 : pos + 8]
        data = payload[pos + 8 : pos + 8 + n]
        expected_crc = struct.unpack_from(">I", payload, pos + 8 + n)[0]
        if binascii.crc32(tag + data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG CRC mismatch")
        pos += 12 + n
        if tag == b"IHDR":
            if dims is not None or n != 13 or seen_data:
                raise ValueError("PNG header invalid")
            width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", data)
            if (depth, color, comp, filt, interlace) != (8, 6, 0, 0, 0):
                raise ValueError("PNG encoding unsupported")
            if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION and width * height <= MAX_PIXELS):
                raise ValueError("PNG dimensions exceed budget")
            dims = (width, height)
        elif tag == b"IDAT":
            if dims is None or seen_end or len(image_data) + n > MAX_ENCODED_BYTES:
                raise ValueError("PNG image data invalid")
            image_data.extend(data)
            seen_data = True
        elif tag == b"IEND":
            if n != 0 or dims is None or not seen_data:
                raise ValueError("PNG terminator invalid")
            seen_end = True
            break
        else:
            raise ValueError("PNG chunk unsupported")
    if not seen_end or pos != len(payload) or not image_data or dims is None:
        raise ValueError("PNG incomplete")
    width, height = dims
    stride = width * 4
    expected = height * (stride + 1)
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(image_data, expected + 1)
    except zlib.error as exc:
        raise ValueError("PNG decompression failed") from exc
    if len(decoded) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("PNG decompressed size invalid")
    rgb = bytearray(width * height * 3)
    for y in range(height):
        start = y * (stride + 1)
        if decoded[start] != 0:
            raise ValueError("PNG row filter unsupported")
        for x in range(width):
            src = start + 1 + x * 4
            dst = (y * width + x) * 3
            rgb[dst:dst + 3] = decoded[src:src + 3]
    return RGBFrame(width, height, bytes(rgb))
