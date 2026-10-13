import binascii
import struct
import zlib

import pytest

from vera_vision import RGBFrame, decode_rgba_png


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", binascii.crc32(kind + body) & 0xFFFFFFFF)


def png(rows, width, height, *, color_type=6, filter_id=0):
    data = b"".join(bytes((filter_id,)) + row for row in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b"")


def test_strict_rgba_png_to_rgb():
    content = png([bytes((11, 22, 33, 44, 55, 66, 77, 88))], 2, 1)
    result = decode_rgba_png(content)
    assert result == RGBFrame(2, 1, bytes((11, 22, 33, 55, 66, 77)))


@pytest.mark.parametrize("corrupt", [
    lambda p: p[:-12],
    lambda p: p + b"junk",
    lambda p: p.replace(b"IHDR", b"XXDR"),
    lambda p: b"not an image",
])
def test_invalid_or_truncated_png_fails_closed(corrupt):
    p = png([bytes((1, 2, 3, 4))], 1, 1)
    with pytest.raises(ValueError):
        decode_rgba_png(corrupt(p))


def test_png_rejects_unsupported_filters_and_encoding():
    with pytest.raises(ValueError, match="filter"):
        decode_rgba_png(png([bytes((1, 2, 3, 4))], 1, 1, filter_id=1))
    with pytest.raises(ValueError, match="unsupported"):
        decode_rgba_png(png([bytes((1, 2, 3, 4))], 1, 1, color_type=2))


def test_decompression_bomb_dimension_fails_before_inflate():
    p = png([bytes((1, 2, 3, 4))], 5000, 5000)
    with pytest.raises(ValueError, match="budget"):
        decode_rgba_png(p)


def test_png_rejects_extra_decompressed_bytes():
    with pytest.raises(ValueError, match="size"):
        decode_rgba_png(png([bytes((1, 2, 3, 4, 5, 6, 7, 8))], 1, 1))
