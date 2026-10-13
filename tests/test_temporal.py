import pytest

from vera_vision import RGBFrame, TemporalAnalyzer


def solid(rgb, size=16):
    return RGBFrame(size, size, bytes(rgb) * (size * size))


def test_red_to_green_with_same_grayscale_mean_is_visible():
    a = TemporalAnalyzer()
    a.observe(solid((255, 0, 0)), session_id="game", step=0, monotonic_seconds=1)
    r = a.observe(solid((0, 255, 0)), session_id="game", step=1, monotonic_seconds=2)
    assert r.status == "PIXEL_CHANGE_DETECTED"
    assert r.change_fraction == 1.0
    assert r.changed_regions == ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT")
    assert r.to_dict()["action_authority"] is False


def test_quadrant_location_and_no_inferred_game_semantics():
    pixels = bytearray(bytes((0, 0, 0)) * 256)
    for y in range(8):
        for x in range(8):
            offset = (y * 16 + x) * 3
            pixels[offset:offset + 3] = bytes((255, 0, 0))
    a = TemporalAnalyzer()
    a.observe(solid((0, 0, 0)), session_id="g", step=0, monotonic_seconds=1)
    r = a.observe(RGBFrame(16, 16, bytes(pixels)), session_id="g", step=1, monotonic_seconds=2)
    assert r.change_fraction == 0.25
    assert r.changed_regions == ("TOP_LEFT",)
    assert r.semantic_state == "UNKNOWN"
    assert r.to_dict()["provenance"] == "SAMPLED_RGB_PIXELS"


def test_identical_frames_are_static_without_caching_full_frame():
    a = TemporalAnalyzer(history_size=2)
    image = solid((12, 34, 56))
    assert a.observe(image, session_id="g", step=0, monotonic_seconds=1).status == "FIRST_FRAME"
    result = a.observe(image, session_id="g", step=1, monotonic_seconds=2)
    assert result.status == "STATIC_SAMPLE" and result.change_fraction == 0
    assert len(a.history) == 2
    assert not hasattr(a.history[0], "rgb")
    assert len(a.history[0].samples) == 768
    assert a.history[0].sha256_prefix == a.history[1].sha256_prefix


def test_stale_session_and_clock_fail_closed():
    a = TemporalAnalyzer(max_gap_seconds=2.0)
    f = solid((0, 0, 0))
    assert a.observe(f, session_id="a", step=0, monotonic_seconds=1).status == "FIRST_FRAME"
    assert a.observe(f, session_id="a", step=1, monotonic_seconds=10).status == "STALE_GAP"
    assert a.observe(f, session_id="b", step=0, monotonic_seconds=11).status == "FIRST_FRAME"
    assert a.observe(f, session_id="b", step=1, monotonic_seconds=11).status == "SAME_TICK"
    assert a.observe(f, session_id="b", step=2, monotonic_seconds=10).status == "CLOCK_REGRESSION"
    assert a.observe(f, session_id="b", step=1, monotonic_seconds=11).status == "STEP_REGRESSION"
    assert a.observe(RGBFrame(8, 8, bytes(8 * 8 * 3)), session_id="b", step=2, monotonic_seconds=12).status == "DIMENSIONS_CHANGED"


def test_strict_inputs_and_bounded_history():
    with pytest.raises(ValueError):
        RGBFrame(True, 10, bytes(30))
    with pytest.raises(ValueError):
        RGBFrame(16, 16, bytes(32))
    with pytest.raises(ValueError):
        TemporalAnalyzer(history_size=100)
    with pytest.raises(ValueError):
        TemporalAnalyzer(max_gap_seconds=float("nan"))
    a = TemporalAnalyzer()
    with pytest.raises(ValueError):
        a.observe(solid((0, 0, 0)), session_id="", step=0)
    with pytest.raises(ValueError):
        a.observe(solid((0, 0, 0)), session_id="s", step=-1)
