"""Regressions for frames captured at a timestamp with no elapsed time.

A SAME_TICK frame is observable but cannot qualify the next comparison
baseline; the latest temporally valid frame must remain the reference.
"""
from vera_vision import RGBFrame, TemporalAnalyzer


def _solid(value: int) -> RGBFrame:
    return RGBFrame(16, 16, bytes((value, value, value)) * 256)


def test_same_tick_does_not_create_phantom_next_frame_change():
    analyzer = TemporalAnalyzer()
    original = _solid(0)
    assert analyzer.observe(original, session_id="vision", step=0, monotonic_seconds=1.0).status == "FIRST_FRAME"
    duplicate_tick = analyzer.observe(_solid(255), session_id="vision", step=1, monotonic_seconds=1.0)
    assert duplicate_tick.status == "SAME_TICK"
    assert duplicate_tick.elapsed_ms == 0
    assert duplicate_tick.change_fraction is None
    assert len(analyzer.history) == 1
    later = analyzer.observe(original, session_id="vision", step=2, monotonic_seconds=2.0)
    assert later.status == "STATIC_SAMPLE"
    assert later.change_fraction == 0.0
    assert len(analyzer.history) == 2


def test_same_tick_does_not_suppress_genuine_later_change():
    analyzer = TemporalAnalyzer()
    analyzer.observe(_solid(0), session_id="vision", step=0, monotonic_seconds=1.0)
    analyzer.observe(_solid(255), session_id="vision", step=1, monotonic_seconds=1.0)
    later = analyzer.observe(_solid(255), session_id="vision", step=2, monotonic_seconds=2.0)
    assert later.status == "PIXEL_CHANGE_DETECTED"
    assert later.change_fraction == 1.0
    assert later.changed_regions == ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT")


def test_repeated_same_tick_frames_leave_only_last_valid_baseline():
    analyzer = TemporalAnalyzer(history_size=2)
    initial = _solid(12)
    analyzer.observe(initial, session_id="vision", step=0, monotonic_seconds=1.0)
    for step in range(1, 5):
        result = analyzer.observe(_solid(100 + step), session_id="vision",
                                  step=step, monotonic_seconds=1.0)
        assert result.status == "SAME_TICK"
        assert len(analyzer.history) == 1
    result = analyzer.observe(initial, session_id="vision", step=5, monotonic_seconds=1.5)
    assert result.status == "STATIC_SAMPLE"
