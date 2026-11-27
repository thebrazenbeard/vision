from vera_vision import RGBFrame, TemporalAnalyzer

def frame(value):
    return RGBFrame(64, 64, bytes([value])*64*64*3)

def test_same_tick_keeps_digest_baseline():
    a=TemporalAnalyzer()
    a.observe(frame(0),session_id="s",step=0,monotonic_seconds=1)
    r=a.observe(frame(255),session_id="s",step=1,monotonic_seconds=1)
    assert r.status=="SAME_TICK" and r.full_frame_bytes_changed is None
    r=a.observe(frame(0),session_id="s",step=2,monotonic_seconds=2)
    assert r.status=="STATIC_SAMPLE" and r.full_frame_bytes_changed is False

def test_same_tick_preserves_step_regression_guard():
    a=TemporalAnalyzer()
    a.observe(frame(0),session_id="s",step=0,monotonic_seconds=1)
    a.observe(frame(255),session_id="s",step=3,monotonic_seconds=1)
    r=a.observe(frame(0),session_id="s",step=2,monotonic_seconds=2)
    assert r.status=="STEP_REGRESSION" and r.full_frame_bytes_changed is None
    r=a.observe(frame(0),session_id="s",step=3,monotonic_seconds=3)
    assert r.status=="STATIC_SAMPLE" and r.full_frame_bytes_changed is False

def test_same_tick_then_real_change():
    a=TemporalAnalyzer()
    a.observe(frame(0),session_id="s",step=0,monotonic_seconds=1)
    a.observe(frame(255),session_id="s",step=1,monotonic_seconds=1)
    r=a.observe(frame(255),session_id="s",step=2,monotonic_seconds=2)
    assert r.status=="PIXEL_CHANGE_DETECTED" and r.full_frame_bytes_changed is True
