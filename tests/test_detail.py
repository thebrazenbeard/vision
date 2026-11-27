from vera_vision import RGBFrame,TemporalAnalyzer

def f(pixel=None, val=1):
    d=bytearray(64*64*3)
    if pixel is not None:d[3*pixel]=val
    return RGBFrame(64,64,bytes(d))

def pair(first,second):
    a=TemporalAnalyzer()
    a.observe(first,session_id="s",step=0,monotonic_seconds=0.)
    return a.observe(second,session_id="s",step=1,monotonic_seconds=.5).to_dict()

def test_unsampled_change():
    r=pair(f(),f(0,255))
    assert r.get("full_frame_bytes_changed") is True
    assert r["status"]=="STATIC_SAMPLE"

def test_subthreshold_change():
    r=pair(f(),f(130,1))
    assert r.get("full_frame_bytes_changed") is True

def test_identical():
    assert pair(f(),f()).get("full_frame_bytes_changed") is False

def test_baseline_and_reset_unknown():
    a=TemporalAnalyzer()
    for session,step,t in [("a",0,0.),("a",1,0.),("b",0,1.),("b",1,20.)]:
        r=a.observe(f(),session_id=session,step=step,monotonic_seconds=t).to_dict()
        assert "full_frame_bytes_changed" in r and r["full_frame_bytes_changed"] is None
        assert r["action_authority"] is False
