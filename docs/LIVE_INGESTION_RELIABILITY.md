# Live ingestion reliability and transport research

Status: **source implementation plus source/CI tests**, not an installed Vera Mono runtime route, not a qualified Chiaki session, and not verified live semantic comprehension.

## Correct observation model

```text
Remote video source (ARK/PS4, Chiaki, RTSP camera, recording)
    -> explicit FFmpeg input -> decoded RGB24 frames
    -> bounded queue, drop stale pending frames under load
    -> observer-host monotonic *decode/receipt* timestamp
    -> pixel evidence and provenance in Vera Vision
    -> optional in-process observation consumer owned by Vera
    -> separately governed reasoning / effect boundary
```

**A local decode timestamp is not source capture time.** It cannot on its own quantify transport latency or the age of an object in the game world. Source presentation timestamps (PTS), clock offset, sender sequence and measured receive/processing delays are a distinct future transport-specific contract. `dropped_frames` currently measures frame sequence gaps attributable to **Vision's bounded decoder queue**, not remote network packet loss or upstream encoder drops.

## What is implemented

- `FFmpegFrames.frames(max_frames, max_seconds, max_idle_seconds)` has separate finite session and frame-idle budgets. If no frame appears, it emits `LIVE_VIDEO_NO_FRAMES_TIMEOUT`; if a feed stalls after frames, it emits `LIVE_VIDEO_FRAME_IDLE_TIMEOUT`. Early EOF of a live stream or Windows window produces `LIVE_VIDEO_SOURCE_ENDED`. Expected EOF in a prerecorded file is not a failure.
- A completed feed isn't considered healthy simply because the watch timer expired. A downstream callback exception explicitly closes the source generator and terminates its owned FFmpeg process.
- Neither FFmpeg stderr nor a stream locator is included in user-facing errors. Child processes start with `shell=False`, without a shell intermediary. Raw decoded frames remain available only through an explicit in-process callback; JSONL events contain no pixels.
- A synthetic FFmpeg-generated file is decoded end-to-end where FFmpeg is installed on the CI runner; tests skip only that integration test if FFmpeg or the required sample encoder is unavailable. All Python-only tests remain required.

## Research and bounded decisions

**MediaMTX.** Its official [FFmpeg reader documentation](https://mediamtx.org/docs/read/ffmpeg) supports RTSP, RTMP, HLS and SRT; [stream read overview](https://mediamtx.org/docs/features/read) details available clients. The current `stream` source covers FFmpeg-accessible MediaMTX URLs **without installing MediaMTX**. MediaMTX itself provides routing/recording/protocol conversions, not the visual understanding engine.

**LiveKit.** Its [track subscription documentation](https://docs.livekit.io/transport/media/subscribe/) and [Python SDK](https://github.com/livekit/python-sdks) expose real decoded video frames via `rtc.VideoStream(track)`. A robust future LiveKit adapter should consume subscribed tracks directly, normalize pixel formats and source-clock/provenance metadata, and feed `RGBFrame` to the same temporal pipeline. It should **not** feed a LiveKit WebRTC room URL to FFmpeg as though it were RTSP. The separate [LiveKit Agents video guide](https://docs.livekit.io/agents/multimodality/vision/video/) documents frame sampling and video-capable real-time model requirements. Merely subscribing to a track is not proof that an attached reasoning model uses it.

**Windows game capture.** Chiaki output remains an application displayed on Windows. FFmpeg `gdigrab` can use an exact window title, but it is not an occlusion-proof private surface and can include overlays. Frame acquisition from another worker's active Chiaki controller session must be coordinated and explicitly scoped. The vision project must never acquire gamepad authority.

**Deep-Live-Cam and LiveVideoCoreSDK.** Useful mainly as research for real-time processing and mobile publishing respectively. Face replacement is outside this perception project; the iOS publisher is not the correct Windows input path.

## Remaining qualification work

1. **Live source probes** with a trusted local recording, Chiaki window and MediaMTX RTSP feed: measure first-frame latency, decoder freeze, recovery time, queue frame age, scene-change accuracy and resource load.
2. **Reconnect policy:** add restart/backoff with bounded attempts, fresh session/generation identity, circuit-breaker and explicit operator stop, only after a representative live failure corpus. Never silently replay stale frames as current.
3. **Temporal semantics:** decouple observer receipt time, source PTS, camera motion and world movement. No collision clearance claim from sampled pixel differences alone.
4. **Vera consumption:** bind and verify the exact installed `vera_core.QualifiedVeraRuntime` consumer route, frame provenance and receipt. The Python callback is currently a prepared seam, not a verified runtime connection.
5. **Real vision:** optional OCR, detectors, optical flow, object tracking and vision-language model inference with privacy constraints and measured confidence/calibration; evaluate on labeled clips, not synthetic color panels alone.

This repo processes video **from** ARK or another source **for** Vera to interpret. It does not give ARK a model or grant autonomous game control.
