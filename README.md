# VISION

**A continuous visual sense for Vera.**

Vision is the system that lets **Vera watch, interpret, and remember visual events over time**—from a game displayed through Chiaki, a desktop application, a camera, or a network video stream. The source provides frames. **Vera is the observer.** Nothing here makes ARK, Chiaki, a streaming server, or a controller into the perceiving agent.

> **Current status:** live-video ingestion and evidence generation are implemented as a standalone source package. Continuous *semantic* comprehension, optical-flow/object tracking, and verified consumption by the installed `vera_core.QualifiedVeraRuntime` remain separate integration/qualification work. This repository does not claim that Vera is already watching ARK.

## What Vision is meant to become

```text
ARK on PS4 ──► Chiaki / PlayStation Remote Play ──┐
Desktop application / camera / media file ───────┤
RTSP / RTMP / SRT / HLS / network video ─────────┘
                         │
                         ▼
               VISION — input and timing
         video decoding • frame quality • bounded queues
                         │
                         ▼
               VISION — visual evidence
        RGB frame sequence • changes • regions • timestamps
                         │
                         ▼
                VERA — interpretation
      scene recognition • tracking • context • memory • goals
                         │
                         ▼
                VERA — reasoning
        decisions conditioned on provenance & uncertainty
                         │
                 [separate effect gate]
                         ▼
               Optional authorized action
```

Vision's long-term scope is **continuous visual attention**, not a screenshot utility: follow the same subject across frames, distinguish camera movement from object movement, reason with temporally ordered scenes, and expose what is seen and what is uncertain to Vera's cognition. A decoder is just the first part of that work.

## Working source implementation

- **Live ingestion:** `FFmpegFrames` launches FFmpeg **without a shell**, decodes fixed-size RGB24 frames from one explicitly selected source, and provides a bounded latest-frame queue so slow downstream inference doesn't silently accumulate minutes of stale images.
- **Inputs:** Windows `gdigrab` exact window title, explicit file, or opt-in RTSP/RTMP/RTMPS/SRT/HTTP(S) media URL. Video streams are not opened automatically. Window capture is desktop capture and may include overlays.
- **Temporal analysis:** `TemporalAnalyzer` samples a 16×16 RGB grid, detects color-sensitive changes, reports 2×2 changed regions, monotonic elapsed time, frame gaps, dimension/clock/step resets, and source provenance. Raw full frames are **not retained** in its history.
- **Perception stream:** `VisionPipeline` provides typed `VisionEvent` records and an **in-process decoded-frame callback** for a separate inference/consumer integration. The CLI emits newline-delimited JSON **without pixels**, source URLs, secrets, or controller operations.
- **Controlled image input:** strict RGBA/filter-zero PNG decoder for known local encoders, with CRC and decompression-size limits. It is intentionally not a general PNG decoder.
- **Explicit limits:** defaults of 320×180 at 2 FPS, bounded to 640×360 and 15 FPS; sessions default to at most 100 observations or 60 seconds (configurable within fixed ceilings). This is bounded continuous sampling, **not yet real-time guaranteed latency**.

The default pipeline performs no model inference or cloud call. Optional explicitly selected local-model inference uses Ollama loopback transport, and the operator must independently verify that the selected model does not forward images elsewhere.

## Try it

Python 3.11+ and FFmpeg on PATH are required for live input. Install the library and tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

**Observe the Chiaki game window on Windows, without issuing any controller input:**

```powershell
vera-vision watch --kind window --source "Chiaki" --fps 2 --frames 120 --seconds 90
```

**Observe a network video published through a local MediaMTX instance:**

```powershell
vera-vision watch --kind stream --source "rtsp://127.0.0.1:8554/ark" --fps 3 --frames 300 --seconds 120
```

**Read a recording while developing offline:**

```bash
vera-vision watch --kind file --source demo.mp4 --fps 2 --frames 80 --seconds 60
```

Output is machine-readable JSONL with `observer="VERA"`, source kind, frame sequence, dropped-frame count, typed temporal evidence and `action_authority=false`. The CLI prints a session-end record when it completes. It does not save or publish images. If FFmpeg is unavailable, a file/window is missing, or a stream cannot be decoded, it fails without revealing FFmpeg stderr or private URLs.

### Optional local scene interpretation

An explicit `--local-vision-model` choice lets Vera's Vision pipeline periodically send selected frames to a **loopback Ollama image-capable model** and receive tightly validated scene categories. For example:

```powershell
vera-vision watch --kind window --source "Chiaki" --fps 2 --seconds 90 --frames 120 --local-vision-model "YOUR_IMAGE_CAPABLE_MODEL" --local-vision-interval 5
```

The default pipeline makes **no model requests**. The local adapter never enables controller effects. A model's descriptions remain **untrusted inference**, not verified reality. It uses no automatic model downloads, no remote inference URL, and no default cloud service. [Local observer architecture and safeguards](docs/LOCAL_VISION_OBSERVER.md) explain exactly what is implemented and what is not qualified.

### Integrating Vera as an actual visual consumer

```python
from vera_vision import FFmpegFrames, VideoSource, VisionPipeline

source = VideoSource("file", "demo.mp4", width=320, height=180, fps=2)

def vera_frame_consumer(packet, observation):
    # This is the explicit integration seam for Vera's visual reasoning.
    # packet.frame is RGBFrame; observation is bounded pixel evidence.
    # An installed QualifiedVeraRuntime does NOT use this callback automatically.
    pass

VisionPipeline().run(
    FFmpegFrames(source).frames(max_frames=100, max_seconds=60),
    session_id="vera-demo",
    source_kind="file",
    on_frame=vera_frame_consumer,
)
```

**Current integration boundary:** that consumer callback and JSONL interface are prepared; an authenticated, qualified live Vera runtime adapter has **not** been installed or selected. Source availability is not runtime consumption. A future adapter must prove frame provenance, freshness, privacy, selected runtime/process identity, and controlled persistence before describing Vera as presently observing.

## Relationship to the supplied repositories

| Project | What it contributes | Our decision |
|---|---|---|
| [MediaMTX](https://github.com/bluenviron/mediamtx) | RTSP, WebRTC, RTMP, SRT, HLS media routing, recording and playback | Optional **transport**. Existing URL-based FFmpeg ingest works with MediaMTX output. MediaMTX is not installed or required. |
| [LiveKit](https://github.com/livekit/livekit) | WebRTC SFU, live audio/video/data distribution | Future interactive multi-participant or remote-vision source. Requires a **track subscriber** (e.g. via a LiveKit client/Agents SDK); the LiveKit server by itself does not do Vera's perception. Not implemented yet. |
| [LiveVideoCoreSDK](https://github.com/runner365/LiveVideoCoreSDK) | iOS live publishing, encoding and low-latency media concepts | Design reference for mobile publishing. Not an appropriate Windows/Chiaki capture dependency. |
| [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) | Real-time camera frame processing and face manipulation | Research reference for a frame-processing loop, **not** a perceptual reasoning engine; no face-swapping dependency or functionality is included. |

## Development priorities

1. **Reliable capture:** prove current Chiaki frame acquisition, restart/reconnect, freshness and dropped-frame reporting, using explicitly authorized local tests.
2. **Semantic perception:** implement optional isolated detectors/vision-language model adapters with uncertainty and sampling limits; never turn model text into unreviewed commands.
3. **Temporal grounding:** optical flow, camera-motion compensation, persistent object identities and occlusion-aware tracking; evaluate against labeled videos before claiming trajectory accuracy.
4. **Vera runtime consumption:** introduce a bounded, permissioned adapter to `vera_core.QualifiedVeraRuntime` with readback receipts showing which process consumed which exact frames.
5. **Optional distribution:** add MediaMTX and LiveKit subscription/publishing adapters only if remote/multiple simultaneous sources are needed; account for privacy, codecs and latency.
6. **Reasoning and agency:** give Vera temporally grounded observations; maintain the separate control/effect authority boundary for any interaction with ARK or other applications.

## Ownership and safeguards

`thebrazenbeard/vision` owns reusable vision ingestion, evidence and inference adapters. `thebrazenbeard/vera-ark` owns the game's capture-specific configuration, gameplay planning, Physica/collision qualification, DS4 controller transport and neutral release. `thebrazenbeard/vera-mono` owns Vera's canonical self-contained runtime. These are ownership boundaries, not proof that the processes are currently connected.

**Pixel differences are not object recognition. Recognition is not verified motion. Motion is not proof of collision clearance. Seeing something is not authorization to act.**

Vision has no controller output, microphone capture, automatic uploads, saved screenshots, credential storage, or default open listener. Never place private frame captures in this public repository.

The source provenance and initial extraction decision are tracked at [Issue #1](https://github.com/thebrazenbeard/vision/issues/1); the earlier ARK perception candidate remains [vera-ark Draft PR #10](https://github.com/thebrazenbeard/vera-ark/pull/10), pending integration qualification.
