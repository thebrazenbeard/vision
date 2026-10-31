# Vision

**Bounded video ingestion and evidence-labeled frame observations for Python applications.**

Vision (`vera-vision`) reads an explicitly selected video source, samples RGB frames through FFmpeg, and emits chronological JSONL observations about pixel changes. An application can also receive decoded frames through an in-process callback. The default path does not call a model or issue control actions.

The package is useful for building a visual-observation pipeline, validating capture behavior, or prototyping a downstream perception adapter. It is **not** a general object tracker, a motion estimator, a scene-understanding model, or an authorization service.

## What is implemented

| Component | Current behavior |
| --- | --- |
| `VideoSource` / `FFmpegFrames` | Explicit file, supported media URL, or exact Windows window title; FFmpeg RGB24 decoding; bounded latest-frame queue that discards stale frames when a consumer falls behind. |
| `TemporalAnalyzer` | Samples a 16×16 RGB grid and reports change fraction, four coarse changed regions, elapsed time, discontinuities, and sampled-pixel provenance. It does **not** identify objects or infer direction of travel. |
| `VisionPipeline` | Produces typed `VisionEvent` records and optional in-process `on_frame` callbacks with decoded pixels. JSONL events omit raw image pixels. |
| `decode_rgba_png` | Strict PNG adapter for supported RGBA8/filter-zero encoders, with chunk CRC and size checks. It is not a general-purpose PNG reader. |
| Optional `LocalOllamaObserver` | Sends rate-limited selected frames to an explicitly named image-capable model at a numeric loopback Ollama endpoint. Returned categories and descriptions are validated but remain **untrusted model inferences**. |

The capture path launches FFmpeg with an argument list (`shell=False`), suppresses its stderr in public errors, and does not automatically open streams. No controller, microphone, persistent image store, credential store, or public listening socket is implemented here. An explicit local-model option does transmit image data to the selected loopback Ollama service; inspect that model's behavior and data policy independently.

## Requirements and installation

- Python **3.11 or newer**.
- FFmpeg available on `PATH` for CLI or live-video inputs. Pixel-only Python analysis does not require FFmpeg.
- Optional: an image-capable local Ollama model when using `--local-vision-model`.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The package declares no mandatory third-party Python runtime dependencies. The test extra installs `pytest`.

## Command-line examples

Observe a local video file for at most 80 frames or 60 seconds:

```bash
vera-vision watch --kind file --source sample.mp4 --fps 2 --frames 80 --seconds 60
```

Observe a deliberately selected RTSP endpoint:

```bash
vera-vision watch --kind stream --source rtsp://127.0.0.1:8554/camera1 --fps 3 --frames 100 --seconds 60
```

On Windows, an exact visible window title can be selected using FFmpeg's `gdigrab` input:

```powershell
vera-vision watch --kind window --source "Example Application" --fps 2 --frames 100 --seconds 60
```

Window capture may include visual overlays and is only available on Windows. Stream URL schemes are limited to RTSP, RTMP, RTMPS, SRT, HTTP, and HTTPS; inline URL credentials are rejected. No source is captured by default.

Each observed frame produces a JSONL `FRAME_OBSERVED` event containing its session, source kind, sequence, dropped-frame count, decode-host monotonic timestamp, and temporal evidence. The CLI also prints a final session record. The schema currently uses the fixed `observer: "VERA"` field for project compatibility; that label is **not** proof of authenticated process identity or any other consumer's participation.

Default decoding is 320×180 at 2 FPS. Input limits cap output size at 640×360 and sampling at 15 FPS. Session bounds are finite (by default 100 observations or 60 seconds), and frame backpressure favors freshness over completeness. These limits do not establish a real-time latency guarantee.

### Optional local visual descriptions

With a user-selected image-capable Ollama model already available locally:

```bash
vera-vision watch --kind file --source sample.mp4 --local-vision-model YOUR_IMAGE_MODEL --local-vision-interval 5 --frames 80 --seconds 60
```

This mode emits additional `SEMANTIC_OBSERVATION` records. The model's confidence is self-reported and uncalibrated. The default mode makes **no model request**. No cloud endpoint, automatic model download, or control effect is configured by this package. See [local interpretation safeguards](docs/LOCAL_VISION_OBSERVER.md).

## Python integration

You can analyze predecoded RGB pixels without running FFmpeg:

```python
from vera_vision import RGBFrame, TemporalAnalyzer

analyzer = TemporalAnalyzer()
frame = RGBFrame(width=2, height=2, rgb=bytes(2 * 2 * 3))
observation = analyzer.observe(
    frame, session_id="sample-session", step=1, monotonic_seconds=1.0
)
print(observation.to_dict()["status"])  # FIRST_FRAME
```

Or connect an explicit video decoder to a frame-processing callback:

```python
from vera_vision import VideoSource, FFmpegFrames, VisionPipeline

source = VideoSource("file", "sample.mp4", width=320, height=180, fps=2)

def consume_frame(packet, temporal_observation):
    # packet.frame contains RGB pixels; the observation contains sampled evidence.
    # The callback alone does not register an external runtime or grant authority.
    pass

VisionPipeline().run(
    FFmpegFrames(source).frames(max_frames=100, max_seconds=60),
    session_id="sample-session", source_kind="file", on_frame=consume_frame,
)
```

The package does not automatically bind that callback to any installed AI-model runtime. A real integration must independently establish provider identity, host-owned permission boundaries, frame provenance/freshness, privacy handling, and readback evidence.

## Evidence boundaries and development

- Sampled color differences are **not** object recognition, optical flow, or collision avoidance.
- Timestamping describes host-side frame decoding; it does **not** authenticate source capture time.
- Optional model summaries are untrusted inferences, never effect permissions.
- Continuous media capture is best effort; reconnect/restart behavior and end-to-end latency need deployment-specific validation.
- Consumer identity, persistence, consent, and any physical or software actions belong to an independently governed host integration.

For source-level design and limitations, see [ingestion reliability](docs/LIVE_INGESTION_RELIABILITY.md) and [local observer safeguards](docs/LOCAL_VISION_OBSERVER.md). The repository's [test workflow](.github/workflows/tests.yml) runs on Windows and Linux with Python 3.11 and 3.12; [workspace-startup checks](.github/workflows/workspace-startup-safety.yml) cover a known editor-autostart class, not all possible host compromise.

## Licensing

This public source tree does not currently contain a root `LICENSE` file. Public visibility alone does not grant reuse or redistribution rights. Confirm permissions with the repository owner before redistributing or embedding the code.
