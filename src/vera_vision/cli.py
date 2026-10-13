"""Command-line live video reader: explicit, bounded, no controller actions."""
from __future__ import annotations

import argparse
import json
import sys
import uuid

from .live import FFmpegFrames, VideoSource, VideoSourceError
from .pipeline import VisionPipeline


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(
        prog="vera-vision", description="Read video FROM a source and produce observations FOR Vera"
    )
    modes = app.add_subparsers(dest="command", required=True)
    watch = modes.add_parser("watch", help="start a bounded live-video observation session")
    watch.add_argument("--kind", choices=("window", "stream", "file"), required=True)
    watch.add_argument("--source", required=True, help="exact window title, approved URL, or media path")
    watch.add_argument("--fps", type=int, default=2)
    watch.add_argument("--width", type=int, default=320)
    watch.add_argument("--height", type=int, default=180)
    watch.add_argument("--frames", type=int, default=100, help="max decoded observations")
    watch.add_argument("--seconds", type=float, default=60, help="max elapsed seconds")
    watch.add_argument("--idle-seconds", type=float, default=10, help="maximum time without a decoded frame")
    watch.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path")
    return app


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        source = VideoSource(args.kind, args.source, args.width, args.height, args.fps)
        # A session belongs to Vera's observer; "ARK" is one possible *source*.
        video = FFmpegFrames(source, binary=args.ffmpeg)
        observed = VisionPipeline().run(
            video.frames(max_frames=args.frames, max_seconds=args.seconds,
                         max_idle_seconds=args.idle_seconds),
            session_id=str(uuid.uuid4()), source_kind=source.kind, output=sys.stdout,
        )
    except (VideoSourceError, ValueError) as exc:
        # Provider URLs, inline credentials, and FFmpeg stderr never enter errors.
        message = str(exc) if isinstance(exc, VideoSourceError) else "INVALID_SOURCE_OR_BOUNDS"
        print(json.dumps({"status": "FAILED", "reason": message}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('{"status":"STOPPED_BY_USER"}', file=sys.stderr)
        return 130
    print(json.dumps({
        "schema": "VERA_VISION_SESSION_END_V1",
        "status": "COMPLETE", "observed_frames": observed,
        "controller_effects": 0, "observer": "VERA",
    }), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
