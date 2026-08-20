"""Stage 3: sample frames from each clip with ffmpeg.

Frames are cached on disk under ``<cache>/frames/<item_id>/frame_<i>.jpg`` and a
clip whose full set is already present is skipped, so a re-run costs nothing.

Extraction runs in a process pool. One unreadable clip is logged and skipped --
a delivery of thousands must not die on a single truncated file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import FramesConfig
from .ingest import ClipRecord

log = logging.getLogger(__name__)

__all__ = ["ClipFrames", "extract_all", "extract_one", "probe_duration", "ffmpeg_available"]


@dataclass(frozen=True, slots=True)
class ClipFrames:
    """The frames sampled for one clip, in temporal order."""

    item_id: str
    paths: tuple[Path, ...]
    reused_cache: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.paths) > 0


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(clip_path: Path, timeout: float = 60.0) -> float | None:
    """Clip length in seconds via ffprobe, or None if it cannot be read."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(clip_path)],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        value = json.loads(proc.stdout)["format"]["duration"]
        seconds = float(value)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return seconds if seconds > 0 else None


def _sample_offsets(duration: float, count: int) -> list[float]:
    """Evenly spaced timestamps across the clip, inclusive of both ends.

    These are the nominal positions the spec asks for -- 0%, 25%, 50%, 75%,
    100% at count=5. The 100% position is past the last decodable frame; that
    is resolved at grab time by :func:`_seek_attempts`, not by fudging the
    arithmetic here.
    """
    if count == 1:
        return [duration / 2.0]
    return [(i / (count - 1)) * duration for i in range(count)]


def _seek_attempts(clip_path: Path, offset: float, duration: float,
                   is_last: bool) -> list[list[str]]:
    """Ordered ffmpeg input arguments to try for one frame.

    A container's nominal duration runs past its final frame -- a 1.5 s clip at
    10 fps ends on the frame at 1.4 s -- so seeking to exactly ``duration``
    decodes nothing. For the closing sample we seek from the end with
    ``-sseof`` and let ffmpeg resolve where the last frame actually is, walking
    the margin outwards until one lands.

    Every other position gets a small backoff ladder too, which covers codecs
    whose keyframe spacing leaves a requested timestamp undecodable.
    """
    clip = str(clip_path)
    attempts: list[list[str]] = []

    if is_last:
        for margin in (0.1, 0.25, 0.5, 1.0):
            m = min(margin, max(duration * 0.5, 0.01))
            attempts.append(["-sseof", f"-{m:.3f}", "-i", clip])

    if not is_last or duration <= 0:
        attempts.append(["-ss", f"{offset:.3f}", "-i", clip])

    for factor in (0.95, 0.85, 0.5):
        earlier = offset * factor
        if earlier > 0.01:
            attempts.append(["-ss", f"{earlier:.3f}", "-i", clip])

    # Frame zero always exists; a clip with no decodable frame at all fails the
    # whole grab rather than silently substituting one.
    attempts.append(["-ss", "0", "-i", clip])
    return attempts


def _frame_dir(cache_dir: Path, item_id: str) -> Path:
    # item_id is manifest-controlled; keep it from escaping the cache root.
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in item_id)
    return cache_dir / "frames" / (safe or "_unnamed")


def _cached_paths(dest: Path, count: int) -> list[Path] | None:
    paths = [dest / f"frame_{i:02d}.jpg" for i in range(count)]
    return paths if all(p.is_file() and p.stat().st_size > 0 for p in paths) else None


def extract_one(clip: ClipRecord, cache_dir: Path, cfg: FramesConfig,
                force: bool = False) -> ClipFrames:
    """Extract (or reuse) the frames for a single clip.

    Runs in a worker process, so it takes only picklable arguments and never
    raises: every failure comes back as a populated ``error``.
    """
    dest = _frame_dir(cache_dir, clip.item_id)

    duration = clip.duration_seconds or probe_duration(clip.clip_path, cfg.timeout_seconds)
    count = cfg.frames_for(duration)

    if not force:
        cached = _cached_paths(dest, count)
        if cached is not None:
            return ClipFrames(clip.item_id, tuple(cached), reused_cache=True)

    if not clip.clip_path.is_file():
        return ClipFrames(clip.item_id, (), error=f"clip not found: {clip.clip_path}")
    if duration is None:
        return ClipFrames(clip.item_id, (), error="could not read clip duration (unplayable?)")

    dest.mkdir(parents=True, exist_ok=True)
    width, height = cfg.resize
    written: list[Path] = []
    offsets = _sample_offsets(duration, count)

    for i, offset in enumerate(offsets):
        out = dest / f"frame_{i:02d}.jpg"
        is_last = (i == len(offsets) - 1) and count > 1
        last_stderr = "no stderr"
        grabbed = False

        for input_args in _seek_attempts(clip.clip_path, offset, duration, is_last):
            cmd = [
                "ffmpeg", "-nostdin", "-y", *input_args,
                "-frames:v", "1",
                # Force exact dimensions; the model wants a fixed input size and
                # aspect distortion is irrelevant to redundancy detection.
                "-vf", f"scale={width}:{height}",
                "-q:v", str(cfg.jpeg_quality),
                str(out),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=cfg.timeout_seconds,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired:
                return ClipFrames(clip.item_id, (),
                                  error=f"ffmpeg timed out at offset {offset:.2f}s")
            except OSError as exc:
                return ClipFrames(clip.item_id, (), error=f"ffmpeg could not run: {exc}")

            if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
                grabbed = True
                break
            last_stderr = ((proc.stderr or "").strip().splitlines() or ["no stderr"])[-1]

        if not grabbed:
            return ClipFrames(clip.item_id, (),
                              error=f"no decodable frame near offset {offset:.2f}s "
                                    f"of {duration:.2f}s: {last_stderr}")
        written.append(out)

    return ClipFrames(clip.item_id, tuple(written))


def extract_all(clips: Sequence[ClipRecord], cache_dir: Path, cfg: FramesConfig,
                force: bool = False) -> dict[str, ClipFrames]:
    """Extract frames for every clip, in parallel. Returns {item_id: ClipFrames}.

    Failures are included in the mapping with ``ok`` False so the caller can
    report exactly which clips dropped out and why.
    """
    if not clips:
        return {}
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH -- frame extraction needs both. "
            "Install ffmpeg, or point the pipeline at pre-extracted frames."
        )

    cache_dir = Path(cache_dir)
    (cache_dir / "frames").mkdir(parents=True, exist_ok=True)
    workers = max(1, min(cfg.workers, len(clips), (os.cpu_count() or 2)))
    results: dict[str, ClipFrames] = {}

    log.info("frames: extracting from %d clip(s) with %d worker(s)%s",
             len(clips), workers, " [force]" if force else "")

    if workers == 1:
        for clip in clips:
            results[clip.item_id] = extract_one(clip, cache_dir, cfg, force)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_one, c, cache_dir, cfg, force): c for c in clips}
            for fut in as_completed(futures):
                clip = futures[fut]
                try:
                    results[clip.item_id] = fut.result()
                except Exception as exc:  # noqa: BLE001 -- a worker crash is one clip's problem
                    results[clip.item_id] = ClipFrames(clip.item_id, (), error=f"worker crashed: {exc}")

    ok = sum(1 for r in results.values() if r.ok)
    reused = sum(1 for r in results.values() if r.reused_cache)
    failed = [r for r in results.values() if not r.ok]
    log.info("frames: %d/%d clip(s) usable (%d reused from cache), %d failed",
             ok, len(clips), reused, len(failed))
    for r in failed[:10]:
        log.warning("frames: %s skipped -- %s", r.item_id, r.error)
    if len(failed) > 10:
        log.warning("frames: ... %d more failure(s)", len(failed) - 10)

    return results
