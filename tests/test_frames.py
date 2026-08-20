"""Stage 3: ffmpeg frame extraction, caching and per-clip failure isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_diversity.config import FramesConfig
from visual_diversity.frames import (_sample_offsets, extract_all, extract_one,
                                     ffmpeg_available, probe_duration)
from visual_diversity.ingest import ClipRecord

from conftest import make_clip, manifest_row, requires_ffmpeg


def _record(item_id: str, path: Path, duration: float | None = None) -> ClipRecord:
    return ClipRecord(**manifest_row(item_id, path),
                      **({"duration_seconds": duration} if duration else {}))


def test_sample_offsets_span_the_clip():
    """The spec's nominal positions: 0%, 25%, 50%, 75%, 100% of duration."""
    offsets = _sample_offsets(10.0, 5)
    assert offsets == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_last_offset_is_resolved_by_seeking_from_the_end():
    """Seeking to exactly `duration` decodes nothing, so the closing sample
    uses -sseof and lets ffmpeg find the real last frame."""
    from visual_diversity.frames import _seek_attempts

    last = _seek_attempts(Path("/tmp/c.mp4"), 1.5, 1.5, is_last=True)
    assert last[0][0] == "-sseof"

    middle = _seek_attempts(Path("/tmp/c.mp4"), 0.75, 1.5, is_last=False)
    assert middle[0][:2] == ["-ss", "0.750"]


def test_sample_offsets_single_frame_takes_the_middle():
    assert _sample_offsets(10.0, 1) == [5.0]


@requires_ffmpeg
def test_probe_duration_reads_a_real_clip(tmp_path: Path):
    clip = make_clip(tmp_path / "c.mp4", seconds=2.0)
    duration = probe_duration(clip)
    assert duration is not None
    assert 1.5 < duration < 2.6


@requires_ffmpeg
def test_probe_duration_returns_none_for_garbage(tmp_path: Path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"definitely not a video")
    assert probe_duration(junk) is None


@requires_ffmpeg
def test_extract_one_writes_the_requested_frames(tmp_path: Path):
    clip = make_clip(tmp_path / "c.mp4", seconds=2.0)
    cfg = FramesConfig(count=4, workers=1, resize=(32, 32))

    result = extract_one(_record("c", clip), tmp_path / "cache", cfg)

    assert result.ok, result.error
    assert len(result.paths) == 4
    assert all(p.is_file() and p.stat().st_size > 0 for p in result.paths)


@requires_ffmpeg
def test_extracted_frames_are_resized(tmp_path: Path):
    from PIL import Image

    clip = make_clip(tmp_path / "c.mp4", seconds=1.0, size="128x96")
    cfg = FramesConfig(count=2, workers=1, resize=(48, 24))

    result = extract_one(_record("c", clip), tmp_path / "cache", cfg)

    with Image.open(result.paths[0]) as img:
        assert img.size == (48, 24)


@requires_ffmpeg
def test_second_run_reuses_the_cache(tmp_path: Path):
    clip = make_clip(tmp_path / "c.mp4", seconds=1.0)
    cfg = FramesConfig(count=3, workers=1, resize=(32, 32))
    cache = tmp_path / "cache"

    first = extract_one(_record("c", clip), cache, cfg)
    assert not first.reused_cache

    second = extract_one(_record("c", clip), cache, cfg)
    assert second.reused_cache
    assert second.paths == first.paths


@requires_ffmpeg
def test_force_bypasses_the_cache(tmp_path: Path):
    clip = make_clip(tmp_path / "c.mp4", seconds=1.0)
    cfg = FramesConfig(count=2, workers=1, resize=(32, 32))
    cache = tmp_path / "cache"

    extract_one(_record("c", clip), cache, cfg)
    forced = extract_one(_record("c", clip), cache, cfg, force=True)
    assert not forced.reused_cache


@requires_ffmpeg
def test_changing_frame_count_invalidates_the_cache(tmp_path: Path):
    clip = make_clip(tmp_path / "c.mp4", seconds=1.0)
    cache = tmp_path / "cache"

    three = extract_one(_record("c", clip), cache, FramesConfig(count=3, workers=1))
    assert len(three.paths) == 3

    five = extract_one(_record("c", clip), cache, FramesConfig(count=5, workers=1))
    assert len(five.paths) == 5
    assert not five.reused_cache


def test_missing_file_becomes_an_error_not_an_exception(tmp_path: Path):
    cfg = FramesConfig(count=2, workers=1)
    result = extract_one(_record("gone", tmp_path / "nope.mp4"), tmp_path / "cache", cfg)
    assert not result.ok
    assert "not found" in result.error


@requires_ffmpeg
def test_unreadable_clip_becomes_an_error(tmp_path: Path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    cfg = FramesConfig(count=2, workers=1)

    result = extract_one(_record("junk", junk), tmp_path / "cache", cfg)

    assert not result.ok
    assert "duration" in result.error


@requires_ffmpeg
def test_extract_all_isolates_one_bad_clip(tmp_path: Path):
    good = make_clip(tmp_path / "good.mp4", seconds=1.0)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"junk")
    cfg = FramesConfig(count=2, workers=2, resize=(32, 32))

    results = extract_all([_record("good", good), _record("bad", bad)],
                          tmp_path / "cache", cfg)

    assert results["good"].ok
    assert not results["bad"].ok
    assert len(results) == 2


def test_extract_all_on_empty_input_is_a_noop(tmp_path: Path):
    assert extract_all([], tmp_path / "cache", FramesConfig()) == {}


@requires_ffmpeg
def test_duration_bucket_controls_frame_count(tmp_path: Path):
    clip = make_clip(tmp_path / "short.mp4", seconds=2.0)
    cfg = FramesConfig(count=5, workers=1, resize=(32, 32),
                       duration_buckets=[{"max_seconds": 5.0, "count": 3}])

    result = extract_one(_record("short", clip), tmp_path / "cache", cfg)

    assert len(result.paths) == 3
