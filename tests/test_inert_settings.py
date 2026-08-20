"""Settings a run cannot act on must be announced, not silently ignored.

In single-frame CLIP mode most of the multi-frame apparatus is inert. A knob
that is quietly ignored is indistinguishable from a knob that works, so the
pipeline states which ones are not in play before it starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_diversity.config import PipelineConfig
from visual_diversity.pipeline import log_inert_settings


def single_frame_clip_config(tmp_path: Path, **overrides) -> PipelineConfig:
    body = {
        "output_dir": tmp_path / "out",
        "cache_dir": tmp_path / "cache",
        "frames": {"count": 1, "duration_buckets": []},
        "pooling": {"mode": "none"},
        "embeddings": {"driver": "secondary", "secondary_clip": {"enabled": True}},
    }
    body.update(overrides)
    return PipelineConfig(**body)


def joined(cfg: PipelineConfig) -> str:
    return " ".join(log_inert_settings(cfg))


def test_single_frame_mode_is_announced(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path))
    assert "frames.count=1" in text
    assert "50% of duration" in text


def test_maxpair_is_reported_inert_at_one_frame(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path))
    assert "maxpair" in text
    assert "identical to the cosine similarity" in text


def test_pooling_bypass_is_explained_as_not_skippable(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path))
    assert "pooling.mode='none'" in text
    assert "not skippable" in text


def test_mean_is_reported_equivalent_at_one_frame(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path, pooling={"mode": "mean"}))
    assert "equivalent to 'none'" in text


def test_primary_backend_is_reported_inert_under_the_secondary_driver(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path))
    assert "driver='secondary'" in text
    assert "no primary model is loaded" in text
    assert "dinov2" in text  # the ignored backend is named, not hidden


def test_empty_duration_buckets_are_announced(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path))
    assert "duration_buckets is empty" in text


def test_disabled_borderline_review_is_announced(tmp_path: Path):
    text = joined(single_frame_clip_config(tmp_path))
    assert "stage 12 does not run" in text


def test_borderline_frames_per_clip_is_reported_capped(tmp_path: Path):
    cfg = single_frame_clip_config(tmp_path, borderline_review={
        "enabled": True, "gray_zone": (0.8, 0.91), "frames_per_clip": 4})
    text = joined(cfg)
    assert "frames_per_clip=4 is capped to 1" in text


def test_multi_frame_run_does_not_claim_single_frame_inertness(tmp_path: Path):
    cfg = PipelineConfig(
        output_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
        frames={"count": 5}, pooling={"mode": "mean"},
        embeddings={"backend": "stub"},
    )
    text = joined(cfg)
    assert "frames.count=1" not in text
    assert "maxpair" not in text
    # The primary is in play, so it must not be reported inert.
    assert "no primary model is loaded" not in text


def test_notes_are_logged_at_info(tmp_path: Path, caplog):
    with caplog.at_level("INFO"):
        log_inert_settings(single_frame_clip_config(tmp_path))
    assert "inert:" in caplog.text
