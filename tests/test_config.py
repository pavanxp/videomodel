"""Config loading, validation and path anchoring."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visual_diversity.config import FramesConfig, PipelineConfig, load_config


def test_defaults_are_valid():
    cfg = PipelineConfig()
    assert cfg.scoring.max_points == 15.0
    assert cfg.search.index_type == "flat"
    assert 0.0 < cfg.clustering.similarity_threshold <= 1.0


def test_shipped_config_parses():
    path = Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"
    cfg = load_config(path)
    assert cfg.embeddings.backend in {"dinov2", "clip", "stub"}
    assert cfg.output_dir.is_absolute()


def test_shipped_config_is_the_single_frame_clip_setup():
    """The delivered configuration: one frame at 50%, CLIP driving, no pooling."""
    path = Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"
    cfg = load_config(path)

    assert cfg.frames.count == 1
    # No bucket may override the count back up.
    assert cfg.frames.duration_buckets == ()
    assert cfg.frames.frames_for(2.0) == 1
    assert cfg.frames.frames_for(600.0) == 1

    assert cfg.embeddings.driver == "secondary"
    assert cfg.embeddings.secondary_clip.enabled is True
    assert cfg.pooling.mode == "none"


def test_secondary_driver_requires_the_secondary_to_be_enabled():
    with pytest.raises(ValidationError, match="secondary_clip.enabled"):
        PipelineConfig(embeddings={"driver": "secondary",
                                   "secondary_clip": {"enabled": False}})


def test_primary_driver_is_the_default_and_needs_no_secondary():
    cfg = PipelineConfig()
    assert cfg.embeddings.driver == "primary"
    assert cfg.embeddings.secondary_clip.enabled is False


def test_a_bucket_may_not_silently_override_single_frame_mode():
    """Buckets take precedence over count, so this would quietly extract 7."""
    with pytest.raises(ValidationError, match="single-frame mode"):
        FramesConfig(count=1, duration_buckets=[{"max_seconds": math.inf, "count": 7}])


def test_all_one_buckets_are_compatible_with_single_frame():
    cfg = FramesConfig(count=1, duration_buckets=[{"max_seconds": math.inf, "count": 1}])
    assert cfg.single_frame is True
    assert cfg.frames_for(999.0) == 1


def test_single_frame_property_is_false_for_multi_frame():
    assert FramesConfig(count=5).single_frame is False
    assert FramesConfig(count=1).single_frame is True


def test_unknown_key_is_rejected(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"scoring": {"max_pointz": 10}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_relative_paths_anchor_to_the_config_file(tmp_path: Path):
    sub = tmp_path / "conf"
    sub.mkdir()
    path = sub / "c.yaml"
    path.write_text(yaml.safe_dump({"output_dir": "./out"}), encoding="utf-8")

    cfg = load_config(path)
    assert cfg.output_dir == (sub / "out").resolve()


def test_gray_zone_must_sit_below_the_duplicate_threshold():
    with pytest.raises(ValidationError, match="gray_zone"):
        PipelineConfig(
            clustering={"similarity_threshold": 0.85},
            borderline_review={"enabled": True, "gray_zone": (0.80, 0.95)},
        )
    # Same overlap is tolerated while the stage is off -- it changes nothing.
    PipelineConfig(
        clustering={"similarity_threshold": 0.85},
        borderline_review={"enabled": False, "gray_zone": (0.80, 0.95)},
    )


def test_gray_zone_bounds_must_be_ordered():
    with pytest.raises(ValidationError):
        PipelineConfig(borderline_review={"gray_zone": (0.9, 0.8)})


def test_duration_buckets_must_ascend():
    with pytest.raises(ValidationError, match="ascending"):
        FramesConfig(duration_buckets=[{"max_seconds": 60, "count": 5},
                                       {"max_seconds": 5, "count": 3}])


def test_frames_for_selects_by_bucket():
    cfg = FramesConfig(count=5, duration_buckets=[
        {"max_seconds": 5.0, "count": 3},
        {"max_seconds": 60.0, "count": 5},
        {"max_seconds": math.inf, "count": 7},
    ])
    assert cfg.frames_for(2.0) == 3
    assert cfg.frames_for(30.0) == 5
    assert cfg.frames_for(600.0) == 7
    # An unknown duration must not be guessed into a bucket.
    assert cfg.frames_for(None) == 5


def test_inf_string_in_yaml_is_coerced(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(
        {"frames": {"duration_buckets": [{"max_seconds": "inf", "count": 7}]}}),
        encoding="utf-8")
    cfg = load_config(path)
    assert math.isinf(cfg.frames.duration_buckets[0].max_seconds)


def test_config_is_immutable():
    cfg = PipelineConfig()
    with pytest.raises(ValidationError):
        cfg.log_level = "DEBUG"  # type: ignore[misc]
