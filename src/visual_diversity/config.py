"""Typed configuration for the whole pipeline.

One pydantic model per stage, assembled into :class:`PipelineConfig`. Every
tunable the stages read lives here, so changing a threshold is a YAML edit and
never a code edit. Unknown keys are rejected rather than ignored -- a silently
misspelled threshold is worse than a loud failure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "PipelineConfig",
    "FramesConfig",
    "EmbeddingsConfig",
    "PoolingConfig",
    "SearchConfig",
    "ClusteringConfig",
    "ScoringConfig",
    "BorderlineReviewConfig",
    "ReportConfig",
    "load_config",
]

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DurationBucket(_Base):
    """Frame count for clips no longer than ``max_seconds``."""

    max_seconds: float = Field(gt=0.0, description="inclusive upper bound, may be inf")
    count: int = Field(ge=1, le=64)


class FramesConfig(_Base):
    count: int = Field(default=5, ge=1, le=64)
    duration_buckets: tuple[DurationBucket, ...] = ()
    resize: tuple[int, int] = (224, 224)
    jpeg_quality: int = Field(default=3, ge=2, le=31)
    workers: int = Field(default=8, ge=1, le=128)
    timeout_seconds: float = Field(default=120.0, gt=0.0)

    @field_validator("resize")
    @classmethod
    def _positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("resize dimensions must be positive")
        return v

    @field_validator("duration_buckets")
    @classmethod
    def _ordered(cls, v: tuple[DurationBucket, ...]) -> tuple[DurationBucket, ...]:
        bounds = [b.max_seconds for b in v]
        if bounds != sorted(bounds):
            raise ValueError("duration_buckets must be ordered by ascending max_seconds")
        return v

    @model_validator(mode="after")
    def _buckets_do_not_contradict_single_frame(self) -> "FramesConfig":
        """A bucket silently overrides ``count``, so at count=1 it would take a
        single-frame run back to multi-frame without anything saying so."""
        if self.count == 1:
            offenders = [b for b in self.duration_buckets if b.count != 1]
            if offenders:
                raise ValueError(
                    f"frames.count=1 (single-frame mode) but {len(offenders)} "
                    f"duration_bucket(s) override it to "
                    f"{sorted({b.count for b in offenders})}. Buckets take precedence "
                    f"over count, so this would silently extract multiple frames. "
                    f"Empty duration_buckets, or set every bucket count to 1."
                )
        return self

    @property
    def single_frame(self) -> bool:
        """True when every clip yields exactly one frame, whatever its length."""
        return self.count == 1 and all(b.count == 1 for b in self.duration_buckets)

    def frames_for(self, duration_seconds: float | None) -> int:
        """Frames to sample from a clip of this length.

        An unknown duration takes the default count -- guessing a bucket from a
        missing value would silently under-sample long clips.
        """
        if duration_seconds is None:
            return self.count
        for bucket in self.duration_buckets:
            if duration_seconds <= bucket.max_seconds:
                return bucket.count
        return self.count


class SecondaryClipConfig(_Base):
    enabled: bool = False
    model_name: str = "openai/clip-vit-large-patch14"


class EmbeddingsConfig(_Base):
    backend: Literal["dinov2", "clip", "stub"] = "dinov2"
    model_name: str = "facebook/dinov2-base"
    batch_size: int = Field(default=32, ge=1, le=1024)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    secondary_clip: SecondaryClipConfig = SecondaryClipConfig()

    #: Which embedder actually feeds pooling -> search -> scoring.
    #:   primary   - `backend`/`model_name` drive the score; the secondary, if
    #:               enabled, is computed and cached alongside.
    #:   secondary - the CLIP secondary drives the score and the primary is not
    #:               built at all, so no DINOv2 weights are loaded.
    driver: Literal["primary", "secondary"] = "primary"

    @model_validator(mode="after")
    def _driver_has_an_embedder(self) -> "EmbeddingsConfig":
        if self.driver == "secondary" and not self.secondary_clip.enabled:
            raise ValueError(
                "embeddings.driver='secondary' requires "
                "embeddings.secondary_clip.enabled=true -- there is otherwise no "
                "embedder to drive the pipeline"
            )
        return self


class PoolingConfig(_Base):
    # mean    - average the frame vectors, then L2-normalise (default).
    # none    - no averaging: take the clip's single frame vector as-is. Only
    #           sensible at one frame per clip.
    # maxpair - compare clips by their most-similar frame pair; used on
    #           borderline pairs, not as the clip representation.
    mode: Literal["mean", "none", "maxpair"] = "mean"


class SearchConfig(_Base):
    index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    top_k: int = Field(default=10, ge=1, le=1000)
    nlist: int = Field(default=100, ge=1)
    nprobe: int = Field(default=10, ge=1)
    hnsw_m: int = Field(default=32, ge=2)


class ClusteringConfig(_Base):
    similarity_threshold: UnitInterval = 0.91
    flag_cluster_size: int = Field(default=3, ge=2)


class ScoringConfig(_Base):
    max_points: float = Field(default=15.0, gt=0.0)
    penalty_exponent: float = Field(default=1.5, ge=1.0, le=4.0)


class BorderlineReviewConfig(_Base):
    enabled: bool = False
    gray_zone: tuple[UnitInterval, UnitInterval] = (0.80, 0.91)
    model: str = "gpt-4o"
    max_pairs: int = Field(default=50, ge=0)
    frames_per_clip: int = Field(default=1, ge=1, le=8)
    max_output_tokens: int = Field(default=300, ge=16)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("gray_zone")
    @classmethod
    def _ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("gray_zone must be (low, high) with low < high")
        return v


class ReportConfig(_Base):
    thumbnails: bool = True


class PipelineConfig(_Base):
    output_dir: Path = Path("./results")
    cache_dir: Path = Path("./.vd_cache")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    random_seed: int = 17

    frames: FramesConfig = FramesConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    pooling: PoolingConfig = PoolingConfig()
    search: SearchConfig = SearchConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    scoring: ScoringConfig = ScoringConfig()
    borderline_review: BorderlineReviewConfig = BorderlineReviewConfig()
    report: ReportConfig = ReportConfig()

    @model_validator(mode="after")
    def _gray_zone_below_threshold(self) -> "PipelineConfig":
        """The gray zone must sit *below* the duplicate threshold.

        Stage 12 exists to adjudicate pairs the clustering rule was not
        confident about. If the zone reached past the threshold it would be
        re-judging pairs already clustered as duplicates, which stage 12 has no
        mandate to undo.
        """
        high = self.borderline_review.gray_zone[1]
        thresh = self.clustering.similarity_threshold
        if self.borderline_review.enabled and high > thresh:
            raise ValueError(
                f"borderline_review.gray_zone upper bound ({high}) must not exceed "
                f"clustering.similarity_threshold ({thresh})"
            )
        return self

    def resolve_paths(self, base: Path) -> "PipelineConfig":
        """Return a copy with relative dirs anchored at ``base``.

        Config files name paths relative to themselves, so a config is portable
        between checkouts.
        """
        updates: dict[str, Any] = {}
        for name in ("output_dir", "cache_dir"):
            path = getattr(self, name)
            if not path.is_absolute():
                updates[name] = (base / path).resolve()
        return self.model_copy(update=updates) if updates else self


def _coerce_inf(node: Any) -> Any:
    """Turn YAML's ``.inf`` / ``"inf"`` spellings into a float, recursively.

    PyYAML already handles ``.inf``, but a quoted ``"inf"`` arrives as a string
    and would fail validation with a type error that does not explain itself.
    """
    if isinstance(node, dict):
        return {k: _coerce_inf(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_inf(v) for v in node]
    if isinstance(node, str) and node.strip().lower() in {"inf", "+inf", ".inf"}:
        return math.inf
    return node


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load and validate a YAML config, anchoring relative paths to its folder.

    ``path`` of None yields the all-defaults config, which is what the tests and
    a bare ``--config``-less run use.
    """
    if path is None:
        cfg = PipelineConfig()
        return cfg.resolve_paths(Path.cwd()) if not overrides else \
            PipelineConfig(**{**cfg.model_dump(), **overrides}).resolve_paths(Path.cwd())

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    if overrides:
        raw.update(overrides)
    return PipelineConfig(**_coerce_inf(raw)).resolve_paths(path.parent)


def as_sequence(value: Sequence[float]) -> tuple[float, ...]:
    """Small helper so callers can treat config tuples uniformly."""
    return tuple(float(v) for v in value)
