"""End-to-end orchestration of stages 1-13.

Every stage logs what came in and what went out, so a shrinking clip count is
always attributable to a named stage rather than discovered at the end.

Resumability is per-stage and content-keyed: frames and embeddings live in
``cache_dir`` keyed by ``item_id``, and a re-run reuses whatever is present
unless ``force`` is set. The cheap stages (pooling, search, clustering, scoring)
are always recomputed -- caching them would buy milliseconds and risk serving a
stale score.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import borderline_review as br
from . import clustering, embeddings, frames as frames_mod, ingest, pooling, report, search
from .config import PipelineConfig, load_config
from .scoring import DiversityScore, score_delivery

log = logging.getLogger(__name__)

__all__ = ["PipelineResult", "run_pipeline", "configure_logging", "log_inert_settings"]


def configure_logging(level: str = "INFO") -> None:
    """Attach a single stderr handler. Idempotent, so repeated calls are safe."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not any(getattr(h, "_visual_diversity", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S"))
        handler._visual_diversity = True  # type: ignore[attr-defined]
        root.addHandler(handler)


@dataclass(slots=True)
class PipelineResult:
    score: DiversityScore
    clusters: clustering.ClusterSet
    outputs: dict[str, Path]
    timings: dict[str, float] = field(default_factory=dict)
    ingested: int = 0
    rejected: int = 0
    scored: int = 0
    borderline: dict[str, Any] | None = None


def log_inert_settings(cfg: PipelineConfig) -> list[str]:
    """Announce every configured setting this run cannot act on, and why.

    A knob that is quietly ignored looks like a knob that is working. In
    single-frame CLIP mode most of the multi-frame apparatus is inert, so it is
    listed once at startup rather than left for someone to discover by changing
    a value and seeing nothing happen.

    Returns the notes (also used by the tests).
    """
    notes: list[str] = []
    frames, emb = cfg.frames, cfg.embeddings

    if frames.single_frame:
        notes.append(
            "frames.count=1 -- one frame per clip at 50% of duration. "
            "Everything that only distinguishes frames *within* a clip is inert."
        )
        if cfg.pooling.mode == "none":
            notes.append(
                "pooling.mode='none' -- no averaging; the frame vector is the clip "
                "vector. The stage still L2-normalises and builds the id-to-row "
                "matrix the index needs, so it is not skippable."
            )
        elif cfg.pooling.mode == "mean":
            notes.append(
                "pooling.mode='mean' -- at one frame the average of a single vector "
                "is that vector, so this is equivalent to 'none'."
            )
        notes.append(
            "pooling 'maxpair' and pooling.max_frame_pair_similarity are inert: "
            "with one frame per clip the most-similar frame pair IS the clip pair, "
            "so it is identical to the cosine similarity search already computes."
        )
        if cfg.borderline_review.enabled and cfg.borderline_review.frames_per_clip > 1:
            notes.append(
                f"borderline_review.frames_per_clip={cfg.borderline_review.frames_per_clip} "
                f"is capped to 1 -- there is only one frame to send."
            )

    if not frames.duration_buckets:
        notes.append("frames.duration_buckets is empty -- clip length does not change "
                     "the frame count.")

    if emb.driver == "secondary":
        notes.append(
            f"embeddings.driver='secondary' -- the CLIP pass "
            f"({emb.secondary_clip.model_name}) scores the delivery. "
            f"embeddings.backend='{emb.backend}' and model_name="
            f"'{emb.model_name}' are inert; no primary model is loaded."
        )

    if not cfg.borderline_review.enabled:
        notes.append("borderline_review.enabled=false -- stage 12 does not run, and "
                     "its gray_zone/max_pairs/model settings are inert.")

    for note in notes:
        log.info("inert: %s", note)
    return notes


@contextmanager
def _stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    log.info("--- stage: %s ---", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        log.info("--- %s finished in %.2fs ---", name, elapsed)


def run_pipeline(manifest_path: str | Path | None = None,
                 config_path: str | Path | None = None,
                 output_dir: str | Path | None = None,
                 *, force: bool = False,
                 judge: br.JudgeFn | None = None) -> PipelineResult:
    """Run every stage in order and write the reports.

    Path precedence, highest first:

        CLI argument  >  settings.py / VD_* env  >  pipeline_config.yaml

    ``judge`` overrides the stage-12 vision judge; the tests pass a fake so the
    full path runs without an API key.
    """
    from .settings import get_settings

    paths = get_settings().paths
    cfg = load_config(config_path)

    overrides: dict[str, Any] = {}
    if paths.output_dir is not None:
        overrides["output_dir"] = paths.output_dir.expanduser().resolve()
    if paths.cache_dir is not None:
        overrides["cache_dir"] = paths.cache_dir.expanduser().resolve()
    # The explicit argument outranks settings.
    if output_dir is not None:
        overrides["output_dir"] = Path(output_dir).expanduser().resolve()
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    if manifest_path is None:
        manifest_path = paths.manifest
    if manifest_path is None:
        raise ValueError(
            "no manifest given: pass one, set VD_MANIFEST, or set DEFAULT_MANIFEST "
            "in visual_diversity/settings.py"
        )

    configure_logging(cfg.log_level)

    out_dir = Path(cfg.output_dir)
    cache_dir = Path(cfg.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Credentials were seeded from .env by get_settings() above, so a missing
    # key fails at startup rather than after the embedding spend.
    creds = get_settings()
    if creds.env_file:
        log.info("visual-diversity: credentials from %s", creds.env_file)
    # Only the real judge needs credentials; an injected one does not.
    if cfg.borderline_review.enabled and judge is None:
        creds.openai.require()

    log.info("visual-diversity: manifest=%s", manifest_path)
    log.info("visual-diversity: output=%s cache=%s force=%s", out_dir, cache_dir, force)
    log_inert_settings(cfg)

    timings: dict[str, float] = {}

    # -- Stages 1-2 --------------------------------------------------------
    with _stage("ingest", timings):
        if paths.input_dir is not None:
            log.info("ingest: relative clip paths resolve against %s", paths.input_dir)
        result = ingest.load_manifest(manifest_path, require_existing_files=True,
                                      input_dir=paths.input_dir)
        ingest.write_rejected_csv(result.rejected, out_dir / "rejected_clips.csv")
        clips = result.clips

    if not clips:
        log.error("ingest: no valid clips; nothing to score")
        return _empty_result(cfg, result.rejected, timings)

    # -- Stage 3 -----------------------------------------------------------
    with _stage("frames", timings):
        frame_map = frames_mod.extract_all(clips, cache_dir, cfg.frames, force=force)
        usable = {k: v for k, v in frame_map.items() if v.ok}

    if not usable:
        log.error("frames: no clip yielded frames; nothing to score")
        return _empty_result(cfg, result.rejected, timings, clips=clips)

    # -- Stages 4-5 --------------------------------------------------------
    with _stage("embeddings", timings):
        driver, side_pass = embeddings.build_driving_embedder(cfg.embeddings)
        frame_vectors = embeddings.embed_clips(usable, driver, cache_dir, force=force)
        if side_pass is not None:
            # Cached alongside, never consumed by the score.
            embeddings.embed_clips(usable, side_pass, cache_dir, force=force)

    if not frame_vectors:
        log.error("embeddings: nothing embedded; nothing to score")
        return _empty_result(cfg, result.rejected, timings, clips=clips)

    # -- Stage 6 -----------------------------------------------------------
    with _stage("pooling", timings):
        vectors = pooling.pool_clips(frame_vectors, mode=cfg.pooling.mode)

    if len(vectors) == 0:
        log.error("pooling: no clip vectors; nothing to score")
        return _empty_result(cfg, result.rejected, timings, clips=clips)

    # -- Stages 7-8 --------------------------------------------------------
    with _stage("search", timings):
        index = search.build_index(vectors, cfg.search)
        neighbors = search.search_neighbors(index, vectors, cfg.search)

    # -- Stage 9 -----------------------------------------------------------
    with _stage("clustering", timings):
        clusters = clustering.cluster_clips(
            neighbors, cfg.clustering.similarity_threshold, vectors.item_ids)

    # -- Stage 12 (before the score is finalised) --------------------------
    borderline_payload: dict[str, Any] = {"enabled": cfg.borderline_review.enabled}
    if cfg.borderline_review.enabled:
        with _stage("borderline_review", timings):
            try:
                outcome = br.review_pairs(neighbors, usable, cfg.borderline_review,
                                          judge=judge, seed=cfg.random_seed)
                clusters = br.apply_corrections(
                    neighbors, clusters, outcome,
                    cfg.clustering.similarity_threshold, vectors.item_ids)
                borderline_payload = {"enabled": True, **outcome.as_dict()}
            except Exception as exc:  # noqa: BLE001 -- review is advisory, never fatal
                log.error("borderline: stage failed (%s); keeping the uncorrected clusters", exc)
                borderline_payload = {"enabled": True, "error": str(exc)}
    else:
        log.info("borderline: disabled in config; skipping stage 12")

    # -- Stages 10-11 ------------------------------------------------------
    with _stage("scoring", timings):
        score = score_delivery(clips, neighbors, clusters, cfg.scoring,
                               item_ids=vectors.item_ids)

    # -- Stage 13 ----------------------------------------------------------
    with _stage("report", timings):
        outputs = report.write_reports(
            score=score, clusters=clusters, clips=clips, rejected=result.rejected,
            cfg=cfg, frames=usable, borderline=borderline_payload, timings=timings)
    outputs["rejected_csv"] = out_dir / "rejected_clips.csv"

    log.info("visual-diversity: score %.2f / %.1f over %d clip(s) in %.2fs total",
             score.score, score.max_points, score.total_clips, sum(timings.values()))

    return PipelineResult(score=score, clusters=clusters, outputs=outputs, timings=timings,
                          ingested=len(clips), rejected=len(result.rejected),
                          scored=score.total_clips, borderline=borderline_payload)


def _empty_result(cfg: PipelineConfig, rejected: Sequence[ingest.RejectedClip],
                  timings: dict[str, float],
                  clips: Sequence[ingest.ClipRecord] = ()) -> PipelineResult:
    """Write a zero-score report rather than returning nothing.

    A run that produced no scoreable clip is a finding about the delivery, and
    the report is where that finding belongs.
    """
    score = DiversityScore(score=0.0, max_points=cfg.scoring.max_points, avg_novelty=0.0,
                           cluster_penalty=0.0, total_clips=0, duplicate_clusters=0,
                           clipped_clips=0)
    empty = clustering.ClusterSet()
    outputs = report.write_reports(score=score, clusters=empty, clips=list(clips),
                                   rejected=list(rejected), cfg=cfg, timings=timings)
    outputs["rejected_csv"] = Path(cfg.output_dir) / "rejected_clips.csv"
    return PipelineResult(score=score, clusters=empty, outputs=outputs, timings=timings,
                          ingested=len(clips), rejected=len(rejected), scored=0)
