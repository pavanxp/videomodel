"""Stages 10-11: per-clip novelty and the delivery-level visual-diversity score.

    novelty(clip)   = 1 - max cosine similarity to any *other* clip
    avg_novelty     = mean(novelty)
    cluster_penalty = sum(size ** exponent for clusters of size > 1) / total_clips
    score           = max(0, (avg_novelty - cluster_penalty) * max_points)

The super-linear exponent is the point of the penalty: ten clusters of two are a
labelling nuisance, one cluster of twenty is a collection failure, and a linear
term would score them the same.

Alongside the headline number the stage attributes redundancy to the worker and
session that produced it, which is what makes the finding actionable upstream as
a collection quota rather than a post-hoc deletion list.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .clustering import ClusterSet
from .config import ScoringConfig
from .ingest import ClipRecord
from .search import NeighborTable

log = logging.getLogger(__name__)

__all__ = [
    "ClipScore",
    "GroupRedundancy",
    "DiversityScore",
    "score_delivery",
    "redundancy_by",
]


@dataclass(frozen=True, slots=True)
class ClipScore:
    item_id: str
    novelty: float
    max_similarity: float
    nearest_item_id: str | None
    cluster_id: int | None
    cluster_size: int


@dataclass(frozen=True, slots=True)
class GroupRedundancy:
    """How much of one group's output is redundant."""

    key: str
    clips: int
    clustered_clips: int
    mean_novelty: float
    #: Share of this group's own clips that sit in a duplicate cluster.
    redundancy_rate: float
    #: Share of *all* clustered clips in the delivery contributed by this group.
    share_of_delivery_redundancy: float

    @property
    def is_clean(self) -> bool:
        return self.clustered_clips == 0


@dataclass(slots=True)
class DiversityScore:
    score: float
    max_points: float
    avg_novelty: float
    cluster_penalty: float
    total_clips: int
    duplicate_clusters: int
    clipped_clips: int
    clip_scores: list[ClipScore] = field(default_factory=list)
    by_worker: list[GroupRedundancy] = field(default_factory=list)
    by_session: list[GroupRedundancy] = field(default_factory=list)
    flagged_for_removal: list[str] = field(default_factory=list)

    @property
    def score_fraction(self) -> float:
        return self.score / self.max_points if self.max_points else 0.0

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "max_points": self.max_points,
            "score_fraction": round(self.score_fraction, 4),
            "avg_novelty": round(self.avg_novelty, 6),
            "cluster_penalty": round(self.cluster_penalty, 6),
            "total_clips": self.total_clips,
            "duplicate_clusters": self.duplicate_clusters,
            "clips_in_duplicate_clusters": self.clipped_clips,
            "flagged_for_removal": self.flagged_for_removal,
        }


def _clip_scores(item_ids: Sequence[str], neighbors: NeighborTable,
                 clusters: ClusterSet) -> list[ClipScore]:
    out: list[ClipScore] = []
    for item_id in item_ids:
        top = neighbors.top(item_id)
        max_sim = top.similarity if top else 0.0
        cluster = clusters.cluster_of(item_id)
        out.append(ClipScore(
            item_id=item_id,
            # Similarity can be mildly negative for genuinely opposed vectors;
            # clamping keeps novelty inside [0, 1] where the score expects it.
            novelty=max(0.0, min(1.0, 1.0 - max_sim)),
            max_similarity=max_sim,
            nearest_item_id=top.item_id if top else None,
            cluster_id=cluster.cluster_id if cluster else None,
            cluster_size=cluster.size if cluster else 1,
        ))
    return out


def redundancy_by(attribute: str, clips: Sequence[ClipRecord],
                  clip_scores: Sequence[ClipScore]) -> list[GroupRedundancy]:
    """Attribute clustered clips to ``worker_id`` or ``session_id``.

    Sorted worst-first, so the top row is the collection source to cap.
    """
    by_id = {c.item_id: c for c in clips}
    scores_by_id = {s.item_id: s for s in clip_scores}

    grouped: dict[str, list[ClipScore]] = defaultdict(list)
    for item_id, score in scores_by_id.items():
        record = by_id.get(item_id)
        if record is None:
            continue
        grouped[str(getattr(record, attribute))].append(score)

    total_clustered = sum(1 for s in clip_scores if s.cluster_size > 1) or 0

    rows: list[GroupRedundancy] = []
    for key, members in grouped.items():
        clustered = sum(1 for s in members if s.cluster_size > 1)
        rows.append(GroupRedundancy(
            key=key,
            clips=len(members),
            clustered_clips=clustered,
            mean_novelty=sum(s.novelty for s in members) / len(members),
            redundancy_rate=clustered / len(members),
            share_of_delivery_redundancy=(clustered / total_clustered) if total_clustered else 0.0,
        ))

    rows.sort(key=lambda r: (-r.clustered_clips, -r.redundancy_rate, r.key))
    return rows


def score_delivery(clips: Sequence[ClipRecord], neighbors: NeighborTable,
                   clusters: ClusterSet, cfg: ScoringConfig,
                   item_ids: Sequence[str] | None = None) -> DiversityScore:
    """Compute the delivery's visual-diversity score and its breakdowns.

    ``item_ids`` defaults to the clips that actually made it through embedding;
    pass it explicitly so clips dropped mid-pipeline are not counted as novel.
    """
    ids = list(item_ids) if item_ids is not None else [c.item_id for c in clips]
    if not ids:
        log.warning("scoring: no clips to score; returning a zero score")
        return DiversityScore(score=0.0, max_points=cfg.max_points, avg_novelty=0.0,
                              cluster_penalty=0.0, total_clips=0,
                              duplicate_clusters=0, clipped_clips=0)

    clip_scores = _clip_scores(ids, neighbors, clusters)
    total = len(ids)

    avg_novelty = sum(s.novelty for s in clip_scores) / total
    dupes = clusters.duplicate_clusters
    penalty = sum(c.size ** cfg.penalty_exponent for c in dupes) / total
    score = max(0.0, (avg_novelty - penalty) * cfg.max_points)

    result = DiversityScore(
        score=score,
        max_points=cfg.max_points,
        avg_novelty=avg_novelty,
        cluster_penalty=penalty,
        total_clips=total,
        duplicate_clusters=len(dupes),
        clipped_clips=sum(c.size for c in dupes),
        clip_scores=clip_scores,
        by_worker=redundancy_by("worker_id", clips, clip_scores),
        by_session=redundancy_by("session_id", clips, clip_scores),
    )

    log.info("scoring: avg_novelty=%.4f  cluster_penalty=%.4f  -> %.2f / %.1f",
             avg_novelty, penalty, score, cfg.max_points)
    if penalty > avg_novelty:
        log.warning("scoring: the cluster penalty (%.4f) exceeds average novelty (%.4f); "
                    "the score floors at 0", penalty, avg_novelty)
    if result.by_worker and result.by_worker[0].clustered_clips:
        worst = result.by_worker[0]
        log.info("scoring: worst worker %s contributes %d clustered clip(s) (%.0f%% of its own "
                 "output, %.0f%% of all delivery redundancy)",
                 worst.key, worst.clustered_clips, 100 * worst.redundancy_rate,
                 100 * worst.share_of_delivery_redundancy)

    return result
