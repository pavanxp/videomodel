"""Stages 10-11: novelty, the cluster penalty and redundancy attribution."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_diversity.clustering import cluster_clips
from visual_diversity.config import ScoringConfig
from visual_diversity.ingest import ClipRecord
from visual_diversity.scoring import redundancy_by, score_delivery
from visual_diversity.search import Neighbor, NeighborTable

from conftest import manifest_row


def clip(item_id: str, worker: str = "w1", session: str = "s1") -> ClipRecord:
    return ClipRecord(**manifest_row(item_id, Path("/tmp/x.mp4"),
                                     worker=worker, session=session))


def table(**rows: list[tuple[str, float]]) -> NeighborTable:
    return NeighborTable({k: [Neighbor(i, s) for i, s in v] for k, v in rows.items()})


def test_novelty_is_one_minus_max_similarity():
    nt = table(a=[("b", 0.25)], b=[("a", 0.25)])
    clusters = cluster_clips(nt, 0.9, ["a", "b"])

    result = score_delivery([clip("a"), clip("b")], nt, clusters, ScoringConfig())

    assert {s.item_id: s.novelty for s in result.clip_scores} == {
        "a": pytest.approx(0.75), "b": pytest.approx(0.75)}


def test_a_lone_clip_is_maximally_novel():
    nt = table(a=[])
    clusters = cluster_clips(nt, 0.9, ["a"])
    result = score_delivery([clip("a")], nt, clusters, ScoringConfig())
    assert result.avg_novelty == pytest.approx(1.0)


def test_novelty_is_clamped_into_the_unit_interval():
    nt = table(a=[("b", -0.4)], b=[("a", -0.4)])
    clusters = cluster_clips(nt, 0.9, ["a", "b"])
    result = score_delivery([clip("a"), clip("b")], nt, clusters, ScoringConfig())
    assert all(0.0 <= s.novelty <= 1.0 for s in result.clip_scores)


def test_perfectly_diverse_delivery_scores_full_marks():
    nt = table(**{c: [(o, 0.0)] for c, o in (("a", "b"), ("b", "a"))})
    clusters = cluster_clips(nt, 0.9, ["a", "b"])

    result = score_delivery([clip("a"), clip("b")], nt, clusters,
                            ScoringConfig(max_points=15.0))

    assert result.score == pytest.approx(15.0)
    assert result.cluster_penalty == 0.0


def test_cluster_penalty_formula():
    # One cluster of 3 out of 4 clips: 3 ** 1.5 / 4.
    nt = table(a=[("b", 0.99), ("c", 0.99)], b=[("a", 0.99)], c=[("a", 0.99)], d=[("a", 0.1)])
    clusters = cluster_clips(nt, 0.9, ["a", "b", "c", "d"])

    result = score_delivery([clip(x) for x in "abcd"], nt, clusters,
                            ScoringConfig(penalty_exponent=1.5))

    assert result.cluster_penalty == pytest.approx(3 ** 1.5 / 4)
    assert result.duplicate_clusters == 1
    assert result.clipped_clips == 3


def test_penalty_is_superlinear_in_cluster_size():
    """One cluster of four must cost more than two clusters of two."""
    big = table(a=[("b", .99), ("c", .99), ("d", .99)], b=[("a", .99)],
                c=[("a", .99)], d=[("a", .99)])
    split = table(a=[("b", .99)], b=[("a", .99)], c=[("d", .99)], d=[("c", .99)])
    ids = ["a", "b", "c", "d"]
    clips = [clip(x) for x in ids]
    cfg = ScoringConfig()

    one_big = score_delivery(clips, big, cluster_clips(big, 0.9, ids), cfg)
    two_small = score_delivery(clips, split, cluster_clips(split, 0.9, ids), cfg)

    assert one_big.cluster_penalty > two_small.cluster_penalty


def test_score_floors_at_zero():
    nt = table(**{c: [(o, 0.99) for o in "abcde" if o != c] for c in "abcde"})
    ids = list("abcde")
    clusters = cluster_clips(nt, 0.9, ids)

    result = score_delivery([clip(x) for x in ids], nt, clusters, ScoringConfig())

    assert result.score == 0.0


def test_max_points_is_configurable():
    nt = table(a=[("b", 0.0)], b=[("a", 0.0)])
    clusters = cluster_clips(nt, 0.9, ["a", "b"])
    result = score_delivery([clip("a"), clip("b")], nt, clusters,
                            ScoringConfig(max_points=20.0))
    assert result.score == pytest.approx(20.0)
    assert result.score_fraction == pytest.approx(1.0)


def test_empty_delivery_scores_zero_without_crashing():
    result = score_delivery([], NeighborTable({}), cluster_clips(NeighborTable({}), 0.9, []),
                            ScoringConfig())
    assert result.score == 0.0
    assert result.total_clips == 0


def test_redundancy_is_attributed_to_the_right_worker():
    nt = table(a=[("b", 0.99)], b=[("a", 0.99)], c=[("a", 0.1)])
    ids = ["a", "b", "c"]
    clusters = cluster_clips(nt, 0.9, ids)
    clips = [clip("a", worker="bad"), clip("b", worker="bad"), clip("c", worker="good")]

    result = score_delivery(clips, nt, clusters, ScoringConfig(), item_ids=ids)
    by_worker = {r.key: r for r in result.by_worker}

    assert by_worker["bad"].clustered_clips == 2
    assert by_worker["bad"].redundancy_rate == pytest.approx(1.0)
    assert by_worker["bad"].share_of_delivery_redundancy == pytest.approx(1.0)
    assert by_worker["good"].is_clean
    # Worst-first ordering makes the top row the actionable one.
    assert result.by_worker[0].key == "bad"


def test_redundancy_by_session_splits_independently():
    nt = table(a=[("b", 0.99)], b=[("a", 0.99)])
    ids = ["a", "b"]
    clusters = cluster_clips(nt, 0.9, ids)
    clips = [clip("a", worker="w1", session="s1"), clip("b", worker="w2", session="s1")]

    rows = redundancy_by("session_id", clips,
                         score_delivery(clips, nt, clusters, ScoringConfig()).clip_scores)

    assert len(rows) == 1
    assert rows[0].key == "s1"
    assert rows[0].clustered_clips == 2


def test_item_ids_limits_scoring_to_surviving_clips():
    """A clip dropped before embedding must not be counted as novel."""
    nt = table(a=[("b", 0.99)], b=[("a", 0.99)])
    clusters = cluster_clips(nt, 0.9, ["a", "b"])
    clips = [clip("a"), clip("b"), clip("dropped")]

    result = score_delivery(clips, nt, clusters, ScoringConfig(), item_ids=["a", "b"])

    assert result.total_clips == 2
    assert "dropped" not in {s.item_id for s in result.clip_scores}
