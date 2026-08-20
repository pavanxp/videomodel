"""Stage 9: threshold graph and connected-component clustering."""

from __future__ import annotations

import pytest

from visual_diversity.clustering import build_graph, cluster_clips
from visual_diversity.search import Neighbor, NeighborTable


def table(**rows: list[tuple[str, float]]) -> NeighborTable:
    return NeighborTable({k: [Neighbor(i, s) for i, s in v] for k, v in rows.items()})


def test_pairs_above_the_threshold_become_one_cluster():
    nt = table(a=[("b", 0.95)], b=[("a", 0.95)], c=[("a", 0.10)])

    result = cluster_clips(nt, threshold=0.9, all_item_ids=["a", "b", "c"])

    assert len(result.duplicate_clusters) == 1
    assert set(result.duplicate_clusters[0].members) == {"a", "b"}
    assert result.cluster_of("c").size == 1


def test_pairs_below_the_threshold_stay_apart():
    nt = table(a=[("b", 0.80)], b=[("a", 0.80)])
    result = cluster_clips(nt, threshold=0.9, all_item_ids=["a", "b"])
    assert result.duplicate_clusters == []


def test_threshold_is_inclusive():
    nt = table(a=[("b", 0.9)], b=[("a", 0.9)])
    assert len(cluster_clips(nt, 0.9, ["a", "b"]).duplicate_clusters) == 1


def test_clustering_is_transitive():
    # A~B and B~C, but A and C are not directly similar.
    nt = table(a=[("b", 0.95)], b=[("a", 0.95), ("c", 0.95)], c=[("b", 0.95)])

    result = cluster_clips(nt, 0.9, ["a", "b", "c"])

    assert len(result.duplicate_clusters) == 1
    assert set(result.duplicate_clusters[0].members) == {"a", "b", "c"}


def test_every_clip_gets_an_assignment_including_singletons():
    nt = table(a=[("b", 0.95)], b=[("a", 0.95)])
    result = cluster_clips(nt, 0.9, ["a", "b", "lonely"])

    assert set(result.assignment) == {"a", "b", "lonely"}
    assert result.cluster_of("lonely").size == 1


def test_cluster_ids_are_stable_and_largest_first():
    nt = table(
        a=[("b", 0.99)], b=[("a", 0.99)],
        x=[("y", 0.99), ("z", 0.99)], y=[("x", 0.99)], z=[("x", 0.99)],
    )
    first = cluster_clips(nt, 0.9, ["a", "b", "x", "y", "z"])
    second = cluster_clips(nt, 0.9, ["a", "b", "x", "y", "z"])

    assert first.clusters[0].size == 3
    assert [c.members for c in first.clusters] == [c.members for c in second.clusters]


def test_similarity_statistics_are_recorded():
    nt = table(a=[("b", 0.95)], b=[("a", 0.95), ("c", 0.91)], c=[("b", 0.91)])
    cluster = cluster_clips(nt, 0.9, ["a", "b", "c"]).duplicate_clusters[0]

    assert cluster.max_similarity == pytest.approx(0.95)
    assert cluster.mean_similarity == pytest.approx((0.95 + 0.91) / 2)


def test_flagged_for_removal_keeps_one_per_cluster():
    nt = table(
        a=[("b", 0.99), ("c", 0.99)], b=[("a", 0.99)], c=[("a", 0.99)],
        x=[("y", 0.99)], y=[("x", 0.99)],
    )
    result = cluster_clips(nt, 0.9, ["a", "b", "c", "x", "y"])

    # Only the cluster of 3 qualifies; one member survives as the keeper.
    assert result.flagged_for_removal(min_size=3) == ["b", "c"]
    # At min_size 2 both clusters qualify, still one keeper each.
    assert result.flagged_for_removal(min_size=2) == ["b", "c", "y"]


def test_clipped_count_totals_clustered_clips():
    nt = table(a=[("b", 0.99)], b=[("a", 0.99)], c=[])
    assert cluster_clips(nt, 0.9, ["a", "b", "c"]).clipped_count == 2


def test_build_graph_only_adds_qualifying_edges():
    nt = table(a=[("b", 0.95), ("c", 0.10)], b=[("a", 0.95)], c=[("a", 0.10)])
    graph = build_graph(nt, 0.9, ["a", "b", "c"])

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 1
    assert graph.has_edge("a", "b")


def test_empty_input_produces_no_clusters():
    result = cluster_clips(NeighborTable({}), 0.9, [])
    assert result.clusters == []
    assert result.duplicate_clusters == []
