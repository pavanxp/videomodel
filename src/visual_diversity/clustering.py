"""Stage 9: group near-duplicate clips into clusters.

Clips are nodes; an edge joins two clips whose cosine similarity reaches the
configured threshold. Duplicate clusters are the connected components.

Connected components are transitive, which is deliberate: A~B and B~C puts all
three in one cluster even if A and C are not directly similar. For near-
duplicate footage that is the right call -- a slow pan across a workspace
produces exactly that chain, and it is one redundant shoot, not two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import networkx as nx

from .search import NeighborTable

log = logging.getLogger(__name__)

__all__ = ["Cluster", "ClusterSet", "build_graph", "cluster_clips"]


@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: int
    members: tuple[str, ...]
    #: Highest pairwise similarity observed inside the cluster.
    max_similarity: float
    #: Mean of the edge similarities that built it.
    mean_similarity: float

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_duplicate(self) -> bool:
        return self.size > 1


@dataclass(slots=True)
class ClusterSet:
    clusters: list[Cluster] = field(default_factory=list)
    #: Every clip, including singletons, mapped to its cluster id.
    assignment: dict[str, int] = field(default_factory=dict)

    @property
    def duplicate_clusters(self) -> list[Cluster]:
        return [c for c in self.clusters if c.is_duplicate]

    @property
    def clipped_count(self) -> int:
        """How many clips sit in a cluster of more than one."""
        return sum(c.size for c in self.duplicate_clusters)

    def cluster_of(self, item_id: str) -> Cluster | None:
        cid = self.assignment.get(item_id)
        return None if cid is None else self.clusters[cid]

    def flagged_for_removal(self, min_size: int) -> list[str]:
        """Members of clusters at or above ``min_size``, minus one keeper each.

        The whole cluster is not condemned -- one representative is worth
        keeping. Members are ordered, so the keeper is stable across runs.
        """
        out: list[str] = []
        for c in self.duplicate_clusters:
            if c.size >= min_size:
                out.extend(c.members[1:])
        return sorted(out)


def build_graph(neighbors: NeighborTable, threshold: float,
                nodes: Iterable[str] | None = None) -> nx.Graph:
    """Similarity graph: an edge wherever similarity >= ``threshold``."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes if nodes is not None else neighbors.neighbors.keys())
    for a, b, sim in neighbors.unique_pairs():
        if sim >= threshold:
            graph.add_edge(a, b, similarity=sim)
    return graph


def cluster_clips(neighbors: NeighborTable, threshold: float,
                  all_item_ids: Sequence[str] | None = None) -> ClusterSet:
    """Connected-component clustering over the similarity graph.

    Every clip gets a cluster id, singletons included, so downstream code can
    look up any clip without a missing-key branch.
    """
    graph = build_graph(neighbors, threshold, all_item_ids)

    # Sort components largest-first, then lexicographically, so cluster ids are
    # stable between runs on the same data.
    components = sorted(
        (sorted(comp) for comp in nx.connected_components(graph)),
        key=lambda m: (-len(m), m[0]),
    )

    clusters: list[Cluster] = []
    assignment: dict[str, int] = {}

    for cid, members in enumerate(components):
        sims = [d["similarity"] for _, _, d in graph.subgraph(members).edges(data=True)]
        clusters.append(Cluster(
            cluster_id=cid,
            members=tuple(members),
            max_similarity=max(sims) if sims else 0.0,
            mean_similarity=(sum(sims) / len(sims)) if sims else 0.0,
        ))
        for m in members:
            assignment[m] = cid

    dupes = [c for c in clusters if c.is_duplicate]
    log.info("clustering: %d cluster(s) at threshold %.3f -- %d duplicate cluster(s) "
             "covering %d clip(s); %d singleton(s)",
             len(clusters), threshold, len(dupes), sum(c.size for c in dupes),
             len(clusters) - len(dupes))
    if dupes:
        biggest = max(dupes, key=lambda c: c.size)
        log.info("clustering: largest cluster holds %d clip(s) (max similarity %.4f)",
                 biggest.size, biggest.max_similarity)

    return ClusterSet(clusters=clusters, assignment=assignment)
