"""Stages 7-8: nearest-neighbour search over the pooled clip vectors.

Vectors are L2-normalised by stage 6, so inner product *is* cosine similarity
and an ``IndexFlatIP`` is the exact answer.

The index sits behind :class:`VectorIndex` so the backend can change without
touching a caller. Three are wired up:

  ``flat``  exact, brute force. Right up to roughly a million clips.
  ``ivf``   coarse-quantised inverted file. Trades recall for speed past that.
  ``hnsw``  graph index. Faster still, higher memory.

When faiss is not installed the search falls back to a numpy brute-force
implementation that returns identical results to ``flat`` -- so the pipeline is
correct without the extra, just slower.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .config import SearchConfig
from .pooling import ClipVectors

log = logging.getLogger(__name__)

__all__ = [
    "VectorIndex",
    "NumpyIndex",
    "FaissIndex",
    "Neighbor",
    "NeighborTable",
    "build_index",
    "search_neighbors",
    "faiss_available",
]


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


@runtime_checkable
class VectorIndex(Protocol):
    """Minimal contract every backend satisfies."""

    def add(self, matrix: np.ndarray) -> None:
        ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities, indices), both (n_queries, k)."""


class NumpyIndex:
    """Exact brute-force inner-product search. No dependencies beyond numpy.

    Chunked over queries so a large delivery does not materialise an
    n x n similarity matrix all at once.
    """

    name = "numpy-flat"

    def __init__(self, dim: int, chunk: int = 1024) -> None:
        self.dim = dim
        self.chunk = chunk
        self._matrix: np.ndarray | None = None

    def add(self, matrix: np.ndarray) -> None:
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None or self._matrix.size == 0:
            raise RuntimeError("index is empty; call add() first")
        base = self._matrix
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, base.shape[0])

        sims_out = np.empty((queries.shape[0], k), dtype=np.float32)
        idx_out = np.empty((queries.shape[0], k), dtype=np.int64)

        for start in range(0, queries.shape[0], self.chunk):
            block = queries[start: start + self.chunk] @ base.T
            # argpartition gives the top-k cheaply; sort only those k.
            part = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(block.shape[0])[:, None]
            order = np.argsort(-block[rows, part], axis=1)
            top = part[rows, order]
            idx_out[start: start + block.shape[0]] = top
            sims_out[start: start + block.shape[0]] = block[rows, top]

        return sims_out, idx_out


class FaissIndex:
    """faiss-backed search. ``index_type`` selects flat / ivf / hnsw."""

    def __init__(self, dim: int, cfg: SearchConfig, n_vectors: int) -> None:
        import faiss

        self._faiss = faiss
        self.name = f"faiss-{cfg.index_type}"
        self.cfg = cfg
        self.dim = dim
        self._needs_training = False

        if cfg.index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif cfg.index_type == "ivf":
            # nlist above the vector count leaves empty cells and faiss warns;
            # clamp so a small delivery still builds a valid index.
            nlist = max(1, min(cfg.nlist, max(1, n_vectors // 39)))
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(cfg.nprobe, nlist)
            self._needs_training = True
        elif cfg.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:  # pragma: no cover - pydantic already constrains this
            raise ValueError(f"unknown index_type {cfg.index_type!r}")

    def add(self, matrix: np.ndarray) -> None:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if self._needs_training and not self.index.is_trained:
            self.index.train(matrix)
        self.index.add(matrix)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        sims, idx = self.index.search(queries, min(k, self.index.ntotal))
        return sims.astype(np.float32), idx.astype(np.int64)


def build_index(vectors: ClipVectors, cfg: SearchConfig) -> VectorIndex:
    """Construct and populate the configured index for these vectors."""
    if len(vectors) == 0:
        raise ValueError("cannot build an index over zero clips")

    if faiss_available():
        index: VectorIndex = FaissIndex(vectors.dim, cfg, len(vectors))
    else:
        if cfg.index_type != "flat":
            log.warning("search: faiss is not installed, so index_type=%r falls back to exact "
                        "numpy search (same results, slower). Install 'visual-diversity[faiss]' "
                        "for the approximate backends.", cfg.index_type)
        else:
            log.info("search: faiss not installed; using exact numpy search")
        index = NumpyIndex(vectors.dim)

    index.add(vectors.matrix)
    log.info("search: indexed %d vector(s) of dim %d with %s",
             len(vectors), vectors.dim, getattr(index, "name", type(index).__name__))
    return index


@dataclass(frozen=True, slots=True)
class Neighbor:
    item_id: str
    similarity: float


@dataclass(frozen=True, slots=True)
class NeighborTable:
    """Top-k neighbours per clip, self-matches removed."""

    neighbors: dict[str, list[Neighbor]]

    def top(self, item_id: str) -> Neighbor | None:
        row = self.neighbors.get(item_id) or []
        return row[0] if row else None

    def max_similarity(self, item_id: str) -> float:
        """Similarity to the nearest *other* clip, or 0.0 when it stands alone."""
        top = self.top(item_id)
        return top.similarity if top else 0.0

    def unique_pairs(self) -> list[tuple[str, str, float]]:
        """Each neighbour relation once, as (a, b, similarity) with a < b.

        k-NN is not symmetric -- B can be in A's top-k without the reverse --
        so the union is taken and the higher similarity kept for any pair seen
        from both sides.
        """
        best: dict[tuple[str, str], float] = {}
        for item_id, row in self.neighbors.items():
            for nb in row:
                key = (item_id, nb.item_id) if item_id < nb.item_id else (nb.item_id, item_id)
                if nb.similarity > best.get(key, -1.0):
                    best[key] = nb.similarity
        return [(a, b, s) for (a, b), s in sorted(best.items())]


def search_neighbors(index: VectorIndex, vectors: ClipVectors,
                     cfg: SearchConfig) -> NeighborTable:
    """Top-k neighbours for every clip, excluding the clip itself."""
    if len(vectors) == 0:
        return NeighborTable({})
    if len(vectors) == 1:
        # A lone clip has no neighbour; it is maximally novel by definition.
        return NeighborTable({vectors.item_ids[0]: []})

    # One extra slot, because the nearest hit is almost always the query itself.
    k = min(cfg.top_k + 1, len(vectors))
    sims, idx = index.search(vectors.matrix, k)

    table: dict[str, list[Neighbor]] = {}
    for row, item_id in enumerate(vectors.item_ids):
        found: list[Neighbor] = []
        for col in range(idx.shape[1]):
            j = int(idx[row, col])
            # faiss pads with -1 when fewer than k results exist.
            if j < 0 or j == row:
                continue
            # Clamp: float error can push an exact match a hair past 1.0.
            found.append(Neighbor(vectors.item_ids[j], float(min(1.0, max(-1.0, sims[row, col])))))
            if len(found) == cfg.top_k:
                break
        table[item_id] = found

    log.info("search: top-%d neighbours resolved for %d clip(s)", cfg.top_k, len(table))
    return NeighborTable(table)
