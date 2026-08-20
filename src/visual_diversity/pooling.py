"""Stage 6: collapse each clip's frame vectors into one clip vector.

Default is mean-pool then L2-normalise, which makes inner product equal cosine
similarity for everything downstream.

Mean pooling deliberately throws away within-clip variation: two clips that each
pan across the same workspace average to nearly the same vector even if no two
individual frames match. :func:`max_frame_pair_similarity` is the counterweight
-- an O(frames^2) comparison used only on the handful of pairs stage 12 flags,
where the cost is affordable and the detail matters.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "l2_normalize",
    "mean_pool",
    "single_frame",
    "pool_clips",
    "max_frame_pair_similarity",
    "ClipVectors",
]


def l2_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale to unit length along ``axis``; a zero vector is left as zeros."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return (vectors / np.maximum(norms, eps)).astype(np.float32, copy=False)


def mean_pool(frame_vectors: np.ndarray) -> np.ndarray:
    """(n_frames, dim) -> (dim,), mean-pooled and L2-normalised."""
    arr = np.asarray(frame_vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"expected a non-empty (n_frames, dim) array, got shape {arr.shape}")
    return l2_normalize(arr.mean(axis=0))


def single_frame(frame_vectors: np.ndarray) -> np.ndarray:
    """(n_frames, dim) -> (dim,), taking the first frame with no averaging.

    The pooling bypass, for one-frame-per-clip runs. Normalisation is *not*
    optional and is applied here too: search scores clips by inner product, and
    that only equals cosine similarity on unit vectors. Skipping it would leave
    every similarity scaled by the vectors' magnitudes and silently invalidate
    the clustering threshold.
    """
    arr = np.asarray(frame_vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"expected a non-empty (n_frames, dim) array, got shape {arr.shape}")
    return l2_normalize(arr[0])


class ClipVectors:
    """Pooled clip vectors in a fixed order, with the id<->row mapping.

    The row order is the contract between this stage, the index and every score
    that comes back keyed by position.
    """

    __slots__ = ("item_ids", "matrix")

    def __init__(self, item_ids: list[str], matrix: np.ndarray) -> None:
        if len(item_ids) != matrix.shape[0]:
            raise ValueError(
                f"{len(item_ids)} id(s) but {matrix.shape[0]} row(s) -- these must match")
        self.item_ids = item_ids
        self.matrix = matrix.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.item_ids)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    def index_of(self, item_id: str) -> int:
        return self.item_ids.index(item_id)

    def vector_for(self, item_id: str) -> np.ndarray:
        return self.matrix[self.index_of(item_id)]


def pool_clips(frame_embeddings: Mapping[str, np.ndarray],
               mode: str = "mean") -> ClipVectors:
    """Reduce each clip's frames to one vector. Ordered by ``item_id``.

    ``mode`` is ``"mean"`` (average the frames) or ``"none"`` (bypass pooling
    and use the clip's single frame).

    ``"maxpair"`` is rejected here rather than quietly falling back. It is a
    comparison between two clips, not a way to reduce one clip to a vector, so
    there is no honest thing for this function to return -- and silently
    mean-pooling instead would report a mode the run did not use.
    """
    if mode == "maxpair":
        raise ValueError(
            "pooling.mode='maxpair' is a pairwise comparison, not a clip "
            "representation; it cannot build the index. Use 'mean' or 'none'. "
            "(At one frame per clip it is identical to plain cosine similarity "
            "and buys nothing.)"
        )
    if mode not in ("mean", "none"):
        raise ValueError(f"unknown pooling mode {mode!r}; expected 'mean' or 'none'")

    item_ids = sorted(frame_embeddings)
    if not item_ids:
        return ClipVectors([], np.zeros((0, 0), dtype=np.float32))

    dims = {int(np.asarray(frame_embeddings[i]).shape[1]) for i in item_ids
            if np.asarray(frame_embeddings[i]).ndim == 2}
    if len(dims) > 1:
        raise ValueError(f"frame embeddings have mixed dimensionality: {sorted(dims)}")

    reduce = single_frame if mode == "none" else mean_pool
    if mode == "none":
        extra = [i for i in item_ids if np.asarray(frame_embeddings[i]).shape[0] > 1]
        if extra:
            log.warning("pooling: mode='none' keeps only the first frame, but %d clip(s) "
                        "carry more than one; the rest are ignored. Set frames.count=1 "
                        "to stop extracting frames that are then discarded.", len(extra))

    pooled: list[np.ndarray] = []
    kept: list[str] = []
    for item_id in item_ids:
        try:
            pooled.append(reduce(frame_embeddings[item_id]))
            kept.append(item_id)
        except ValueError as exc:
            log.warning("pooling: %s skipped -- %s", item_id, exc)

    matrix = np.vstack(pooled) if pooled else np.zeros((0, 0), dtype=np.float32)
    log.info("pooling: %d clip vector(s) of dim %d [mode=%s]", len(kept),
             matrix.shape[1] if matrix.size else 0, mode)
    return ClipVectors(kept, matrix)


def max_frame_pair_similarity(frames_a: np.ndarray, frames_b: np.ndarray) -> float:
    """Highest cosine similarity between any frame of A and any frame of B.

    The alternative comparison mode. Where mean pooling asks "do these clips
    average to the same thing", this asks "do they share any single moment",
    which is the better question for near-duplicate footage shot from a slightly
    different angle.
    """
    a = l2_normalize(np.asarray(frames_a, dtype=np.float32), axis=1)
    b = l2_normalize(np.asarray(frames_b, dtype=np.float32), axis=1)
    if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
        raise ValueError("both inputs must be non-empty (n_frames, dim) arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}")
    return float(np.max(a @ b.T))
