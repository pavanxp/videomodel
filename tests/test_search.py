"""Stages 7-8: index construction and neighbour search."""

from __future__ import annotations

import numpy as np
import pytest

from visual_diversity.config import SearchConfig
from visual_diversity.pooling import ClipVectors, pool_clips
from visual_diversity.search import (NumpyIndex, VectorIndex, build_index, faiss_available,
                                     search_neighbors)


def _orthogonal(n: int, dim: int = 8) -> ClipVectors:
    matrix = np.eye(n, dim, dtype=np.float32)
    return ClipVectors([f"c{i}" for i in range(n)], matrix)


def test_numpy_index_satisfies_the_protocol():
    assert isinstance(NumpyIndex(4), VectorIndex)


def test_numpy_index_ranks_by_inner_product():
    vectors = _orthogonal(4)
    index = NumpyIndex(vectors.dim)
    index.add(vectors.matrix)

    sims, idx = index.search(vectors.matrix[:1], k=4)

    # Row 0 is its own nearest neighbour at similarity 1.
    assert idx[0, 0] == 0
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert sims[0, 1:].max() == pytest.approx(0.0, abs=1e-6)


def test_numpy_index_chunking_matches_a_single_pass():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 6)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    whole = NumpyIndex(6, chunk=1000)
    whole.add(matrix)
    chunked = NumpyIndex(6, chunk=7)
    chunked.add(matrix)

    s1, i1 = whole.search(matrix, k=5)
    s2, i2 = chunked.search(matrix, k=5)

    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, atol=1e-6)


def test_empty_index_raises_on_search():
    with pytest.raises(RuntimeError, match="empty"):
        NumpyIndex(4).search(np.zeros((1, 4), dtype=np.float32), k=1)


def test_build_index_rejects_zero_clips():
    with pytest.raises(ValueError, match="zero clips"):
        build_index(ClipVectors([], np.zeros((0, 0), dtype=np.float32)), SearchConfig())


def test_search_excludes_self_matches(toy_vectors):
    vectors = pool_clips(toy_vectors)
    cfg = SearchConfig(top_k=3)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    for item_id, row in table.neighbors.items():
        assert item_id not in [n.item_id for n in row]
        assert len(row) <= 3


def test_identical_clips_are_each_others_top_neighbour(toy_vectors):
    vectors = pool_clips(toy_vectors)
    cfg = SearchConfig(top_k=4)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    assert table.top("a").item_id == "b"
    assert table.top("b").item_id == "a"
    assert table.max_similarity("a") == pytest.approx(1.0, abs=1e-4)


def test_neighbours_are_sorted_by_descending_similarity(toy_vectors):
    vectors = pool_clips(toy_vectors)
    cfg = SearchConfig(top_k=4)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    for row in table.neighbors.values():
        sims = [n.similarity for n in row]
        assert sims == sorted(sims, reverse=True)


def test_single_clip_has_no_neighbours():
    vectors = _orthogonal(1)
    cfg = SearchConfig(top_k=5)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    assert table.neighbors == {"c0": []}
    assert table.max_similarity("c0") == 0.0


def test_unique_pairs_deduplicates_and_orders(toy_vectors):
    vectors = pool_clips(toy_vectors)
    cfg = SearchConfig(top_k=4)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    pairs = table.unique_pairs()
    keys = [(a, b) for a, b, _ in pairs]

    assert len(keys) == len(set(keys))
    assert all(a < b for a, b in keys)


def test_similarity_is_clamped_to_one(toy_vectors):
    vectors = pool_clips(toy_vectors)
    cfg = SearchConfig(top_k=4)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    for row in table.neighbors.values():
        for n in row:
            assert -1.0 <= n.similarity <= 1.0


def test_top_k_larger_than_the_delivery_is_safe():
    vectors = _orthogonal(3)
    cfg = SearchConfig(top_k=50)
    table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    assert all(len(row) == 2 for row in table.neighbors.values())


@pytest.mark.skipif(not faiss_available(), reason="faiss not installed")
def test_faiss_flat_agrees_with_numpy(toy_vectors):
    vectors = pool_clips(toy_vectors)
    cfg = SearchConfig(top_k=3, index_type="flat")

    faiss_table = search_neighbors(build_index(vectors, cfg), vectors, cfg)

    numpy_index = NumpyIndex(vectors.dim)
    numpy_index.add(vectors.matrix)
    numpy_table = search_neighbors(numpy_index, vectors, cfg)

    for item_id in vectors.item_ids:
        assert ([n.item_id for n in faiss_table.neighbors[item_id]]
                == [n.item_id for n in numpy_table.neighbors[item_id]])
