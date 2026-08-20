"""Stage 6: pooling and the alternative max-frame-pair comparison."""

from __future__ import annotations

import numpy as np
import pytest

from visual_diversity.pooling import (ClipVectors, l2_normalize, max_frame_pair_similarity,
                                      mean_pool, pool_clips, single_frame)


def test_l2_normalize_gives_unit_rows():
    out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32), axis=1)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_l2_normalize_leaves_a_zero_vector_alone():
    out = l2_normalize(np.zeros((1, 4), dtype=np.float32), axis=1)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out, np.zeros((1, 4), dtype=np.float32))


def test_mean_pool_averages_then_normalises():
    frames = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pooled = mean_pool(frames)
    assert pooled.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(pooled), 1.0, atol=1e-6)
    np.testing.assert_allclose(pooled, [0.70710678, 0.70710678], atol=1e-5)


def test_mean_pool_rejects_empty_input():
    with pytest.raises(ValueError):
        mean_pool(np.zeros((0, 4), dtype=np.float32))


def test_pool_clips_orders_ids_and_matches_rows(toy_vectors):
    vectors = pool_clips(toy_vectors)
    assert vectors.item_ids == sorted(toy_vectors)
    assert vectors.matrix.shape == (5, 8)
    np.testing.assert_allclose(np.linalg.norm(vectors.matrix, axis=1),
                               np.ones(5), atol=1e-6)


def test_identical_clips_pool_to_the_same_vector(toy_vectors):
    vectors = pool_clips(toy_vectors)
    np.testing.assert_allclose(vectors.vector_for("a"), vectors.vector_for("b"), atol=1e-6)
    assert float(vectors.vector_for("a") @ vectors.vector_for("b")) > 0.999


def test_pool_clips_on_empty_input():
    vectors = pool_clips({})
    assert len(vectors) == 0
    assert vectors.dim == 0


def test_mixed_dimensionality_is_an_error():
    with pytest.raises(ValueError, match="mixed dimensionality"):
        pool_clips({"a": np.zeros((2, 4), dtype=np.float32),
                    "b": np.zeros((2, 8), dtype=np.float32)})


def test_clip_vectors_rejects_a_mismatched_length():
    with pytest.raises(ValueError, match="must match"):
        ClipVectors(["a", "b"], np.zeros((3, 4), dtype=np.float32))


def test_single_frame_takes_the_frame_unaveraged():
    frames = np.array([[3.0, 4.0], [100.0, 0.0]], dtype=np.float32)
    out = single_frame(frames)
    # Direction of frame 0 only -- frame 1 must not pull it.
    np.testing.assert_allclose(out, [0.6, 0.8], atol=1e-6)


def test_single_frame_still_normalises():
    """Pooling is bypassed; normalisation is not. Inner-product search only
    equals cosine similarity on unit vectors."""
    out = single_frame(np.array([[3.0, 4.0]], dtype=np.float32))
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)


def test_single_frame_rejects_empty():
    with pytest.raises(ValueError):
        single_frame(np.zeros((0, 4), dtype=np.float32))


def test_pool_clips_mode_none_bypasses_averaging():
    frames = {"a": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}

    mean = pool_clips(frames, mode="mean").vector_for("a")
    bypassed = pool_clips(frames, mode="none").vector_for("a")

    np.testing.assert_allclose(mean, [0.70710678, 0.70710678], atol=1e-5)
    np.testing.assert_allclose(bypassed, [1.0, 0.0], atol=1e-6)


def test_pool_clips_mode_none_matches_mean_at_one_frame():
    """With a single frame the two modes must agree -- the intended setup."""
    frames = {"a": np.array([[2.0, 5.0, 1.0]], dtype=np.float32)}
    np.testing.assert_allclose(pool_clips(frames, mode="none").matrix,
                               pool_clips(frames, mode="mean").matrix, atol=1e-6)


def test_pool_clips_mode_none_warns_when_frames_are_discarded(caplog):
    frames = {"a": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    with caplog.at_level("WARNING"):
        pool_clips(frames, mode="none")
    assert "keeps only the first frame" in caplog.text


def test_maxpair_is_rejected_rather_than_silently_mean_pooling(toy_vectors):
    """It is a pair comparator, not a clip reducer -- falling back to mean would
    report a mode the run did not actually use."""
    with pytest.raises(ValueError, match="pairwise comparison"):
        pool_clips(toy_vectors, mode="maxpair")


def test_unknown_pooling_mode_is_rejected(toy_vectors):
    with pytest.raises(ValueError, match="unknown pooling mode"):
        pool_clips(toy_vectors, mode="median")


def test_pool_clips_defaults_to_mean(toy_vectors):
    np.testing.assert_allclose(pool_clips(toy_vectors).matrix,
                               pool_clips(toy_vectors, mode="mean").matrix, atol=1e-6)


def test_max_frame_pair_finds_a_shared_moment_that_mean_pooling_hides():
    dim = 8
    shared = np.zeros(dim, dtype=np.float32)
    shared[0] = 1.0
    a_only = np.zeros(dim, dtype=np.float32)
    a_only[1] = 1.0
    b_only = np.zeros(dim, dtype=np.float32)
    b_only[2] = 1.0

    # Each clip is mostly its own content plus one identical frame.
    a = np.vstack([a_only, a_only, shared])
    b = np.vstack([b_only, b_only, shared])

    pooled_similarity = float(mean_pool(a) @ mean_pool(b))
    pair_similarity = max_frame_pair_similarity(a, b)

    assert pair_similarity == pytest.approx(1.0, abs=1e-5)
    assert pooled_similarity < 0.5
    assert pair_similarity > pooled_similarity


def test_max_frame_pair_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        max_frame_pair_similarity(np.zeros((2, 4), dtype=np.float32),
                                  np.zeros((2, 8), dtype=np.float32))


def test_max_frame_pair_rejects_empty():
    with pytest.raises(ValueError):
        max_frame_pair_similarity(np.zeros((0, 4), dtype=np.float32),
                                  np.zeros((2, 4), dtype=np.float32))
