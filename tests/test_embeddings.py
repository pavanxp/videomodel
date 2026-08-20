"""Stages 4-5: embedder backends, caching and device fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visual_diversity.config import EmbeddingsConfig
from visual_diversity.embeddings import (STUB_DIM, Embedder, StubEmbedder, build_driving_embedder,
                                         build_embedder, build_secondary_embedder, embed_clips,
                                         resolve_device)
from visual_diversity.frames import ClipFrames

from conftest import write_jpeg


def test_stub_satisfies_the_embedder_protocol():
    assert isinstance(StubEmbedder(), Embedder)


def test_stub_is_deterministic_and_normalised(frame_dir: Path):
    p = write_jpeg(frame_dir / "a.jpg", (10, 20, 30))
    e = StubEmbedder()

    first = e.embed_paths([p])
    second = e.embed_paths([p])

    assert first.shape == (1, STUB_DIM)
    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0, atol=1e-5)


def test_identical_images_share_a_vector_and_different_ones_do_not(frame_dir: Path):
    same_a = write_jpeg(frame_dir / "sa.jpg", (200, 10, 10))
    same_b = write_jpeg(frame_dir / "sb.jpg", (200, 10, 10))
    other = write_jpeg(frame_dir / "o.jpg", (10, 10, 200))

    vecs = StubEmbedder().embed_paths([same_a, same_b, other])

    np.testing.assert_allclose(vecs[0], vecs[1], atol=1e-6)
    assert float(vecs[0] @ vecs[2]) < 0.9


def test_empty_input_returns_an_empty_array():
    assert StubEmbedder().embed_paths([]).shape == (0, STUB_DIM)


def test_build_embedder_returns_the_stub_backend():
    assert isinstance(build_embedder(EmbeddingsConfig(backend="stub")), StubEmbedder)


def test_secondary_is_none_when_disabled():
    assert build_secondary_embedder(EmbeddingsConfig(backend="stub")) is None


def test_primary_driver_returns_the_primary_and_no_side_pass():
    driver, side = build_driving_embedder(EmbeddingsConfig(backend="stub"))
    assert isinstance(driver, StubEmbedder)
    assert side is None


def test_secondary_driver_never_constructs_the_primary():
    """driver='secondary' must not load DINOv2 -- the point is CLIP alone.

    transformers is absent here, so the secondary raises; what matters is that
    the error names CLIP, proving the primary was never the thing being built.
    """
    cfg = EmbeddingsConfig(backend="stub", driver="secondary",
                           secondary_clip={"enabled": True,
                                           "model_name": "openai/clip-vit-large-patch14"})
    with pytest.raises(RuntimeError, match="clip"):
        build_driving_embedder(cfg)


def test_cache_is_partitioned_by_embedder_name(frame_dir: Path, tmp_path: Path):
    """Two embedders must not read each other's vectors.

    A primary and a secondary that both reported the family name "clip" would
    share one cache directory and serve each other's weights' output.
    """
    frames = {"a": _frames(frame_dir, "a", [(5, 5, 5)])}
    cache = tmp_path / "cache"

    class Alpha(StubEmbedder):
        name = "clip_model-alpha"

    class Beta(StubEmbedder):
        name = "clip_model-beta"
        dim = STUB_DIM

        def embed_paths(self, paths):
            return np.ones((len(paths), self.dim), dtype=np.float32)

    embed_clips(frames, Alpha(), cache)
    beta_out = embed_clips(frames, Beta(), cache)

    assert (cache / "embeddings" / "clip_model-alpha" / "a.npy").is_file()
    assert (cache / "embeddings" / "clip_model-beta" / "a.npy").is_file()
    # Beta got its own vectors, not Alpha's cached ones.
    np.testing.assert_array_equal(beta_out["a"], np.ones((1, STUB_DIM), dtype=np.float32))


def test_resolve_device_never_raises():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    # A cuda request on a CPU-only box warns and degrades rather than failing.
    assert resolve_device("cuda") in {"cpu", "cuda"}


def _frames(frame_dir: Path, item: str, colours: list[tuple[int, int, int]]) -> ClipFrames:
    paths = tuple(write_jpeg(frame_dir / item / f"frame_{i:02d}.jpg", c)
                  for i, c in enumerate(colours))
    return ClipFrames(item, paths)


def test_embed_clips_shapes_and_skips_failed_clips(frame_dir: Path, tmp_path: Path):
    frames = {
        "ok": _frames(frame_dir, "ok", [(1, 2, 3), (4, 5, 6)]),
        "broken": ClipFrames("broken", (), error="ffmpeg died"),
    }

    out = embed_clips(frames, StubEmbedder(), tmp_path / "cache")

    assert set(out) == {"ok"}
    assert out["ok"].shape == (2, STUB_DIM)


def test_embeddings_are_cached_and_reused(frame_dir: Path, tmp_path: Path):
    frames = {"a": _frames(frame_dir, "a", [(9, 9, 9), (8, 8, 8)])}
    cache = tmp_path / "cache"

    first = embed_clips(frames, StubEmbedder(), cache)
    cached_file = cache / "embeddings" / "stub" / "a.npy"
    assert cached_file.is_file()

    # A counting embedder proves the second run does not recompute.
    class Counting(StubEmbedder):
        calls = 0

        def embed_paths(self, paths):
            Counting.calls += 1
            return super().embed_paths(paths)

    second = embed_clips(frames, Counting(), cache)
    assert Counting.calls == 0
    np.testing.assert_array_equal(first["a"], second["a"])


def test_force_recomputes(frame_dir: Path, tmp_path: Path):
    frames = {"a": _frames(frame_dir, "a", [(9, 9, 9)])}
    cache = tmp_path / "cache"
    embed_clips(frames, StubEmbedder(), cache)

    class Counting(StubEmbedder):
        calls = 0

        def embed_paths(self, paths):
            Counting.calls += 1
            return super().embed_paths(paths)

    embed_clips(frames, Counting(), cache, force=True)
    assert Counting.calls == 1


def test_stale_cache_with_the_wrong_frame_count_is_discarded(frame_dir: Path, tmp_path: Path):
    cache = tmp_path / "cache"
    two = {"a": _frames(frame_dir, "a", [(1, 1, 1), (2, 2, 2)])}
    embed_clips(two, StubEmbedder(), cache)

    three = {"a": _frames(frame_dir, "a3", [(1, 1, 1), (2, 2, 2), (3, 3, 3)])}
    three = {"a": ClipFrames("a", three["a"].paths)}

    out = embed_clips(three, StubEmbedder(), cache)
    assert out["a"].shape[0] == 3


def test_a_failing_embedder_skips_the_clip_rather_than_the_run(frame_dir: Path, tmp_path: Path):
    frames = {
        "bad": _frames(frame_dir, "bad", [(1, 1, 1)]),
        "good": _frames(frame_dir, "good", [(2, 2, 2)]),
    }

    class Flaky(StubEmbedder):
        def embed_paths(self, paths):
            if any("bad" in str(p) for p in paths):
                raise RuntimeError("boom")
            return super().embed_paths(paths)

    out = embed_clips(frames, Flaky(), tmp_path / "cache")
    assert set(out) == {"good"}
