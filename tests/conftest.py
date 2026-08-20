"""Shared synthetic fixtures.

Nothing here touches a real video, a real model or a real API. Clips are tiny
generated MP4s (or bare JPEG frames where the test does not need decoding), and
the vision judge is always a stub.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from visual_diversity.config import PipelineConfig

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch, tmp_path_factory):
    """Keep every test away from the developer's real ``.env`` and VD_* vars.

    Without this the settings loader walks up the tree, finds the repository's
    real credential file and seeds it into the run -- so results depend on whose
    machine is running, a real secret sits one failed assertion from the output,
    and a stray ``VD_OUTPUT_DIR`` would redirect where tests write.
    """
    from visual_diversity.settings import ENV_FILE_VAR, clear_settings_cache, get_settings

    for name in ("VD_INPUT_DIR", "VD_OUTPUT_DIR", "VD_CACHE_DIR", "VD_MANIFEST",
                 "VD_S3_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_FILE_VAR,
                       str(tmp_path_factory.mktemp("noenv") / "absent.env"))
    get_settings(reload=True)
    yield
    clear_settings_cache()


# --------------------------------------------------------------------------
# Images and frames
# --------------------------------------------------------------------------
def write_jpeg(path: Path, colour: tuple[int, int, int], size: int = 32) -> Path:
    """A flat single-colour JPEG. Same colour -> byte-identical file."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    return d


# --------------------------------------------------------------------------
# Synthetic video clips
# --------------------------------------------------------------------------
def make_clip(path: Path, colour: str = "red", seconds: float = 2.0,
              size: str = "64x64", rate: int = 10) -> Path:
    """Render a solid-colour test clip with ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi",
         "-i", f"color=c={colour}:s={size}:r={rate}:d={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    return path


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------
MANIFEST_COLUMNS = ["item_id", "session_id", "parent_video_id", "worker_id",
                    "timestamp", "clip_path"]


def write_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys([*MANIFEST_COLUMNS, *(k for r in rows for k in r)]))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def manifest_row(item_id: str, clip_path: Path, *, worker: str = "w1",
                 session: str = "s1", parent: str = "pv1",
                 timestamp: str = "2026-08-01T10:00:00Z") -> dict:
    return {
        "item_id": item_id,
        "session_id": session,
        "parent_video_id": parent,
        "worker_id": worker,
        "timestamp": timestamp,
        "clip_path": str(clip_path),
    }


@pytest.fixture
def mini_delivery(tmp_path: Path) -> dict:
    """Eight clips: three identical reds (a duplicate cluster), then five distinct.

    Returns {"manifest": Path, "root": Path, "duplicates": [item_ids]}.
    """
    if not FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    root = tmp_path / "delivery"
    clips_dir = root / "clips"
    rows = []

    # One redundant shoot, split across two workers so attribution is testable.
    duplicates = ["dup_a", "dup_b", "dup_c"]
    for i, item in enumerate(duplicates):
        p = make_clip(clips_dir / f"{item}.mp4", colour="red", seconds=1.5)
        rows.append(manifest_row(item, p, worker="w_redundant" if i < 2 else "w_mixed",
                                 session="s_redundant"))

    for i, colour in enumerate(["blue", "green", "yellow", "magenta", "white"]):
        item = f"uniq_{colour}"
        p = make_clip(clips_dir / f"{item}.mp4", colour=colour, seconds=1.5)
        rows.append(manifest_row(item, p, worker="w_mixed" if i % 2 else "w_clean",
                                 session=f"s_{i}"))

    manifest = write_manifest(root / "manifest.csv", rows)
    return {"manifest": manifest, "root": root, "duplicates": duplicates,
            "clip_count": len(rows)}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@pytest.fixture
def stub_config(tmp_path: Path) -> PipelineConfig:
    """Defaults, rewired for an offline test run: stub embedder, small caches."""
    return PipelineConfig(
        output_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        log_level="WARNING",
        embeddings={"backend": "stub"},
        frames={"count": 3, "workers": 1, "resize": (32, 32)},
        search={"top_k": 5},
    )


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------
@pytest.fixture
def toy_vectors() -> dict[str, np.ndarray]:
    """Frame embeddings for five clips: a=b (identical), c near them, d/e far."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 8)).astype(np.float32)
    near = base + rng.normal(scale=0.01, size=(3, 8)).astype(np.float32)
    return {
        "a": base.copy(),
        "b": base.copy(),
        "c": near,
        "d": rng.normal(size=(3, 8)).astype(np.float32),
        "e": rng.normal(size=(3, 8)).astype(np.float32),
    }
