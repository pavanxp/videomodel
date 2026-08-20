"""Stages 4-5: turn frames into vectors.

Three backends behind one :class:`Embedder` protocol:

  ``dinov2``  facebook/dinov2-base CLS token (768-dim) -- the default, and what
              the core score is calibrated on.
  ``clip``    openai/clip-vit-large-patch14 image tower -- a semantic second
              opinion. Optional, and never feeds the core score.
  ``stub``    a deterministic hash of the decoded pixels. No torch, no network.
              Visually identical frames land on identical vectors and unrelated
              frames land far apart, which is all the pipeline's own tests need.

torch/transformers are imported lazily inside the backend that needs them, so
the package installs and the suite runs without the ``[embed]`` extra.

Vectors are cached to ``<cache>/embeddings/<backend>/<item_id>.npy`` keyed by
item and frame index; a re-run reloads instead of recomputing.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import EmbeddingsConfig
from .frames import ClipFrames

log = logging.getLogger(__name__)

__all__ = [
    "Embedder",
    "StubEmbedder",
    "HFImageEmbedder",
    "build_embedder",
    "build_secondary_embedder",
    "build_driving_embedder",
    "embed_clips",
    "resolve_device",
    "STUB_DIM",
]

#: Stub vectors are small: they exist to exercise wiring, not to be informative.
STUB_DIM = 64


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of image paths into a (n, dim) float32 array."""

    name: str
    dim: int

    def embed_paths(self, paths: Sequence[Path]) -> np.ndarray:
        ...


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device, degrading to CPU with a warning rather than failing."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if preference == "cuda":
            log.warning("device: cuda requested but torch is not installed; using cpu")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        log.warning("device: cuda requested but no CUDA device is visible; falling back to cpu")
    else:
        log.info("device: no CUDA device visible; using cpu")
    return "cpu"


class StubEmbedder:
    """Deterministic content hash of the decoded image. No model, no network.

    Not a perceptual embedding: it is stable under re-encoding of the same
    pixels and otherwise pseudo-random. That is exactly the property the
    pipeline's tests need -- identical frames cluster, different frames do not.
    """

    name = "stub"
    dim = STUB_DIM

    def embed_paths(self, paths: Sequence[Path]) -> np.ndarray:
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        for i, path in enumerate(paths):
            out[i] = self._vector(path)
        return out

    def _vector(self, path: Path) -> np.ndarray:
        try:
            from PIL import Image

            with Image.open(path) as img:
                # Downsample hard so JPEG noise does not separate identical
                # frames. Kept in RGB: luminance alone collapses distinct
                # colours onto each other (pure red and mid green have almost
                # the same grey value), which would fabricate duplicates.
                small = img.convert("RGB").resize((16, 16))
                payload = small.tobytes()
        except Exception:  # noqa: BLE001 -- unreadable frame: fall back to the bytes
            payload = path.read_bytes() if path.is_file() else path.name.encode()

        # Stretch the digest to dim floats deterministically.
        buf = bytearray()
        counter = 0
        while len(buf) < self.dim * 4:
            buf += hashlib.sha256(payload + counter.to_bytes(4, "little")).digest()
            counter += 1
        vec = np.frombuffer(bytes(buf[: self.dim * 4]), dtype=np.uint32).astype(np.float32)
        vec = vec / np.float32(2**32) - np.float32(0.5)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


class HFImageEmbedder:
    """DINOv2 or CLIP image tower via HuggingFace ``transformers``.

    Imports torch/transformers on construction, not at module import, so the
    rest of the package stays usable without the ``[embed]`` extra.
    """

    def __init__(self, model_name: str, *, kind: str = "dinov2",
                 batch_size: int = 32, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                f"the '{kind}' backend needs torch and transformers. "
                f"Install the extra:  pip install 'visual-diversity[embed]'"
            ) from exc

        # The cache is keyed by this name, so it has to identify the *weights*,
        # not just the family: a primary and a secondary both reporting "clip"
        # with different model_names would share one cache directory and serve
        # each other's vectors.
        self.name = f"{kind}_{model_name.replace('/', '_')}"
        self.kind = kind
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._torch = torch

        log.info("embeddings: loading %s (%s) on %s", model_name, kind, self.device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        cfg = self._model.config
        # CLIP wraps two towers; the image side carries its own hidden size.
        vision_cfg = getattr(cfg, "vision_config", cfg)
        self.dim = int(getattr(vision_cfg, "hidden_size", getattr(cfg, "hidden_size", 768)))

    def embed_paths(self, paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start: start + self.batch_size]
            images = []
            for p in batch:
                with Image.open(p) as img:
                    images.append(img.convert("RGB").copy())
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                chunks.append(self._forward(inputs))
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def _forward(self, inputs) -> np.ndarray:
        """One batch -> (n, dim). CLIP and DINOv2 expose different heads."""
        if self.kind == "clip":
            feats = self._model.get_image_features(**inputs)
        else:
            out = self._model(**inputs)
            # DINOv2's CLS token is position 0 of the last hidden state.
            feats = out.last_hidden_state[:, 0, :]
        return feats.detach().cpu().numpy()


def build_embedder(cfg: EmbeddingsConfig) -> Embedder:
    """Construct the configured primary embedder."""
    if cfg.backend == "stub":
        log.info("embeddings: using the deterministic stub backend (no model)")
        return StubEmbedder()
    return HFImageEmbedder(cfg.model_name, kind=cfg.backend,
                           batch_size=cfg.batch_size, device=cfg.device)


def build_secondary_embedder(cfg: EmbeddingsConfig) -> Embedder | None:
    """The optional CLIP second opinion, or None when disabled."""
    if not cfg.secondary_clip.enabled:
        return None
    log.info("embeddings: secondary CLIP pass enabled (%s)", cfg.secondary_clip.model_name)
    return HFImageEmbedder(cfg.secondary_clip.model_name, kind="clip",
                           batch_size=cfg.batch_size, device=cfg.device)


def build_driving_embedder(cfg: EmbeddingsConfig) -> tuple[Embedder, Embedder | None]:
    """Return (driver, side_pass) according to ``cfg.driver``.

    The driver is what feeds pooling, search, clustering and scoring. The side
    pass, when present, is embedded and cached but never consumed by the score.

    With ``driver='secondary'`` the primary is never constructed, so no DINOv2
    weights are downloaded or loaded -- the point of the setting is to run CLIP
    alone, not to run both and ignore one.
    """
    if cfg.driver == "secondary":
        secondary = build_secondary_embedder(cfg)
        if secondary is None:  # pragma: no cover - the config validator forbids it
            raise RuntimeError("driver='secondary' but secondary_clip is disabled")
        log.info("embeddings: the secondary CLIP pass is driving the score; "
                 "the primary '%s' backend is not loaded", cfg.backend)
        return secondary, None

    return build_embedder(cfg), build_secondary_embedder(cfg)


def _cache_path(cache_dir: Path, backend: str, item_id: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in item_id)
    return cache_dir / "embeddings" / backend / f"{safe or '_unnamed'}.npy"


def embed_clips(frames: dict[str, ClipFrames], embedder: Embedder, cache_dir: Path,
                force: bool = False) -> dict[str, np.ndarray]:
    """Embed every usable clip's frames. Returns {item_id: (n_frames, dim) array}.

    Cache validity requires the stored frame count to match what the frame stage
    produced -- otherwise a changed ``frames.count`` would silently reuse stale
    vectors of the wrong length.
    """
    cache_dir = Path(cache_dir)
    root = cache_dir / "embeddings" / embedder.name
    root.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    reused = 0
    computed = 0

    for item_id, cf in frames.items():
        if not cf.ok:
            continue
        path = _cache_path(cache_dir, embedder.name, item_id)

        if not force and path.is_file():
            try:
                cached = np.load(path)
                if cached.ndim == 2 and cached.shape[0] == len(cf.paths):
                    out[item_id] = cached.astype(np.float32, copy=False)
                    reused += 1
                    continue
                log.debug("embeddings: %s cache shape %s does not match %d frame(s); recomputing",
                          item_id, cached.shape, len(cf.paths))
            except Exception as exc:  # noqa: BLE001 -- a corrupt cache entry is recoverable
                log.debug("embeddings: unreadable cache for %s (%s); recomputing", item_id, exc)

        try:
            vecs = embedder.embed_paths(list(cf.paths))
        except Exception as exc:  # noqa: BLE001 -- one clip must not kill the stage
            log.warning("embeddings: %s failed -- %s", item_id, exc)
            continue

        if vecs.ndim != 2 or vecs.shape[0] != len(cf.paths):
            log.warning("embeddings: %s produced %s for %d frame(s); skipped",
                        item_id, vecs.shape, len(cf.paths))
            continue

        np.save(path, vecs)
        out[item_id] = vecs
        computed += 1

    log.info("embeddings: %d clip(s) embedded with %s (%d reused, %d computed, dim=%d)",
             len(out), embedder.name, reused, computed,
             next(iter(out.values())).shape[1] if out else 0)
    return out
