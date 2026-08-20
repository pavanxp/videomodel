"""Visual-diversity scoring for egocentric video deliveries.

Measures how visually redundant a delivery is and reports the duplicate clusters
driving it, so redundancy can be traced back to the worker and session that
produced it.

Stage map (one module per responsibility):

    ingest             1-2   manifest load + row validation
    frames             3     ffmpeg frame sampling, cached
    embeddings         4-5   DINOv2 / CLIP / stub vectors, cached
    pooling            6     mean-pool + L2 normalise per clip
    search             7-8   FAISS (or numpy) top-k neighbours
    clustering         9     threshold graph -> connected components
    borderline_review  12    GPT-4o adjudication of gray-zone pairs
    scoring            10-11 novelty + cluster penalty -> score
    report             13    JSON + Markdown scorecard
    pipeline           --    orchestration

The heavy dependencies (torch/transformers, faiss, openai) are optional extras.
Without them the pipeline still runs end to end using the stub embedder and the
exact numpy search backend.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import PipelineConfig, load_config
from .pipeline import PipelineResult, configure_logging, run_pipeline
from .settings import Settings, get_settings

__all__ = [
    "__version__",
    "PipelineConfig",
    "load_config",
    "PipelineResult",
    "run_pipeline",
    "configure_logging",
    "Settings",
    "get_settings",
]
