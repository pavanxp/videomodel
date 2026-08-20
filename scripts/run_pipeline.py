#!/usr/bin/env python3
"""CLI wrapper that works from a checkout, without installing the package.

    python scripts/run_pipeline.py --manifest manifest.csv \
        --config config/pipeline_config.yaml --output-dir ./results [--force]

Once installed (``pip install -e .``) the same CLI is on PATH as
``visual-diversity``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running straight from the repo.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visual_diversity.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
