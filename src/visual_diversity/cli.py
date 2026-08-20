"""Command-line entry point.

Installed as ``visual-diversity``; also reachable as
``python scripts/run_pipeline.py`` without installing.

Exit codes:
    0  the pipeline ran and a report was written
    1  the run produced no scoreable clip
    2  usage / configuration error (bad path, invalid config)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import configure_logging, run_pipeline

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visual-diversity",
        description="Score a delivery of egocentric clips for visual diversity.",
    )
    p.add_argument("--manifest", default=None,
                   help="clip manifest (.csv or .json) with item_id, session_id, "
                        "parent_video_id, worker_id, timestamp, clip_path. "
                        "Defaults to VD_MANIFEST / DEFAULT_MANIFEST in settings.py")
    p.add_argument("--config", default=None,
                   help="pipeline YAML config (default: built-in defaults)")
    p.add_argument("--output-dir", default=None,
                   help="where reports are written (overrides the config)")
    p.add_argument("--force", action="store_true",
                   help="ignore cached frames and embeddings and recompute everything")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="override the config's log level")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 -- any config problem is a usage error
        print(f"error: could not load config: {exc}", file=sys.stderr)
        return EXIT_USAGE

    configure_logging(args.log_level or cfg.log_level)

    from .settings import get_settings

    manifest_arg = args.manifest or get_settings().paths.manifest
    if manifest_arg is None:
        print("error: no manifest. Pass --manifest, set VD_MANIFEST, or set "
              "DEFAULT_MANIFEST in visual_diversity/settings.py", file=sys.stderr)
        return EXIT_USAGE

    manifest = Path(manifest_arg).expanduser()
    if not manifest.is_file():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = run_pipeline(manifest, args.config, args.output_dir, force=args.force)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:
        # e.g. ffmpeg missing, or an embedding extra that is not installed.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print()
    print(f"Visual diversity: {result.score.score:.2f} / {result.score.max_points:.0f}")
    print(f"  clips ingested        {result.ingested}")
    print(f"  clips rejected        {result.rejected}")
    print(f"  clips scored          {result.scored}")
    print(f"  duplicate clusters    {result.score.duplicate_clusters}")
    print(f"  clips in clusters     {result.score.clipped_clips}")
    for name, path in result.outputs.items():
        print(f"  {name:<20}  {path}")

    if result.scored == 0:
        print("\nwarning: no clip reached scoring; the score is not meaningful.",
              file=sys.stderr)
        return EXIT_EMPTY
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
