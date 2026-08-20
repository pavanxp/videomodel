"""Stages 1-2: load the delivery manifest and validate every row.

A row that cannot be trusted never enters the pipeline. It is written to
``rejected_clips.csv`` with the reason instead, so the delivery's own accounting
gaps stay visible rather than quietly shrinking the denominator.

Accepts CSV or JSON. CSV is read with the stdlib :mod:`csv` module and JSON with
:mod:`json`; neither needs pandas, which keeps the core install light.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

__all__ = [
    "ClipRecord",
    "RejectedClip",
    "IngestResult",
    "REQUIRED_COLUMNS",
    "load_manifest",
    "write_rejected_csv",
]

#: Every column a manifest row must carry a non-empty value for.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "item_id",
    "session_id",
    "parent_video_id",
    "worker_id",
    "timestamp",
    "clip_path",
)


class ClipRecord(BaseModel):
    """One validated clip. ``duration_seconds`` is optional metadata."""

    model_config = ConfigDict(extra="allow", frozen=True)

    item_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    parent_video_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    clip_path: Path
    duration_seconds: float | None = None

    @field_validator("item_id", "session_id", "parent_video_id", "worker_id",
                     "timestamp", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _blank_is_none(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    @field_validator("duration_seconds")
    @classmethod
    def _positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("duration_seconds must be positive when present")
        return v

    @field_validator("timestamp")
    @classmethod
    def _parseable(cls, v: str) -> str:
        """Accept ISO-8601 or a bare unix epoch; reject anything unparseable.

        The timestamp is what lets downstream quota fixes reason about *when*
        redundant footage was collected, so an unusable value is a defect.
        """
        try:
            float(v)
            return v
        except ValueError:
            pass
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"timestamp is neither ISO-8601 nor a unix epoch: {v!r}") from exc
        return v


@dataclass(frozen=True, slots=True)
class RejectedClip:
    """A manifest row that failed validation, and why."""

    row_number: int
    item_id: str
    reason: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IngestResult:
    clips: list[ClipRecord]
    rejected: list[RejectedClip]

    @property
    def total_rows(self) -> int:
        return len(self.clips) + len(self.rejected)


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            # A short row leaves trailing keys as None; normalise to "" so the
            # missing-field check below reports them uniformly.
            yield {k: ("" if v is None else v) for k, v in row.items() if k is not None}


def _read_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Accept either a bare list of records or {"clips": [...]} / {"items": [...]}.
    if isinstance(payload, dict):
        for key in ("clips", "items", "records", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            raise ValueError(
                f"{path}: object manifest must hold a list under one of "
                f"'clips', 'items', 'records' or 'data'"
            )
    if not isinstance(payload, list):
        raise ValueError(f"{path}: manifest must be a list of records")
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: every manifest record must be an object")
        yield row


def _missing_fields(row: dict[str, Any]) -> list[str]:
    missing = []
    for col in REQUIRED_COLUMNS:
        val = row.get(col)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(col)
    return missing


def _describe(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def load_manifest(path: str | Path, *, require_existing_files: bool = False,
                  input_dir: Path | None = None) -> IngestResult:
    """Read and validate a manifest.

    ``require_existing_files`` additionally rejects rows whose ``clip_path`` is
    not on disk. Off by default so a manifest can be validated before the media
    is staged; the pipeline turns it on.

    Duplicate ``item_id`` values are rejected after the first occurrence: the
    id keys every cache entry downstream, so a repeat would silently overwrite
    another clip's frames and embeddings. Repeated ids are also exactly the
    source-accounting defect this pipeline is meant to surface.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows: Iterable[dict[str, Any]] = _read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        rows = _read_json(path)
    else:
        raise ValueError(f"unsupported manifest type {suffix!r}; expected .csv or .json")

    clips: list[ClipRecord] = []
    rejected: list[RejectedClip] = []
    seen: dict[str, int] = {}

    for n, row in enumerate(rows, start=1):
        item_id = str(row.get("item_id") or "").strip()

        missing = _missing_fields(row)
        if missing:
            rejected.append(RejectedClip(n, item_id, f"missing required field(s): {', '.join(missing)}", row))
            continue

        if item_id in seen:
            rejected.append(RejectedClip(
                n, item_id, f"duplicate item_id (first seen at row {seen[item_id]})", row))
            continue

        try:
            clip = ClipRecord(**row)
        except ValidationError as exc:
            rejected.append(RejectedClip(n, item_id, _describe(exc), row))
            continue

        # A relative clip_path is anchored to input_dir, so moving the media
        # is a settings change rather than a manifest rewrite. Absolute paths
        # are left alone.
        if input_dir is not None and not clip.clip_path.is_absolute():
            clip = clip.model_copy(update={"clip_path": input_dir / clip.clip_path})

        if require_existing_files and not clip.clip_path.is_file():
            rejected.append(RejectedClip(n, item_id, f"clip_path not found: {clip.clip_path}", row))
            continue

        seen[item_id] = n
        clips.append(clip)

    log.info("ingest: %d row(s) read -> %d valid, %d rejected",
             len(clips) + len(rejected), len(clips), len(rejected))
    for rej in rejected[:10]:
        log.debug("ingest: row %d (%s) rejected -- %s", rej.row_number, rej.item_id or "<no id>", rej.reason)
    if len(rejected) > 10:
        log.debug("ingest: ... %d more rejected row(s)", len(rejected) - 10)

    return IngestResult(clips=clips, rejected=rejected)


def write_rejected_csv(rejected: Sequence[RejectedClip], dest: str | Path) -> Path:
    """Write the rejected rows to CSV. Always writes, even when empty.

    An empty file with a header is a positive statement that nothing was
    dropped; an absent file is ambiguous.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_number", "item_id", "reason"])
        for rej in rejected:
            writer.writerow([rej.row_number, rej.item_id, rej.reason])
    log.info("ingest: wrote %d rejected row(s) -> %s", len(rejected), dest)
    return dest
