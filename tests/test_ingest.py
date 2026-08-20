"""Stages 1-2: manifest loading and row validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from visual_diversity import ingest
from visual_diversity.ingest import REQUIRED_COLUMNS, load_manifest, write_rejected_csv

from conftest import manifest_row, write_manifest


def test_valid_rows_load(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("a", clip)])

    result = load_manifest(manifest)

    assert len(result.clips) == 1
    assert not result.rejected
    assert result.clips[0].item_id == "a"
    assert result.clips[0].clip_path == clip


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_each_required_field_is_enforced(tmp_path: Path, missing: str):
    clip = tmp_path / "a.mp4"
    clip.touch()
    row = manifest_row("a", clip)
    row[missing] = ""
    manifest = write_manifest(tmp_path / "m.csv", [row])

    result = load_manifest(manifest)

    assert not result.clips
    assert len(result.rejected) == 1
    assert missing in result.rejected[0].reason


def test_duplicate_item_ids_are_rejected_after_the_first(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    manifest = write_manifest(tmp_path / "m.csv", [
        manifest_row("same", clip), manifest_row("same", clip), manifest_row("other", clip)])

    result = load_manifest(manifest)

    assert [c.item_id for c in result.clips] == ["same", "other"]
    assert len(result.rejected) == 1
    assert "duplicate item_id" in result.rejected[0].reason


def test_unparseable_timestamp_is_rejected(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    manifest = write_manifest(tmp_path / "m.csv",
                              [manifest_row("a", clip, timestamp="last tuesday")])

    result = load_manifest(manifest)

    assert not result.clips
    assert "timestamp" in result.rejected[0].reason


def test_unix_epoch_timestamp_is_accepted(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    manifest = write_manifest(tmp_path / "m.csv",
                              [manifest_row("a", clip, timestamp="1784876817.88")])

    assert len(load_manifest(manifest).clips) == 1


def test_require_existing_files_rejects_absent_media(tmp_path: Path):
    manifest = write_manifest(tmp_path / "m.csv",
                              [manifest_row("a", tmp_path / "nope.mp4")])

    assert len(load_manifest(manifest).clips) == 1
    result = load_manifest(manifest, require_existing_files=True)
    assert not result.clips
    assert "not found" in result.rejected[0].reason


def test_json_manifest_list_and_wrapped(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    rows = [manifest_row("a", clip)]

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(rows), encoding="utf-8")
    assert len(load_manifest(bare).clips) == 1

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"clips": rows}), encoding="utf-8")
    assert len(load_manifest(wrapped).clips) == 1


def test_unsupported_extension_raises(tmp_path: Path):
    bad = tmp_path / "m.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported manifest type"):
        load_manifest(bad)


def test_missing_manifest_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "absent.csv")


def test_rejected_csv_is_written_even_when_empty(tmp_path: Path):
    dest = write_rejected_csv([], tmp_path / "rejected.csv")
    rows = list(csv.reader(dest.open(encoding="utf-8")))
    assert rows == [["row_number", "item_id", "reason"]]


def test_rejected_csv_carries_reasons(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    manifest = write_manifest(tmp_path / "m.csv",
                              [manifest_row("", clip), manifest_row("ok", clip)])
    result = load_manifest(manifest)

    dest = write_rejected_csv(result.rejected, tmp_path / "rejected.csv")
    rows = list(csv.DictReader(dest.open(encoding="utf-8")))

    assert len(rows) == 1
    assert "item_id" in rows[0]["reason"]


def test_optional_duration_is_parsed_and_validated(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.touch()
    good = manifest_row("a", clip) | {"duration_seconds": "12.5"}
    bad = manifest_row("b", clip) | {"duration_seconds": "-3"}
    blank = manifest_row("c", clip) | {"duration_seconds": ""}
    manifest = write_manifest(tmp_path / "m.csv", [good, bad, blank])

    result = load_manifest(manifest)
    by_id = {c.item_id: c for c in result.clips}

    assert by_id["a"].duration_seconds == 12.5
    assert by_id["c"].duration_seconds is None
    assert [r.item_id for r in result.rejected] == ["b"]
