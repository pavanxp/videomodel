"""End-to-end runs over a synthetic mini-delivery.

Real ffmpeg, real caching, real reports -- but the stub embedder and a fake
vision judge, so no model download and no OpenAI call. The pipeline is driven
through its public entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from visual_diversity.borderline_review import PairVerdict
from visual_diversity.cli import main
from visual_diversity.pipeline import run_pipeline

from conftest import make_clip, manifest_row, requires_ffmpeg, write_manifest


def write_config(path: Path, tmp_path: Path, **overrides) -> Path:
    body = {
        "output_dir": str(tmp_path / "results"),
        "cache_dir": str(tmp_path / "cache"),
        "log_level": "WARNING",
        "embeddings": {"backend": "stub"},
        "frames": {"count": 3, "workers": 2, "resize": [32, 32]},
        "search": {"top_k": 5},
        "clustering": {"similarity_threshold": 0.99, "flag_cluster_size": 2},
        "report": {"thumbnails": True},
    }
    body.update(overrides)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


@requires_ffmpeg
def test_full_run_scores_and_reports(mini_delivery, tmp_path: Path):
    config = write_config(tmp_path / "c.yaml", tmp_path)

    result = run_pipeline(mini_delivery["manifest"], config)

    assert result.ingested == mini_delivery["clip_count"]
    assert result.rejected == 0
    assert result.scored == mini_delivery["clip_count"]

    # The three identical red clips must land in one cluster.
    dupes = result.clusters.duplicate_clusters
    assert len(dupes) == 1
    assert set(dupes[0].members) == set(mini_delivery["duplicates"])

    assert 0.0 <= result.score.score <= result.score.max_points
    for key in ("json", "markdown", "rejected_csv"):
        assert result.outputs[key].is_file()


@requires_ffmpeg
def test_single_frame_no_pooling_run(mini_delivery, tmp_path: Path):
    """The delivered setup: one frame at 50%, pooling bypassed.

    The secondary CLIP driver cannot run offline, so the stub stands in for it;
    what this covers is the frames.count=1 + pooling.mode=none path that the
    driver setting is paired with.
    """
    config = write_config(
        tmp_path / "c.yaml", tmp_path,
        frames={"count": 1, "workers": 2, "resize": [32, 32]},
        pooling={"mode": "none"},
    )

    result = run_pipeline(mini_delivery["manifest"], config)

    assert result.scored == mini_delivery["clip_count"]
    # One frame per clip, taken at the midpoint.
    for item in mini_delivery["duplicates"]:
        assert len(list((tmp_path / "cache" / "frames" / item).glob("frame_*.jpg"))) == 1

    # Identical clips still cluster with pooling switched off.
    dupes = result.clusters.duplicate_clusters
    assert len(dupes) == 1
    assert set(dupes[0].members) == set(mini_delivery["duplicates"])


@requires_ffmpeg
def test_bypassed_pooling_still_yields_unit_vectors(mini_delivery, tmp_path: Path):
    """Similarities must stay in [-1, 1]; the thresholds depend on it."""
    config = write_config(
        tmp_path / "c.yaml", tmp_path,
        frames={"count": 1, "workers": 1, "resize": [32, 32]},
        pooling={"mode": "none"},
    )
    result = run_pipeline(mini_delivery["manifest"], config)

    for s in result.score.clip_scores:
        assert -1.0 <= s.max_similarity <= 1.0
        assert 0.0 <= s.novelty <= 1.0


@requires_ffmpeg
def test_at_one_frame_pooling_modes_agree(mini_delivery, tmp_path: Path):
    """mean and none must produce the same score when there is one frame."""
    common = {"frames": {"count": 1, "workers": 1, "resize": [32, 32]}}
    mean_cfg = write_config(tmp_path / "mean.yaml", tmp_path / "m", pooling={"mode": "mean"},
                            **common)
    none_cfg = write_config(tmp_path / "none.yaml", tmp_path / "n", pooling={"mode": "none"},
                            **common)

    mean_run = run_pipeline(mini_delivery["manifest"], mean_cfg)
    none_run = run_pipeline(mini_delivery["manifest"], none_cfg)

    assert none_run.score.score == pytest.approx(mean_run.score.score)


@requires_ffmpeg
def test_settings_paths_drive_the_run(mini_delivery, tmp_path: Path, monkeypatch):
    """VD_* settings redirect output, cache and manifest without touching config."""
    from visual_diversity.settings import get_settings

    config = write_config(tmp_path / "c.yaml", tmp_path)   # points at tmp_path/results
    monkeypatch.setenv("VD_OUTPUT_DIR", str(tmp_path / "settings_out"))
    monkeypatch.setenv("VD_CACHE_DIR", str(tmp_path / "settings_cache"))
    monkeypatch.setenv("VD_MANIFEST", str(mini_delivery["manifest"]))
    get_settings(reload=True)

    # No manifest argument: it comes from settings.
    result = run_pipeline(None, config)

    assert result.outputs["json"].parent == tmp_path / "settings_out"
    assert (tmp_path / "settings_cache" / "frames").is_dir()
    assert not (tmp_path / "results").exists()


@requires_ffmpeg
def test_explicit_output_dir_beats_settings(mini_delivery, tmp_path: Path, monkeypatch):
    from visual_diversity.settings import get_settings

    config = write_config(tmp_path / "c.yaml", tmp_path)
    monkeypatch.setenv("VD_OUTPUT_DIR", str(tmp_path / "from_settings"))
    get_settings(reload=True)

    result = run_pipeline(mini_delivery["manifest"], config,
                          output_dir=tmp_path / "from_argument")

    assert result.outputs["json"].parent == tmp_path / "from_argument"


@requires_ffmpeg
def test_input_dir_anchors_relative_clip_paths(mini_delivery, tmp_path: Path, monkeypatch):
    """Move the media, change one setting -- no manifest rewrite."""
    import csv

    from visual_diversity.settings import get_settings

    rows = list(csv.DictReader(mini_delivery["manifest"].open(encoding="utf-8")))
    clips_root = Path(rows[0]["clip_path"]).parent
    for r in rows:                                   # absolute -> bare filename
        r["clip_path"] = Path(r["clip_path"]).name
    relative_manifest = write_manifest(tmp_path / "relative.csv", rows)

    monkeypatch.setenv("VD_INPUT_DIR", str(clips_root))
    get_settings(reload=True)

    config = write_config(tmp_path / "c.yaml", tmp_path)
    result = run_pipeline(relative_manifest, config)

    assert result.rejected == 0
    assert result.scored == mini_delivery["clip_count"]


@requires_ffmpeg
def test_relative_paths_fail_without_an_input_dir(mini_delivery, tmp_path: Path):
    """The counterpart: unanchored relative paths are rejected, not guessed at."""
    import csv

    rows = list(csv.DictReader(mini_delivery["manifest"].open(encoding="utf-8")))
    for r in rows:
        r["clip_path"] = Path(r["clip_path"]).name
    relative_manifest = write_manifest(tmp_path / "relative.csv", rows)

    result = run_pipeline(relative_manifest, write_config(tmp_path / "c.yaml", tmp_path))

    assert result.rejected == mini_delivery["clip_count"]
    assert result.scored == 0


def test_no_manifest_anywhere_is_a_clear_error(tmp_path: Path):
    with pytest.raises(ValueError, match="no manifest"):
        run_pipeline(None, None)


@requires_ffmpeg
def test_reports_reconcile_with_the_result(mini_delivery, tmp_path: Path):
    config = write_config(tmp_path / "c.yaml", tmp_path)
    result = run_pipeline(mini_delivery["manifest"], config)

    payload = json.loads(result.outputs["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["score"] == pytest.approx(round(result.score.score, 4))
    assert payload["summary"]["clips_scored"] == result.scored
    assert len(payload["duplicate_clusters"]) == len(result.clusters.duplicate_clusters)

    markdown = result.outputs["markdown"].read_text(encoding="utf-8")
    assert f"{result.score.score:.2f}" in markdown


@requires_ffmpeg
def test_redundancy_is_attributed_to_the_redundant_worker(mini_delivery, tmp_path: Path):
    config = write_config(tmp_path / "c.yaml", tmp_path)
    result = run_pipeline(mini_delivery["manifest"], config)

    worst = result.score.by_worker[0]
    assert worst.key == "w_redundant"
    assert worst.clustered_clips == 2

    clean = {r.key: r for r in result.score.by_worker}["w_clean"]
    assert clean.is_clean


@requires_ffmpeg
def test_thumbnails_are_emitted_for_clusters(mini_delivery, tmp_path: Path):
    config = write_config(tmp_path / "c.yaml", tmp_path)
    result = run_pipeline(mini_delivery["manifest"], config)

    payload = json.loads(result.outputs["json"].read_text(encoding="utf-8"))
    thumb = payload["duplicate_clusters"][0].get("thumbnail")

    assert thumb is not None
    assert (result.outputs["json"].parent / thumb).is_file()


@requires_ffmpeg
def test_rerun_reuses_the_cache_and_agrees(mini_delivery, tmp_path: Path):
    config = write_config(tmp_path / "c.yaml", tmp_path)

    first = run_pipeline(mini_delivery["manifest"], config)
    cache_files = sorted((tmp_path / "cache").rglob("*.npy"))
    assert cache_files

    second = run_pipeline(mini_delivery["manifest"], config)

    assert second.score.score == pytest.approx(first.score.score)
    assert second.scored == first.scored


@requires_ffmpeg
def test_force_recomputes_and_still_agrees(mini_delivery, tmp_path: Path):
    config = write_config(tmp_path / "c.yaml", tmp_path)
    first = run_pipeline(mini_delivery["manifest"], config)
    forced = run_pipeline(mini_delivery["manifest"], config, force=True)
    assert forced.score.score == pytest.approx(first.score.score)


@requires_ffmpeg
def test_bad_rows_are_rejected_without_stopping_the_run(mini_delivery, tmp_path: Path):
    import csv

    rows = list(csv.DictReader(mini_delivery["manifest"].open(encoding="utf-8")))
    rows.append(manifest_row("no_worker", Path(rows[0]["clip_path"])) | {"worker_id": ""})
    rows.append(manifest_row("absent_media", tmp_path / "nope.mp4"))
    manifest = write_manifest(tmp_path / "m2.csv", rows)

    config = write_config(tmp_path / "c.yaml", tmp_path)
    result = run_pipeline(manifest, config)

    assert result.rejected == 2
    assert result.scored == mini_delivery["clip_count"]

    import csv as _csv
    reasons = list(_csv.DictReader(result.outputs["rejected_csv"].open(encoding="utf-8")))
    assert {r["item_id"] for r in reasons} == {"no_worker", "absent_media"}


@requires_ffmpeg
def test_an_unreadable_clip_is_skipped_not_fatal(mini_delivery, tmp_path: Path):
    import csv

    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    rows = list(csv.DictReader(mini_delivery["manifest"].open(encoding="utf-8")))
    rows.append(manifest_row("junk", junk))
    manifest = write_manifest(tmp_path / "m3.csv", rows)

    config = write_config(tmp_path / "c.yaml", tmp_path)
    result = run_pipeline(manifest, config)

    # It passed ingest (the file exists) but produced no frames, so it is not scored.
    assert result.ingested == mini_delivery["clip_count"] + 1
    assert result.scored == mini_delivery["clip_count"]


@requires_ffmpeg
def test_borderline_stage_runs_with_a_mocked_judge(mini_delivery, tmp_path: Path):
    """Stage 12 wired end to end, with the OpenAI call replaced by a fake."""
    config = write_config(
        tmp_path / "c.yaml", tmp_path,
        clustering={"similarity_threshold": 0.999, "flag_cluster_size": 2},
        borderline_review={"enabled": True, "gray_zone": [0.05, 0.999],
                           "max_pairs": 5, "frames_per_clip": 1},
    )

    calls: list[tuple[str, str]] = []

    def judge(pair, frames_a, frames_b):
        calls.append((pair.item_a, pair.item_b))
        assert frames_a and frames_b
        # Merge only the identical red clips.
        redundant = pair.item_a.startswith("dup") and pair.item_b.startswith("dup")
        return PairVerdict(pair, "REDUNDANT" if redundant else "DISTINCT", 0.95, "fake")

    result = run_pipeline(mini_delivery["manifest"], config, judge=judge)

    assert calls, "the judge should have been consulted"
    assert result.borderline["enabled"] is True
    assert result.borderline["reviewed"] <= 5

    if result.borderline["merged"]:
        merged_members = {m for c in result.clusters.duplicate_clusters for m in c.members}
        assert merged_members <= set(mini_delivery["duplicates"])


@requires_ffmpeg
def test_borderline_never_runs_without_an_api_key_when_disabled(mini_delivery, tmp_path: Path,
                                                               monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = write_config(tmp_path / "c.yaml", tmp_path)

    result = run_pipeline(mini_delivery["manifest"], config)

    assert result.borderline == {"enabled": False}


@requires_ffmpeg
def test_a_borderline_failure_does_not_sink_the_run(mini_delivery, tmp_path: Path):
    config = write_config(
        tmp_path / "c.yaml", tmp_path,
        clustering={"similarity_threshold": 0.999, "flag_cluster_size": 2},
        borderline_review={"enabled": True, "gray_zone": [0.05, 0.999], "max_pairs": 3},
    )

    def judge(pair, fa, fb):
        raise RuntimeError("API exploded")

    result = run_pipeline(mini_delivery["manifest"], config, judge=judge)

    # The stage caught it, recorded it, and the score still exists.
    assert result.scored == mini_delivery["clip_count"]
    assert result.outputs["json"].is_file()


@requires_ffmpeg
def test_empty_manifest_writes_a_zero_report(tmp_path: Path):
    manifest = write_manifest(tmp_path / "empty.csv", [])
    config = write_config(tmp_path / "c.yaml", tmp_path)

    result = run_pipeline(manifest, config)

    assert result.scored == 0
    assert result.score.score == 0.0
    assert result.outputs["json"].is_file()
    text = result.outputs["markdown"].read_text(encoding="utf-8")
    assert "No clips reached the scoring stage" in text


@requires_ffmpeg
def test_cli_end_to_end(mini_delivery, tmp_path: Path, capsys):
    config = write_config(tmp_path / "c.yaml", tmp_path)

    code = main(["--manifest", str(mini_delivery["manifest"]),
                 "--config", str(config),
                 "--output-dir", str(tmp_path / "cli_out")])

    assert code == 0
    out = capsys.readouterr().out
    assert "Visual diversity:" in out
    assert (tmp_path / "cli_out" / "visual_diversity_report.json").is_file()


def test_cli_reports_a_missing_manifest(tmp_path: Path, capsys):
    code = main(["--manifest", str(tmp_path / "nope.csv")])
    assert code == 2
    assert "manifest not found" in capsys.readouterr().err


@requires_ffmpeg
def test_cli_signals_an_empty_run(tmp_path: Path, capsys):
    manifest = write_manifest(tmp_path / "empty.csv", [])
    config = write_config(tmp_path / "c.yaml", tmp_path)

    code = main(["--manifest", str(manifest), "--config", str(config)])

    assert code == 1
    assert "no clip reached scoring" in capsys.readouterr().err
