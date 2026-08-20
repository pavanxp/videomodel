"""Stage 13: report payload and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_diversity.clustering import cluster_clips
from visual_diversity.config import PipelineConfig
from visual_diversity.ingest import ClipRecord, RejectedClip
from visual_diversity.report import MAX_TABLE_ROWS, build_payload, write_markdown, write_reports
from visual_diversity.scoring import score_delivery
from visual_diversity.search import Neighbor, NeighborTable

from conftest import manifest_row


def table(**rows: list[tuple[str, float]]) -> NeighborTable:
    return NeighborTable({k: [Neighbor(i, s) for i, s in v] for k, v in rows.items()})


def clip(item_id: str, worker: str = "w1") -> ClipRecord:
    return ClipRecord(**manifest_row(item_id, Path("/tmp/x.mp4"), worker=worker))


@pytest.fixture
def scored(tmp_path: Path):
    nt = table(a=[("b", 0.99), ("c", 0.99)], b=[("a", 0.99)], c=[("a", 0.99)], d=[("a", 0.1)])
    ids = ["a", "b", "c", "d"]
    clips = [clip("a", "bad"), clip("b", "bad"), clip("c", "bad"), clip("d", "good")]
    cfg = PipelineConfig(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    clusters = cluster_clips(nt, cfg.clustering.similarity_threshold, ids)
    score = score_delivery(clips, nt, clusters, cfg.scoring, item_ids=ids)
    return {"score": score, "clusters": clusters, "clips": clips, "cfg": cfg}


def test_payload_summary_reconciles(scored):
    payload = build_payload(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=[], cfg=scored["cfg"])
    s = payload["summary"]

    assert s["clips_scored"] == 4
    assert s["clips_ingested"] == 4
    assert s["duplicate_clusters"] == 1
    assert s["clips_in_duplicate_clusters"] == 3
    assert s["manifest_rows"] == s["clips_ingested"] + s["clips_rejected"]


def test_payload_records_the_config_that_produced_it(scored):
    payload = build_payload(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=[], cfg=scored["cfg"])
    cfg = payload["config"]

    assert cfg["similarity_threshold"] == scored["cfg"].clustering.similarity_threshold
    assert cfg["penalty_exponent"] == scored["cfg"].scoring.penalty_exponent
    assert "embedding_backend" in cfg


def test_rejected_rows_reach_the_payload(scored):
    rejected = [RejectedClip(3, "bad_row", "missing required field(s): worker_id", {})]
    payload = build_payload(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=rejected, cfg=scored["cfg"])

    assert payload["summary"]["clips_rejected"] == 1
    assert payload["rejected_clips"][0]["reason"].startswith("missing required")


def test_clip_scores_are_ordered_least_novel_first(scored):
    payload = build_payload(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=[], cfg=scored["cfg"])
    novelties = [c["novelty"] for c in payload["clip_scores"]]
    assert novelties == sorted(novelties)


def test_write_reports_emits_both_files(scored, tmp_path: Path):
    outputs = write_reports(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=[], cfg=scored["cfg"])

    assert outputs["json"].is_file()
    assert outputs["markdown"].is_file()

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["clips_scored"] == 4


def test_markdown_states_the_score_and_the_clusters(scored, tmp_path: Path):
    payload = build_payload(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=[], cfg=scored["cfg"])
    dest = write_markdown(payload, tmp_path / "r.md")
    text = dest.read_text(encoding="utf-8")

    assert "# Visual diversity report" in text
    assert "## Score:" in text
    assert "Duplicate clusters" in text
    assert "bad" in text  # the worst worker is named


def test_markdown_declares_truncation(tmp_path: Path):
    """A cut-down table must say how much it cut."""
    n = MAX_TABLE_ROWS + 10
    ids = [f"c{i:03d}" for i in range(n * 2)]
    nt = NeighborTable({})
    rows = {}
    for i in range(n):
        a, b = ids[2 * i], ids[2 * i + 1]
        rows[a] = [Neighbor(b, 0.99)]
        rows[b] = [Neighbor(a, 0.99)]
    nt = NeighborTable(rows)

    cfg = PipelineConfig(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    clusters = cluster_clips(nt, cfg.clustering.similarity_threshold, ids)
    clips = [clip(i) for i in ids]
    score = score_delivery(clips, nt, clusters, cfg.scoring, item_ids=ids)

    payload = build_payload(score=score, clusters=clusters, clips=clips, rejected=[], cfg=cfg)
    text = write_markdown(payload, tmp_path / "r.md").read_text(encoding="utf-8")

    assert f"Showing {MAX_TABLE_ROWS} of {n}" in text
    # The JSON keeps every row even though the Markdown does not.
    assert len(payload["duplicate_clusters"]) == n


def test_markdown_handles_a_clean_delivery(tmp_path: Path):
    nt = table(a=[("b", 0.1)], b=[("a", 0.1)])
    cfg = PipelineConfig(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    clusters = cluster_clips(nt, cfg.clustering.similarity_threshold, ["a", "b"])
    clips = [clip("a"), clip("b")]
    score = score_delivery(clips, nt, clusters, cfg.scoring, item_ids=["a", "b"])

    payload = build_payload(score=score, clusters=clusters, clips=clips, rejected=[], cfg=cfg)
    text = write_markdown(payload, tmp_path / "r.md").read_text(encoding="utf-8")

    assert "No clip pair reached the duplicate threshold" in text
    assert "Every manifest row validated" in text


def test_markdown_flags_an_empty_run(tmp_path: Path):
    from visual_diversity.clustering import ClusterSet
    from visual_diversity.scoring import DiversityScore

    cfg = PipelineConfig(output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    score = DiversityScore(score=0.0, max_points=15.0, avg_novelty=0.0, cluster_penalty=0.0,
                           total_clips=0, duplicate_clusters=0, clipped_clips=0)

    payload = build_payload(score=score, clusters=ClusterSet(), clips=[], rejected=[], cfg=cfg)
    text = write_markdown(payload, tmp_path / "r.md").read_text(encoding="utf-8")

    assert "No clips reached the scoring stage" in text


def test_borderline_section_appears_only_when_reviewed(scored, tmp_path: Path):
    payload = build_payload(score=scored["score"], clusters=scored["clusters"],
                            clips=scored["clips"], rejected=[], cfg=scored["cfg"],
                            borderline={"enabled": True, "considered": 2, "reviewed": 2,
                                        "skipped_for_cap": 0, "errors": 0, "merged": 1,
                                        "verdicts": [{"item_a": "a", "item_b": "d",
                                                      "similarity": 0.85,
                                                      "verdict": "REDUNDANT",
                                                      "confidence": 0.9,
                                                      "reason": "same bench"}]})
    text = write_markdown(payload, tmp_path / "r.md").read_text(encoding="utf-8")

    assert "## Borderline review" in text
    assert "same bench" in text
