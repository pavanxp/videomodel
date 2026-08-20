"""Stage 13: write the scorecard and the duplicate-cluster report.

Three artefacts:

  ``visual_diversity_report.json``  the machine-readable result
  ``visual_diversity_report.md``    the same content for a human reader
  ``rejected_clips.csv``            written by the ingest stage, referenced here

Every figure in the Markdown comes from the same JSON payload, so the two cannot
drift. Where a table omits rows for length, it says how many it omitted -- a
truncated list that does not admit it reads as a complete one.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .clustering import ClusterSet
from .config import PipelineConfig
from .frames import ClipFrames
from .ingest import ClipRecord, RejectedClip
from .scoring import DiversityScore, GroupRedundancy

log = logging.getLogger(__name__)

__all__ = ["build_payload", "write_json", "write_markdown", "write_reports", "MAX_TABLE_ROWS"]

#: Rows shown per table in the Markdown before truncating (JSON is never cut).
MAX_TABLE_ROWS = 25


def _cluster_payload(clusters: ClusterSet, cfg: PipelineConfig,
                     thumbs: Mapping[str, str] | None) -> list[dict[str, Any]]:
    out = []
    for c in sorted(clusters.duplicate_clusters, key=lambda x: (-x.size, x.members[0])):
        entry: dict[str, Any] = {
            "cluster_id": c.cluster_id,
            "size": c.size,
            "members": list(c.members),
            "max_similarity": round(c.max_similarity, 4),
            "mean_similarity": round(c.mean_similarity, 4),
            "flagged": c.size >= cfg.clustering.flag_cluster_size,
        }
        if thumbs and str(c.cluster_id) in thumbs:
            entry["thumbnail"] = thumbs[str(c.cluster_id)]
        out.append(entry)
    return out


def _group_payload(rows: Sequence[GroupRedundancy]) -> list[dict[str, Any]]:
    return [{
        "key": r.key,
        "clips": r.clips,
        "clustered_clips": r.clustered_clips,
        "redundancy_rate": round(r.redundancy_rate, 4),
        "share_of_delivery_redundancy": round(r.share_of_delivery_redundancy, 4),
        "mean_novelty": round(r.mean_novelty, 4),
    } for r in rows]


def _copy_thumbnails(clusters: ClusterSet, frames: Mapping[str, ClipFrames],
                     dest_dir: Path) -> dict[str, str]:
    """One representative frame per duplicate cluster, copied beside the report."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for c in clusters.duplicate_clusters:
        cf = frames.get(c.members[0])
        if not (cf and cf.ok):
            continue
        src = cf.paths[len(cf.paths) // 2]
        dst = dest_dir / f"cluster_{c.cluster_id:04d}.jpg"
        try:
            shutil.copyfile(src, dst)
            out[str(c.cluster_id)] = f"thumbnails/{dst.name}"
        except OSError as exc:
            log.debug("report: could not copy thumbnail for cluster %d (%s)", c.cluster_id, exc)
    if out:
        log.info("report: copied %d cluster thumbnail(s)", len(out))
    return out


def build_payload(*, score: DiversityScore, clusters: ClusterSet, clips: Sequence[ClipRecord],
                  rejected: Sequence[RejectedClip], cfg: PipelineConfig,
                  frames: Mapping[str, ClipFrames] | None = None,
                  borderline: dict[str, Any] | None = None,
                  thumbnails: Mapping[str, str] | None = None,
                  timings: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Assemble the single dict both output formats render from."""
    flagged = clusters.flagged_for_removal(cfg.clustering.flag_cluster_size)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            **score.as_dict(),
            "manifest_rows": len(clips) + len(rejected),
            "clips_ingested": len(clips),
            "clips_rejected": len(rejected),
            "clips_scored": score.total_clips,
            "flagged_for_removal": len(flagged),
        },
        "config": {
            "similarity_threshold": cfg.clustering.similarity_threshold,
            "flag_cluster_size": cfg.clustering.flag_cluster_size,
            "penalty_exponent": cfg.scoring.penalty_exponent,
            "max_points": cfg.scoring.max_points,
            "embedding_backend": cfg.embeddings.backend,
            "embedding_model": cfg.embeddings.model_name,
            "index_type": cfg.search.index_type,
            "top_k": cfg.search.top_k,
            "frames_per_clip": cfg.frames.count,
            "pooling_mode": cfg.pooling.mode,
            "borderline_review_enabled": cfg.borderline_review.enabled,
        },
        "duplicate_clusters": _cluster_payload(clusters, cfg, thumbnails),
        "clips_flagged_for_removal": flagged,
        "redundancy_by_worker": _group_payload(score.by_worker),
        "redundancy_by_session": _group_payload(score.by_session),
        "clip_scores": [asdict(s) for s in sorted(score.clip_scores, key=lambda s: s.novelty)],
        "rejected_clips": [
            {"row_number": r.row_number, "item_id": r.item_id, "reason": r.reason}
            for r in rejected
        ],
        "borderline_review": borderline or {"enabled": False},
        "timings_seconds": dict(timings or {}),
    }


def write_json(payload: Mapping[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("report: wrote %s", dest)
    return dest


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not rows:
        return ["_none_", ""]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out.append("")
    return out


def _truncated(rows: Sequence[Any], limit: int = MAX_TABLE_ROWS) -> tuple[Sequence[Any], str]:
    if len(rows) <= limit:
        return rows, ""
    return rows[:limit], f"_Showing {limit} of {len(rows)}; the JSON report carries all of them._\n"


def write_markdown(payload: Mapping[str, Any], dest: Path) -> Path:
    s = payload["summary"]
    cfg = payload["config"]
    lines: list[str] = []

    lines.append("# Visual diversity report")
    lines.append("")
    lines.append(f"_Generated {payload['generated_at']}_")
    lines.append("")

    lines.append(f"## Score: {s['score']:.2f} / {s['max_points']:.0f}")
    lines.append("")
    lines += _table(
        ["Metric", "Value"],
        [["Average novelty", f"{s['avg_novelty']:.4f}"],
         ["Cluster penalty", f"{s['cluster_penalty']:.4f}"],
         ["Clips scored", s["clips_scored"]],
         ["Duplicate clusters", s["duplicate_clusters"]],
         ["Clips in a duplicate cluster", s["clips_in_duplicate_clusters"]],
         ["Clips flagged for removal", s["flagged_for_removal"]],
         ["Manifest rows rejected at ingest", s["clips_rejected"]]],
    )
    lines.append(f"`score = max(0, (avg_novelty - cluster_penalty)) x {s['max_points']:.0f}`, "
                 f"where `cluster_penalty = sum(size ** {cfg['penalty_exponent']}) / clips` "
                 f"over clusters larger than one.")
    lines.append("")

    if s["clips_scored"] == 0:
        lines.append("> No clips reached the scoring stage, so the score is not meaningful. "
                     "Check the rejected-clips list and the frame-extraction log.")
        lines.append("")

    lines.append("## How it was measured")
    lines.append("")
    # The stub backend loads no model, so naming one would misdescribe the run.
    embedding = (cfg["embedding_backend"] if cfg["embedding_backend"] == "stub"
                 else f"{cfg['embedding_backend']} (`{cfg['embedding_model']}`)")
    lines += _table(["Setting", "Value"], [
        ["Embedding", embedding],
        ["Frames per clip", cfg["frames_per_clip"]],
        ["Pooling", cfg["pooling_mode"]],
        ["Index", f"{cfg['index_type']}, top-{cfg['top_k']}"],
        ["Duplicate threshold", cfg["similarity_threshold"]],
        ["Flag clusters at size", cfg["flag_cluster_size"]],
        ["Borderline review", "on" if cfg["borderline_review_enabled"] else "off"],
    ])

    clusters = payload["duplicate_clusters"]
    lines.append(f"## Duplicate clusters ({len(clusters)})")
    lines.append("")
    if clusters:
        shown, note = _truncated(clusters)
        lines += _table(
            ["Cluster", "Size", "Max sim", "Flagged", "Members"],
            [[c["cluster_id"], c["size"], f"{c['max_similarity']:.4f}",
              "yes" if c["flagged"] else "",
              ", ".join(c["members"][:6]) + (" ..." if len(c["members"]) > 6 else "")]
             for c in shown],
        )
        if note:
            lines.append(note)
    else:
        lines.append("_No clip pair reached the duplicate threshold._")
        lines.append("")

    for title, key, label in (("worker", "redundancy_by_worker", "Worker"),
                              ("session", "redundancy_by_session", "Session")):
        rows = [r for r in payload[key] if r["clustered_clips"] > 0]
        lines.append(f"## Redundancy by {title}")
        lines.append("")
        if rows:
            shown, note = _truncated(rows)
            lines += _table(
                [label, "Clips", "Clustered", "Own redundancy", "Share of all redundancy"],
                [[r["key"], r["clips"], r["clustered_clips"],
                  f"{100 * r['redundancy_rate']:.0f}%",
                  f"{100 * r['share_of_delivery_redundancy']:.0f}%"] for r in shown],
            )
            if note:
                lines.append(note)
        else:
            lines.append(f"_No {title} contributed a clustered clip._")
            lines.append("")

    flagged = payload["clips_flagged_for_removal"]
    lines.append(f"## Clips flagged for removal or re-shoot ({len(flagged)})")
    lines.append("")
    if flagged:
        lines.append(f"Members of clusters of {cfg['flag_cluster_size']} or more, "
                     f"excluding one keeper per cluster.")
        lines.append("")
        shown, note = _truncated(flagged, 50)
        lines.append("```")
        lines.extend(str(x) for x in shown)
        lines.append("```")
        if note:
            lines.append(note)
    else:
        lines.append("_Nothing flagged._")
        lines.append("")

    br = payload.get("borderline_review") or {}
    if br.get("reviewed"):
        lines.append("## Borderline review")
        lines.append("")
        lines.append(f"{br['reviewed']} of {br['considered']} gray-zone pair(s) reviewed "
                     f"({br.get('skipped_for_cap', 0)} skipped for the cap, "
                     f"{br.get('errors', 0)} unresolved); "
                     f"{br.get('merged', 0)} merged as redundant.")
        lines.append("")
        shown, note = _truncated(br.get("verdicts", []))
        lines += _table(
            ["A", "B", "Sim", "Verdict", "Conf", "Reason"],
            [[v["item_a"], v["item_b"], f"{v['similarity']:.3f}", v["verdict"],
              f"{v['confidence']:.2f}", v["reason"]] for v in shown],
        )
        if note:
            lines.append(note)

    rejected = payload["rejected_clips"]
    lines.append(f"## Rejected at ingest ({len(rejected)})")
    lines.append("")
    if rejected:
        shown, note = _truncated(rejected)
        lines += _table(["Row", "Item", "Reason"],
                        [[r["row_number"], r["item_id"] or "_(none)_", r["reason"]] for r in shown])
        if note:
            lines.append(note)
        lines.append("Full list: `rejected_clips.csv`.")
        lines.append("")
    else:
        lines.append("_Every manifest row validated._")
        lines.append("")

    timings = payload.get("timings_seconds") or {}
    if timings:
        lines.append("## Stage timings")
        lines.append("")
        lines += _table(["Stage", "Seconds"],
                        [[k, f"{v:.2f}"] for k, v in timings.items()])

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    log.info("report: wrote %s", dest)
    return dest


def write_reports(*, score: DiversityScore, clusters: ClusterSet, clips: Sequence[ClipRecord],
                  rejected: Sequence[RejectedClip], cfg: PipelineConfig,
                  frames: Mapping[str, ClipFrames] | None = None,
                  borderline: dict[str, Any] | None = None,
                  timings: Mapping[str, float] | None = None) -> dict[str, Path]:
    """Write both reports (and thumbnails) into ``cfg.output_dir``."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thumbs = None
    if cfg.report.thumbnails and frames:
        thumbs = _copy_thumbnails(clusters, frames, out_dir / "thumbnails")

    payload = build_payload(score=score, clusters=clusters, clips=clips, rejected=rejected,
                            cfg=cfg, frames=frames, borderline=borderline,
                            thumbnails=thumbs, timings=timings)

    return {
        "json": write_json(payload, out_dir / "visual_diversity_report.json"),
        "markdown": write_markdown(payload, out_dir / "visual_diversity_report.md"),
    }
