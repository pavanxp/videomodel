"""Stage 12: gray-zone selection, sampling, judging and cluster correction.

The OpenAI client is never constructed: every test injects a fake judge, and the
parser is exercised directly against representative reply shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_diversity.borderline_review import (BorderlinePair, PairVerdict, ReviewOutcome,
                                                _parse_verdict, _representative,
                                                apply_corrections, review_pairs, sample_pairs,
                                                select_gray_zone_pairs)
from visual_diversity.clustering import cluster_clips
from visual_diversity.config import BorderlineReviewConfig
from visual_diversity.frames import ClipFrames
from visual_diversity.search import Neighbor, NeighborTable

from conftest import write_jpeg


def table(**rows: list[tuple[str, float]]) -> NeighborTable:
    return NeighborTable({k: [Neighbor(i, s) for i, s in v] for k, v in rows.items()})


def frames_for(tmp_path: Path, *items: str, n: int = 3) -> dict[str, ClipFrames]:
    out = {}
    for k, item in enumerate(items):
        paths = tuple(write_jpeg(tmp_path / item / f"f{i}.jpg", (k * 40 % 255, i * 10, 7))
                      for i in range(n))
        out[item] = ClipFrames(item, paths)
    return out


# --------------------------------------------------------------------------
# Selection and sampling
# --------------------------------------------------------------------------
def test_only_gray_zone_pairs_are_selected():
    nt = table(a=[("b", 0.85)], b=[("a", 0.85), ("c", 0.95)], c=[("b", 0.95), ("d", 0.5)],
               d=[("c", 0.5)])
    cfg = BorderlineReviewConfig(gray_zone=(0.80, 0.91))

    pairs = select_gray_zone_pairs(nt, cfg)

    assert [(p.item_a, p.item_b) for p in pairs] == [("a", "b")]


def test_gray_zone_bounds_are_low_inclusive_high_exclusive():
    nt = table(a=[("b", 0.80)], b=[("a", 0.80), ("c", 0.91)], c=[("b", 0.91)])
    pairs = select_gray_zone_pairs(nt, BorderlineReviewConfig(gray_zone=(0.80, 0.91)))
    assert [(p.item_a, p.item_b) for p in pairs] == [("a", "b")]


def test_sampling_is_a_noop_under_the_cap():
    pairs = [BorderlinePair("a", "b", 0.85), BorderlinePair("c", "d", 0.87)]
    kept, skipped = sample_pairs(pairs, cap=10)
    assert kept == pairs
    assert skipped == 0


def test_sampling_respects_the_cap_and_reports_the_remainder():
    pairs = [BorderlinePair(f"a{i}", f"b{i}", 0.80 + i * 0.001) for i in range(100)]
    kept, skipped = sample_pairs(pairs, cap=10, seed=1)
    assert len(kept) == 10
    assert skipped == 90


def test_sampling_spans_the_zone_rather_than_taking_the_top():
    """A proportional sample must reach the low end of the gray zone."""
    pairs = [BorderlinePair(f"a{i}", f"b{i}", 0.80 + i * 0.001) for i in range(100)]
    kept, _ = sample_pairs(pairs, cap=12, seed=1)
    sims = [p.similarity for p in kept]
    assert min(sims) < 0.83
    assert max(sims) > 0.87


def test_sampling_is_deterministic_for_a_seed():
    pairs = [BorderlinePair(f"a{i}", f"b{i}", 0.80 + i * 0.001) for i in range(60)]
    first, _ = sample_pairs(pairs, cap=9, seed=42)
    second, _ = sample_pairs(pairs, cap=9, seed=42)
    assert first == second


def test_zero_cap_reviews_nothing():
    pairs = [BorderlinePair("a", "b", 0.85)]
    kept, skipped = sample_pairs(pairs, cap=0)
    assert kept == []
    assert skipped == 1


# --------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("reply,expected", [
    ('{"verdict":"REDUNDANT","confidence":0.9,"reason":"same bench"}', "REDUNDANT"),
    ('{"verdict":"DISTINCT","confidence":0.7,"reason":"different tool"}', "DISTINCT"),
    ('```json\n{"verdict": "REDUNDANT", "confidence": 0.8, "reason": "x"}\n```', "REDUNDANT"),
    ('Sure! {"verdict":"DISTINCT","confidence":0.6,"reason":"y"} hope that helps',
     "DISTINCT"),
    ("These look REDUNDANT to me.", "REDUNDANT"),
    ("no idea", "UNKNOWN"),
])
def test_parse_verdict_handles_real_reply_shapes(reply: str, expected: str):
    verdict, confidence, _ = _parse_verdict(reply)
    assert verdict == expected
    assert 0.0 <= confidence <= 1.0


def test_parse_verdict_clamps_a_silly_confidence():
    _, confidence, _ = _parse_verdict('{"verdict":"DISTINCT","confidence":9.5}')
    assert confidence == 1.0


# --------------------------------------------------------------------------
# Representative frame choice
# --------------------------------------------------------------------------
def test_representative_takes_the_middle_frame(tmp_path: Path):
    cf = frames_for(tmp_path, "a", n=5)["a"]
    assert _representative(cf, 1) == [cf.paths[2]]


def test_representative_spreads_across_the_clip(tmp_path: Path):
    cf = frames_for(tmp_path, "a", n=5)["a"]
    picked = _representative(cf, 3)
    assert picked == [cf.paths[0], cf.paths[2], cf.paths[4]]


# --------------------------------------------------------------------------
# The stage end to end, with a fake judge
# --------------------------------------------------------------------------
def test_review_pairs_uses_the_injected_judge(tmp_path: Path):
    nt = table(a=[("b", 0.85)], b=[("a", 0.85)])
    frames = frames_for(tmp_path, "a", "b")
    seen = []

    def judge(pair, fa, fb):
        seen.append(pair)
        return PairVerdict(pair, "REDUNDANT", 0.9, "same work")

    outcome = review_pairs(nt, frames, BorderlineReviewConfig(gray_zone=(0.8, 0.91)),
                           judge=judge)

    assert len(seen) == 1
    assert outcome.reviewed == 1
    assert len(outcome.merged_pairs) == 1


def test_no_gray_zone_pairs_short_circuits(tmp_path: Path):
    nt = table(a=[("b", 0.99)], b=[("a", 0.99)])

    def judge(pair, fa, fb):  # pragma: no cover - must not be called
        raise AssertionError("judge should not run")

    outcome = review_pairs(nt, frames_for(tmp_path, "a", "b"),
                           BorderlineReviewConfig(gray_zone=(0.8, 0.91)), judge=judge)

    assert outcome.considered == 0
    assert outcome.reviewed == 0


def test_missing_frames_count_as_an_error_not_a_crash(tmp_path: Path):
    nt = table(a=[("b", 0.85)], b=[("a", 0.85)])
    frames = {"a": frames_for(tmp_path, "a")["a"],
              "b": ClipFrames("b", (), error="extraction failed")}

    outcome = review_pairs(nt, frames, BorderlineReviewConfig(gray_zone=(0.8, 0.91)),
                           judge=lambda p, fa, fb: PairVerdict(p, "REDUNDANT", 1.0, ""))

    assert outcome.errors == 1
    assert outcome.reviewed == 0


def test_unknown_verdicts_are_counted_and_do_not_merge(tmp_path: Path):
    nt = table(a=[("b", 0.85)], b=[("a", 0.85)])
    outcome = review_pairs(nt, frames_for(tmp_path, "a", "b"),
                           BorderlineReviewConfig(gray_zone=(0.8, 0.91)),
                           judge=lambda p, fa, fb: PairVerdict(p, "UNKNOWN", 0.0, "api down"))

    assert outcome.errors == 1
    assert outcome.merged_pairs == []


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------
def test_redundant_verdict_merges_a_pair_the_threshold_missed():
    nt = table(a=[("b", 0.85)], b=[("a", 0.85)])
    ids = ["a", "b"]
    before = cluster_clips(nt, 0.9, ids)
    assert before.duplicate_clusters == []

    outcome = ReviewOutcome(
        [PairVerdict(BorderlinePair("a", "b", 0.85), "REDUNDANT", 0.9, "same")], 1, 1, 0, 0)

    after = apply_corrections(nt, before, outcome, 0.9, ids)

    assert len(after.duplicate_clusters) == 1
    assert set(after.duplicate_clusters[0].members) == {"a", "b"}


def test_distinct_verdict_changes_nothing():
    nt = table(a=[("b", 0.85)], b=[("a", 0.85)])
    ids = ["a", "b"]
    before = cluster_clips(nt, 0.9, ids)

    outcome = ReviewOutcome(
        [PairVerdict(BorderlinePair("a", "b", 0.85), "DISTINCT", 0.9, "different")], 1, 1, 0, 0)

    after = apply_corrections(nt, before, outcome, 0.9, ids)

    assert after.duplicate_clusters == before.duplicate_clusters


def test_a_forced_merge_is_transitive_with_existing_clusters():
    """Merging A-B must also pull in whatever B was already chained to."""
    nt = table(a=[("b", 0.85)], b=[("a", 0.85), ("c", 0.95)], c=[("b", 0.95)])
    ids = ["a", "b", "c"]
    before = cluster_clips(nt, 0.9, ids)
    assert set(before.duplicate_clusters[0].members) == {"b", "c"}

    outcome = ReviewOutcome(
        [PairVerdict(BorderlinePair("a", "b", 0.85), "REDUNDANT", 0.9, "same")], 1, 1, 0, 0)
    after = apply_corrections(nt, before, outcome, 0.9, ids)

    assert len(after.duplicate_clusters) == 1
    assert set(after.duplicate_clusters[0].members) == {"a", "b", "c"}


def test_no_verdicts_leaves_clusters_untouched():
    nt = table(a=[("b", 0.95)], b=[("a", 0.95)])
    ids = ["a", "b"]
    before = cluster_clips(nt, 0.9, ids)

    after = apply_corrections(nt, before, ReviewOutcome([], 0, 0, 0, 0), 0.9, ids)

    assert after is before
