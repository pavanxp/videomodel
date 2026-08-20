"""Stage 12: adjudicate gray-zone pairs with a vision model.

Cosine similarity in the 0.80-0.91 band is genuinely ambiguous -- too close to
dismiss, too far apart to call a duplicate. This stage shows both clips'
representative frames to GPT-4o and asks the question the metric cannot answer:
is this *meaningfully different work*, or the same activity from a slightly
different setup?

Placement matters. It runs after stage 9 has clustered and before stage 10
finalises the score, and it only ever *adds* edges -- a REDUNDANT verdict merges
two clips the threshold missed. A DISTINCT verdict changes nothing, because the
pair was below the duplicate threshold to begin with; there is no cluster to
break. That keeps the model's influence monotone and auditable.

Cost is bounded two ways: a hard cap on pairs per run, and proportional sampling
across similarity bands when the gray zone is larger than the cap, so the sample
spans the whole zone rather than crowding one end.

The API key is read from ``OPENAI_API_KEY``. It is never accepted as an argument
and never logged.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from .clustering import ClusterSet, cluster_clips
from .config import BorderlineReviewConfig
from .frames import ClipFrames
from .search import NeighborTable

log = logging.getLogger(__name__)

__all__ = [
    "BorderlinePair",
    "PairVerdict",
    "ReviewOutcome",
    "select_gray_zone_pairs",
    "sample_pairs",
    "review_pairs",
    "apply_corrections",
    "OpenAIVisionJudge",
    "PROMPT",
]

Verdict = Literal["DISTINCT", "REDUNDANT", "UNKNOWN"]

PROMPT = """You are auditing an egocentric (head-mounted camera) video delivery for \
redundancy.

You are shown representative frames from two different clips, A then B.

Decide whether they capture MEANINGFULLY DIFFERENT WORK, or the SAME ACTIVITY \
recorded from a similar angle or setup.

Judge the work being performed, not incidental differences. Treat as REDUNDANT: \
the same task at the same station with only a change of camera angle, lighting, \
object colour, or moment within the same action. Treat as DISTINCT: a different \
task, a different stage of a multi-step workflow, a different workstation, or a \
different tool or material that changes what is being learned.

Respond with a JSON object and nothing else:
{"verdict": "DISTINCT" | "REDUNDANT", "confidence": 0.0-1.0, "reason": "<one short sentence>"}"""


@dataclass(frozen=True, slots=True)
class BorderlinePair:
    item_a: str
    item_b: str
    similarity: float


@dataclass(frozen=True, slots=True)
class PairVerdict:
    pair: BorderlinePair
    verdict: Verdict
    confidence: float
    reason: str

    @property
    def merges(self) -> bool:
        return self.verdict == "REDUNDANT"


@dataclass(slots=True)
class ReviewOutcome:
    verdicts: list[PairVerdict]
    considered: int
    reviewed: int
    skipped_for_cap: int
    errors: int

    @property
    def merged_pairs(self) -> list[BorderlinePair]:
        return [v.pair for v in self.verdicts if v.merges]

    def as_dict(self) -> dict:
        return {
            "considered": self.considered,
            "reviewed": self.reviewed,
            "skipped_for_cap": self.skipped_for_cap,
            "errors": self.errors,
            "merged": len(self.merged_pairs),
            "verdicts": [
                {"item_a": v.pair.item_a, "item_b": v.pair.item_b,
                 "similarity": round(v.pair.similarity, 4), "verdict": v.verdict,
                 "confidence": round(v.confidence, 3), "reason": v.reason}
                for v in self.verdicts
            ],
        }


def select_gray_zone_pairs(neighbors: NeighborTable,
                           cfg: BorderlineReviewConfig) -> list[BorderlinePair]:
    """Pairs whose similarity lands in [low, high). Highest similarity first."""
    low, high = cfg.gray_zone
    pairs = [BorderlinePair(a, b, sim)
             for a, b, sim in neighbors.unique_pairs() if low <= sim < high]
    pairs.sort(key=lambda p: (-p.similarity, p.item_a, p.item_b))
    return pairs


def sample_pairs(pairs: Sequence[BorderlinePair], cap: int,
                 seed: int = 17, bands: int = 4) -> tuple[list[BorderlinePair], int]:
    """Trim to ``cap`` pairs, spread proportionally across similarity bands.

    Taking the top-``cap`` by similarity would sample only the top of the zone,
    where the answer is least in doubt. Splitting the zone into bands and
    drawing from each in proportion keeps the sample representative.

    Returns (kept, skipped).
    """
    if cap <= 0:
        return [], len(pairs)
    if len(pairs) <= cap:
        return list(pairs), 0

    lo = min(p.similarity for p in pairs)
    hi = max(p.similarity for p in pairs)
    width = (hi - lo) or 1e-9

    buckets: list[list[BorderlinePair]] = [[] for _ in range(bands)]
    for p in pairs:
        idx = min(bands - 1, int((p.similarity - lo) / width * bands))
        buckets[idx].append(p)

    rng = random.Random(seed)
    kept: list[BorderlinePair] = []
    # Largest bucket last, so rounding remainders land where there is most to draw from.
    order = sorted(range(bands), key=lambda i: len(buckets[i]))
    remaining_cap = cap
    remaining_total = len(pairs)

    for i in order:
        bucket = buckets[i]
        if not bucket:
            continue
        take = min(len(bucket), max(1, round(remaining_cap * len(bucket) / remaining_total)))
        take = min(take, remaining_cap)
        kept.extend(rng.sample(bucket, take) if take < len(bucket) else bucket)
        remaining_cap -= take
        remaining_total -= len(bucket)
        if remaining_cap <= 0:
            break

    kept.sort(key=lambda p: (-p.similarity, p.item_a, p.item_b))
    return kept, len(pairs) - len(kept)


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_verdict(text: str) -> tuple[Verdict, float, str]:
    """Pull the structured answer out of the reply, tolerating stray prose."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start: end + 1])
            verdict = str(obj.get("verdict", "")).upper()
            if verdict in ("DISTINCT", "REDUNDANT"):
                conf = float(obj.get("confidence", 0.0))
                return verdict, max(0.0, min(1.0, conf)), str(obj.get("reason", ""))[:400]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Bare-word fallback before giving up entirely.
    upper = raw.upper()
    if "REDUNDANT" in upper and "DISTINCT" not in upper:
        return "REDUNDANT", 0.5, "unstructured reply"
    if "DISTINCT" in upper and "REDUNDANT" not in upper:
        return "DISTINCT", 0.5, "unstructured reply"
    return "UNKNOWN", 0.0, f"unparseable reply: {raw[:120]}"


class OpenAIVisionJudge:
    """Calls GPT-4o with both clips' frames and parses the verdict.

    The client is built lazily so importing this module never requires the
    ``[llm]`` extra or an API key.
    """

    def __init__(self, cfg: BorderlineReviewConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        # Credentials come from settings, which seeds itself from a .env when
        # one is present -- so the key is never read from two places.
        from .settings import get_settings

        creds = get_settings().openai.require()
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "the openai package is required for borderline review. "
                "Install:  pip install 'visual-diversity[llm]'"
            ) from exc

        kwargs: dict[str, str] = {"api_key": creds.api_key.get_secret_value()}  # type: ignore[union-attr]
        if creds.base_url:
            kwargs["base_url"] = creds.base_url
        if creds.organization:
            kwargs["organization"] = creds.organization
        self._client = OpenAI(**kwargs)
        return self._client

    def __call__(self, pair: BorderlinePair,
                 frames_a: Sequence[Path], frames_b: Sequence[Path]) -> PairVerdict:
        client = self._ensure_client()

        content: list[dict] = [{"type": "text", "text": PROMPT},
                               {"type": "text", "text": "Clip A:"}]
        for p in frames_a:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_encode(p)}"}})
        content.append({"type": "text", "text": "Clip B:"})
        for p in frames_b:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_encode(p)}"}})

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.cfg.max_output_tokens,
                    temperature=0.0,
                )
                verdict, conf, reason = _parse_verdict(resp.choices[0].message.content or "")
                return PairVerdict(pair, verdict, conf, reason)
            except Exception as exc:  # noqa: BLE001 -- transport/rate-limit/parse
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))

        log.warning("borderline: %s vs %s failed after %d attempt(s) -- %s",
                    pair.item_a, pair.item_b, self.cfg.max_retries + 1, last_error)
        return PairVerdict(pair, "UNKNOWN", 0.0, f"api error: {last_error}")


JudgeFn = Callable[[BorderlinePair, Sequence[Path], Sequence[Path]], PairVerdict]


def review_pairs(neighbors: NeighborTable, frames: Mapping[str, ClipFrames],
                 cfg: BorderlineReviewConfig, *, judge: JudgeFn | None = None,
                 seed: int = 17) -> ReviewOutcome:
    """Select, sample and judge the gray-zone pairs.

    ``judge`` is injectable so tests can exercise the whole stage without an API
    key; the default builds an :class:`OpenAIVisionJudge`.
    """
    candidates = select_gray_zone_pairs(neighbors, cfg)
    if not candidates:
        log.info("borderline: no pairs in the gray zone [%.2f, %.2f)", *cfg.gray_zone)
        return ReviewOutcome([], 0, 0, 0, 0)

    selected, skipped = sample_pairs(candidates, cfg.max_pairs, seed=seed)
    log.info("borderline: %d pair(s) in the gray zone [%.2f, %.2f); reviewing %d, skipping %d "
             "(cap=%d)", len(candidates), *cfg.gray_zone, len(selected), skipped, cfg.max_pairs)

    judge = judge or OpenAIVisionJudge(cfg)
    verdicts: list[PairVerdict] = []
    errors = 0

    for pair in selected:
        fa, fb = frames.get(pair.item_a), frames.get(pair.item_b)
        if not (fa and fa.ok and fb and fb.ok):
            log.debug("borderline: %s vs %s skipped -- frames unavailable",
                      pair.item_a, pair.item_b)
            errors += 1
            continue
        n = cfg.frames_per_clip
        verdict = judge(pair, _representative(fa, n), _representative(fb, n))
        if verdict.verdict == "UNKNOWN":
            errors += 1
        verdicts.append(verdict)

    merged = sum(1 for v in verdicts if v.merges)
    log.info("borderline: %d verdict(s) -- %d REDUNDANT, %d DISTINCT, %d unresolved",
             len(verdicts), merged,
             sum(1 for v in verdicts if v.verdict == "DISTINCT"), errors)

    return ReviewOutcome(verdicts, len(candidates), len(verdicts), skipped, errors)


def _representative(cf: ClipFrames, n: int) -> list[Path]:
    """Pick ``n`` frames spread across the clip, middle frame first.

    The middle of a clip is the most likely to show the work rather than the
    approach or the walk-away.
    """
    paths = list(cf.paths)
    if n >= len(paths):
        return paths
    if n == 1:
        return [paths[len(paths) // 2]]
    step = (len(paths) - 1) / (n - 1)
    return [paths[round(i * step)] for i in range(n)]


def apply_corrections(neighbors: NeighborTable, clusters: ClusterSet,
                      outcome: ReviewOutcome, threshold: float,
                      all_item_ids: Sequence[str]) -> ClusterSet:
    """Re-cluster with the REDUNDANT verdicts forced in as edges.

    Implemented by promoting each merged pair's similarity to the threshold and
    re-running stage 9, rather than by mutating the cluster set. Re-running is
    what keeps transitivity correct: merging A-B can also pull in whatever B was
    already chained to.
    """
    merged = outcome.merged_pairs
    if not merged:
        log.info("borderline: no corrections to apply; clusters unchanged")
        return clusters

    forced = {(p.item_a, p.item_b) if p.item_a < p.item_b else (p.item_b, p.item_a)
              for p in merged}

    patched: dict[str, list] = {k: list(v) for k, v in neighbors.neighbors.items()}
    from .search import Neighbor  # local import: avoids a cycle at module load

    for a, b in forced:
        for src, dst in ((a, b), (b, a)):
            row = patched.setdefault(src, [])
            row = [n for n in row if n.item_id != dst]
            row.append(Neighbor(dst, max(threshold, 1.0 if threshold >= 1.0 else threshold)))
            patched[src] = row

    before = len(clusters.duplicate_clusters)
    corrected = cluster_clips(NeighborTable(patched), threshold, all_item_ids)
    log.info("borderline: %d forced merge(s) took duplicate clusters from %d to %d",
             len(forced), before, len(corrected.duplicate_clusters))
    return corrected
