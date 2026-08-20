# visual-diversity

Scores a delivery of egocentric video clips for **visual diversity** — how much
of the footage is genuinely different work versus near-duplicate variations —
and reports the duplicate clusters driving the number, attributed back to the
worker and session that produced them.

The output is a score out of 15 (configurable), a duplicate-cluster report, a
per-worker/session redundancy breakdown, and a list of clips to drop or re-shoot.

---

## Install

Python 3.11+ and `ffmpeg`/`ffprobe` on `PATH`.

```bash
cd egoeval_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # core: runs end to end
pip install -e '.[embed]'             # + DINOv2 / CLIP (torch, transformers)
pip install -e '.[faiss]'             # + approximate search at scale
pip install -e '.[llm]'               # + stage 12 borderline review
pip install -e '.[all]'               # everything, including dev tools
```

The core install deliberately excludes torch, faiss and openai. Without them the
pipeline still runs: search falls back to an exact numpy implementation (same
results as `IndexFlatIP`, just slower), and the `stub` embedding backend needs no
model. **You need `[embed]` for a real score** — the stub is for wiring tests and
dry runs, not for judging a delivery.

> **The shipped config requires `[embed]`.** It drives the score with CLIP (see
> *Current configuration* below), so `pip install -e '.[embed]'` is mandatory to
> run it as delivered. Without it the run stops with an install hint and exit 2.

### Current configuration

`config/pipeline_config.yaml` ships set to:

| | Setting | Effect |
|---|---|---|
| Frames | `frames.count: 1`, `duration_buckets: []` | one frame per clip, taken at **50% of duration** |
| Embedding | `embeddings.driver: secondary` | the **CLIP** block scores the delivery; DINOv2 is never loaded |
| Pooling | `pooling.mode: none` | averaging bypassed — the single frame vector *is* the clip vector |
| Search onward | unchanged | k-NN, clustering, scoring and reporting as described below |

To go back to the multi-frame DINOv2 setup: `driver: primary`, `pooling.mode:
mean`, `frames.count: 5`, and restore the `duration_buckets` list (an example is
commented into the file).

---

## Run

```bash
python scripts/run_pipeline.py \
  --manifest manifest.csv \
  --config config/pipeline_config.yaml \
  --output-dir ./results \
  [--force]
```

Installed, the same CLI is on `PATH` as `visual-diversity`.

| Flag | Meaning |
|---|---|
| `--manifest` | clip manifest, `.csv` or `.json` |
| `--config` | pipeline YAML; omit for built-in defaults |
| `--output-dir` | overrides `output_dir` in the config |
| `--force` | ignore cached frames and embeddings, recompute everything |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

Exit codes: `0` ran and reported · `1` no clip reached scoring · `2` usage or
config error.

### Manifest

Required columns — a row missing any of them is rejected, never silently
dropped:

```csv
item_id,session_id,parent_video_id,worker_id,timestamp,clip_path
clip_0001,sess_A,pv_1,worker_7,2026-08-01T10:00:00Z,/data/clips/clip_0001.mp4
```

`timestamp` accepts ISO-8601 or a bare unix epoch. `duration_seconds` is
optional; when absent it is probed with ffprobe. Any extra columns are kept.
JSON manifests may be a bare list or an object with a `clips` / `items` /
`records` / `data` key.

---

## Credentials

Two kinds of configuration, kept apart deliberately:

| | File | Contains | Committed? |
|---|---|---|---|
| Behaviour | `config/pipeline_config.yaml` | thresholds, model choices, frame counts | **yes** |
| Credentials | `.env` → [settings.py](src/visual_diversity/settings.py) | keys, endpoints, buckets | **never** |

```bash
cp .env.example .env      # then fill it in; .env is gitignored
python -m visual_diversity.settings          # show what is configured
python -m visual_diversity.settings --export-aws   # mirror DO_SPACES_* to AWS_*
```

The status output prints **set / not set**, never a value:

```
env_file              /home/you/repo/.env
DO_SPACES_ENDPOINT    https://sfo3.digitaloceanspaces.com
DO_SPACES_BUCKET      your-bucket
DO_SPACES_SECRET_KEY  set
OPENAI_API_KEY        — not set
object storage : ready
borderline LLM : not configured
```

### If you supply more than one set of credentials

Three families are recognised, tried in this order:

| Family | Variables | Endpoint |
|---|---|---|
| `digitalocean` | `DO_SPACES_ACCESS_KEY` / `_SECRET_KEY` / `_ENDPOINT` / `_REGION` / `_BUCKET` / `_BUCKET_URL` | required |
| `aws` | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL` / `AWS_DEFAULT_REGION` | optional — derived from region |
| `s3` | `S3_ACCESS_KEY` / `S3_SECRET_KEY` / `S3_ENDPOINT_URL` / `S3_REGION` / `S3_BUCKET` | required |

**Credentials resolve as a set, never field by field.** The first family with
*both* an access key and a secret key wins outright, and its endpoint and region
come from that same family. Losing families are named in a warning and otherwise
ignored:

```
WARNING credentials found for 2 provider(s) -- using 'digitalocean'; ignoring 'aws'.
        Endpoint and region come from 'digitalocean' only, so keys are never
        paired with another provider's host.
```

This rule matters. Per-field fallback would let a DigitalOcean key pair combine
with an `AWS_ENDPOINT_URL` that happened to be set — sending your Spaces secret
to `amazonaws.com` while the status output cheerfully reported *ready*. A
credential and the host it authenticates to are one decision, so they are read
together. If the chosen family is incomplete the run reports `not configured`
and names the missing variable, rather than borrowing one from elsewhere.

`bucket` and `bucket_url` are exempt — they name a location rather than
authorise anything, so a generic `S3_BUCKET` may fill in for a family that
doesn't define its own.

`--export-aws` mirrors the selected family into the standard `AWS_*` names
**inside the running Python process** — so boto3, and any subprocess the
pipeline launches (ffmpeg inherits the environment), see them without a second
copy of the secret in the file. It does **not** reach a shell you type into
afterwards; environment changes do not escape the process. For `aws` or `rclone`
at a prompt, source the file instead:

```bash
set -a; . /path/to/.env; set +a
```

**Discovery** walks up from the working directory, so a `.env` at the repo root
is found from anywhere inside it. Override with `VD_ENV_FILE=/path/to/.env`.

**Precedence**: a real environment variable always beats the file. An explicit
`export` for a one-off run is never silently overridden.

**Redaction**: secrets are held as pydantic `SecretStr`, so a stray `print`, a
`repr`, a `model_dump()` or an exception traceback shows `**********`. Loading
logs key *names* only. Tests assert this against a canary value.

Nothing is read at import time. `settings.s3.require()` and
`settings.openai.require()` raise at the point of use, naming exactly which
variable is absent — and the pipeline calls the OpenAI one at **startup** when
stage 12 is enabled, so a missing key fails before the embedding spend rather
than after it.

## Output

Written to `output_dir`:

| File | Contents |
|---|---|
| `visual_diversity_report.json` | the complete result — every cluster, every clip score |
| `visual_diversity_report.md` | the same content for a human reader |
| `rejected_clips.csv` | manifest rows dropped at ingest, with reasons |
| `thumbnails/` | one representative frame per duplicate cluster |

The Markdown truncates long tables and says how many rows it omitted; the JSON
never truncates.

---

## How the score works

```
novelty(clip)   = 1 - max cosine similarity to any other clip
avg_novelty     = mean(novelty)
cluster_penalty = sum(size ** 1.5 for clusters of size > 1) / total_clips
score           = max(0, (avg_novelty - cluster_penalty)) * 15
```

The exponent is super-linear on purpose: ten clusters of two are a labelling
nuisance, one cluster of twenty is a collection failure, and a linear term would
score them identically. Both the exponent and `max_points` are configurable.

Note the penalty divides by clip count, so on very small deliveries a single
cluster dominates and the score floors at 0. That is the formula behaving as
specified, not a bug — the score is meaningful at delivery scale.

### Stages

| Stage | Module | What it does |
|---|---|---|
| 1–2 | `ingest` | load manifest, validate every row, reject the rest |
| 3 | `frames` | ffmpeg-sample N frames per clip, cached, parallel |
| 4–5 | `embeddings` | DINOv2 CLS token or CLIP image features, cached |
| 6 | `pooling` | frames → one L2-normalised vector per clip (`mean`, or bypassed) |
| 7–8 | `search` | FAISS (or numpy) top-k neighbours |
| 9 | `clustering` | threshold graph → connected components |
| 12 | `borderline_review` | GPT-4o adjudicates gray-zone pairs |
| 10–11 | `scoring` | novelty + cluster penalty + attribution |
| 13 | `report` | JSON + Markdown scorecard |

Stage 12 runs between 9 and 10, as specified: it corrects cluster membership
before the score is finalised.

---

## Configuration

Everything tunable lives in `config/pipeline_config.yaml`; no stage reads a
threshold from code. Relative paths resolve against the config file's own
directory. Unknown keys are rejected rather than ignored, so a typo fails loudly
instead of silently keeping a default.

The settings you are most likely to touch:

| Key | Default | Effect |
|---|---|---|
| `clustering.similarity_threshold` | `0.91` | at/above this, two clips are duplicates |
| `clustering.flag_cluster_size` | `3` | clusters this large get a remove/re-shoot list |
| `scoring.penalty_exponent` | `1.5` | how hard large clusters are punished |
| `scoring.max_points` | `15.0` | score ceiling |
| `frames.count` | **`1`** | frames per clip, evenly spaced; at 1 the sample is the midpoint |
| `frames.duration_buckets` | **`[]`** | per-length overrides; empty means `count` applies to every clip |
| `embeddings.driver` | **`secondary`** | which embedder scores: `primary` or `secondary` |
| `embeddings.backend` | `dinov2` | the primary: `dinov2` · `clip` · `stub`. Ignored while `driver: secondary` |
| `pooling.mode` | **`none`** | `mean` · `none` (bypass) · `maxpair` |
| `search.index_type` | `flat` | `flat` · `ivf` · `hnsw` |
| `search.top_k` | `10` | neighbours retrieved per clip |
| `borderline_review.enabled` | `false` | stage 12 on/off |
| `borderline_review.gray_zone` | `[0.80, 0.91]` | the ambiguous band |
| `borderline_review.max_pairs` | `50` | hard cap on API calls per run |

### Scaling past ~1M clips

Switch `search.index_type` to `ivf` or `hnsw`. Nothing else changes — the index
sits behind a `VectorIndex` protocol and callers never see the backend.

### Borderline review (stage 12)

Off by default because it costs money. When on, pairs in the gray zone are shown
to GPT-4o, which decides whether they are the same activity or genuinely
different work. A `REDUNDANT` verdict merges the pair and re-clusters; a
`DISTINCT` verdict changes nothing, so the model's influence is monotone and
auditable.

If the gray zone holds more pairs than `max_pairs`, the sample is drawn
proportionally across similarity bands — taking the top-N by similarity would
only sample the end of the zone where the answer is least in doubt. Sampling is
seeded by `random_seed`, so a re-run reviews the same pairs.

The key is read from `OPENAI_API_KEY`. It is never a config value and never
logged:

```bash
export OPENAI_API_KEY=sk-...
```

A failure in this stage is logged and the uncorrected clusters are kept — an
advisory stage never sinks a run.

---

## Caching and resumability

Frames and embeddings are cached under `cache_dir`, keyed by `item_id`. A re-run
reuses whatever is present; `--force` recomputes. The embedding cache is
invalidated automatically when the frame count changes, so editing
`frames.count` will not serve stale vectors.

The cheap stages (pooling, search, clustering, scoring) always recompute —
caching them would save milliseconds and risk reporting a stale score.

Failures are isolated per clip: an unreadable file is logged and skipped, and
the run continues.

---

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

167 tests, no network and no model download: the embedder is the deterministic
stub, the vision judge is injected as a fake, and clips are tiny generated MP4s.
One test is skipped unless faiss is installed — it checks the faiss and numpy
backends return identical neighbours.

The CLIP driver itself cannot be exercised offline, so the end-to-end tests cover
the `frames.count: 1` + `pooling.mode: none` path with the stub standing in, and
assert separately that `driver: secondary` never constructs the primary.

---

## Notes

- `src/visual_diversity/config.py` and `cli.py` are additions to the module list
  in the original spec: the pydantic config models needed a home of their own,
  and the CLI logic lives in the package so it is importable and testable, with
  `scripts/run_pipeline.py` as the thin wrapper the spec asks for.
- Manifests are read with the stdlib `csv`/`json` modules rather than pandas,
  which keeps the core install light. Validated rows come back as a typed list
  of pydantic `ClipRecord` models.
