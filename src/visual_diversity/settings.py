"""Credentials and environment-driven settings, in one place.

Separate from :mod:`visual_diversity.config` on purpose:

  * ``config.py``   -- *how the pipeline behaves*. Thresholds, model choices,
                       frame counts. Belongs in version control.
  * ``settings.py`` -- *what it needs to reach the outside world*. Credentials,
                       endpoints, buckets. Must never be committed.

Nothing here is ever hardcoded. Values come from the process environment, which
can be seeded from a ``.env`` file. **A real environment variable always wins
over the file**, so an explicit ``export`` is never silently overridden.

Secrets are held as :class:`~pydantic.SecretStr`, so a stray ``print`` or an
exception traceback shows ``**********`` rather than the key. Use
:meth:`Settings.describe` to report what is configured -- it reports set/unset,
never values.

Names follow what the surrounding repo already uses: ``DO_SPACES_*`` for the
DigitalOcean Spaces bucket, with the standard ``AWS_*`` spellings accepted as
aliases (they are the same credentials, and boto3/rclone/aws-cli read those).

Check what is configured:

    python -m visual_diversity.settings
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, SecretStr

log = logging.getLogger(__name__)

__all__ = [
    "Settings",
    "S3Settings",
    "OpenAISettings",
    "Provider",
    "PROVIDERS",
    "PathSettings",
    "get_settings",
    "clear_settings_cache",
    "load_env_file",
    "find_env_file",
    "parse_env_text",
    "ENV_FILENAMES",
]

# ===========================================================================
# EDIT HERE -- where a run reads from and writes to.
#
# Leave a value as "" to fall back to the pipeline config / CLI flag. Anything
# set here wins over pipeline_config.yaml but still loses to an explicit CLI
# flag and to the matching environment variable, so a one-off run never has to
# edit this file.
#
#     CLI flag  >  environment variable  >  this file  >  pipeline_config.yaml
#
# Credentials are NOT here on purpose -- they belong in `.env`, which is
# gitignored. This file is committed.
# ===========================================================================

#: Root the manifest's *relative* ``clip_path`` values resolve against.
#: Absolute paths in the manifest ignore this. Env: VD_INPUT_DIR
DEFAULT_INPUT_DIR = ""

#: Where reports land. Env: VD_OUTPUT_DIR  ·  CLI: --output-dir
DEFAULT_OUTPUT_DIR = ""

#: Frames and embeddings. Wiping this only costs recomputation. Env: VD_CACHE_DIR
DEFAULT_CACHE_DIR = ""

#: Default manifest, so --manifest can be omitted. Env: VD_MANIFEST
DEFAULT_MANIFEST = ""

#: Force which credential family to use: "digitalocean" | "aws" | "s3".
#: Leave "" to auto-select the first family with a complete key pair.
#: Setting this makes the choice explicit rather than a consequence of which
#: variables happen to be filled in -- and it fails loudly if that family is
#: incomplete instead of quietly falling through to another provider's keys.
#: Env: VD_S3_PROVIDER
DEFAULT_S3_PROVIDER = ""

#: Filenames searched for, in order, walking up from the start directory.
ENV_FILENAMES = (".env", ".env.local")

#: Explicit override: point at a specific file and skip the search.
ENV_FILE_VAR = "VD_ENV_FILE"


# ---------------------------------------------------------------------------
# .env loading (stdlib only -- no new dependency for a 20-line format)
# ---------------------------------------------------------------------------
def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines.

    Handles ``export KEY=value``, ``#`` comments, blank lines, and single or
    double quoted values. Deliberately does *not* do shell interpolation --
    a ``$VAR`` inside a value stays literal, because guessing at expansion is
    how a credential file starts producing surprises.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate a ``.env`` by walking up from ``start`` (default: cwd).

    Walking up matters here: this package lives in a subdirectory of a repo
    whose ``.env`` sits at the top, so a run started from the package folder
    still finds it.
    """
    override = os.environ.get(ENV_FILE_VAR)
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in ENV_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_env_file(path: Path | None = None, *, override: bool = False,
                  start: Path | None = None) -> tuple[Path | None, list[str]]:
    """Seed ``os.environ`` from a ``.env``. Returns (path, keys applied).

    Existing environment variables are preserved unless ``override`` is set --
    an exported credential is an explicit act and outranks a file.

    Only key *names* are returned and logged; values never are.
    """
    path = path or find_env_file(start)
    if path is None:
        return None, []

    try:
        parsed = parse_env_text(path.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("settings: could not read %s -- %s", path, exc)
        return None, []

    applied: list[str] = []
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)

    if applied:
        log.info("settings: loaded %d value(s) from %s (%s)",
                 len(applied), path, ", ".join(sorted(applied)))
    else:
        log.debug("settings: %s had nothing to add; environment already set", path)
    return path, applied


def _first(*names: str) -> str | None:
    """First non-empty environment value among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _secret(*names: str) -> SecretStr | None:
    value = _first(*names)
    return SecretStr(value) if value else None


# ---------------------------------------------------------------------------
# Setting groups
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    model_config = ConfigDict(frozen=True)


class Provider(NamedTuple):
    """One credential family. Its security-relevant fields are read together."""

    name: str
    access_key: tuple[str, ...]
    secret_key: tuple[str, ...]
    endpoint_url: tuple[str, ...]
    region: tuple[str, ...]
    bucket: tuple[str, ...]
    bucket_url: tuple[str, ...] = ()
    #: AWS proper derives its endpoint from the region; everyone else needs one.
    endpoint_required: bool = True


#: Tried in order. The first family with a **complete key pair** wins outright.
PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="digitalocean",
        access_key=("DO_SPACES_ACCESS_KEY",),
        secret_key=("DO_SPACES_SECRET_KEY",),
        endpoint_url=("DO_SPACES_ENDPOINT",),
        region=("DO_SPACES_REGION",),
        bucket=("DO_SPACES_BUCKET",),
        bucket_url=("DO_SPACES_BUCKET_URL",),
    ),
    Provider(
        name="aws",
        access_key=("AWS_ACCESS_KEY_ID",),
        secret_key=("AWS_SECRET_ACCESS_KEY",),
        endpoint_url=("AWS_ENDPOINT_URL",),
        region=("AWS_DEFAULT_REGION", "AWS_REGION"),
        bucket=("AWS_S3_BUCKET",),
        endpoint_required=False,
    ),
    Provider(
        name="s3",
        access_key=("S3_ACCESS_KEY", "S3_ACCESS_KEY_ID"),
        secret_key=("S3_SECRET_KEY", "S3_SECRET_ACCESS_KEY"),
        endpoint_url=("S3_ENDPOINT_URL",),
        region=("S3_REGION",),
        bucket=("S3_BUCKET",),
    ),
)


class S3Settings(_Base):
    """Object storage: DigitalOcean Spaces, AWS S3, or any S3-compatible endpoint.

    **Credentials are resolved as a set, never field by field.** The first
    family in :data:`PROVIDERS` that supplies *both* an access key and a secret
    key wins, and its endpoint and region are taken from that same family.

    That rule exists because the obvious alternative is dangerous: falling back
    per field lets a DigitalOcean key pair combine with an ``AWS_ENDPOINT_URL``
    that happens to be set, which would send a Spaces secret to Amazon. A
    credential and the host it authenticates to are one decision, so they are
    read together.

    ``bucket`` and ``bucket_url`` are exempt -- they name a location rather than
    authorise anything, so a generic ``S3_BUCKET`` may fill in for a family that
    does not define its own.
    """

    provider: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    bucket: str | None = None
    bucket_url: str | None = None
    access_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    #: Families that also had a complete key pair and were not chosen.
    shadowed: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "S3Settings":
        forced = (_first("VD_S3_PROVIDER") or DEFAULT_S3_PROVIDER.strip() or "").lower()
        if forced:
            chosen = next((p for p in PROVIDERS if p.name == forced), None)
            if chosen is None:
                raise ValueError(
                    f"unknown storage provider {forced!r}; expected one of "
                    + ", ".join(repr(p.name) for p in PROVIDERS)
                    + ". Set VD_S3_PROVIDER or DEFAULT_S3_PROVIDER in settings.py."
                )
            log.info("settings: object-storage provider pinned to '%s'", chosen.name)
            # No fallback: an explicit choice that is incomplete must surface as
            # incomplete, never silently resolve to a different provider's keys.
            return cls(
                provider=chosen.name,
                endpoint_url=_first(*chosen.endpoint_url),
                region=_first(*chosen.region),
                bucket=_first(*chosen.bucket, "S3_BUCKET"),
                bucket_url=_first(*chosen.bucket_url) if chosen.bucket_url else None,
                access_key=_secret(*chosen.access_key),
                secret_key=_secret(*chosen.secret_key),
            )

        complete = [p for p in PROVIDERS
                    if _first(*p.access_key) and _first(*p.secret_key)]

        if not complete:
            # Nothing usable. Still surface a partially-filled family so
            # `missing()` can name what is absent rather than shrugging.
            partial = next((p for p in PROVIDERS
                            if any(_first(*names) for names in
                                   (p.access_key, p.secret_key, p.endpoint_url))), PROVIDERS[0])
            return cls(
                provider=None,
                endpoint_url=_first(*partial.endpoint_url),
                region=_first(*partial.region),
                bucket=_first(*partial.bucket, "S3_BUCKET"),
                bucket_url=_first(*partial.bucket_url) if partial.bucket_url else None,
                access_key=_secret(*partial.access_key),
                secret_key=_secret(*partial.secret_key),
            )

        chosen, *rest = complete
        if rest:
            log.warning(
                "settings: credentials found for %d provider(s) -- using '%s'; "
                "ignoring %s. Endpoint and region come from '%s' only, so keys are "
                "never paired with another provider's host.",
                len(complete), chosen.name,
                ", ".join(f"'{p.name}'" for p in rest), chosen.name)
        else:
            log.info("settings: object-storage credentials from '%s'", chosen.name)

        return cls(
            provider=chosen.name,
            endpoint_url=_first(*chosen.endpoint_url),
            region=_first(*chosen.region),
            # Not a credential: a generic bucket name may fill in.
            bucket=_first(*chosen.bucket, "S3_BUCKET"),
            bucket_url=_first(*chosen.bucket_url) if chosen.bucket_url else None,
            access_key=_secret(*chosen.access_key),
            secret_key=_secret(*chosen.secret_key),
            shadowed=tuple(p.name for p in rest),
        )

    @property
    def _provider(self) -> Provider | None:
        return next((p for p in PROVIDERS if p.name == self.provider), None)

    @property
    def configured(self) -> bool:
        """True when a client could actually be built from this.

        AWS proper needs no explicit endpoint -- boto3 derives it from the
        region -- so the endpoint requirement is per provider.
        """
        if not (self.access_key and self.secret_key):
            return False
        provider = self._provider
        if provider is not None and not provider.endpoint_required:
            return True
        return bool(self.endpoint_url)

    def missing(self) -> list[str]:
        provider = self._provider or PROVIDERS[0]
        gaps = []
        if not self.access_key:
            gaps.append(provider.access_key[0])
        if not self.secret_key:
            gaps.append(provider.secret_key[0])
        if provider.endpoint_required and not self.endpoint_url:
            gaps.append(provider.endpoint_url[0])
        return gaps

    def require(self) -> "S3Settings":
        """Return self, or raise naming exactly what is absent."""
        gaps = self.missing()
        if gaps:
            raise RuntimeError(
                "object storage is not configured; missing: " + ", ".join(gaps)
                + ". Set them in the environment or a .env file "
                  "(see .env.example)."
            )
        return self

    def boto3_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``boto3.client('s3', **kwargs)``.

        Returned rather than used: this package does not depend on boto3, so
        the caller supplies the client.
        """
        self.require()
        kwargs: dict[str, Any] = {
            "endpoint_url": self.endpoint_url,
            "aws_access_key_id": self.access_key.get_secret_value(),  # type: ignore[union-attr]
            "aws_secret_access_key": self.secret_key.get_secret_value(),  # type: ignore[union-attr]
        }
        if self.region:
            kwargs["region_name"] = self.region
        return kwargs

    def apply_to_environment(self) -> list[str]:
        """Export the canonical ``AWS_*`` spellings for tools that expect them.

        boto3, aws-cli, rclone and ffmpeg's S3 support all read the standard
        names. When credentials arrived as ``DO_SPACES_*`` this mirrors them
        across so those tools work without a second copy of the secret.

        Never overwrites a variable that is already set. Returns the names
        written.
        """
        mapping = {
            "AWS_ACCESS_KEY_ID": self.access_key.get_secret_value() if self.access_key else None,
            "AWS_SECRET_ACCESS_KEY": (self.secret_key.get_secret_value()
                                      if self.secret_key else None),
            "AWS_DEFAULT_REGION": self.region,
            "AWS_ENDPOINT_URL": self.endpoint_url,
        }
        written = []
        for name, value in mapping.items():
            if value and not os.environ.get(name):
                os.environ[name] = value
                written.append(name)
        if written:
            log.info("settings: exported %s for S3-aware tooling", ", ".join(written))
        return written


class OpenAISettings(_Base):
    """Credentials for the stage-12 borderline review."""

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        return cls(
            api_key=_secret("OPENAI_API_KEY"),
            base_url=_first("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            organization=_first("OPENAI_ORG_ID", "OPENAI_ORGANIZATION"),
        )

    @property
    def configured(self) -> bool:
        return self.api_key is not None

    def require(self) -> "OpenAISettings":
        if not self.configured:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; borderline review cannot run. "
                "Set it in the environment or a .env file, or disable "
                "borderline_review in the pipeline config."
            )
        return self


class PathSettings(_Base):
    """Where a run reads from and writes to.

    Every field is optional. ``None`` means "not set here" and the pipeline
    config (or the CLI flag) decides instead -- which is why editing the
    constants at the top of this module is additive and never breaks an
    existing config.
    """

    input_dir: Path | None = None
    output_dir: Path | None = None
    cache_dir: Path | None = None
    manifest: Path | None = None

    @staticmethod
    def _pick(env_names: Iterable[str], fallback: str) -> Path | None:
        value = _first(*env_names) or (fallback.strip() or None)
        return Path(value).expanduser() if value else None

    @classmethod
    def from_env(cls) -> "PathSettings":
        return cls(
            input_dir=cls._pick(("VD_INPUT_DIR",), DEFAULT_INPUT_DIR),
            output_dir=cls._pick(("VD_OUTPUT_DIR",), DEFAULT_OUTPUT_DIR),
            cache_dir=cls._pick(("VD_CACHE_DIR",), DEFAULT_CACHE_DIR),
            manifest=cls._pick(("VD_MANIFEST",), DEFAULT_MANIFEST),
        )

    def resolve_clip(self, clip_path: Path) -> Path:
        """Anchor a relative ``clip_path`` from the manifest to ``input_dir``.

        Absolute paths are returned untouched, so a manifest of absolute paths
        is unaffected by this setting.
        """
        if clip_path.is_absolute() or self.input_dir is None:
            return clip_path
        return self.input_dir / clip_path


class Settings(_Base):
    """Everything the pipeline needs from outside itself."""

    s3: S3Settings = Field(default_factory=S3Settings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    #: The .env that seeded the environment, when one was found.
    env_file: Path | None = None

    @classmethod
    def load(cls, *, env_file: Path | None = None, start: Path | None = None,
             override: bool = False) -> "Settings":
        """Seed from a ``.env`` if present, then read the environment."""
        path, _ = load_env_file(env_file, override=override, start=start)
        return cls(s3=S3Settings.from_env(), openai=OpenAISettings.from_env(),
                   paths=PathSettings.from_env(), env_file=path)

    def describe(self) -> dict[str, str]:
        """Set/unset per setting. Never returns a secret value."""
        def mark(value: Any) -> str:
            if value is None or value == "":
                return "— not set"
            return "set" if isinstance(value, SecretStr) else str(value)

        return {
            "env_file": str(self.env_file) if self.env_file else "— none found",
            "input_dir  (VD_INPUT_DIR)": mark(self.paths.input_dir),
            "output_dir (VD_OUTPUT_DIR)": mark(self.paths.output_dir),
            "cache_dir  (VD_CACHE_DIR)": mark(self.paths.cache_dir),
            "manifest   (VD_MANIFEST)": mark(self.paths.manifest),
            "storage provider": (self.s3.provider or "— none")
                                + (f"  (ignoring {', '.join(self.s3.shadowed)})"
                                   if self.s3.shadowed else ""),
            "  endpoint": mark(self.s3.endpoint_url),
            "  region": mark(self.s3.region),
            "  bucket": mark(self.s3.bucket),
            "  bucket_url": mark(self.s3.bucket_url),
            "  access_key": mark(self.s3.access_key),
            "  secret_key": mark(self.s3.secret_key),
            "OPENAI_API_KEY": mark(self.openai.api_key),
            "OPENAI_BASE_URL": mark(self.openai.base_url),
        }

    def report(self) -> str:
        """Human-readable status block. Safe to print or log."""
        rows = self.describe()
        width = max(len(k) for k in rows)
        lines = ["visual-diversity settings", "-" * (width + 24)]
        lines += [f"{k.ljust(width)}  {v}" for k, v in rows.items()]
        lines.append("-" * (width + 24))
        lines.append(f"object storage : {'ready' if self.s3.configured else 'not configured'}")
        lines.append(f"borderline LLM : {'ready' if self.openai.configured else 'not configured'}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings.load()


def get_settings(*, reload: bool = False) -> Settings:
    """Process-wide settings, loaded once.

    ``reload=True`` re-reads the environment -- used by tests, and after a
    deliberate change to os.environ.
    """
    if reload:
        _cached_settings.cache_clear()
    return _cached_settings()


def clear_settings_cache() -> None:
    """Drop the cached settings without re-reading the environment.

    Distinct from ``get_settings(reload=True)`` on purpose: reloading *parses*,
    and parsing can raise (an unknown ``VD_S3_PROVIDER``, say). Test teardown
    only needs to invalidate, and must never fail on the way out.
    """
    _cached_settings.cache_clear()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """``python -m visual_diversity.settings`` -- print redacted status."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m visual_diversity.settings",
        description="Show which credentials are configured. Values are never printed.")
    parser.add_argument("--env-file", default=None, help="explicit .env to load")
    parser.add_argument("--export-aws", action="store_true",
                        help="mirror DO_SPACES_* into the standard AWS_* names")
    parser.add_argument("--require", choices=["s3", "openai"], action="append", default=[],
                        help="exit non-zero unless this is configured (repeatable)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.load(env_file=Path(args.env_file) if args.env_file else None)

    if args.export_aws:
        settings.s3.apply_to_environment()

    print(settings.report())

    failed = False
    for need in args.require:
        try:
            (settings.s3 if need == "s3" else settings.openai).require()
        except RuntimeError as exc:
            print(f"\nerror: {exc}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
