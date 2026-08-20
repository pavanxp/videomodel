"""Credential loading, precedence and redaction.

No real credential is used anywhere here; every value is a fake with a
recognisable marker so the leak tests can search for it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from visual_diversity.settings import (ENV_FILE_VAR, OpenAISettings, PathSettings, S3Settings,
                                       Settings, find_env_file, get_settings, load_env_file,
                                       parse_env_text, clear_settings_cache)

# A value that must never appear in output. Distinctive on purpose.
CANARY = "sk-CANARY-must-never-be-printed-9f3a"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Every test starts from a known-empty credential environment.

    Critically, this also points the loader at a path that does not exist.
    Without it the discovery walk finds the developer's real ``.env`` further up
    the tree and seeds live credentials into the suite -- which makes results
    depend on whose machine is running and risks a real secret reaching test
    output. Tests that exercise discovery delete the override themselves.
    """
    for name in ("DO_SPACES_ENDPOINT", "DO_SPACES_REGION", "DO_SPACES_BUCKET",
                 "DO_SPACES_BUCKET_URL", "DO_SPACES_ACCESS_KEY", "DO_SPACES_SECRET_KEY",
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
                 "AWS_REGION", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL", "S3_BUCKET",
                 "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                 "OPENAI_ORG_ID", "OPENAI_ORGANIZATION",
                 "VD_INPUT_DIR", "VD_OUTPUT_DIR", "VD_CACHE_DIR", "VD_MANIFEST",
                 "VD_S3_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_FILE_VAR, str(tmp_path / "no-such-file.env"))
    get_settings(reload=True)
    yield
    clear_settings_cache()


def test_the_suite_is_isolated_from_any_real_env_file():
    """Guard on the fixture above: no real credential may reach these tests."""
    settings = Settings.load()
    assert settings.env_file is None
    assert settings.s3.configured is False
    assert settings.openai.configured is False


# --------------------------------------------------------------------------
# .env parsing
# --------------------------------------------------------------------------
def test_parse_handles_the_common_shapes():
    parsed = parse_env_text(
        "\n".join([
            "# a comment",
            "",
            "PLAIN=value",
            "export EXPORTED=value2",
            'DQUOTED="quoted value"',
            "SQUOTED='single'",
            "  SPACED  =  trimmed  ",
            "EMPTY=",
            "WITH_EQUALS=a=b=c",
            "novalue",
        ])
    )
    assert parsed == {
        "PLAIN": "value", "EXPORTED": "value2", "DQUOTED": "quoted value",
        "SQUOTED": "single", "SPACED": "trimmed", "EMPTY": "", "WITH_EQUALS": "a=b=c",
    }


def test_parse_does_not_interpolate():
    """A literal $VAR must stay literal -- guessing at expansion surprises."""
    assert parse_env_text("KEY=$OTHER/path")["KEY"] == "$OTHER/path"


# --------------------------------------------------------------------------
# Discovery and precedence
# --------------------------------------------------------------------------
def test_env_file_is_found_by_walking_up(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(ENV_FILE_VAR, raising=False)
    root = tmp_path / "repo"
    deep = root / "pkg" / "src"
    deep.mkdir(parents=True)
    (root / ".env").write_text("OPENAI_API_KEY=from-root\n", encoding="utf-8")

    assert find_env_file(start=deep) == root / ".env"


def test_explicit_override_variable_wins(tmp_path: Path, monkeypatch):
    chosen = tmp_path / "custom.env"
    chosen.write_text("OPENAI_API_KEY=x\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=y\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VAR, str(chosen))

    assert find_env_file(start=tmp_path) == chosen


def test_no_env_file_is_not_an_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ENV_FILE_VAR, str(tmp_path / "absent.env"))
    path, applied = load_env_file()
    assert path is None
    assert applied == []


def test_real_environment_beats_the_file(tmp_path: Path, monkeypatch):
    """An explicit export is a deliberate act and must not be overridden."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")

    load_env_file(env)

    import os
    assert os.environ["OPENAI_API_KEY"] == "from-environment"


def test_override_flag_forces_the_file_to_win(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")

    load_env_file(env, override=True)

    import os
    assert os.environ["OPENAI_API_KEY"] == "from-file"


# --------------------------------------------------------------------------
# Reading settings
# --------------------------------------------------------------------------
def test_s3_reads_do_spaces_names(monkeypatch):
    monkeypatch.setenv("DO_SPACES_ENDPOINT", "https://blr1.example.com")
    monkeypatch.setenv("DO_SPACES_REGION", "blr1")
    monkeypatch.setenv("DO_SPACES_BUCKET", "clips")
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "AK")
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", "SK")

    s3 = S3Settings.from_env()

    assert s3.configured
    assert s3.bucket == "clips"
    assert s3.access_key.get_secret_value() == "AK"


def test_s3_falls_back_to_aws_names(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SK")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example.com")

    s3 = S3Settings.from_env()

    assert s3.configured
    assert s3.endpoint_url == "https://s3.example.com"


def test_do_spaces_takes_precedence_over_aws(monkeypatch):
    for name, value in (("DO_SPACES_ACCESS_KEY", "preferred"),
                        ("DO_SPACES_SECRET_KEY", "preferred-secret"),
                        ("DO_SPACES_ENDPOINT", "https://do.example.com"),
                        ("AWS_ACCESS_KEY_ID", "fallback"),
                        ("AWS_SECRET_ACCESS_KEY", "fallback-secret")):
        monkeypatch.setenv(name, value)

    s3 = S3Settings.from_env()

    assert s3.provider == "digitalocean"
    assert s3.access_key.get_secret_value() == "preferred"
    assert s3.shadowed == ("aws",)


def test_a_key_is_never_paired_with_another_providers_endpoint(monkeypatch):
    """The whole point of set-based resolution.

    With a DigitalOcean key pair and only an AWS endpoint set, per-field
    fallback would send a Spaces secret to amazonaws.com. The chosen family
    supplies the endpoint, or there is none.
    """
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "DO-KEY")
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", "DO-SECRET")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.us-east-1.amazonaws.com")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    s3 = S3Settings.from_env()

    assert s3.provider == "digitalocean"
    assert s3.access_key.get_secret_value() == "DO-KEY"
    assert s3.endpoint_url is None          # NOT the AWS one
    assert s3.region is None                # NOT us-east-1
    assert s3.configured is False           # and it does not claim to be ready
    assert "DO_SPACES_ENDPOINT" in " ".join(s3.missing())


def test_all_three_families_at_once_picks_one_cleanly(monkeypatch):
    for name, value in (
        ("DO_SPACES_ACCESS_KEY", "DO-KEY"), ("DO_SPACES_SECRET_KEY", "DO-SECRET"),
        ("DO_SPACES_ENDPOINT", "https://sfo3.digitaloceanspaces.com"),
        ("DO_SPACES_REGION", "sfo3"), ("DO_SPACES_BUCKET", "example-bucket"),
        ("AWS_ACCESS_KEY_ID", "AWS-KEY"), ("AWS_SECRET_ACCESS_KEY", "AWS-SECRET"),
        ("AWS_ENDPOINT_URL", "https://s3.amazonaws.com"), ("AWS_DEFAULT_REGION", "us-east-1"),
        ("S3_ACCESS_KEY", "S3-KEY"), ("S3_SECRET_KEY", "S3-SECRET"),
        ("S3_ENDPOINT_URL", "https://minio.example.com"),
    ):
        monkeypatch.setenv(name, value)

    s3 = S3Settings.from_env()

    # Every field comes from digitalocean; nothing is blended.
    assert s3.provider == "digitalocean"
    assert s3.access_key.get_secret_value() == "DO-KEY"
    assert s3.secret_key.get_secret_value() == "DO-SECRET"
    assert s3.endpoint_url == "https://sfo3.digitaloceanspaces.com"
    assert s3.region == "sfo3"
    assert s3.shadowed == ("aws", "s3")
    assert s3.configured is True


def test_multiple_families_are_warned_about(monkeypatch, caplog):
    for name, value in (("DO_SPACES_ACCESS_KEY", "K"), ("DO_SPACES_SECRET_KEY", "S"),
                        ("DO_SPACES_ENDPOINT", "https://do.example.com"),
                        ("AWS_ACCESS_KEY_ID", "K2"), ("AWS_SECRET_ACCESS_KEY", "S2")):
        monkeypatch.setenv(name, value)

    with caplog.at_level(logging.WARNING):
        S3Settings.from_env()

    assert "using 'digitalocean'" in caplog.text
    assert "ignoring 'aws'" in caplog.text


def test_aws_needs_no_explicit_endpoint(monkeypatch):
    """Real S3 derives its endpoint from the region, unlike Spaces or MinIO."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "K")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "S")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    s3 = S3Settings.from_env()

    assert s3.provider == "aws"
    assert s3.endpoint_url is None
    assert s3.configured is True
    assert s3.missing() == []


def test_a_lone_access_key_is_not_a_usable_family(monkeypatch):
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "half")
    s3 = S3Settings.from_env()
    assert s3.provider is None
    assert s3.configured is False
    assert "DO_SPACES_SECRET_KEY" in " ".join(s3.missing())


def test_provider_can_be_pinned_explicitly(monkeypatch):
    """With two complete families, VD_S3_PROVIDER decides instead of the order."""
    for name, value in (("DO_SPACES_ACCESS_KEY", "DO-KEY"),
                        ("DO_SPACES_SECRET_KEY", "DO-SECRET"),
                        ("DO_SPACES_ENDPOINT", "https://do.example.com"),
                        ("AWS_ACCESS_KEY_ID", "AWS-KEY"),
                        ("AWS_SECRET_ACCESS_KEY", "AWS-SECRET"),
                        ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VD_S3_PROVIDER", "aws")

    s3 = S3Settings.from_env()

    assert s3.provider == "aws"
    assert s3.access_key.get_secret_value() == "AWS-KEY"
    assert s3.region == "us-east-1"


def test_pinned_provider_does_not_fall_through_when_incomplete(monkeypatch):
    """An explicit choice that is incomplete stays incomplete.

    Falling back would hand the run another provider's keys under the name the
    operator explicitly rejected.
    """
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "DO-KEY")
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", "DO-SECRET")
    monkeypatch.setenv("DO_SPACES_ENDPOINT", "https://do.example.com")
    monkeypatch.setenv("VD_S3_PROVIDER", "aws")          # aws has no keys set

    s3 = S3Settings.from_env()

    assert s3.provider == "aws"
    assert s3.access_key is None
    assert s3.configured is False
    assert "AWS_ACCESS_KEY_ID" in " ".join(s3.missing())


def test_pinned_provider_accepts_the_module_constant(monkeypatch):
    monkeypatch.setattr("visual_diversity.settings.DEFAULT_S3_PROVIDER", "s3")
    monkeypatch.setenv("S3_ACCESS_KEY", "K")
    monkeypatch.setenv("S3_SECRET_KEY", "S")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://minio.example.com")

    assert S3Settings.from_env().provider == "s3"


def test_unknown_pinned_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("VD_S3_PROVIDER", "gcs")
    with pytest.raises(ValueError, match="unknown storage provider"):
        S3Settings.from_env()


def test_generic_bucket_fills_in_for_a_family_without_one(monkeypatch):
    """A bucket names a location; it authorises nothing, so it may fall back."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "K")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "S")
    monkeypatch.setenv("S3_BUCKET", "shared-bucket")

    s3 = S3Settings.from_env()

    assert s3.provider == "aws"
    assert s3.bucket == "shared-bucket"


def test_blank_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert OpenAISettings.from_env().configured is False


def test_require_names_what_is_missing():
    with pytest.raises(RuntimeError) as exc:
        S3Settings().require()
    message = str(exc.value)
    assert "DO_SPACES_ACCESS_KEY" in message
    assert "DO_SPACES_ENDPOINT" in message


def test_openai_require_points_at_the_fix():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAISettings().require()


def test_boto3_kwargs_shape(monkeypatch):
    monkeypatch.setenv("DO_SPACES_ENDPOINT", "https://e.example.com")
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "AK")
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", "SK")
    monkeypatch.setenv("DO_SPACES_REGION", "blr1")

    kwargs = S3Settings.from_env().boto3_kwargs()

    assert kwargs == {
        "endpoint_url": "https://e.example.com",
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
        "region_name": "blr1",
    }


def test_boto3_kwargs_refuses_when_unconfigured():
    with pytest.raises(RuntimeError, match="not configured"):
        S3Settings().boto3_kwargs()


# --------------------------------------------------------------------------
# Mirroring into the standard AWS names
# --------------------------------------------------------------------------
def test_apply_to_environment_mirrors_do_spaces(monkeypatch):
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "AK")
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", "SK")
    monkeypatch.setenv("DO_SPACES_ENDPOINT", "https://e.example.com")
    monkeypatch.setenv("DO_SPACES_REGION", "blr1")

    written = S3Settings.from_env().apply_to_environment()

    import os
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AK"
    assert os.environ["AWS_ENDPOINT_URL"] == "https://e.example.com"
    assert "AWS_SECRET_ACCESS_KEY" in written


def test_apply_to_environment_never_clobbers(monkeypatch):
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", "from-spaces")
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", "SK")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "already-there")

    written = S3Settings.from_env().apply_to_environment()

    import os
    assert os.environ["AWS_ACCESS_KEY_ID"] == "already-there"
    assert "AWS_ACCESS_KEY_ID" not in written


# --------------------------------------------------------------------------
# Redaction -- the failure that actually matters
# --------------------------------------------------------------------------
def test_repr_and_str_do_not_leak(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", CANARY)

    settings = Settings.load()

    for rendered in (repr(settings), str(settings),
                     repr(settings.openai), str(settings.s3)):
        assert CANARY not in rendered


def test_describe_and_report_do_not_leak(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    monkeypatch.setenv("DO_SPACES_ACCESS_KEY", CANARY)
    monkeypatch.setenv("DO_SPACES_SECRET_KEY", CANARY)

    settings = Settings.load()

    described = " ".join(f"{k}={v}" for k, v in settings.describe().items())
    assert CANARY not in described
    assert CANARY not in settings.report()
    # It still says the value is present.
    described_map = settings.describe()
    assert described_map["OPENAI_API_KEY"] == "set"
    assert described_map["  access_key"] == "set"
    assert described_map["  bucket"].startswith("—")


def test_loading_logs_names_not_values(tmp_path: Path, caplog):
    env = tmp_path / ".env"
    env.write_text(f"OPENAI_API_KEY={CANARY}\n", encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        load_env_file(env)

    assert "OPENAI_API_KEY" in caplog.text
    assert CANARY not in caplog.text


def test_model_dump_does_not_leak(monkeypatch):
    """Serialising the model must not expose the secret either."""
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    dumped = str(Settings.load().model_dump())
    assert CANARY not in dumped


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def test_paths_default_to_unset_so_the_config_decides():
    paths = PathSettings.from_env()
    assert paths.input_dir is None
    assert paths.output_dir is None
    assert paths.cache_dir is None
    assert paths.manifest is None


def test_paths_read_their_env_vars(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VD_INPUT_DIR", str(tmp_path / "clips"))
    monkeypatch.setenv("VD_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("VD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VD_MANIFEST", str(tmp_path / "m.csv"))

    paths = PathSettings.from_env()

    assert paths.input_dir == tmp_path / "clips"
    assert paths.output_dir == tmp_path / "out"
    assert paths.cache_dir == tmp_path / "cache"
    assert paths.manifest == tmp_path / "m.csv"


def test_env_var_beats_the_module_constant(monkeypatch, tmp_path: Path):
    """So a one-off run never has to edit settings.py."""
    monkeypatch.setattr("visual_diversity.settings.DEFAULT_OUTPUT_DIR", "/from/the/file")
    monkeypatch.setenv("VD_OUTPUT_DIR", str(tmp_path / "from-env"))

    assert PathSettings.from_env().output_dir == tmp_path / "from-env"


def test_module_constant_is_used_when_no_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("visual_diversity.settings.DEFAULT_OUTPUT_DIR", str(tmp_path / "edited"))
    assert PathSettings.from_env().output_dir == tmp_path / "edited"


def test_blank_constant_means_unset(monkeypatch):
    monkeypatch.setattr("visual_diversity.settings.DEFAULT_INPUT_DIR", "   ")
    assert PathSettings.from_env().input_dir is None


def test_resolve_clip_anchors_relative_paths(tmp_path: Path):
    paths = PathSettings(input_dir=tmp_path / "clips")
    assert paths.resolve_clip(Path("a/b.mp4")) == tmp_path / "clips" / "a" / "b.mp4"


def test_resolve_clip_leaves_absolute_paths_alone(tmp_path: Path):
    paths = PathSettings(input_dir=tmp_path / "clips")
    absolute = tmp_path / "elsewhere" / "c.mp4"
    assert paths.resolve_clip(absolute) == absolute


def test_resolve_clip_is_a_noop_without_an_input_dir():
    assert PathSettings().resolve_clip(Path("rel.mp4")) == Path("rel.mp4")


def test_paths_appear_in_the_status_report(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VD_OUTPUT_DIR", str(tmp_path / "out"))
    report = Settings.load().report()
    assert "output_dir" in report
    assert str(tmp_path / "out") in report


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def test_get_settings_is_cached_until_reloaded(monkeypatch):
    first = get_settings(reload=True)
    assert first.openai.configured is False

    monkeypatch.setenv("OPENAI_API_KEY", "later")
    assert get_settings().openai.configured is False      # still the cached view
    assert get_settings(reload=True).openai.configured is True
