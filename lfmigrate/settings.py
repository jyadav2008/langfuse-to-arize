"""Tunable settings, all overridable by environment variable.

Why this module exists
----------------------
Behaviour used to be split between a 12-key allowlist in ``config.py`` and
hard-coded constants scattered across modules. That had two consequences for
anyone deploying this against their own stack:

* A knob like the Flight port was *supported* by ``regions.py`` but could not
  be set, because ``config`` dropped the key.
* Anything not on the allowlist was ignored **silently** — set
  ``ARIZE_FLIGHT_PORT=443`` and nothing happened, with no warning. That is the
  same failure shape as every other trap in this codebase: looks applied, does
  nothing.

So every tunable is declared here once, with its type, default and purpose.
:func:`unknown_keys` then reports anything that *looks* like configuration for
this tool but matches no declared setting, so a typo is loud instead of inert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

#: Prefixes that mark an environment variable as "intended for this tool".
#: A variable with one of these prefixes that matches no declared setting is
#: almost certainly a typo or a stale name, and is reported.
OWNED_PREFIXES = ("LANGFUSE_", "ARIZE_", "LFMIGRATE_")


@dataclass(frozen=True)
class Setting:
    key: str
    default: Any
    cast: Callable[[str], Any]
    help: str
    secret: bool = False
    required: bool = False


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: str) -> int:
    return int(str(value).strip())


def _as_str(value: str) -> str:
    return str(value).strip()


SETTINGS: tuple[Setting, ...] = (
    # ── Source: Langfuse ────────────────────────────────────────────────────
    Setting("LANGFUSE_HOST", "https://cloud.langfuse.com", _as_str,
            "Langfuse base URL. Include any path prefix for a self-hosted deployment.",
            required=True),
    Setting("LANGFUSE_PUBLIC_KEY", None, _as_str,
            "Langfuse public key (pk-...).", secret=True, required=True),
    Setting("LANGFUSE_SECRET_KEY", None, _as_str,
            "Langfuse secret key (sk-...).", secret=True, required=True),
    Setting("LANGFUSE_SOURCE_MODE", "api", _as_str,
            "Where to read history from. Only 'api' is implemented; 'blob' and "
            "'clickhouse' fail fast with a clear error."),
    Setting("LANGFUSE_API_GENERATION", None, _as_str,
            "Pin the API generation (v4 | v3) instead of probing for it."),
    Setting("LANGFUSE_TIMEOUT_SECONDS", 60, _as_int,
            "HTTP timeout for Langfuse requests."),
    Setting("LANGFUSE_PAGE_LIMIT", 1000, _as_int,
            "Records per page when reading observations. v4 allows up to 1000."),
    Setting("LANGFUSE_SCORE_PAGE_LIMIT", 100, _as_int,
            "Records per page when reading scores. v4 caps this at 100."),
    Setting("LANGFUSE_OBSERVATION_FIELDS", None, _as_str,
            "Override the observation field groups requested. The default asks for "
            "everything; narrowing it risks migrating spans with no content."),
    Setting("LANGFUSE_SCORE_FIELDS", None, _as_str,
            "Override the score field groups. Must include 'subject' on v4 or score "
            "linkage is lost."),
    Setting("LANGFUSE_METRICS_CHUNK_DAYS", 60, _as_int,
            "Days per metrics request when discovering the history range. A span of "
            "two years in one request returns empty rather than erroring."),
    Setting("LANGFUSE_EXPAND_METADATA_KEYS", None, _as_str,
            "Comma-separated metadata keys to return untruncated. Langfuse truncates "
            "metadata values over 200 characters by default."),

    # ── ClickHouse (self-hosted fast path) ─────────────────────────────────

    # ── Destination: Arize AX ──────────────────────────────────────────────
    Setting("ARIZE_API_KEY", None, _as_str, "Arize AX API key.",
            secret=True, required=True),
    Setting("ARIZE_SPACE_ID", None, _as_str, "Arize AX space id.", required=True),
    Setting("ARIZE_PROJECT_NAME", None, _as_str,
            "Destination project, used verbatim with no timestamp. Leave "
            "unset to auto-generate a fresh, never-reused name."),
    Setting("ARIZE_PROJECT_PREFIX", "langfuse-backfill", _as_str,
            "Prefix for the auto-generated name. Defaults to the Langfuse "
            "project name, so the two systems read alike."),
    Setting("ARIZE_AI_INTEGRATION_ID", None, _as_str,
            "AX AI integration that migrated LLM judges run on. Unset: judges "
            "are not migrated."),
    Setting("LFMIGRATE_EVAL_INTEGRATIONS", None, _as_str,
            "Per-provider integrations, e.g. openai=<id>,anthropic=<id>."),
    Setting("LFMIGRATE_EVAL_DEFAULT_MODEL", None, _as_str,
            "Model for judges that use the Langfuse project default."),
    Setting("LFMIGRATE_EVAL_EXPORT_DIR", "out/code-evaluators", _as_str,
            "Where code evaluators are written for manual porting."),
    Setting("LFMIGRATE_CREATE_EVAL_TASKS", False, _as_bool,
            "Schedule evaluation rules as AX tasks. Off: tasks bill the model "
            "provider."),
    Setting("ARIZE_REGION", "us", _as_str,
            "Region the space is homed in: us | us-central-1a | us-east-1b | eu."),
    Setting("ARIZE_API_HOST", None, _as_str, "Override the REST host."),
    Setting("ARIZE_OTLP_HOST", None, _as_str, "Override the OTLP host."),
    Setting("ARIZE_FLIGHT_HOST", None, _as_str, "Override the Arrow Flight host."),
    Setting("ARIZE_FLIGHT_PORT", 443, _as_int, "Arrow Flight port."),

    # ── Migration behaviour ────────────────────────────────────────────────
    Setting("LFMIGRATE_INGEST_PATH", "auto", _as_str,
            "auto (default; arrow when history is older than 31 days or "
            "over ~830k spans, else otlp) | otlp (visible in seconds) | "
            "arrow (~1.8x upload rate)."),
    Setting("LFMIGRATE_OTLP_SHARDS", 8, _as_int,
            "Concurrent OTLP exporters; the rate is flat from 4 upward."),
    Setting("LFMIGRATE_BATCH_SIZE", 10000, _as_int,
            "Spans per upload batch. 10k matches the SDK's own chunk size; "
            "500 is ~4.5x slower."),
    Setting("LFMIGRATE_BATCH_MAX_MB", 32, _as_int,
            "Byte budget per batch; splits a batch of unusually large spans."),
    Setting("LFMIGRATE_VALIDATE_UPLOAD", True, _as_bool,
            "SDK-side dataframe validation. Costs ~31% of upload time; safe "
            "to disable after a successful trial run."),
    Setting("LFMIGRATE_MAX_PAST_YEARS", 2, _as_int,
            "Arize retention window in years. Raise only if Arize has widened it "
            "for this space."),
    Setting("LFMIGRATE_VERIFY_TIMEOUT_SECONDS", 900, _as_int,
            "How long to poll readback before declaring a day unverified. "
            "Indexing has been measured at around five minutes."),
    Setting("LFMIGRATE_VERIFY_INTERVAL_SECONDS", 30, _as_int,
            "Poll interval while waiting for indexing."),
    Setting("LFMIGRATE_SHARD_ROOT", "out/manifests", _as_str,
            "Where shards and manifests are written."),
    Setting("LFMIGRATE_VERIFY_MODE", "end", _as_str,
            "Readback verification: end (one check, default) | each-day "
            "(blocks ~5 min per day) | off."),
    Setting("LFMIGRATE_QUIET_SDK_WARNINGS", True, _as_bool,
            "Hide the SDK's per-endpoint [BETA] warnings."),
)

BY_KEY = {s.key: s for s in SETTINGS}
REQUIRED = tuple(s.key for s in SETTINGS if s.required)
SECRET_KEYS = frozenset(s.key for s in SETTINGS if s.secret)


def unknown_keys(values: dict) -> list[str]:
    """Keys that look like ours but match no declared setting.

    Reported rather than ignored: a silently dropped override is
    indistinguishable from one that had no effect.
    """
    return sorted(
        key for key in values
        if key.upper().startswith(OWNED_PREFIXES) and key not in BY_KEY
    )


def coerce(key: str, raw: str):
    setting = BY_KEY.get(key)
    if setting is None:
        return raw
    try:
        return setting.cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key}={raw!r} is not valid for this setting "
            f"({setting.help.split('.')[0]})."
        ) from exc


def defaults() -> dict:
    return {s.key: s.default for s in SETTINGS if s.default is not None}


def describe() -> list[dict]:
    """Full settings reference, for `lfmigrate config --list`."""
    return [
        {"key": s.key, "default": "<unset>" if s.default is None else s.default,
         "secret": s.secret, "required": s.required, "help": s.help}
        for s in SETTINGS
    ]


def parse_pairs(raw: str | None) -> dict[str, str]:
    """Parse a 'A=1,B=2' override string."""
    out: dict[str, str] = {}
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected KEY=VALUE pairs, got {chunk!r}.")
        key, value = chunk.split("=", 1)
        out[key.strip()] = value.strip()
    return out

#: Prefix -> section title. Ordered most-specific-first, because LANGFUSE_
#: would otherwise swallow LANGFUSE_CLICKHOUSE_.
SECTION_PREFIXES = (
    ("LANGFUSE_CLICKHOUSE_", "Source: ClickHouse direct (self-hosted v4 fast path)"),
    ("LANGFUSE_", "Source: Langfuse"),
    ("ARIZE_", "Destination: Arize AX"),
    ("LFMIGRATE_", "Migration behaviour"),
)

#: Display order, which is NOT the matching order: what a customer must fill in
#: comes first, and the optional ClickHouse path comes last.
SECTION_ORDER = (
    "Source: Langfuse",
    "Destination: Arize AX",
    "Migration behaviour",
    "Source: ClickHouse direct (self-hosted v4 fast path)",
)


def _section_of(key: str) -> str:
    for prefix, title in SECTION_PREFIXES:
        if key.startswith(prefix):
            return title
    return "Other"


def short_help(text: str, limit: int = 52) -> str:
    """A one-line hint. Full prose lives in `config --list`."""
    flat = " ".join(text.split())
    # Split on sentence end, but not on an abbreviation like "e.g." or "i.e.",
    # which would clip the hint mid-phrase.
    parts, first = flat.split(". "), ""
    for part in parts:
        first = part if not first else f"{first}. {part}"
        if not first.rstrip(".").lower().endswith(("e.g", "i.e", "etc")):
            break
    first = first.rstrip(".")
    if len(first) <= limit:
        return first
    clipped = first[:limit].rsplit(" ", 1)[0]
    return clipped + "…"


def render_env_template() -> str:
    """Generate a compact .env template from the declared settings.

    Generated rather than hand-maintained so it cannot drift from the settings
    it documents -- a stale template is worse than none, because it implies a
    knob exists that the tool ignores. Tests assert the committed file matches
    this output exactly.
    """
    width = max(len(s.key) for s in SETTINGS) + 1
    lines = [
        "# lfmigrate configuration. Copy to .env and chmod 600 "
        "(the loader refuses group/world-readable files).",
        "# Required settings are active; the rest are commented at their default. "
        "No CLI flag takes a secret.",
        "#",
        "# Full reference:  python -m lfmigrate config --list",
        "# This file:       python -m lfmigrate config --example > .env.example",
    ]
    rendered: set[str] = set()
    titles = list(SECTION_ORDER) + [
        t for t in dict.fromkeys(_section_of(s.key) for s in SETTINGS)
        if t not in SECTION_ORDER
    ]
    for title in titles:
        group = [s for s in SETTINGS
                 if _section_of(s.key) == title and s.key not in rendered]
        if not group:
            continue
        lines += ["", f"# ── {title} " + "─" * max(0, 62 - len(title))]
        for setting in group:
            rendered.add(setting.key)
            default = setting.default
            if isinstance(default, bool):
                default = "true" if default else "false"
            shown = "" if default is None else default
            assignment = f"{setting.key}={shown}"
            if not setting.required:
                assignment = "# " + assignment
            note = short_help(setting.help)
            if setting.secret:
                note = "secret — " + note
            if setting.required:
                note = "REQUIRED — " + note
            pad = " " * max(1, width + 2 - len(assignment))
            lines.append(f"{assignment}{pad}# {note}")
    return "\n".join(lines).rstrip() + "\n"


def _wrap(text: str, width: int = 72) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width=width) or [""]
