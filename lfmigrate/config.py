"""Configuration and credential loading.

Security rules this module enforces, not merely documents:

* Credentials are read from an env file or the process environment only. There
  is no CLI flag that accepts a secret, so secrets cannot reach ``argv`` (world
  readable via ``ps``), shell history, or a CI command log.
* The env file must be owner-only (0600). A group/world-readable credential
  file is refused rather than warned about.
* ``__repr__``/``__str__`` on the config object redact secret values, so an
  accidental ``print(config)`` or a traceback frame cannot leak a key.
* :func:`redact` is the only sanctioned way to put a credential-adjacent value
  in output.

Secret keys are classified by name suffix rather than an allowlist, so a new
secret added later is redacted by default instead of leaking until someone
remembers to list it.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

#: Any key containing one of these fragments is treated as secret.
_SECRET_FRAGMENTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")

from . import settings as S  # noqa: E402  (after _SECRET_FRAGMENTS, by design)

#: Declared in settings.py, which is the single source of truth for every
#: tunable. Keys that look like ours but are not declared are REPORTED, not
#: dropped -- see Config.warnings.
KNOWN_KEYS = tuple(S.BY_KEY)

#: Credential files in the wild name the space inconsistently. Accept both.
_ALIASES = {"SPACE_ID": "ARIZE_SPACE_ID", "ARIZE_SPACE": "ARIZE_SPACE_ID"}


class ConfigError(Exception):
    """Raised for missing or unsafe configuration. Never contains a secret."""


def is_secret(key: str) -> bool:
    """Secret by declaration, or by name shape as a backstop.

    The name-shape fallback matters: a secret added later is redacted by
    default rather than leaking until someone remembers to declare it.
    """
    if key in S.SECRET_KEYS:
        return True
    upper = key.upper()
    return any(fragment in upper for fragment in _SECRET_FRAGMENTS)


def redact(value: str | None, *, keep: int = 4) -> str:
    """Render a secret as a non-reversible fingerprint.

    Shows only the length and a short prefix. The prefix is useful because AX
    and Langfuse keys are type-tagged (``ak-``, ``pk-``, ``sk-``), so it lets a
    human confirm they supplied the right KIND of key without exposing it.
    """
    if not value:
        return "<unset>"
    prefix = value[:keep]
    return f"{prefix}…<{len(value)} chars>"


@dataclass
class Config:
    values: dict = field(default_factory=dict)
    source_path: Path | None = None
    #: Keys that look like configuration for this tool but match no declared
    #: setting. Surfaced by preflight so a typo cannot pass as applied.
    warnings: list = field(default_factory=list)

    def get(self, key: str, default=None):
        """Resolved value: explicit setting, else declared default, else default."""
        if key in self.values:
            return self.values[key]
        setting = S.BY_KEY.get(key)
        if setting is not None and setting.default is not None:
            return setting.default
        return default

    def int_(self, key: str, default=None) -> int | None:
        value = self.get(key, default)
        return None if value is None else int(value)

    def bool_(self, key: str, default=False) -> bool:
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def pairs(self, key: str) -> dict:
        """A 'A=1,B=2' setting parsed into a dict."""
        return S.parse_pairs(self.get(key))

    def csv(self, key: str) -> list:
        raw = self.get(key)
        return [p.strip() for p in str(raw).split(",") if p.strip()] if raw else []

    def require(self, *keys: str) -> list[str]:
        """Return the values for ``keys``, raising if any is missing."""
        missing = [k for k in keys if not self.values.get(k)]
        if missing:
            raise ConfigError(
                "Missing required configuration: "
                + ", ".join(sorted(missing))
                + ". Add them to an owner-only env file and pass --env-file; "
                "do not pass credentials on the command line."
            )
        return [self.values[k] for k in keys]

    def overrides(self) -> dict:
        """Explicit host overrides for regions.resolve()."""
        mapping = {
            "api_host": "ARIZE_API_HOST",
            "otlp_host": "ARIZE_OTLP_HOST",
            "flight_host": "ARIZE_FLIGHT_HOST",
            "flight_port": "ARIZE_FLIGHT_PORT",
        }
        out = {}
        for target, key in mapping.items():
            # Only an EXPLICIT value counts as an override; the declared
            # default for flight_port must not masquerade as one.
            if self.values.get(key):
                out[target] = self.values[key]
        return out

    def safe_summary(self) -> dict:
        """Loggable view: secrets fingerprinted, everything else verbatim."""
        return {
            key: (redact(value) if is_secret(key) else value)
            for key, value in sorted(self.values.items())
        }

    # Redacting repr/str so an accidental print or a traceback cannot leak.
    def __repr__(self) -> str:
        return f"Config({self.safe_summary()!r})"

    __str__ = __repr__


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ``# comment`` from an env value.

    Standard .env behaviour, and load-bearing here: the generated template puts
    an aligned hint after each assignment, so without this a key is read with
    its own documentation appended -- an API key arrives 40 characters too long
    and authentication fails for a reason nobody would guess.

    Only a ``#`` preceded by whitespace starts a comment, so a value may
    legitimately contain one (``pass#word``). Quoted values are left intact;
    quote stripping happens after this.
    """
    if not value or not value.strip():
        return ""
    stripped = value.lstrip()
    if stripped.startswith("#"):
        # The whole value is a comment, i.e. the setting was left unset.
        return ""
    value = stripped
    if value[0] in "\"'":
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            # Keep the quoted span; drop anything after it.
            return value[: end + 1]
        return value
    cut = len(value)
    for index, char in enumerate(value):
        if char == "#" and index > 0 and value[index - 1] in " \t":
            cut = index
            break
    return value[:cut].strip()


def _parse_env_file(path: Path) -> dict:
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        # Comment stripped from the RAW value: stripping whitespace first would
        # move a '#' to position 0, where it is no longer recognisable as a
        # comment, and an unfilled "KEY=   # REQUIRED ..." line would be read as
        # its own documentation.
        value = _strip_inline_comment(value).strip()
        # Strip one layer of matching quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def check_file_permissions(path: Path) -> None:
    """Refuse a credential file readable by anyone but its owner."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
        raise ConfigError(
            f"Credential file {path} is group- or world-accessible "
            f"(mode {stat.filemode(mode)}). Run: chmod 600 {path}"
        )


def load(env_file: str | Path | None = None, environ: dict | None = None,
         *, require_secure_file: bool = True) -> Config:
    """Load configuration from an env file plus the process environment.

    The env file wins over the ambient environment: an explicitly supplied file
    is a deliberate act, whereas an inherited variable is often a leftover from
    another project pointing at the wrong space.
    """
    environ = os.environ if environ is None else environ
    values: dict = {}
    seen: dict = {}

    for key, value in environ.items():
        canonical = _ALIASES.get(key, key)
        if value:
            seen[canonical] = value
        if canonical in KNOWN_KEYS and value:
            values[canonical] = S.coerce(canonical, value)

    source_path = None
    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ConfigError(f"Env file not found: {path}")
        if require_secure_file:
            check_file_permissions(path)
        source_path = path
        for key, value in _parse_env_file(path).items():
            canonical = _ALIASES.get(key, key)
            if value:
                seen[canonical] = value
            if canonical in KNOWN_KEYS and value:
                values[canonical] = S.coerce(canonical, value)

    warnings = [
        f"{key} is not a recognised setting and was ignored. "
        f"Run 'lfmigrate config --list' for the full list."
        for key in S.unknown_keys(seen)
    ]
    # Defaults are resolved lazily by Config.get so that `values` holds only
    # what was explicitly supplied -- which is what overrides() must key on.
    return Config(values=values, source_path=source_path, warnings=warnings)
