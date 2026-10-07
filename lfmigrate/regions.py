"""Arize AX region routing, resolved to EXPLICIT hosts.

Why this module exists
----------------------
AX spaces are region-homed and the SDKs default to US. A mismatch does not say
"wrong region" -- it reports:

    OTLP span path    PERMISSION_DENIED: invalid Space ID
    Arrow Flight path auth-error: invalid token

Both read as dead credentials. That costs hours.

Worse, the three planes do not always agree for a given region, so passing
``region=Region(...)`` wholesale to ``ArizeClient`` is not safe: for
``us-central-1a`` it derives ``api.us-central-1a.arize.com``, whose TLS
certificate does not match the hostname, and every control-plane call fails
with ``SSLCertVerificationError``. Observed:

    HTTPSConnectionPool(host='api.us-central-1a.arize.com', port=443):
    certificate is not valid for 'api.us-central-1a.arize.com'

So this module resolves a region name to explicit per-plane hosts and the
caller passes those. Defaults come from the SDK's own ``REGION_ENDPOINTS`` table
so regions added upstream work without editing this file; ``_EXCEPTIONS`` then
overrides only the planes where the SDK's table is known to be wrong in
practice.

``ArizeClient`` also reads ``ARIZE_REGION`` from the process environment and
re-derives hosts from it, which silently defeats explicit overrides. Callers
must use :func:`client_kwargs`, which strips it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_API_HOST = "api.arize.com"
DEFAULT_OTLP_HOST = "otlp.arize.com"
DEFAULT_FLIGHT_HOST = "flight.arize.com"
FLIGHT_PORT = 443


@dataclass(frozen=True)
class RegionHosts:
    name: str
    api_host: str
    otlp_host: str
    flight_host: str
    flight_port: int = FLIGHT_PORT
    #: False when these hosts have not been confirmed against a live space.
    #: The CLI warns rather than guessing silently.
    verified: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


# Aliases a human is likely to type -> SDK region value.
ALIASES = {
    "us": None,  # default/global US endpoints, no region suffix
    "us-east-1b": "us-east-1b",
    "us-central-1a": "us-central-1a",
    "eu": "eu-west-1a",
    "eu-west-1a": "eu-west-1a",
    "ca": "ca-central-1a",
    "ca-central-1a": "ca-central-1a",
}

# Planes where the SDK's derived host is wrong in practice, or where we have
# positive confirmation against a live space. Keyed by SDK region value
# (None = the default US endpoints).
#
# Provenance of each entry is recorded because an unverified guess here is
# indistinguishable from a verified fact at call time.
_EXCEPTIONS: dict[str | None, dict] = {
    None: {
        "verified": True,
        "notes": ("Default global US endpoints.",),
    },
    "us-central-1a": {
        # Confirmed against a live space: OTLP and the control plane must use
        # the DEFAULT hosts; only Flight is region-specific.
        "api_host": DEFAULT_API_HOST,
        "otlp_host": DEFAULT_OTLP_HOST,
        "verified": True,
        "notes": (
            "api.us-central-1a.arize.com fails TLS hostname verification; use api.arize.com.",
            "OTLP accepted on otlp.arize.com; otlp.eu-west-1a rejects with 'invalid Space ID'.",
            "Flight MUST be flight.us-central-1a.arize.com; flight.arize.com returns "
            "'auth-error: invalid token'.",
        ),
    },
    "eu-west-1a": {
        "verified": True,
        "notes": (
            "All three planes confirmed on region-specific hosts against a live space.",
        ),
    },
}


def _sdk_endpoints() -> dict[str | None, dict]:
    """Per-region hosts from the installed SDK, if it exposes them."""
    out: dict[str | None, dict] = {}
    try:
        from arize.regions import REGION_ENDPOINTS  # type: ignore
    except Exception:
        return out
    for region, endpoints in (REGION_ENDPOINTS or {}).items():
        value = getattr(region, "value", region) or None
        if not value:
            continue
        out[value] = {
            "api_host": getattr(endpoints, "api_host", None),
            "otlp_host": getattr(endpoints, "otlp_host", None),
            "flight_host": getattr(endpoints, "flight_host", None),
            "flight_port": getattr(endpoints, "flight_port", None) or FLIGHT_PORT,
        }
    return out


def known_regions() -> list[str]:
    names = {"us"}
    names.update(k for k in ALIASES if k != "us")
    names.update(k for k in _sdk_endpoints() if k)
    return sorted(names)


def resolve(region: str | None, overrides: dict | None = None) -> RegionHosts:
    """Resolve a region name to explicit per-plane hosts.

    Precedence: explicit ``overrides`` > ``_EXCEPTIONS`` > SDK table > defaults.
    Raises ``KeyError`` on an unknown region rather than silently falling back
    to US, because a silent US fallback is the failure this module exists to
    prevent.
    """
    name = (region or "us").strip().lower()
    if name not in ALIASES and name not in _sdk_endpoints():
        raise KeyError(
            f"Unknown AX region {name!r}. Known: {', '.join(known_regions())}. "
            "Pass explicit ARIZE_API_HOST / ARIZE_OTLP_HOST / ARIZE_FLIGHT_HOST "
            "for a self-hosted deployment."
        )
    sdk_value = ALIASES.get(name, name)

    hosts = {
        "api_host": DEFAULT_API_HOST,
        "otlp_host": DEFAULT_OTLP_HOST,
        "flight_host": DEFAULT_FLIGHT_HOST,
        "flight_port": FLIGHT_PORT,
    }
    for key, value in (_sdk_endpoints().get(sdk_value) or {}).items():
        if value:
            hosts[key] = value

    exception = dict(_EXCEPTIONS.get(sdk_value) or {})
    verified = bool(exception.pop("verified", False))
    notes = tuple(exception.pop("notes", ()))
    hosts.update({k: v for k, v in exception.items() if v})

    for key in ("api_host", "otlp_host", "flight_host", "flight_port"):
        value = (overrides or {}).get(key)
        if value:
            hosts[key] = value
            verified = False
            notes = notes + (f"{key} overridden by configuration.",)

    return RegionHosts(name=name, verified=verified, notes=notes, **hosts)


def otlp_endpoint(hosts: RegionHosts) -> str:
    """Full OTLP endpoint URL for arize.otel.register(endpoint=...)."""
    return f"https://{hosts.otlp_host}/v1"


def client_kwargs(hosts: RegionHosts) -> dict:
    """Kwargs for ``ArizeClient`` that pin every plane explicitly.

    Deliberately does NOT pass ``region=``: that re-derives hosts and would
    reintroduce the TLS failure this module routes around. Also clears
    ``ARIZE_REGION`` from the environment, which the client reads on its own and
    which would otherwise override these values.
    """
    os.environ.pop("ARIZE_REGION", None)
    return {
        "api_host": hosts.api_host,
        "api_scheme": "https",
        "flight_host": hosts.flight_host,
        "flight_port": hosts.flight_port,
        "flight_scheme": "grpc+tls",
    }
