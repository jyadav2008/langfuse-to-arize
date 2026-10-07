"""Deterministic ID derivation for Langfuse -> AX.

AX requires W3C trace context shapes: 32 hex characters for a trace ID, 16 for
a span ID. Langfuse emits either already-conforming hex (its OpenTelemetry
SDKs) or UUIDs/opaque strings (older SDKs, and trace records).

Two properties matter more than they appear to:

Determinism
    A re-export must produce byte-identical IDs, otherwise a resumed or
    repeated run creates DUPLICATE spans rather than overwriting. There is no
    evidence AX de-duplicates by span ID -- the Phoenix migration tooling
    assumes it does not and rejects manifests containing duplicates -- so
    stability here is the only thing preventing double-counting.

Namespacing
    Derived IDs are salted with a purpose prefix. A synthesized root span ID is
    derived from a *trace* ID, so without a namespace it could in principle
    collide with a real observation's derived ID. Namespacing makes the
    derivations disjoint by construction instead of by probability.
"""

from __future__ import annotations

import hashlib
import re

_HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")
_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")

#: Purpose salts. Changing one of these changes every ID it produces and will
#: duplicate previously migrated data, so they are versioned and frozen.
NS_SPAN = "lfmigrate:v1:span:"
NS_TRACE = "lfmigrate:v1:trace:"
NS_ROOT = "lfmigrate:v1:root:"

#: All-zero IDs are invalid trace context and signal an upstream bug.
_ZERO16 = "0" * 16
_ZERO32 = "0" * 32


def _digest(namespace: str, value: str, width: int) -> str:
    return hashlib.sha256((namespace + value).encode("utf-8")).hexdigest()[:width]


def _normalize(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def span_id(value) -> str | None:
    """16 hex chars. Passes through conforming input, hashes anything else."""
    text = _normalize(value)
    if text is None:
        return None
    candidate = text.lower().replace("-", "")
    if _HEX16.fullmatch(candidate) and candidate != _ZERO16:
        return candidate
    return _digest(NS_SPAN, text, 16)


def trace_id(value) -> str | None:
    """32 hex chars. Passes through conforming input, hashes anything else."""
    text = _normalize(value)
    if text is None:
        return None
    candidate = text.lower().replace("-", "")
    if _HEX32.fullmatch(candidate) and candidate != _ZERO32:
        return candidate
    return _digest(NS_TRACE, text, 32)


def root_span_id(trace_identifier) -> str | None:
    """Span ID for the root span synthesized from a Langfuse trace record.

    Langfuse models a trace as its own record carrying name, input, output,
    sessionId, userId and tags, with its top-level observations pointing at no
    parent. Without a synthesized root, three things break:

    * trace-level input/output/tags are lost outright;
    * a trace with two top-level observations renders as two separate roots,
      which breaks the trace tree;
    * a trace-level score has no single span to attach to.

    Derived under its own namespace so it cannot collide with a derived
    observation ID.
    """
    text = _normalize(trace_identifier)
    if text is None:
        return None
    return _digest(NS_ROOT, text, 16)
