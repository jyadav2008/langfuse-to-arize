"""The contract a history source must satisfy.

This exists because "pluggable source" was previously aspirational: the
pipeline imported ``LangfuseAPI`` directly and called nine methods on it, so
adding a second source meant editing the orchestration. Declaring the surface
here makes the dependency explicit and checkable, and keeps the pipeline honest
about what it actually needs.

Three sources are anticipated, and all of them can satisfy this:

``api``
    The Langfuse public REST API. Works for Cloud and self-hosted, v3 and v4.

``clickhouse``
    Direct reads against a self-hosted v4 ClickHouse. Faster for bulk history,
    and the only realistic option at several million rows. Note that v4's
    legacy ``observations`` table is empty -- the full-fidelity data lives in
    ``events_full``.

``blob``
    An unpacked Langfuse blob/Parquet export. Useful where API access is
    restricted or rate limits make pagination impractical.

The pipeline depends on this protocol and nothing narrower, so a new source
needs no orchestration changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class HistorySource(Protocol):
    """A readable source of Langfuse history.

    Implementations must be safe to call repeatedly: the pipeline may re-export
    a day when resuming, and an export that is not reproducible would produce
    different span IDs and therefore duplicates downstream.
    """

    def detect(self) -> dict:
        """Identify the source generation/capabilities.

        Returns a dict with at least ``name`` (e.g. ``"v4"``). The pipeline
        records this in each shard manifest so a resumed migration can tell
        whether the source changed underneath it.
        """
        ...

    def history_range(self, *, max_past_years: int = 2) -> dict:
        """Discover what history exists, without being told.

        Returns ``{"earliest", "latest", "detected", "daily_counts", ...}``.
        ``detected=False`` with a ``reason`` is a valid answer and must not
        raise -- the caller falls back to scanning the retention window.

        ``daily_counts`` doubles as the reconciliation oracle, so implementations
        should populate it where the source can answer cheaply.
        """
        ...

    def day_windows(self, earliest: datetime,
                    latest: datetime) -> list[tuple[datetime, datetime]]:
        """Contiguous UTC day windows covering the range, oldest first."""
        ...

    def observations(self, start: datetime, end: datetime,
                     extra: dict | None = None) -> Iterator[dict]:
        """Observation records with start time in ``[start, end)``.

        Must yield the FULL record -- input, output, metadata, model, usage and
        cost. A source that returns a summary projection produces spans that
        reconcile perfectly by count and carry no content, which is the worst
        available failure mode.
        """
        ...

    def scores_for_traces(self, trace_ids: list[str]) -> Iterator[dict]:
        """Scores belonging to specific traces.

        Keyed on trace rather than on a timestamp window deliberately: a score's
        own timestamp need not fall on the same day as the trace it annotates
        (a human annotation can land days later), so windowing by score
        timestamp orphans scores whose trace sits in another shard.
        """
        ...

    def scores_for_sessions(self, session_ids: list[str]) -> Iterator[dict]:
        """Session-level scores, which are addressed separately from traces."""
        ...

    def close(self) -> None:
        """Release connections. Must be safe to call more than once."""
        ...


#: Methods the pipeline actually calls. Used by the conformance check so a new
#: source fails loudly at construction rather than part-way through a migration.
REQUIRED_METHODS = (
    "detect",
    "history_range",
    "day_windows",
    "observations",
    "scores_for_traces",
    "scores_for_sessions",
    "close",
)


class SourceNotImplemented(Exception):
    """Raised for a configured source mode that has no implementation yet."""


def check_conformance(source) -> None:
    """Verify a source satisfies the contract before any migration starts.

    Deliberately eager. Discovering a missing method on day 40 of a backfill,
    after thousands of spans are already in the destination, is far worse than
    failing at startup.
    """
    missing = [name for name in REQUIRED_METHODS
               if not callable(getattr(source, name, None))]
    if missing:
        raise SourceNotImplemented(
            f"{type(source).__name__} does not satisfy HistorySource; "
            f"missing: {', '.join(missing)}"
        )
