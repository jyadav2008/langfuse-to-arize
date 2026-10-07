"""Upload a shard's spans to Arize AX.

``expand_indexed`` and the row-shaping in ``span_row`` are adapted from
Arize's Phoenix migration tooling
(arize-skills/skills/arize-phoenix-migration/scripts/migrate.py), which is the
reference implementation for this upload path. They are vendored rather than
imported because this is standalone tooling with no dependency on that repo.

The exception taxonomy is the important part and is also inherited:

``PreparationError``
    The rows cannot be represented. Deterministic -- the same input will fail
    again, so the batch is reset to pending and nothing was sent.

``RejectedUpload``
    AX refused the request (4xx, or an auth failure). Also deterministic: fix
    the cause and re-run. Nothing landed.

``AmbiguousUpload``
    A timeout, 5xx, or transport failure AFTER the request went out. It is
    unknown whether the write landed. This must NOT be retried blindly -- the
    batch stays ``uncertain`` and only a readback can resolve it.

Collapsing that third case into "just retry" is what produces duplicate spans,
because the upload path does not de-duplicate by span ID.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from . import manifest as M

#: Mirrors the server-side limit in
#: go/pkg/lib/validation/tracing.go:spanHasValidStartAndEndTimes. Overridable
#: per space via timeRangeAllowedBySpaceYears, which requires an Arize-side
#: code change and deploy -- so it must be checked BEFORE a long migration.
DEFAULT_MAX_PAST_YEARS = 2

_INDEXED = re.compile(r"\A(llm\.(?:input_messages|output_messages|tools))\.(\d+)\.(.+)\Z")
_TOOL_CALL = re.compile(r"\Amessage\.tool_calls\.(\d+)\.(.+)\Z")


class UploadError(Exception):
    pass


class PreparationError(UploadError):
    """Rows could not be prepared. Nothing was sent."""


class RejectedUpload(UploadError):
    """AX refused the request. Nothing landed."""


class AmbiguousUpload(UploadError):
    """Outcome unknown. Resolve by readback; never retry blindly."""


def nanos(value) -> int:
    """UTC nanoseconds from an ISO-8601 string, datetime, or epoch number."""
    if value is None:
        raise PreparationError("Span is missing a timestamp.")
    if isinstance(value, (int, float)):
        # Heuristic on magnitude: s / ms / us / ns.
        number = float(value)
        for threshold, scale in ((1e11, 1e9), (1e14, 1e6), (1e17, 1e3)):
            if number < threshold:
                return int(number * scale)
        return int(number)
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise PreparationError(f"Unparseable timestamp: {value!r}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def expand_indexed(attributes: dict) -> dict:
    """Fold ``llm.input_messages.0.message.role`` keys into lists of dicts.

    AX renders a chat view for list-of-dict message columns and an opaque JSON
    blob for flat keys, so this is the difference between a usable trace view
    and an unreadable one.

    Adapted from the Phoenix migration tooling.
    """
    result = dict(attributes)
    groups: dict[str, dict[int, dict]] = {}
    for key, value in attributes.items():
        match = _INDEXED.match(key)
        if not match:
            continue
        prefix, index, suffix = match.groups()
        groups.setdefault(prefix, {}).setdefault(int(index), {})[suffix] = value
        result.pop(key)
    for prefix, items in groups.items():
        expanded = []
        for index in range(max(items) + 1):
            entry: dict = {}
            calls: dict[int, dict] = {}
            for key, value in items.get(index, {}).items():
                call = _TOOL_CALL.match(key)
                if call:
                    calls.setdefault(int(call.group(1)), {})[call.group(2)] = value
                else:
                    entry[key] = value
            if calls:
                entry["message.tool_calls"] = [
                    calls.get(i, {}) for i in range(max(calls) + 1)
                ]
            expanded.append(entry)
        result[prefix] = expanded
    return result


def span_row(span: dict) -> dict:
    """One manifest span -> one AX dataframe row.

    Adapted from the Phoenix migration tooling. ``attributes.metadata`` must be
    a DICT: the schema types it as such and a ``json.dumps`` string fails
    validation and rejects the whole batch without naming the column.
    """
    context = span.get("context") or {}
    if not context.get("trace_id") or not context.get("span_id"):
        raise PreparationError("Span is missing context.trace_id or context.span_id.")

    start = nanos(span.get("start_time"))
    end = nanos(span.get("end_time") or span.get("start_time"))
    if end < start:
        raise PreparationError(
            f"Span {context['span_id']!r} ends before it starts."
        )

    row = {
        "context.trace_id": context["trace_id"],
        "context.span_id": context["span_id"],
        "parent_id": span.get("parent_id") or None,
        "name": str(span.get("name") or ""),
        "start_time": start,
        "end_time": end,
        "status_code": span.get("status_code") or "UNSET",
        "status_message": span.get("status_message") or "",
        "span_kind": span.get("span_kind") or "UNKNOWN",
    }

    attributes = dict(span.get("attributes") or {})
    metadata = attributes.pop("metadata", {})
    for key, value in expand_indexed(attributes).items():
        if key.startswith("metadata."):
            continue
        if key in ("session.id", "user.id") and value is not None:
            value = str(value)
        row["attributes." + key] = value

    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {"original_metadata": metadata}
    if not isinstance(metadata, dict):
        metadata = {} if metadata is None else {"original_metadata": metadata}
    for key, value in (span.get("attributes") or {}).items():
        if key.startswith("metadata."):
            metadata[key[len("metadata."):]] = value
    row["attributes.metadata"] = metadata

    if span.get("events"):
        row["events"] = span["events"]
    return row


def to_dataframe(spans: list[dict]):
    """Build the upload frame, typing token counts as nullable integers.

    Without ``Int64``, a batch mixing LLM and non-LLM spans coerces token
    counts to float via Arrow and the column type is wrong in AX.
    """
    import pandas as pd

    frame = pd.DataFrame([span_row(s) for s in spans])
    for column in frame.columns:
        if column.startswith("attributes.llm.token_count."):
            frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def check_time_window(spans, *, max_past_years: int = DEFAULT_MAX_PAST_YEARS,
                      now: datetime | None = None) -> dict:
    """Find spans AX will reject for being too old, before uploading any.

    AX rejects spans whose start time is outside the allowed past window, and
    the window is widened only by an Arize-side code change to
    ``timeRangeAllowedBySpaceYears`` (keyed by space KEY) plus a deploy. On a
    multi-day migration that lead time has to be discovered up front, not at
    the first rejected batch.
    """
    moment = now or datetime.now(timezone.utc)
    # Calendar-accurate enough for a boundary check; the server uses AddDate.
    cutoff = moment - timedelta(days=365 * max_past_years)
    offending: list[dict] = []
    oldest = None
    total = 0
    for span in spans:
        total += 1
        try:
            start = nanos(span.get("start_time"))
        except PreparationError:
            offending.append({"span_id": (span.get("context") or {}).get("span_id"),
                              "reason": "unparseable start_time"})
            continue
        when = datetime.fromtimestamp(start / 1e9, tz=timezone.utc)
        if oldest is None or when < oldest:
            oldest = when
        if when < cutoff:
            offending.append({
                "span_id": (span.get("context") or {}).get("span_id"),
                "start_time": when.isoformat(),
                "reason": f"older than {max_past_years}y cutoff {cutoff.isoformat()}",
            })
    return {
        "span_count": total,
        "oldest": oldest.isoformat() if oldest else None,
        "cutoff": cutoff.isoformat(),
        "rejected_count": len(offending),
        # Bounded sample: the full list could be millions of entries.
        "rejected_sample": offending[:20],
        "ok": not offending,
    }


def upload_batch(client, spans: list[dict], destination: dict,
                 evals_frame=None, *, timeout: float = 60,
                 validate: bool = True) -> None:
    """Upload one batch, mapping SDK failures into the three-way taxonomy."""
    try:
        from arize.exceptions.auth import AuthenticationError
        from arize.exceptions.base import ValidationFailure
        from arize.exceptions.http import APIError
    except Exception:  # pragma: no cover
        AuthenticationError = ValidationFailure = APIError = ()  # type: ignore

    import pyarrow as pa

    try:
        frame = to_dataframe(spans)
    except PreparationError:
        raise
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise PreparationError(f"Arrow conversion failed: {type(exc).__name__}") from None

    kwargs = {
        "space_id": destination["space_id"],
        "project_name": destination["project_name"],
        "dataframe": frame,
        "timeout": timeout,
        # SDK-side validation is ~31% of the call (measured: 10k spans took
        # 9.14s with it, 6.35s without). Kept ON by default -- it is what
        # turns a malformed column into a clean PreparationError instead of an
        # opaque server rejection -- but it is a real lever on a long backfill
        # once a trial run has proven the mapping.
        "validate": validate,
    }
    # Evals ride with their spans in the same call: it removes the indexing
    # wait and the risk of a score arriving before the span it attaches to.
    if evals_frame is not None and len(evals_frame):
        kwargs["evals_dataframe"] = evals_frame

    try:
        client.spans.log(**kwargs)
    except AuthenticationError as exc:  # type: ignore[misc]
        raise RejectedUpload(
            f"AX rejected the upload (HTTP {getattr(exc, 'status_code', 'auth')}). "
            "Nothing landed; fix credentials or permissions and re-run."
        ) from None
    except ValidationFailure:  # type: ignore[misc]
        raise PreparationError(
            "AX SDK validation failed before upload. Nothing was sent. "
            "Check column types -- attributes.metadata must be a dict, and "
            "eval/annotation names must match [a-zA-Z0-9_ ]+."
        ) from None
    except APIError as exc:  # type: ignore[misc]
        status = getattr(exc, "status_code", 0) or 0
        if 400 <= status < 500 and status != 408:
            raise RejectedUpload(
                f"AX rejected the upload (HTTP {status}). Nothing landed."
            ) from None
        raise AmbiguousUpload(
            f"AX upload outcome unknown (HTTP {status or 'transport'}). "
            "Verify by readback before any retry."
        ) from None
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        raise PreparationError("Arrow conversion failed during upload.") from None
    except UploadError:
        # Already classified. Must be re-raised BEFORE the catch-all below,
        # otherwise a deterministic rejection gets re-wrapped as ambiguous and
        # a batch that provably never landed is quarantined for no reason.
        raise
    except Exception as exc:
        raise AmbiguousUpload(
            f"AX upload outcome unknown ({type(exc).__name__}). "
            "Verify by readback before any retry."
        ) from None


def upload_batch_otlp(spans: list[dict], destination: dict,
                      api_key: str | None, *, shards: int | None = None,
                      session=None) -> None:
    """Send one batch over OTLP, mapped into the same failure taxonomy.

    Every OTLP failure is AMBIGUOUS by construction. There is no HTTP status
    and no per-span acknowledgement: the exporter may have delivered some,
    all or none of a batch before failing, and the queue can drop spans while
    ``force_flush`` still reports success. So nothing here is ever classified
    as "provably did not land" -- the batch stays ``uncertain`` on disk and is
    resolved by read-back, never by a blind retry.

    The one exception is a shaping error, which happens before any span is
    emitted and is therefore deterministic.
    """
    from . import otlp

    if session is None and not api_key:
        raise PreparationError(
            "The OTLP path needs ARIZE_API_KEY to build its exporter.")
    try:
        for span in spans:
            span_row(span)          # same shaping contract as the Arrow path
    except PreparationError:
        raise
    try:
        if session is not None:
            # Reused across every batch of the run: provider construction is
            # ~10x the cost of the emission itself, so building a pool per
            # batch was the dominant cost before this.
            session.upload(spans)
        else:
            kwargs = {}
            if shards:
                kwargs["shards"] = shards
            otlp.upload(spans, destination, api_key, **kwargs)
    except otlp.OtlpError as exc:
        raise AmbiguousUpload(
            f"OTLP batch outcome unknown ({exc}). Verify by readback before "
            f"any retry.") from None
    except Exception as exc:
        raise AmbiguousUpload(
            f"OTLP batch outcome unknown ({type(exc).__name__}). Verify by "
            f"readback before any retry.") from None


def import_shard(shard: M.Shard, client, destination: dict, *,
                 max_batches: int | None = None,
                 evals_by_span: dict | None = None,
                 validate: bool = True,
                 # Defaults to the Arrow path: this function's other
                 # arguments (client, validate, evals_by_span) are Arrow-path
                 # concepts, so Arrow is the unsurprising default here. The
                 # pipeline selects OTLP explicitly.
                 ingest_path: str = "arrow",
                 api_key: str | None = None,
                 otlp_shards: int | None = None,
                 otlp_session=None,
                 progress=None) -> dict:
    """Upload a shard's pending batches under the persisted ledger.

    ``uncertain`` is written to disk BEFORE each attempt, so a process death
    mid-upload is discoverable on restart instead of being silently re-sent.
    """
    man = M.load_manifest(shard)
    man = M.bind_destination(shard, man, destination)
    pending = M.pending_batches(man)
    done = 0

    for batch in pending:
        if max_batches is not None and done >= max_batches:
            break
        spans = M.read_span_range(shard, batch["start"], batch["end"])

        # Shape every row before claiming the batch, so a mapping error leaves
        # the ledger untouched rather than stranding a batch as uncertain.
        try:
            for span in spans:
                span_row(span)
        except PreparationError:
            M.mark(shard, man, batch, M.PENDING)
            raise

        # Evals ride with the spans only on the Arrow path. OTLP has no
        # equivalent side-channel, so they are deferred and applied through
        # Flight update once the project is queryable -- which on OTLP is
        # seconds, not minutes.
        evals_frame = (_evals_frame_for(spans, evals_by_span)
                       if ingest_path == "arrow" else None)

        M.mark(shard, man, batch, M.UNCERTAIN)
        try:
            if ingest_path == "otlp":
                upload_batch_otlp(spans, destination, api_key,
                                  shards=otlp_shards, session=otlp_session)
            else:
                upload_batch(client, spans, destination, evals_frame,
                             validate=validate)
        except (PreparationError, RejectedUpload):
            M.mark(shard, man, batch, M.PENDING)   # provably nothing landed
            raise
        except AmbiguousUpload:
            raise                                   # stays uncertain on disk
        M.mark(shard, man, batch, M.SUBMITTED)
        done += 1
        if progress:
            progress(M.shard_progress(man))

    return M.shard_progress(man)


def _evals_frame_for(spans: list[dict], evals_by_span: dict | None):
    """Eval rows for just the spans in this batch.

    Scoped to the batch so an eval always travels with the span it attaches
    to, in the same call.
    """
    if not evals_by_span:
        return None
    import pandas as pd

    rows = []
    for span in spans:
        span_id = (span.get("context") or {}).get("span_id")
        row = evals_by_span.get(span_id)
        if row:
            rows.append(row)
    return pd.DataFrame(rows) if rows else None
