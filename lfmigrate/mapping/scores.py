"""Langfuse scores -> AX evaluations and annotations.

Column contracts, read from the installed SDK rather than assumed:

    eval.<name>.label | .score | .explanation
    trace_eval.<name>.*           attaches at the trace's root span
    session_eval.<name>.*         attaches at the session
    annotation.<name>.label | .score | .text | .updated_by | .updated_at

Three constraints that bite:

1. ``<name>`` must match ``[a-zA-Z0-9_\\s]+``. No dots, no hyphens. Langfuse
   score names routinely contain both ("answer-relevance", "toxicity.v2"), so
   names must be sanitised -- and sanitisation can collide, which would
   silently merge two distinct metrics. Collisions are raised, not resolved.

2. Annotations have NO ``.explanation``. A Langfuse annotation comment belongs
   in ``.text``; sending ``.explanation`` rejects the batch.

3. Annotations require at least one of label/score/text non-null per row, so a
   comment-only annotation is valid but an empty one is not.

Session-level scores use the ``session_eval.`` prefix. ``spans.annotate`` has
no granularity parameter and ``AnnotateRecordInput`` carries only
``record_id``/``values``, so the column convention -- not the annotate call --
is the usable path for session granularity.
"""

from __future__ import annotations

import re
from typing import Any

from .ids import root_span_id, span_id

#: Mirrors _EVAL_NAME_REGEX / _ANNOTATION_NAME_REGEX in arize.spans.columns.
_ALLOWED_NAME = re.compile(r"\A[a-zA-Z0-9_\s]+\Z")
_DISALLOWED = re.compile(r"[^a-zA-Z0-9_\s]+")

EVAL_SUFFIXES = ("label", "score", "explanation")
ANNOTATION_SUFFIXES = ("label", "score", "text", "updated_by", "updated_at")

#: Langfuse score sources that represent human judgement rather than an
#: automated evaluator. Everything else becomes an eval.
ANNOTATION_SOURCES = {"ANNOTATION"}


class ScoreMappingError(Exception):
    pass


def sanitise_name(name: str, *, seen: dict[str, str] | None = None) -> str:
    """Make a Langfuse score name valid as an AX eval/annotation column name.

    ``seen`` maps sanitised -> original. Pass the same dict across a whole
    migration to detect collisions: two different Langfuse metrics mapping to
    one AX column would silently overwrite each other, which is worse than
    failing.
    """
    raw = (name or "").strip()
    if not raw:
        raise ScoreMappingError("Score has an empty name.")
    cleaned = _DISALLOWED.sub("_", raw).strip()
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    if not cleaned or not _ALLOWED_NAME.fullmatch(cleaned):
        raise ScoreMappingError(f"Score name {name!r} cannot be represented as a column.")
    if seen is not None:
        previous = seen.setdefault(cleaned, raw)
        if previous != raw:
            raise ScoreMappingError(
                f"Score names {previous!r} and {raw!r} both sanitise to {cleaned!r}. "
                "Rename one in Langfuse, or supply an explicit name mapping, "
                "otherwise the two metrics would silently merge in AX."
            )
    return cleaned


def _value_parts(score: dict) -> tuple[float | None, str | None]:
    """(numeric score, categorical label) for a Langfuse score.

    Langfuse data types: NUMERIC, CATEGORICAL, BOOLEAN. Boolean carries both a
    0/1 value and a string label, and both are kept so the metric is filterable
    *and* averageable in AX.
    """
    data_type = str(score.get("dataType") or score.get("data_type") or "").upper()
    value = score.get("value")
    string_value = score.get("stringValue") or score.get("string_value")

    numeric: float | None = None
    label: str | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
    if string_value is not None:
        label = str(string_value)

    if data_type == "CATEGORICAL" and label is None and value is not None:
        label = str(value)
    if data_type == "BOOLEAN":
        if numeric is not None and label is None:
            label = "true" if numeric else "false"
        if label is not None and numeric is None:
            numeric = 1.0 if label.strip().lower() in ("true", "1", "yes") else 0.0
    if numeric is None and label is None and value is not None:
        label = str(value)
    return numeric, label


def subject_of(score: dict) -> tuple[str | None, str | None, str | None]:
    """(kind, id, trace_id) for a score, across source versions.

    Langfuse v4 nests the linkage in a ``subject`` object -- verified against a
    live v4.53.0 instance:

        {"kind": "trace",       "id": "<traceId>"}
        {"kind": "observation", "id": "<obsId>", "traceId": "<traceId>"}
        {"kind": "session",     "id": "<sessionId>"}

    The flat ``traceId``/``observationId``/``sessionId`` fields that v3 and
    Cloud return are ABSENT in v4, so reading only those would classify every
    v4 score as unattachable.
    """
    subject = score.get("subject")
    if isinstance(subject, dict) and subject.get("kind"):
        kind = str(subject["kind"]).lower()
        return kind, subject.get("id"), subject.get("traceId") or subject.get("trace_id")
    # v3 / Cloud flat shape.
    observation = score.get("observationId") or score.get("observation_id")
    trace = score.get("traceId") or score.get("trace_id")
    session = score.get("sessionId") or score.get("session_id")
    if session:
        return "session", session, None
    if observation:
        return "observation", observation, trace
    if trace:
        return "trace", trace, trace
    return None, None, None


#: Langfuse subject kinds that this tool maps. ``dataset_run`` scores belong to
#: experiments rather than traces and are reported as skipped rather than being
#: forced onto a span.
_GRANULARITY_BY_KIND = {"observation": "span", "trace": "trace", "session": "session"}


def granularity_of(score: dict) -> str:
    """'span' | 'trace' | 'session' for a Langfuse score."""
    kind, identifier, _trace = subject_of(score)
    if kind in _GRANULARITY_BY_KIND and identifier:
        return _GRANULARITY_BY_KIND[kind]
    if kind:
        raise ScoreMappingError(
            f"Score {score.get('id')!r} has subject kind {kind!r}, which does not "
            "map to a span, trace or session."
        )
    raise ScoreMappingError(
        f"Score {score.get('id')!r} references no observation, trace or session."
    )


def is_annotation(score: dict) -> bool:
    return str(score.get("source") or "").upper() in ANNOTATION_SOURCES


def target_span_id(score: dict, root_resolver=None) -> str | None:
    """Which AX span this score attaches to.

    A span-level score attaches to its own observation. A trace-level score
    attaches to that trace's ROOT span -- but which span that is depends on the
    source version, so it must be resolved rather than assumed:

    * Langfuse v4 is observation-centric with no trace-read API, so the root is
      normally the real observation whose ``parentObservationId`` is null.
    * A synthetic root exists only when the trace has zero or several parentless
      observations, or when a v3-style trace record supplied trace-level fields.

    ``root_resolver`` is a callable ``traceId -> span_id`` built during export
    (see ``mapping.spans.root_target_id``). Without it this falls back to the
    synthetic-root derivation, which is correct for v3 but would misanchor
    trace-level scores on v4.
    """
    kind, identifier, subject_trace = subject_of(score)
    if kind == "observation" and identifier:
        return span_id(identifier)
    trace = subject_trace if kind == "trace" else None
    if kind == "trace":
        trace = identifier
    if not trace:
        return None
    if root_resolver is not None:
        resolved = root_resolver(trace)
        if resolved:
            return resolved
        raise ScoreMappingError(
            f"Trace-level score {score.get('id')!r} references trace {trace!r}, "
            "which is not present in this export shard. Its root span is unknown, "
            "so attaching it would guess at an anchor."
        )
    return root_span_id(trace)


def eval_columns(score: dict, *, seen: dict[str, str] | None = None) -> dict[str, Any]:
    """``eval.``/``trace_eval.``/``session_eval.`` columns for one score."""
    name = sanitise_name(score.get("name"), seen=seen)
    prefix = {"span": "eval", "trace": "trace_eval", "session": "session_eval"}[
        granularity_of(score)
    ]
    numeric, label = _value_parts(score)
    columns: dict[str, Any] = {}
    if numeric is not None:
        columns[f"{prefix}.{name}.score"] = numeric
    if label is not None:
        columns[f"{prefix}.{name}.label"] = label
    comment = score.get("comment")
    if comment:
        columns[f"{prefix}.{name}.explanation"] = str(comment)
    if not columns:
        raise ScoreMappingError(f"Score {score.get('id')!r} has no usable value.")
    return columns


def annotation_columns(score: dict, *, seen: dict[str, str] | None = None) -> dict[str, Any]:
    """``annotation.`` columns for one human-authored Langfuse score.

    The comment maps to ``.text`` (there is no ``.explanation`` for
    annotations) and authorship/time are preserved in ``.updated_by`` /
    ``.updated_at`` so the human provenance survives migration.
    """
    name = sanitise_name(score.get("name"), seen=seen)
    numeric, label = _value_parts(score)
    comment = score.get("comment")
    columns: dict[str, Any] = {}
    if numeric is not None:
        columns[f"annotation.{name}.score"] = numeric
    if label is not None:
        columns[f"annotation.{name}.label"] = label
    if comment:
        columns[f"annotation.{name}.text"] = str(comment)
    if not columns:
        # At least one of label/score/text must be non-null per row.
        raise ScoreMappingError(
            f"Annotation {score.get('id')!r} has no label, score or text; "
            "AX requires at least one."
        )
    author = score.get("authorUserId") or score.get("author_user_id")
    if author:
        columns[f"annotation.{name}.updated_by"] = str(author)
    updated = score.get("timestamp") or score.get("createdAt") or score.get("created_at")
    if updated:
        columns[f"annotation.{name}.updated_at"] = updated
    return columns


def partition(scores: list[dict], *, seen: dict[str, str] | None = None,
              root_resolver=None) -> dict:
    """Group scores into the three frames the upload stage needs.

    Returns ``{"evals": {span_id: {...}}, "annotations": {...},
    "session_evals": {session_id: {...}}, "skipped": [...]}``.

    Session scores are keyed by session id, not span id: they are uploaded
    against the session, so they cannot be merged into the span frames.
    """
    seen = {} if seen is None else seen
    evals: dict[str, dict] = {}
    annotations: dict[str, dict] = {}
    session_evals: dict[str, dict] = {}
    skipped: list[dict] = []

    for score in scores:
        try:
            granularity = granularity_of(score)
            if granularity == "session":
                key = str(subject_of(score)[1])
                bucket = session_evals.setdefault(key, {"session.id": key})
                bucket.update(eval_columns(score, seen=seen))
                continue
            target = target_span_id(score, root_resolver)
            if not target:
                raise ScoreMappingError("No target span could be resolved.")
            if is_annotation(score):
                bucket = annotations.setdefault(target, {"context.span_id": target})
                bucket.update(annotation_columns(score, seen=seen))
            else:
                bucket = evals.setdefault(target, {"context.span_id": target})
                bucket.update(eval_columns(score, seen=seen))
        except ScoreMappingError as exc:
            # Never drop a score silently: an unmappable score is reported so
            # the count reconciliation at verify time stays honest.
            skipped.append({"id": score.get("id"), "name": score.get("name"),
                            "reason": str(exc)})

    return {
        "evals": evals,
        "annotations": annotations,
        "session_evals": session_evals,
        "skipped": skipped,
    }
