"""Langfuse trace/observation -> Phoenix-shaped span dict.

The intermediate shape is deliberately the one Arize's Phoenix migration
tooling already consumes::

    {"context": {"trace_id", "span_id"}, "parent_id", "name",
     "start_time", "end_time", "status_code", "status_message",
     "span_kind", "attributes": {...}, "events": [...]}

Emitting that shape means the proven upload, batching, resume and verification
machinery applies unchanged, and the only Langfuse-specific code in the whole
utility is this module plus the source adapters.

Attributes are emitted FLAT, including indexed message keys
(``llm.input_messages.0.message.role``). The upload layer folds those into
list-of-dict columns, which is what makes AX render a chat view rather than a
JSON blob.
"""

from __future__ import annotations

import json
from typing import Any

from .ids import root_span_id, span_id, trace_id

#: Canonical OpenInference span kinds, read from the installed semantic
#: conventions so this cannot drift from the SDK.
try:
    from openinference.semconv.trace import OpenInferenceSpanKindValues as _Kinds

    VALID_SPAN_KINDS = frozenset(m.value for m in _Kinds)
except Exception:  # pragma: no cover - semconv always present in practice
    VALID_SPAN_KINDS = frozenset(
        {"AGENT", "CHAIN", "EMBEDDING", "EVALUATOR", "GUARDRAIL", "LLM",
         "PROMPT", "RERANKER", "RETRIEVER", "TOOL", "UNKNOWN"}
    )

#: Langfuse observation type -> OpenInference span kind.
#:
#: EVENT is the one judgement call. A Langfuse EVENT is a zero-duration point
#: log, not a unit of work; UNKNOWN is the literal mapping but renders poorly
#: and is easy to mistake for a defect. CHAIN keeps it visible with correct
#: parentage, and the original type is always preserved in metadata, so nothing
#: is lost either way. Override via ``extra_kinds`` if a customer disagrees.
SPAN_KIND_BY_TYPE = {
    "GENERATION": "LLM",
    "SPAN": "CHAIN",
    "EVENT": "CHAIN",
    "AGENT": "AGENT",
    "TOOL": "TOOL",
    "CHAIN": "CHAIN",
    "RETRIEVER": "RETRIEVER",
    "EMBEDDING": "EMBEDDING",
    "GUARDRAIL": "GUARDRAIL",
    "EVALUATOR": "EVALUATOR",
    "RERANKER": "RERANKER",
}

# Fail at import rather than at hour three of a 3M-row upload. The SDK does not
# validate span_kind, so a bad value here would reach the server unchallenged.
_unknown = sorted(set(SPAN_KIND_BY_TYPE.values()) - VALID_SPAN_KINDS)
if _unknown:  # pragma: no cover
    raise RuntimeError(f"SPAN_KIND_BY_TYPE maps to invalid span kinds: {_unknown}")

#: Namespaced provenance key. Never collides with customer metadata, and a
#: collision is treated as an error rather than silently overwritten.
PROVENANCE_KEY = "langfuse_migration"


class MappingError(Exception):
    """Raised when a source record cannot be represented faithfully."""


def _text_and_mime(value: Any) -> tuple[str | None, str | None]:
    """Render a value for input.value/output.value plus its mime type."""
    if value is None:
        return None, None
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False), "application/json"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                json.loads(stripped)
                return value, "application/json"
            except ValueError:
                pass
        return value, "text/plain"
    return str(value), "text/plain"


def _coerce_messages(value: Any) -> list[dict] | None:
    """Best-effort extraction of chat messages from a Langfuse input/output."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                value = json.loads(stripped)
            except ValueError:
                return None
        else:
            return None
    if isinstance(value, dict):
        for key in ("messages", "input", "prompt"):
            inner = value.get(key)
            if isinstance(inner, list):
                value = inner
                break
        else:
            # A single message object.
            if "role" in value and ("content" in value or "tool_calls" in value):
                value = [value]
            else:
                return None
    if not isinstance(value, list):
        return None
    messages = [m for m in value if isinstance(m, dict) and "role" in m]
    return messages or None


def _emit_messages(attrs: dict, prefix: str, messages: list[dict]) -> None:
    """Write indexed message keys that the upload layer folds into LIST_DICT."""
    for index, message in enumerate(messages):
        base = f"{prefix}.{index}.message"
        role = message.get("role")
        if role is not None:
            attrs[f"{base}.role"] = str(role)
        content = message.get("content")
        if content is not None:
            if isinstance(content, (dict, list)):
                content = json.dumps(content, default=str, ensure_ascii=False)
            attrs[f"{base}.content"] = str(content)
        if message.get("name") is not None:
            attrs[f"{base}.name"] = str(message["name"])
        for call_index, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            call_base = f"{base}.tool_calls.{call_index}.tool_call"
            function = call.get("function") or {}
            if call.get("id") is not None:
                attrs[f"{call_base}.id"] = str(call["id"])
            if function.get("name") is not None:
                attrs[f"{call_base}.function.name"] = str(function["name"])
            arguments = function.get("arguments")
            if arguments is not None:
                if isinstance(arguments, (dict, list)):
                    arguments = json.dumps(arguments, default=str, ensure_ascii=False)
                attrs[f"{call_base}.function.arguments"] = str(arguments)


def _first_number(mapping: Any, *keys) -> float | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _emit_usage(attrs: dict, observation: dict) -> None:
    """Token counts from either the v3 usageDetails or the legacy usage shape."""
    details = observation.get("usageDetails")
    usage = observation.get("usage")
    prompt = (
        _first_number(details, "input", "prompt", "promptTokens")
        or _first_number(usage, "input", "promptTokens", "prompt_tokens")
    )
    completion = (
        _first_number(details, "output", "completion", "completionTokens")
        or _first_number(usage, "output", "completionTokens", "completion_tokens")
    )
    total = (
        _first_number(details, "total", "totalTokens")
        or _first_number(usage, "total", "totalTokens", "total_tokens")
    )
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    for suffix, value in (("prompt", prompt), ("completion", completion), ("total", total)):
        if value is not None:
            attrs[f"llm.token_count.{suffix}"] = int(value)

    # Any remaining usageDetails keys (cached tokens, reasoning tokens, audio)
    # are preserved rather than dropped, since they drive cost reconciliation.
    if isinstance(details, dict):
        for key, value in details.items():
            if key in ("input", "output", "total", "prompt", "completion"):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                attrs[f"llm.token_count.{key}"] = int(value)


def _emit_cost(attrs: dict, observation: dict) -> None:
    """Carry Langfuse's own calculated cost across.

    AX computes cost on ingest in the OTLP receiver path only; the Arrow upload
    path used for backfill does not. Cost Config is also applied once at
    arrival and never recalculated. So historical cost must be carried over
    explicitly or the cost view is empty for all migrated history -- which
    tends to be the first thing anyone checks after a backfill.
    """
    details = observation.get("costDetails")
    prompt = (
        _first_number(details, "input", "prompt")
        or _first_number(observation, "calculatedInputCost", "inputCost")
    )
    completion = (
        _first_number(details, "output", "completion")
        or _first_number(observation, "calculatedOutputCost", "outputCost")
    )
    total = (
        _first_number(details, "total")
        or _first_number(observation, "calculatedTotalCost", "totalCost")
    )
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    for suffix, value in (("prompt", prompt), ("completion", completion), ("total", total)):
        if value is not None:
            attrs[f"llm.cost.{suffix}"] = float(value)


def _status(level: Any, status_message: Any) -> tuple[str, str]:
    """Langfuse level/statusMessage -> OTel status.

    Only ERROR maps to ERROR. Langfuse WARNING and DEBUG are observability
    levels, not failures, and promoting them would inflate the error rate the
    customer sees immediately after migrating -- an alarming and wrong first
    impression.
    """
    text = str(status_message) if status_message else ""
    if str(level or "").upper() == "ERROR":
        return "ERROR", text
    return ("OK" if level else "UNSET"), text


def _provenance(record: dict, kind: str, extra: dict | None = None) -> dict:
    return {
        "source": "langfuse",
        "record_kind": kind,
        "original_id": record.get("id"),
        "original_trace_id": record.get("traceId") or record.get("id"),
        "original_parent_id": record.get("parentObservationId"),
        "original_type": record.get("type"),
        **(extra or {}),
    }


def _metadata(record: dict, kind: str, extra: dict | None = None) -> dict:
    """Metadata as a DICT, never a JSON string.

    The AX schema types ``attributes.metadata`` as DICT and validates
    ``is_dict_of(...)``. A ``json.dumps``-ed string fails validation and
    rejects the entire batch -- a trap worth stating plainly because the
    rejection message does not name the column.
    """
    raw = record.get("metadata")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {"original_metadata": raw}
    if not isinstance(raw, dict):
        raw = {} if raw is None else {"original_metadata": raw}
    metadata = dict(raw)
    if PROVENANCE_KEY in metadata:
        raise MappingError(
            f"Source metadata already uses the reserved {PROVENANCE_KEY!r} key; "
            "refusing to overwrite customer data."
        )
    metadata[PROVENANCE_KEY] = _provenance(record, kind, extra)
    return metadata


def _common_attrs(attrs: dict, trace: dict | None, record: dict) -> None:
    """Session/user/tags, preferring the observation then falling back to trace.

    Langfuse carries sessionId/userId on the TRACE record; whether they are
    denormalised onto observations varies by SDK version and export mode. AX
    sessions do not appear at all without session.id on the span, so the
    fallback is what makes the session view work.
    """
    source = {}
    if trace:
        source.update({k: v for k, v in trace.items() if v is not None})
    source.update({k: v for k, v in record.items() if v is not None})
    for target, keys in (
        ("session.id", ("sessionId", "session_id")),
        ("user.id", ("userId", "user_id")),
    ):
        for key in keys:
            if source.get(key) is not None:
                attrs[target] = str(source[key])
                break
    tags = (trace or {}).get("tags") or record.get("tags")
    if isinstance(tags, list) and tags:
        attrs["tag.tags"] = [str(t) for t in tags]


def observation_to_span(
    observation: dict,
    trace: dict | None = None,
    *,
    attach_to_synth_root: bool = True,
    extra_kinds: dict | None = None,
) -> dict:
    """Map one Langfuse observation to a Phoenix-shaped span dict."""
    source_trace_id = observation.get("traceId") or (trace or {}).get("id")
    if not source_trace_id:
        raise MappingError(f"Observation {observation.get('id')!r} has no traceId.")
    start = observation.get("startTime") or observation.get("start_time")
    if not start:
        raise MappingError(f"Observation {observation.get('id')!r} has no startTime.")
    # Zero-duration observations (Langfuse EVENT) are legal: end == start.
    end = (
        observation.get("endTime")
        or observation.get("end_time")
        or observation.get("completionStartTime")
        or start
    )

    kinds = {**SPAN_KIND_BY_TYPE, **(extra_kinds or {})}
    obs_type = str(observation.get("type") or "SPAN").upper()
    kind = kinds.get(obs_type, "UNKNOWN")

    parent = observation.get("parentObservationId")
    if parent:
        parent_id = span_id(parent)
    elif attach_to_synth_root:
        # Re-parent top-level observations onto the synthesized trace root so a
        # trace with several top-level observations renders as ONE tree.
        parent_id = root_span_id(source_trace_id)
    else:
        parent_id = None

    attrs: dict[str, Any] = {"openinference.span.kind": kind}

    input_value, input_mime = _text_and_mime(observation.get("input"))
    if input_value is not None:
        attrs["input.value"] = input_value
        attrs["input.mime_type"] = input_mime
    output_value, output_mime = _text_and_mime(observation.get("output"))
    if output_value is not None:
        attrs["output.value"] = output_value
        attrs["output.mime_type"] = output_mime

    if kind == "LLM":
        if observation.get("model"):
            attrs["llm.model_name"] = str(observation["model"])
        if observation.get("modelParameters"):
            attrs["llm.invocation_parameters"] = json.dumps(
                observation["modelParameters"], default=str, ensure_ascii=False
            )
        if observation.get("promptName"):
            attrs["llm.prompt_template.template"] = str(observation["promptName"])
        if observation.get("promptVersion") is not None:
            attrs["llm.prompt_template.version"] = str(observation["promptVersion"])
        _emit_usage(attrs, observation)
        _emit_cost(attrs, observation)
        for prefix, value in (
            ("llm.input_messages", observation.get("input")),
            ("llm.output_messages", observation.get("output")),
        ):
            messages = _coerce_messages(value)
            if messages:
                _emit_messages(attrs, prefix, messages)

    _common_attrs(attrs, trace, observation)
    status_code, status_message = _status(
        observation.get("level"), observation.get("statusMessage")
    )
    attrs["metadata"] = _metadata(
        observation, "observation", {"langfuse_span_kind_source": obs_type}
    )

    return {
        "context": {
            "trace_id": trace_id(source_trace_id),
            "span_id": span_id(observation.get("id")),
        },
        "parent_id": parent_id,
        "name": str(observation.get("name") or obs_type.lower()),
        "start_time": start,
        "end_time": end,
        "status_code": status_code,
        "status_message": status_message,
        "span_kind": kind,
        "attributes": attrs,
        "events": [],
    }


def trace_to_root_span(trace: dict, *, observations: list[dict] | None = None) -> dict:
    """Synthesize the root span for a Langfuse trace record.

    Carries the trace's own name/input/output/tags, which are otherwise lost,
    and gives trace-level scores and the session view a single anchor. Its
    interval is widened to cover its children so the tree is well-formed even
    when the trace record's own timestamps are narrower than its observations.
    """
    source_id = trace.get("id")
    if not source_id:
        raise MappingError("Trace record has no id.")

    starts = [o.get("startTime") for o in (observations or []) if o.get("startTime")]
    ends = [
        o.get("endTime") or o.get("startTime")
        for o in (observations or [])
        if o.get("endTime") or o.get("startTime")
    ]
    start = min([t for t in [trace.get("timestamp")] + starts if t], default=None)
    end = max([t for t in [trace.get("timestamp")] + ends if t], default=start)
    if not start:
        raise MappingError(f"Trace {source_id!r} has no usable timestamp.")

    attrs: dict[str, Any] = {"openinference.span.kind": "CHAIN"}
    input_value, input_mime = _text_and_mime(trace.get("input"))
    if input_value is not None:
        attrs["input.value"] = input_value
        attrs["input.mime_type"] = input_mime
    output_value, output_mime = _text_and_mime(trace.get("output"))
    if output_value is not None:
        attrs["output.value"] = output_value
        attrs["output.mime_type"] = output_mime
    if trace.get("release"):
        attrs["langfuse.release"] = str(trace["release"])
    if trace.get("version"):
        attrs["langfuse.version"] = str(trace["version"])

    _common_attrs(attrs, trace, trace)
    attrs["metadata"] = _metadata(
        trace, "trace", {"synthesized_root": True,
                         "observation_count": len(observations or [])}
    )

    return {
        "context": {"trace_id": trace_id(source_id), "span_id": root_span_id(source_id)},
        "parent_id": None,
        "name": str(trace.get("name") or "trace"),
        "start_time": start,
        "end_time": end,
        "status_code": "UNSET",
        "status_message": "",
        "span_kind": "CHAIN",
        "attributes": attrs,
        "events": [],
    }

# ─── Trace assembly ──────────────────────────────────────────────────────────


def parentless(observations: list[dict]) -> list[dict]:
    return [o for o in observations if not o.get("parentObservationId")]


def needs_synthetic_root(observations: list[dict], trace: dict | None = None) -> bool:
    """Whether this trace requires a synthesized grouping root.

    Langfuse v4 is observation-centric and has NO trace-read API: the official
    guidance is to group observation rows by ``traceId`` and "reconstruct
    [input/output] from the root observation of each trace: the row with
    ``parentObservationId == null``". So in v4 the root observation IS the
    root, and synthesizing another one would insert a redundant parent layer
    above it.

    A synthetic root is required only when the trace does not already have
    exactly one natural root:

    * 0 parentless rows -- an orphaned subtree (its root fell outside the
      export window, or was never written). Without a root the spans have a
      dangling parent and render detached.
    * 2+ parentless rows -- the trace would render as several separate trees,
      and a trace-level score would have no single anchor.

    A v3-style export that supplies a real ``trace`` record always gets a
    synthetic root, because that record carries name/input/output/tags that
    exist nowhere else.
    """
    if trace is not None:
        return True
    return len(parentless(observations)) != 1


def root_target_id(trace_identifier, observations: list[dict],
                   trace: dict | None = None) -> str:
    """The span ID that represents this trace's root.

    This is what trace-level scores must attach to. Resolving it here -- rather
    than assuming ``root_span_id(traceId)`` -- is what keeps trace-level scores
    anchored correctly in v4, where there is usually no synthetic root at all.
    """
    if needs_synthetic_root(observations, trace):
        return root_span_id(trace_identifier)
    return span_id(parentless(observations)[0]["id"])


def build_trace_spans(trace_identifier, observations: list[dict],
                      trace: dict | None = None,
                      *, extra_kinds: dict | None = None) -> list[dict]:
    """Map one whole Langfuse trace to a well-formed span tree.

    Operating per-trace rather than per-observation is deliberate: the root
    decision, re-parenting and interval widening all need the full sibling set,
    and AX requires a trace's spans to arrive together anyway.
    """
    if not observations and trace is None:
        return []
    synthetic = needs_synthetic_root(observations, trace)
    built: list[dict] = []

    if synthetic:
        # A v4 export has no trace record; fabricate the minimum needed to
        # describe the group, and mark it so nobody mistakes it for source data.
        record = trace if trace is not None else {
            "id": trace_identifier,
            "name": _derived_group_name(observations),
            "metadata": {},
        }
        built.append(trace_to_root_span(record, observations=observations))

    parent_for_roots = root_span_id(trace_identifier) if synthetic else None
    for observation in observations:
        span = observation_to_span(
            observation, trace,
            attach_to_synth_root=synthetic,
            extra_kinds=extra_kinds,
        )
        if not observation.get("parentObservationId"):
            span["parent_id"] = parent_for_roots
        built.append(span)
    return built


def _derived_group_name(observations: list[dict]) -> str:
    roots = parentless(observations)
    if len(roots) == 1 and roots[0].get("name"):
        return str(roots[0]["name"])
    for observation in observations:
        if observation.get("name"):
            return str(observation["name"])
    return "trace"
