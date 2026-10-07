"""Langfuse configuration objects -> Arize AX call arguments.

Three resource families, all **space-scoped** in AX rather than project-scoped,
which is why they migrate once per space and not once per day like spans:

``prompts``
    Langfuse ``/api/public/v2/prompts`` -> ``client.prompts.create`` /
    ``create_version``.

``datasets``
    Langfuse ``/api/public/v2/datasets`` + ``/api/public/dataset-items`` ->
    ``client.datasets.create`` / ``append_examples``.

``score configs`` -> ``annotation configs``
    Langfuse ``/api/public/score-configs`` ->
    ``client.annotation_configs.create_categorical`` / ``create_continuous``.
    A score config is the *definition* of a metric; in AX that is an
    annotation config. The LLM judges and code evaluators that write those
    metrics are separate resources -- see ``mapping/evaluators.py``.

Everything here is pure: it takes decoded JSON and returns kwargs. Nothing
calls the network, so every shape below is unit-testable against a fixture.
"""

from __future__ import annotations

import json
from typing import Any

from .scores import sanitise_name

#: Langfuse templates variables as ``{{variable}}``, which is Mustache. AX
#: accepts MUSTACHE as an input variable format, so templates are carried over
#: BYTE-FOR-BYTE and never rewritten. Translating to F_STRING would mean
#: escaping every literal brace in the prompt body -- JSON examples in a
#: system prompt are common, and a missed escape corrupts the prompt silently.
MUSTACHE = "MUSTACHE"

#: AX message roles are uppercase (SDK 8.40.0 flipped the generated enums).
_ROLES = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
    "function": "TOOL",
}

#: model name prefix -> AX LlmProvider. Checked longest-first.
_PROVIDERS = (
    ("gpt-", "OPEN_AI"),
    ("o1", "OPEN_AI"),
    ("o3", "OPEN_AI"),
    ("o4", "OPEN_AI"),
    ("chatgpt", "OPEN_AI"),
    ("text-davinci", "OPEN_AI"),
    ("claude", "ANTHROPIC"),
    ("gemini", "VERTEX_AI"),
    ("bison", "VERTEX_AI"),
    ("anthropic.", "AWS_BEDROCK"),
    ("amazon.", "AWS_BEDROCK"),
    ("meta.", "AWS_BEDROCK"),
    ("mistral.", "AWS_BEDROCK"),
)

#: AX requires a provider on every prompt version; Langfuse does not record
#: one. CUSTOM is the honest answer for a model we cannot place, and it is
#: accepted by the API rather than guessed wrong.
DEFAULT_PROVIDER = "CUSTOM"

#: Invocation params AX accepts on a prompt version. Langfuse `config` is free
#: JSON, so anything outside this set would be rejected -- it is reported as
#: dropped rather than silently discarded.
_INVOCATION_KEYS = (
    "temperature", "max_tokens", "max_completion_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "stop", "response_format",
    "tool_config", "top_k", "thinking_level", "thinking_budget",
    "reasoning_effort", "verbosity",
)

#: Langfuse `config` keys that are NOT invocation params and are expected to be
#: absent from the AX payload. Listed so they are not reported as surprises.
_CONFIG_NON_PARAMS = ("model", "provider", "api_version", "deployment")


class ResourceMappingError(Exception):
    """A resource cannot be represented in AX. Never silently dropped."""


# ----------------------------------------------------------------- prompts


def provider_for_model(model: str | None) -> str:
    """Best-effort AX provider for a Langfuse model string."""
    if not model:
        return DEFAULT_PROVIDER
    lowered = str(model).strip().lower()
    for prefix, provider in _PROVIDERS:
        if lowered.startswith(prefix):
            return provider
    return DEFAULT_PROVIDER


def prompt_messages(prompt: dict) -> list[dict]:
    """Langfuse prompt body -> AX ``LLMMessageRequest`` dicts.

    Langfuse has two prompt types and they are shaped differently:

    ``chat``
        ``prompt`` is a list of ``{"role", "content"}``. Roles are uppercased.

    ``text``
        ``prompt`` is a plain string with no role at all. AX has no
        role-less prompt, so it becomes a single USER message. USER rather than
        SYSTEM because a Langfuse text prompt is the content *sent* to the
        model, not instructions wrapped around it -- mapping it to SYSTEM would
        change how a provider weighs it.
    """
    body = prompt.get("prompt")
    kind = (prompt.get("type") or "").strip().lower()

    if isinstance(body, str):
        if not body.strip():
            raise ResourceMappingError(
                f"Prompt {prompt.get('name')!r} version "
                f"{prompt.get('version')} has an empty body."
            )
        return [{"role": "USER", "content": body}]

    if isinstance(body, list):
        messages = []
        for index, entry in enumerate(body):
            if not isinstance(entry, dict):
                raise ResourceMappingError(
                    f"Prompt {prompt.get('name')!r} message {index} is "
                    f"{type(entry).__name__}, expected an object."
                )
            raw_role = str(entry.get("role") or "").strip().lower()
            role = _ROLES.get(raw_role)
            if role is None:
                raise ResourceMappingError(
                    f"Prompt {prompt.get('name')!r} message {index} has role "
                    f"{entry.get('role')!r}, which has no AX equivalent "
                    f"(known: {', '.join(sorted(set(_ROLES.values())))})."
                )
            content = entry.get("content")
            messages.append({"role": role,
                             "content": "" if content is None else str(content)})
        if not messages:
            raise ResourceMappingError(
                f"Prompt {prompt.get('name')!r} has no messages.")
        return messages

    raise ResourceMappingError(
        f"Prompt {prompt.get('name')!r} has a {type(body).__name__} body; "
        f"expected a string (type={kind or 'text'}) or a list of messages."
    )


def invocation_params(config: Any) -> tuple[dict, list[str]]:
    """Split a Langfuse ``config`` into AX invocation params and leftovers.

    Returns ``(params, dropped)``. Langfuse config is free-form JSON while AX
    has a closed set, so the leftovers are RETURNED for reporting instead of
    being dropped quietly -- a prompt that migrates without its
    ``response_format`` behaves differently and should say so.
    """
    if not isinstance(config, dict):
        return {}, []
    params, dropped = {}, []
    for key, value in config.items():
        if key in _INVOCATION_KEYS:
            if value is not None:
                params[key] = value
        elif key not in _CONFIG_NON_PARAMS:
            dropped.append(key)
    return params, sorted(dropped)


def prompt_version_payload(prompt: dict) -> dict:
    """AX kwargs for one Langfuse prompt version.

    ``_dropped_config`` is metadata for the caller's report, not an API
    argument, and is stripped before the call.
    """
    model = None
    config = prompt.get("config")
    if isinstance(config, dict):
        model = config.get("model")
    params, dropped = invocation_params(config)

    commit = (prompt.get("commitMessage") or "").strip()
    if not commit:
        # AX requires a commit message. Langfuse does not, so synthesise one
        # that records where the version came from rather than sending "".
        commit = f"Migrated from Langfuse (version {prompt.get('version')})"

    payload: dict = {
        "commit_message": commit,
        "input_variable_format": MUSTACHE,
        "provider": provider_for_model(model),
        "messages": prompt_messages(prompt),
        "_dropped_config": dropped,
    }
    if model:
        payload["model"] = str(model)
    if params:
        payload["invocation_params"] = params
    return payload


def prompt_labels(prompt: dict) -> list[str]:
    """Labels worth carrying over.

    ``latest`` is excluded: in Langfuse it is maintained automatically and in
    AX it would become a stale manual label pinned to whichever version
    happened to migrate last.
    """
    labels = prompt.get("labels")
    if not isinstance(labels, list):
        return []
    return [str(label) for label in labels
            if str(label).strip() and str(label).strip().lower() != "latest"]


# ---------------------------------------------------------------- datasets


def _flat(value: Any) -> Any:
    """Scalars pass through; containers become compact JSON.

    AX dataset examples are tabular, so a nested dict in a cell has to be
    serialised. JSON (not ``str()``) so it round-trips: ``str()`` on a dict
    emits single quotes and ``None``, which no JSON parser will read back.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def dataset_example(item: dict, *, expand_input: bool = True) -> dict:
    """One Langfuse dataset item -> one AX dataset example row.

    A Langfuse ``input`` is usually an object (``{"question": "..."}``). Its
    keys are promoted to columns by default, because an AX experiment task
    reads columns -- leaving the whole object JSON-encoded in one cell means
    every task has to parse it first. A non-object input keeps a single
    ``input`` column.

    Provenance (``langfuse_item_id`` and any source trace/observation) is
    carried so a migrated example can still be traced back.
    """
    example: dict = {}
    source = item.get("input")
    if expand_input and isinstance(source, dict) and source:
        for key, value in source.items():
            name = str(key).strip() or "input"
            example[name] = _flat(value)
    else:
        example["input"] = _flat(source)

    expected = item.get("expectedOutput")
    if expected is not None:
        example["expected_output"] = _flat(expected)

    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata:
        for key, value in metadata.items():
            # Prefixed so dataset metadata cannot shadow an input column of
            # the same name -- silently overwriting an input would change what
            # the experiment actually tests.
            example[f"metadata_{str(key).strip()}"] = _flat(value)
    elif metadata is not None:
        example["metadata"] = _flat(metadata)

    if item.get("id"):
        example["langfuse_item_id"] = str(item["id"])
    if item.get("sourceTraceId"):
        example["langfuse_source_trace_id"] = str(item["sourceTraceId"])
    if item.get("sourceObservationId"):
        example["langfuse_source_observation_id"] = str(item["sourceObservationId"])
    return example


def dataset_examples(items: list[dict], *, expand_input: bool = True,
                     include_archived: bool = False) -> list[dict]:
    """Rows for a whole dataset, with a uniform column set.

    Columns are unioned and missing cells filled with ``None``: AX builds one
    table, and rows with differing keys would otherwise produce a ragged frame
    where a column's presence depends on row order.
    """
    rows = []
    for item in items:
        status = str(item.get("status") or "ACTIVE").upper()
        if status != "ACTIVE" and not include_archived:
            continue
        rows.append(dataset_example(item, expand_input=expand_input))
    if not rows:
        return []
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{key: row.get(key) for key in columns} for row in rows]


# ------------------------------------------------- score / annotation configs


def annotation_config_payload(config: dict) -> dict:
    """Langfuse score config -> AX annotation config kwargs.

    ``{"kind": "categorical"|"continuous", "name": ..., ...}``. ``kind`` picks
    the SDK method (``create_categorical`` / ``create_continuous``) and is not
    sent on the wire.

    The name is run through the SAME sanitiser as migrated score columns. That
    is the point of migrating these at all: an annotation config only binds to
    data when its name matches the ``eval.<name>`` / ``annotation.<name>``
    columns the span migration wrote. Sanitising differently here would
    produce configs that look right in the UI and match nothing.
    """
    name = sanitise_name(config.get("name"))
    data_type = str(config.get("dataType") or "").strip().upper()
    categories = config.get("categories")

    if data_type in ("CATEGORICAL", "BOOLEAN"):
        if not isinstance(categories, list) or not categories:
            raise ResourceMappingError(
                f"Score config {config.get('name')!r} is {data_type} but has "
                f"no categories, so there are no allowed values to create."
            )
        values = []
        for entry in categories:
            if not isinstance(entry, dict):
                raise ResourceMappingError(
                    f"Score config {config.get('name')!r} has a "
                    f"{type(entry).__name__} category, expected an object.")
            label = entry.get("label")
            if label is None or not str(label).strip():
                raise ResourceMappingError(
                    f"Score config {config.get('name')!r} has a category with "
                    f"no label.")
            value: dict = {"label": str(label)}
            score = entry.get("value")
            if score is not None:
                value["score"] = float(score)
            values.append(value)
        return {"kind": "categorical", "name": name, "values": values}

    if data_type == "NUMERIC":
        minimum, maximum = config.get("minValue"), config.get("maxValue")
        if minimum is None or maximum is None:
            # AX requires both bounds; Langfuse allows an unbounded numeric
            # config. Refused rather than invented: a guessed 0..1 range would
            # mark legitimate out-of-range scores invalid in the UI.
            raise ResourceMappingError(
                f"Score config {config.get('name')!r} is NUMERIC but has "
                f"minValue={minimum!r} maxValue={maximum!r}. AX continuous "
                f"annotation configs require both bounds. Set a range in "
                f"Langfuse, or skip this config."
            )
        return {"kind": "continuous", "name": name,
                "minimum_score": float(minimum), "maximum_score": float(maximum)}

    raise ResourceMappingError(
        f"Score config {config.get('name')!r} has dataType {data_type!r}, "
        f"which has no AX annotation config equivalent "
        f"(known: CATEGORICAL, BOOLEAN, NUMERIC)."
    )


#: Langfuse score-config fields with no AX annotation-config equivalent.
#: Surfaced in the migration report so the loss is stated, not discovered.
UNMAPPED_CONFIG_FIELDS = ("description",)


def unmapped_fields(config: dict) -> list[str]:
    """Which lossy fields this particular config actually carries."""
    return [field for field in UNMAPPED_CONFIG_FIELDS
            if config.get(field) not in (None, "", [], {})]
