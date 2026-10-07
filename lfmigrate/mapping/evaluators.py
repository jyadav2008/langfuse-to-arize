"""Langfuse evaluators and evaluation rules -> Arize AX evaluators and tasks.

Langfuse v4 exposes both through ``/api/public/v2/evaluators`` and
``/api/public/v2/evaluation-rules`` (an earlier note in this repo claimed there
was no public evaluator API; it had probed two guessed paths that do not exist).

The two systems model the same ideas differently, and every difference below
was confirmed against a live instance on each side rather than read off a spec:

=====================  ===============================  ==============================
Concept                Langfuse v4                      Arize AX
=====================  ===============================  ==============================
LLM judge prompt       chat messages, or one string     ONE template string
Variables              Mustache ``{{var}}``             Python f-string ``{var}``
Literal braces         plain text                       must be doubled (``{{``)
Output                 NUMERIC / BOOLEAN / CATEGORICAL  ``classification_choices``
                                                        (label -> score), else freeform
Model                  ``modelConfig`` or project        ``llm_config`` with a REQUIRED
                       default eval model               AI integration id
Variable binding       ``variableMapping`` on            ``column_mappings`` on the
                       evaluator or rule                task's evaluator entry
Code evaluator         ``evaluate(observation)``         a Python *class*; no TypeScript
                       function, PYTHON or TYPESCRIPT
Where it runs          evaluation rule (sampling,        evaluation task (sampling,
                       filters)                         query filter)
=====================  ===============================  ==============================

Everything here is pure: decoded JSON in, kwargs out.
"""

from __future__ import annotations

import re
from typing import Any

from .scores import ScoreMappingError, sanitise_name


class EvaluatorMappingError(Exception):
    """An evaluator cannot be represented in AX. Never silently dropped."""


# ------------------------------------------------------------- templates

#: A Langfuse Mustache variable. Langfuse allows surrounding whitespace.
_MUSTACHE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")

#: What AX recognises as an f-string expression.
_FSTRING_VAR = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_.]*)\}(?!\})")


def mustache_to_fstring(text: str) -> tuple[str, list[str]]:
    """Convert a Langfuse template to an AX f-string template.

    Returns ``(template, variables)``. Two rules, and both are load-bearing:

    * ``{{var}}`` becomes ``{var}``. AX rejects a template with no f-string
      expression ("template must contain at least one f-string expression like
      {variable_name}"), and under f-string rules ``{{input}}`` is an escaped
      literal, not a variable -- verified: AX returned 400 for it.
    * Every OTHER brace is doubled. A judge prompt that shows the model a JSON
      example is ordinary, and a raw ``{"k": 1}`` is an invalid f-string field
      at render time. Doubled, it renders back to exactly ``{"k": 1}``.

    The conversion is done in one pass over the original text, so a brace that
    belongs to a variable is never escaped and an escaped brace is never
    re-read as a variable.
    """
    out: list[str] = []
    variables: list[str] = []
    position = 0
    for match in _MUSTACHE.finditer(text or ""):
        literal = text[position:match.start()]
        out.append(literal.replace("{", "{{").replace("}", "}}"))
        name = match.group(1)
        out.append("{" + name + "}")
        if name not in variables:
            variables.append(name)
        position = match.end()
    out.append((text or "")[position:].replace("{", "{{").replace("}", "}}"))
    return "".join(out), variables


def flatten_prompt(prompt: Any) -> str:
    """Langfuse chat messages -> one block of text, roles kept as headers.

    AX template evaluators take a single string. Concatenating with explicit
    role headers keeps the system instructions recognisable as instructions
    instead of silently merging them into the user turn. A plain-string
    Langfuse prompt (the API's shortcut form) passes through unchanged.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list) or not prompt:
        raise EvaluatorMappingError("prompt is empty or not a list of messages")
    if len(prompt) == 1 and isinstance(prompt[0], dict) \
            and str(prompt[0].get("role", "")).lower() == "user":
        return str(prompt[0].get("content") or "")
    parts = []
    for message in prompt:
        if not isinstance(message, dict):
            raise EvaluatorMappingError("prompt message is not an object")
        role = str(message.get("role") or "user").upper()
        parts.append(f"[{role}]\n{message.get('content') or ''}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- output

#: Steps used to discretise a bounded NUMERIC judge. AX template evaluators
#: have no numeric output type -- without ``classification_choices`` they
#: produce freeform text -- so a 0..1 judge becomes 11 choices to keep its
#: score numeric and queryable.
NUMERIC_STEPS = 11


def _number_label(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def output_choices(output: dict, score_config: dict | None = None
                   ) -> tuple[dict | None, list[str]]:
    """Langfuse ``outputDefinition`` -> AX ``classification_choices``.

    Returns ``(choices or None, notes)``. ``None`` means freeform output.

    CATEGORICAL labels carry no scores in Langfuse, but a score config of the
    same name usually does (the Tenor seed's ``citation_validity`` has
    all_cited=1, partially_cited=0.5, uncited=0). That is preferred; positional
    scores are a labelled fallback, never silent.
    """
    notes: list[str] = []
    if not isinstance(output, dict):
        raise EvaluatorMappingError("outputDefinition is missing")
    kind = str(output.get("dataType") or "").upper()

    if kind == "BOOLEAN":
        return {"true": 1.0, "false": 0.0}, notes

    if kind == "CATEGORICAL":
        labels = [str(c) for c in (output.get("categories") or []) if str(c).strip()]
        if len(labels) < 2:
            raise EvaluatorMappingError("CATEGORICAL output needs at least two categories")
        if output.get("shouldAllowMultipleMatches"):
            notes.append("multi-label output is not supported by AX template "
                         "evaluators; migrated as single-label")
        configured = {}
        for entry in (score_config or {}).get("categories") or []:
            if isinstance(entry, dict) and entry.get("label") is not None \
                    and entry.get("value") is not None:
                configured[str(entry["label"])] = float(entry["value"])
        if configured and all(label in configured for label in labels):
            return {label: configured[label] for label in labels}, notes
        top = len(labels) - 1
        choices = {label: round(1 - i / top, 4) for i, label in enumerate(labels)}
        notes.append("category scores assigned by position (first=1, last=0) -- "
                     "no matching score config found; review")
        return choices, notes

    if kind == "NUMERIC":
        low, high = output.get("minValue"), output.get("maxValue")
        if low is None or high is None or float(high) <= float(low):
            notes.append("unbounded NUMERIC output migrated as freeform; AX "
                         "template evaluators have no numeric output type")
            return None, notes
        low, high = float(low), float(high)
        step = (high - low) / (NUMERIC_STEPS - 1)
        choices = {_number_label(low + i * step): round(low + i * step, 6)
                   for i in range(NUMERIC_STEPS)}
        notes.append(f"NUMERIC {low:g}..{high:g} discretised to {NUMERIC_STEPS} "
                     f"choices; AX template evaluators have no numeric output type")
        return choices, notes

    raise EvaluatorMappingError(f"unsupported outputDefinition dataType {kind!r}")


def _guidance(output: dict, choices: dict | None) -> str:
    """Langfuse's output instructions, restated for a single template."""
    lines = []
    reasoning = (output or {}).get("scoreReasoningInstructions")
    value = (output or {}).get("scoreValueInstructions")
    if value:
        lines.append(f"Scoring guidance: {value}")
    if reasoning:
        lines.append(f"Reasoning guidance: {reasoning}")
    if choices:
        lines.append("Respond with exactly one of: " + ", ".join(choices))
    return "\n".join(lines)


# --------------------------------------------------------------- mapping

#: Langfuse variable source -> AX span column.
_SOURCE_COLUMNS = {
    "input": "attributes.input.value",
    "output": "attributes.output.value",
}

#: Sources with no span column equivalent on a migrated span.
_UNMAPPABLE = ("tool_calls", "expected_output", "experiment_item_metadata")


def column_mappings(variable_mapping: list | None,
                    variables: list[str]) -> tuple[dict[str, str], list[str]]:
    """Langfuse ``variableMapping`` -> AX task ``column_mappings``.

    ``metadata`` with a ``jsonPath`` maps under ``attributes.metadata.``, which
    is where the span migration writes Langfuse observation metadata, so a
    migrated evaluator reads the same field it read in Langfuse.
    """
    mappings: dict[str, str] = {}
    notes: list[str] = []
    for entry in variable_mapping or []:
        if not isinstance(entry, dict):
            continue
        variable = entry.get("variable")
        source = str(entry.get("source") or "").lower()
        if not variable:
            continue
        if source in _SOURCE_COLUMNS:
            mappings[variable] = _SOURCE_COLUMNS[source]
        elif source == "metadata":
            path = str(entry.get("jsonPath") or "").strip()
            path = re.sub(r"^\$\.?", "", path)
            mappings[variable] = "attributes.metadata" + (f".{path}" if path else "")
        elif source in _UNMAPPABLE:
            notes.append(f"variable {variable!r} reads {source!r}, which has no "
                         f"span column after migration -- map it manually")
        else:
            notes.append(f"variable {variable!r} has unknown source {source!r}")
    for variable in variables:
        if variable not in mappings and variable in ("input", "output"):
            mappings[variable] = _SOURCE_COLUMNS[variable]
    unbound = [v for v in variables if v not in mappings]
    if unbound:
        notes.append(f"unbound variable(s) {', '.join(unbound)} -- set "
                     f"column_mappings on the AX task")
    return mappings, notes


def parse_integrations(raw: str | None) -> dict[str, str]:
    """``openai=<id>,anthropic=<id>`` -> {provider: integration id}."""
    out: dict[str, str] = {}
    for part in str(raw or "").split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            if key.strip() and value.strip():
                out[key.strip().lower()] = value.strip()
    return out


def template_evaluator_payload(evaluator: dict, *, integrations: dict[str, str],
                               default_integration: str | None,
                               default_model: str | None,
                               score_config: dict | None = None) -> dict:
    """One Langfuse LLM-as-judge version -> AX ``create_template_*`` kwargs.

    Raises :class:`EvaluatorMappingError` when the judge cannot be created --
    above all when there is no AX AI integration to run it with. That is
    deliberate: the integration decides whose credentials and whose bill the
    judge runs on, so it is never inferred.

    The ``_column_mappings`` and ``_notes`` keys are for the caller's report
    and for task creation; they are stripped before the SDK call.
    """
    if evaluator.get("type") != "llm_as_judge":
        raise EvaluatorMappingError(f"not an LLM judge: {evaluator.get('type')!r}")

    model_config = evaluator.get("modelConfig") or {}
    provider = str(model_config.get("provider") or "").lower()
    model = model_config.get("model") or default_model
    integration = integrations.get(provider) or default_integration
    if not integration:
        raise EvaluatorMappingError(
            "no AX AI integration to run this judge with -- set "
            "ARIZE_AI_INTEGRATION_ID (or LFMIGRATE_EVAL_INTEGRATIONS="
            f"{provider or 'provider'}=<id>)")
    if not model:
        raise EvaluatorMappingError(
            "judge has no model and uses the Langfuse project default -- set "
            "LFMIGRATE_EVAL_DEFAULT_MODEL")

    output = evaluator.get("outputDefinition") or {}
    choices, notes = output_choices(output, score_config)
    body = flatten_prompt(evaluator.get("prompt"))
    guidance = _guidance(output, choices)
    if guidance:
        body = f"{body}\n\n{guidance}"
    template, variables = mustache_to_fstring(body)
    if not variables:
        raise EvaluatorMappingError(
            "prompt has no {{variables}}; AX requires at least one")

    mappings, mapping_notes = column_mappings(
        evaluator.get("variableMapping"), evaluator.get("variables") or variables)
    notes += mapping_notes
    if not model_config:
        notes.append(f"used default model {model!r} (judge relied on the "
                     f"Langfuse project default)")

    try:
        column = sanitise_name(evaluator.get("name"))
    except ScoreMappingError as exc:
        raise EvaluatorMappingError(str(exc)) from None

    config: dict = {
        # Same sanitiser as migrated score columns: future AX runs of this
        # judge write the column the migrated history already uses.
        "name": column,
        "template": template,
        "include_explanations": True,
        "use_function_calling": True,
        "data_granularity": "SPAN",
        "llm_config": {"ai_integration_id": integration, "model_name": str(model),
                       "invocation_parameters": {}, "provider_parameters": {}},
    }
    if choices is not None:
        config["classification_choices"] = choices
    return {"template_config": config, "_column_mappings": mappings,
            "_variables": variables, "_notes": notes}


# ---------------------------------------------------------- code evaluators

_SUFFIX = {"PYTHON": "py", "TYPESCRIPT": "ts"}


def code_evaluator_export(evaluator: dict) -> tuple[str, str, str]:
    """A Langfuse code evaluator -> (filename, file content, reason).

    Exported for a manual port rather than created in AX. AX runs a Python
    evaluator *class* with span columns mapped to its arguments; Langfuse runs
    an ``evaluate(observation)`` function, in Python or TypeScript. There is no
    TypeScript runtime in AX, and generating a Python wrapper against an
    unverified class contract would produce evaluators that look migrated and
    fail when a task first runs them.
    """
    language = str(evaluator.get("sourceCodeLanguage") or "").upper()
    suffix = _SUFFIX.get(language, "txt")
    comment = "#" if suffix == "py" else "//"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(evaluator.get("name") or "evaluator"))
    reason = ("TypeScript has no AX runtime; port to an AX Python evaluator class"
              if language == "TYPESCRIPT" else
              "AX runs a Python evaluator class, not Langfuse's "
              "evaluate(observation) function; port the body into one")
    header = "\n".join(f"{comment} {line}" for line in (
        f"Migrated from Langfuse code evaluator {evaluator.get('name')!r} "
        f"(id {evaluator.get('id')}, version {evaluator.get('version')}).",
        f"Language: {language or 'unknown'}.",
        f"NOT created in Arize AX: {reason}.",
        f"Description: {evaluator.get('description') or '-'}",
    ))
    return f"{name}.{suffix}", f"{header}\n\n{evaluator.get('sourceCode') or ''}", reason


# --------------------------------------------------------- evaluation rules

def rule_task_plan(rule: dict, evaluators_by_id: dict[str, dict],
                   ax_ids_by_lf_id: dict[str, str],
                   mappings_by_lf_id: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """A Langfuse evaluation rule -> AX evaluation-task payloads.

    AX tasks hold one evaluator type, so a mixed rule splits; code evaluators
    are exported, never created, so they cannot be scheduled. A rule with
    filters is NOT turned into a task: Langfuse's structured filters have no
    verified translation to an AX query filter, and dropping a filter would
    evaluate -- and bill -- more traffic than the rule ever did.
    """
    notes: list[str] = []
    if rule.get("filter"):
        return [], [f"rule {rule.get('name')!r} has filters with no verified AX "
                    f"translation; not scheduled -- recreate the filter on the "
                    f"AX task by hand"]
    entries = []
    for assignment in rule.get("evaluatorAssignments") or []:
        lf_id = assignment.get("evaluatorId")
        evaluator = evaluators_by_id.get(lf_id) or {}
        if evaluator.get("type") != "llm_as_judge":
            notes.append(f"{evaluator.get('name') or lf_id!r} is a code evaluator; "
                         f"exported for manual port, not scheduled")
            continue
        ax_id = ax_ids_by_lf_id.get(lf_id)
        if not ax_id:
            notes.append(f"{evaluator.get('name') or lf_id!r} was not created in AX; "
                         f"not scheduled")
            continue
        mapping = dict(mappings_by_lf_id.get(lf_id) or {})
        if assignment.get("variableMapping"):
            override, more = column_mappings(assignment["variableMapping"],
                                             list(mapping))
            mapping.update(override)
            notes += more
        entry = {"evaluator_id": ax_id}
        if mapping:
            entry["column_mappings"] = mapping
        entries.append(entry)
    if not entries:
        return [], notes
    sampling = rule.get("sampling")
    task = {"name": str(rule.get("name") or "langfuse-rule"),
            "task_type": "TEMPLATE_EVALUATION",
            "evaluators": entries,
            "sampling_rate": float(sampling) if sampling is not None else None,
            "is_continuous": bool(rule.get("enabled"))}
    return [task], notes
