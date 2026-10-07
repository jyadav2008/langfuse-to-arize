"""Migrate Langfuse prompts, datasets and score configs into an AX space.

Separate from the span migration for a reason that is easy to miss: spans are
**project**-scoped and these are **space**-scoped. A space holds many projects,
so these resources migrate once per space, are not sharded by day, and are not
timestamp-named the way a destination project is.

That makes the idempotency rule different too. A project gets a fresh
timestamped name every run precisely so a name is never reused; a prompt called
``tenor_intent_router`` has to keep its name or nothing downstream resolves it.
So the rule here is **skip what already exists**, matching the
``if_exists="skip"`` convention, and never overwrite a resource a human may
have edited since.

Nothing here raises on a single failure. One malformed prompt must not strand
the other forty resources, so failures are collected per resource and reported
together with an exit code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .mapping import evaluators as EV
from .mapping import resources as R
from .mapping.scores import ScoreMappingError

#: Resource families, in dependency order. Annotation configs come first so
#: that a space whose spans are already migrated gets its metric definitions
#: bound as early as possible; datasets last because they are the largest.
FAMILIES = ("annotation_configs", "prompts", "datasets", "evaluators",
            "evaluation_rules")


@dataclass
class EvaluatorOptions:
    """How Langfuse evaluators are recreated in AX.

    ``create_tasks`` is off by default on purpose. An AX evaluation task runs
    LLM judges on live traffic, so creating one starts spending against the
    customer's model provider. Evaluator *definitions* are inert and migrate
    by default; scheduling them is an explicit choice.
    """

    integrations: dict = field(default_factory=dict)
    default_integration: str | None = None
    default_model: str | None = None
    export_dir: Path = Path("out/code-evaluators")
    create_tasks: bool = False
    project: str | None = None


class ResourceReport:
    """What happened, per family. Plain enough to serialise to JSON."""

    def __init__(self):
        self.families: dict[str, dict] = {}

    def family(self, name: str) -> dict:
        return self.families.setdefault(
            name, {"created": [], "skipped": [], "failed": [], "notes": [],
                   "exported": []})

    @property
    def failed(self) -> list[str]:
        out = []
        for name, data in self.families.items():
            out += [f"{name}: {entry}" for entry in data["failed"]]
        return out

    def as_dict(self) -> dict:
        return {"families": self.families, "failed": self.failed,
                "ok": not self.failed}


def _existing_names(lister, space: str, attribute: str) -> set[str]:
    """Names already in the space, paginated.

    Returns a set so the caller can decide to skip. A failure to LIST is
    deliberately allowed to propagate: if we cannot tell what exists, creating
    blindly risks duplicate-name errors for every item, and a clear failure up
    front beats forty confusing ones.
    """
    names: set[str] = set()
    cursor = None
    while True:
        kwargs = {"space": space, "limit": 50}
        if cursor:
            kwargs["cursor"] = cursor
        response = lister(**kwargs)
        items = getattr(response, attribute, None) or []
        for item in items:
            name = getattr(item, "name", None)
            if name:
                names.add(str(name))
        pagination = getattr(response, "pagination", None)
        cursor = getattr(pagination, "next_cursor", None) if pagination else None
        if not cursor or not getattr(pagination, "has_more", False):
            break
    return names


# ------------------------------------------------- annotation configs


def migrate_annotation_configs(client, space: str, source, report: ResourceReport,
                               *, dry_run: bool = False, log=print) -> None:
    """Langfuse score configs -> AX annotation configs."""
    data = report.family("annotation_configs")
    configs = source.score_configs()
    if not configs:
        log("  annotation configs: none in Langfuse")
        return

    existing = _existing_names(client.annotation_configs.list, space,
                               "annotation_configs")
    lossy: set[str] = set()

    for config in configs:
        raw_name = config.get("name")
        try:
            payload = R.annotation_config_payload(config)
        except (R.ResourceMappingError, ScoreMappingError) as exc:
            data["failed"].append(f"{raw_name!r}: {exc}")
            log(f"  annotation config {raw_name!r}: REFUSED -- {exc}")
            continue

        lossy.update(R.unmapped_fields(config))
        name = payload["name"]
        if name in existing:
            data["skipped"].append(name)
            continue
        if dry_run:
            data["created"].append(name)
            continue

        try:
            if payload["kind"] == "categorical":
                from arize.annotation_configs.types import (
                    CategoricalAnnotationValueRequest,
                )

                client.annotation_configs.create_categorical(
                    name=name, space=space,
                    values=[CategoricalAnnotationValueRequest(**value)
                            for value in payload["values"]])
            else:
                client.annotation_configs.create_continuous(
                    name=name, space=space,
                    minimum_score=payload["minimum_score"],
                    maximum_score=payload["maximum_score"])
        except Exception as exc:
            from .pipeline import _reason

            data["failed"].append(f"{name!r}: {_reason(exc)}")
            log(f"  annotation config {name!r}: FAILED -- {_reason(exc)}")
            continue
        data["created"].append(name)
        existing.add(name)

    if lossy:
        # Stated, not discovered. A description is the only thing a Langfuse
        # score config carries that AX has nowhere to put.
        note = (f"{', '.join(sorted(lossy))} not carried over: AX annotation "
                f"configs have no such field")
        data["notes"].append(note)
    log(f"  annotation configs: {len(data['created'])} created, "
        f"{len(data['skipped'])} already present, {len(data['failed'])} failed")


# ------------------------------------------------------------- prompts


def migrate_prompts(client, space: str, source, report: ResourceReport,
                    *, dry_run: bool = False, log=print) -> None:
    """Langfuse prompts -> AX prompts, oldest version first.

    Version order matters and is not cosmetic: AX appends versions, so
    migrating v2 before v1 would invert the history and leave the newest
    prompt pointing at the oldest body.
    """
    data = report.family("prompts")
    summaries = source.prompts()
    if not summaries:
        log("  prompts: none in Langfuse")
        return

    existing = _existing_names(client.prompts.list, space, "prompts")
    dropped_config: set[str] = set()

    for summary in summaries:
        name = str(summary.get("name") or "").strip()
        if not name:
            data["failed"].append("a prompt has no name")
            continue
        if name in existing:
            data["skipped"].append(name)
            continue

        versions = sorted(int(v) for v in (summary.get("versions") or []))
        if not versions:
            data["failed"].append(f"{name!r}: no versions")
            continue

        payloads = []
        try:
            for number in versions:
                body = source.prompt_version(name, number)
                if body is None:
                    raise R.ResourceMappingError(
                        f"version {number} could not be fetched")
                payload = R.prompt_version_payload(body)
                payload["_labels"] = R.prompt_labels(body)
                payloads.append(payload)
        except Exception as exc:
            # Covers both a mapping refusal and a failed version fetch: either
            # way this prompt is skipped and the other prompts continue.
            data["failed"].append(f"{name!r}: {exc}")
            log(f"  prompt {name!r}: REFUSED -- {exc}")
            continue

        for payload in payloads:
            dropped_config.update(payload["_dropped_config"])

        if dry_run:
            data["created"].append(f"{name} ({len(payloads)} version(s))")
            continue

        try:
            _create_prompt(client, space, name, payloads)
        except Exception as exc:
            from .pipeline import _reason

            data["failed"].append(f"{name!r}: {_reason(exc)}")
            log(f"  prompt {name!r}: FAILED -- {_reason(exc)}")
            continue
        data["created"].append(f"{name} ({len(payloads)} version(s))")
        existing.add(name)

    if dropped_config:
        data["notes"].append(
            f"Langfuse config keys with no AX invocation param: "
            f"{', '.join(sorted(dropped_config))}")
    data["notes"].append(
        "templates carried over verbatim as MUSTACHE ({{variable}}); "
        "no brace rewriting")
    log(f"  prompts: {len(data['created'])} created, "
        f"{len(data['skipped'])} already present, {len(data['failed'])} failed")


def _create_prompt(client, space: str, name: str, payloads: list[dict]) -> None:
    """Create the prompt from its first version, then append the rest."""
    from arize.prompts.types import InvocationParamsRequest, LLMMessageRequest

    def call_kwargs(payload: dict) -> dict:
        kwargs = {
            "commit_message": payload["commit_message"],
            "input_variable_format": payload["input_variable_format"],
            "provider": payload["provider"],
            "messages": [LLMMessageRequest(**message)
                         for message in payload["messages"]],
        }
        if payload.get("model"):
            kwargs["model"] = payload["model"]
        if payload.get("invocation_params"):
            kwargs["invocation_params"] = InvocationParamsRequest(
                **payload["invocation_params"])
        return kwargs

    first, rest = payloads[0], payloads[1:]
    created = client.prompts.create(space=space, name=name, **call_kwargs(first))
    _apply_labels(client, created, first.get("_labels"))

    for payload in rest:
        version = client.prompts.create_version(
            prompt=name, space=space, **call_kwargs(payload))
        _apply_labels(client, version, payload.get("_labels"))


def _apply_labels(client, created, labels) -> None:
    """Attach Langfuse labels to the version just created.

    Best-effort: a label is metadata, and failing the whole prompt because a
    label would not attach would be the wrong trade.
    """
    if not labels:
        return
    version_id = getattr(created, "id", None)
    version = getattr(created, "version", None)
    if version is not None:
        version_id = getattr(version, "id", version_id)
    if not version_id:
        return
    try:
        client.prompts.set_labels(version_id=str(version_id), labels=list(labels))
    except Exception:
        pass


# ------------------------------------------------------------ datasets


def migrate_datasets(client, space: str, source, report: ResourceReport,
                     *, dry_run: bool = False, expand_input: bool = True,
                     log=print) -> None:
    """Langfuse datasets -> AX datasets, with their items as examples."""
    data = report.family("datasets")
    datasets = source.datasets()
    if not datasets:
        log("  datasets: none in Langfuse")
        return

    existing = _existing_names(client.datasets.list, space, "datasets")
    lossy: set[str] = set()

    for dataset in datasets:
        name = str(dataset.get("name") or "").strip()
        if not name:
            data["failed"].append("a dataset has no name")
            continue
        if name in existing:
            data["skipped"].append(name)
            continue
        if dataset.get("description"):
            lossy.add("description")
        if dataset.get("metadata"):
            lossy.add("metadata")

        try:
            items = source.dataset_items(name)
        except Exception as exc:
            from .pipeline import _reason

            data["failed"].append(f"{name!r}: items unreadable: {_reason(exc)}")
            continue

        rows = R.dataset_examples(items, expand_input=expand_input)
        if not rows:
            # An empty dataset is not an error, but AX requires examples on
            # create, so there is nothing to send. Reported so it is visible.
            data["notes"].append(f"{name!r} has no active items; not created")
            continue
        if dry_run:
            data["created"].append(f"{name} ({len(rows)} example(s))")
            continue

        try:
            client.datasets.create(name=name, space=space, examples=rows)
        except Exception as exc:
            from .pipeline import _reason

            data["failed"].append(f"{name!r}: {_reason(exc)}")
            log(f"  dataset {name!r}: FAILED -- {_reason(exc)}")
            continue
        data["created"].append(f"{name} ({len(rows)} example(s))")
        existing.add(name)

    if lossy:
        data["notes"].append(
            f"dataset-level {', '.join(sorted(lossy))} not carried over: AX "
            f"datasets have no such field (item metadata IS carried, as "
            f"metadata_* columns)")
    log(f"  datasets: {len(data['created'])} created, "
        f"{len(data['skipped'])} already present, {len(data['failed'])} failed")


# --------------------------------------------------------------- driver


def evaluator_options(cfg, *, project: str | None = None) -> EvaluatorOptions:
    """Build evaluator options from configuration, in one place."""
    return EvaluatorOptions(
        integrations=EV.parse_integrations(cfg.get("LFMIGRATE_EVAL_INTEGRATIONS")),
        default_integration=cfg.get("ARIZE_AI_INTEGRATION_ID") or None,
        default_model=cfg.get("LFMIGRATE_EVAL_DEFAULT_MODEL") or None,
        export_dir=Path(cfg.get("LFMIGRATE_EVAL_EXPORT_DIR") or "out/code-evaluators"),
        create_tasks=cfg.bool_("LFMIGRATE_CREATE_EVAL_TASKS", False),
        project=project or cfg.get("ARIZE_PROJECT_NAME") or None,
    )


# ------------------------------------------------------------ evaluators


def migrate_evaluators(client, space: str, source, report: ResourceReport, *,
                       options: EvaluatorOptions, context: dict,
                       dry_run: bool = False, log=print) -> None:
    """Langfuse evaluators -> AX template evaluators; code exported for porting.

    Records the AX id and column mappings of every judge in ``context`` so the
    rule family can schedule them -- including judges that already existed and
    were skipped, or a re-run could never schedule anything.
    """
    data = report.family("evaluators")
    lister = getattr(source, "evaluators", None)
    evaluators = lister() if callable(lister) else []
    if not evaluators:
        log("  evaluators: none in Langfuse")
        return
    context.setdefault("ax_ids", {})
    context.setdefault("mappings", {})
    context["by_id"] = {e.get("id"): e for e in evaluators}

    score_configs = {}
    try:
        score_configs = {c.get("name"): c for c in source.score_configs()}
    except Exception:
        pass
    existing = _existing_names(client.evaluators.list, space, "evaluators")
    judges_unconfigured = not (options.default_integration or options.integrations)

    for evaluator in evaluators:
        name = str(evaluator.get("name") or "").strip()
        kind = evaluator.get("type")
        if kind == "code":
            filename, content, reason = EV.code_evaluator_export(evaluator)
            if not dry_run:
                options.export_dir.mkdir(parents=True, exist_ok=True)
                (options.export_dir / filename).write_text(content)
            data["exported"].append(f"{name} -> {options.export_dir / filename}")
            continue
        if kind != "llm_as_judge":
            data["failed"].append(f"{name!r}: unknown evaluator type {kind!r}")
            continue
        if judges_unconfigured:
            # A choice, not a failure: without an AI integration there is
            # nothing to run a judge with, and that must be configured, never
            # guessed. Reported once below rather than per judge.
            data["skipped"].append(f"{name} (no AI integration configured)")
            continue

        versions = []
        try:
            versions = source.evaluator_versions(evaluator["id"])
        except Exception:
            versions = []
        versions = [{**evaluator, **v} for v in versions] or [evaluator]
        try:
            payloads = [EV.template_evaluator_payload(
                version, integrations=options.integrations,
                default_integration=options.default_integration,
                default_model=options.default_model,
                score_config=score_configs.get(name)) for version in versions]
        except EV.EvaluatorMappingError as exc:
            data["failed"].append(f"{name!r}: {exc}")
            log(f"  evaluator {name!r}: REFUSED -- {exc}")
            continue
        for payload in payloads:
            for note in payload["_notes"]:
                data["notes"].append(f"{name}: {note}")
        context["mappings"][evaluator["id"]] = payloads[-1]["_column_mappings"]

        if name in existing:
            data["skipped"].append(name)
            if not dry_run:
                try:
                    found = client.evaluators.get(evaluator=name, space=space)
                    context["ax_ids"][evaluator["id"]] = found.id
                except Exception:
                    pass
            continue
        if dry_run:
            data["created"].append(f"{name} ({len(payloads)} version(s))")
            context["ax_ids"][evaluator["id"]] = f"<dry-run:{name}>"
            continue
        try:
            first, rest = payloads[0], payloads[1:]
            created = client.evaluators.create_template_evaluator(
                name=name, space=space,
                commit_message=f"Migrated from Langfuse (version "
                               f"{versions[0].get('version', 1)})",
                template_config=first["template_config"],
                description=evaluator.get("description") or None)
            for version, payload in zip(versions[1:], rest):
                client.evaluators.create_template_version(
                    evaluator=created.id, space=space,
                    commit_message=f"Migrated from Langfuse (version "
                                   f"{version.get('version')})",
                    template_config=payload["template_config"])
        except Exception as exc:
            from .pipeline import _reason

            data["failed"].append(f"{name!r}: {_reason(exc)}")
            log(f"  evaluator {name!r}: FAILED -- {_reason(exc)}")
            continue
        context["ax_ids"][evaluator["id"]] = created.id
        data["created"].append(f"{name} ({len(payloads)} version(s))")
        existing.add(name)

    if judges_unconfigured and any("no AI integration" in s for s in data["skipped"]):
        data["notes"].append(
            "LLM judges NOT migrated: set ARIZE_AI_INTEGRATION_ID (or "
            "LFMIGRATE_EVAL_INTEGRATIONS=openai=<id>,...) to the AX AI "
            "integration the judges should run on")
    if data["exported"]:
        data["notes"].append(
            "code evaluators exported for manual port, not created: AX runs a "
            "Python evaluator class, Langfuse an evaluate(observation) function "
            "(and AX has no TypeScript runtime)")
    log(f"  evaluators: {len(data['created'])} created, "
        f"{len(data['skipped'])} skipped, {len(data['exported'])} exported, "
        f"{len(data['failed'])} failed")


def migrate_evaluation_rules(client, space: str, source, report: ResourceReport,
                             *, options: EvaluatorOptions, context: dict,
                             dry_run: bool = False, log=print) -> None:
    """Langfuse evaluation rules -> AX evaluation tasks, only on opt-in."""
    data = report.family("evaluation_rules")
    lister = getattr(source, "evaluation_rules", None)
    rules = lister() if callable(lister) else []
    if not rules:
        log("  evaluation rules: none in Langfuse")
        return
    for rule in rules:
        tasks, notes = EV.rule_task_plan(rule, context.get("by_id") or {},
                                         context.get("ax_ids") or {},
                                         context.get("mappings") or {})
        data["notes"] += [f"{rule.get('name')}: {n}" for n in notes]
        for task in tasks:
            summary = (f"{task['name']} (sampling {task['sampling_rate']}, "
                       f"{len(task['evaluators'])} judge(s), "
                       f"{'continuous' if task['is_continuous'] else 'one-off'})")
            if not options.create_tasks:
                data["skipped"].append(f"{summary} -- not scheduled")
                continue
            if not options.project:
                data["failed"].append(f"{summary}: no destination project")
                continue
            if dry_run:
                data["created"].append(summary)
                continue
            try:
                existing = _existing_names(client.tasks.list, space, "tasks")
                if task["name"] in existing:
                    data["skipped"].append(task["name"])
                    continue
                client.tasks.create_evaluation_task(
                    name=task["name"], task_type=task["task_type"],
                    evaluators=task["evaluators"], project=options.project,
                    space=space, sampling_rate=task["sampling_rate"],
                    is_continuous=task["is_continuous"])
            except Exception as exc:
                from .pipeline import _reason

                data["failed"].append(f"{task['name']!r}: {_reason(exc)}")
                continue
            data["created"].append(summary)
    if not options.create_tasks and data["skipped"]:
        data["notes"].append(
            "evaluation rules were planned, not scheduled: an AX task runs LLM "
            "judges on live traffic and bills the model provider. Set "
            "LFMIGRATE_CREATE_EVAL_TASKS=true to create them")
    log(f"  evaluation rules: {len(data['created'])} scheduled, "
        f"{len(data['skipped'])} planned/skipped, {len(data['failed'])} failed")


def migrate_all(client, space: str, source, *, dry_run: bool = False,
                families=FAMILIES, expand_input: bool = True,
                evaluator_options: EvaluatorOptions | None = None,
                log=print) -> dict:
    """Migrate every resource family. Never raises for one bad resource."""
    report = ResourceReport()
    options = evaluator_options or EvaluatorOptions()
    context: dict = {}
    handlers = {
        "annotation_configs": lambda: migrate_annotation_configs(
            client, space, source, report, dry_run=dry_run, log=log),
        "prompts": lambda: migrate_prompts(
            client, space, source, report, dry_run=dry_run, log=log),
        "datasets": lambda: migrate_datasets(
            client, space, source, report, dry_run=dry_run,
            expand_input=expand_input, log=log),
        "evaluators": lambda: migrate_evaluators(
            client, space, source, report, options=options, context=context,
            dry_run=dry_run, log=log),
        "evaluation_rules": lambda: migrate_evaluation_rules(
            client, space, source, report, options=options, context=context,
            dry_run=dry_run, log=log),
    }
    for family in families:
        handler = handlers.get(family)
        if handler is None:
            report.family(family)["failed"].append("unknown resource family")
            continue
        try:
            handler()
        except Exception as exc:
            from .pipeline import _reason

            report.family(family)["failed"].append(f"family aborted: {_reason(exc)}")
            log(f"  {family}: ABORTED -- {_reason(exc)}")
    return report.as_dict()
