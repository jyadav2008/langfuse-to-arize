"""End-to-end orchestration: credentials in, verified migration out.

The operator supplies credentials. Everything else -- API generation, history
range, day sharding, destination naming, resume point, retention eligibility --
is discovered or derived here.

Ordering is deliberate. Every irreversible step is gated behind a check that
can fail cheaply:

1. ``preflight`` does only read-only work on both sides and creates nothing.
2. Export writes local shards; still nothing in AX.
3. Import is the first write, one day at a time, oldest first.
4. Verification reads back before the next day proceeds, so a systemic problem
   stops after one day instead of three million rows.
"""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import manifest as M
from . import otlp, regions, upload
from .mapping import scores as SC
from .mapping import spans as SP
from .source import base as source_base
from .source.langfuse_api import LangfuseAPI, group_by_trace

#: AX indexing measured at ~5 minutes; the reference tooling waits up to 900s.
#: Reporting "missing" early on a large migration invites a re-upload, which
#: duplicates data, so this is deliberately patient.
VERIFY_TIMEOUT_SECONDS = 900
VERIFY_INTERVAL_SECONDS = 30

#: First poll delay. Read-back is often possible within seconds, so a fixed
#: 30 s interval wasted up to 28 s per check whenever the data was already
#: there -- and the deferred score frames waited a further fixed 30 s per retry.
POLL_FIRST_DELAY = 2.0

#: The only columns a read-back needs. A full export carries every attribute
#: (89 columns on the Tenor data); polling with all of them made each check
#: ~2.4x slower at 7.6k rows, and the cost scales linearly toward 1M.
VERIFY_COLUMNS = ["context.span_id", "parent_id", "start_time",
                  "attributes.openinference.span.kind"]


def poll_delays(cap: float = VERIFY_INTERVAL_SECONDS, first: float = POLL_FIRST_DELAY):
    """Exponential backoff: 2, 4, 8, 16, then ``cap`` forever.

    Short first so data that is already queryable is confirmed in seconds;
    capped so a slow index is still checked twice a minute rather than at
    ever-growing gaps.
    """
    delay = float(first)
    while True:
        yield max(0.0, min(delay, float(cap)))
        delay *= 2


def _sleep_until_next(delays, deadline: float) -> None:
    """Sleep for the next backoff step, never past the deadline."""
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(min(next(delays), remaining))


class PipelineError(Exception):
    pass


# --------------------------------------------------------------- helpers


def _parse(moment) -> datetime | None:
    if not moment:
        return None
    if isinstance(moment, datetime):
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(moment).replace("Z", "+00:00"))
    except ValueError:
        return None


def default_project_name(prefix: str = "langfuse-backfill") -> str:
    """A fresh, non-colliding destination name.

    Never reuses a name: re-sending to a previously DELETED project name lands
    spans in a tombstoned datasource where ingest reports success and the
    project stays permanently empty. A timestamp suffix makes that impossible
    by construction.
    """
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


#: Source mode -> builder. Adding a source means adding an entry here and an
#: implementation satisfying source.base.HistorySource -- no pipeline changes.
def _build_api_source(cfg):
    host, public, secret = cfg.require(
        "LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    kwargs = {"host": host, "public_key": public, "secret_key": secret,
              "timeout": cfg.int_("LANGFUSE_TIMEOUT_SECONDS"),
              "page_limit": cfg.int_("LANGFUSE_PAGE_LIMIT"),
              "score_page_limit": cfg.int_("LANGFUSE_SCORE_PAGE_LIMIT"),
              "metrics_chunk_days": cfg.int_("LANGFUSE_METRICS_CHUNK_DAYS"),
              "expand_metadata_keys": cfg.get("LANGFUSE_EXPAND_METADATA_KEYS"),
              "pinned_generation": cfg.get("LANGFUSE_API_GENERATION")}
    for key, field_name in (("LANGFUSE_OBSERVATION_FIELDS", "observation_fields"),
                            ("LANGFUSE_SCORE_FIELDS", "score_fields")):
        if cfg.get(key):
            kwargs[field_name] = cfg.get(key)
    return LangfuseAPI(**kwargs)


def _unimplemented(mode):
    def build(_cfg):
        raise source_base.SourceNotImplemented(
            f"LANGFUSE_SOURCE_MODE={mode!r} is declared but not implemented yet. "
            f"Use 'api' for now."
        )
    return build


SOURCE_BUILDERS = {
    "api": _build_api_source,
    "clickhouse": _unimplemented("clickhouse"),
    "blob": _unimplemented("blob"),
}


def make_source(cfg):
    mode = str(cfg.get("LANGFUSE_SOURCE_MODE") or "api").strip().lower()
    builder = SOURCE_BUILDERS.get(mode)
    if builder is None:
        raise PipelineError(
            f"Unknown LANGFUSE_SOURCE_MODE {mode!r}. "
            f"Choose one of: {', '.join(sorted(SOURCE_BUILDERS))}."
        )
    source = builder(cfg)
    # Eager: a missing method found on day 40 of a backfill, with data already
    # in the destination, is far worse than failing at startup.
    source_base.check_conformance(source)
    return source


#: Characters an AX project name may carry through unchanged. Deliberately
#: narrow: the name reaches a Flight datasource identifier and a URL, and a
#: Langfuse project name is free text that may hold spaces, slashes or emoji.
_NAME_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


def sanitise_project_name(name: str, *, limit: int = 60) -> str:
    """Make a Langfuse project name usable as an AX project name.

    Keeps the name recognisable rather than encoding it: the whole point is
    that an operator can look at both systems side by side and see the same
    thing. Unsafe runs collapse to a single hyphen so "Tenor / LevFin  Prod"
    reads as "Tenor-LevFin-Prod", not "TenorLevFinProd".
    """
    out, previous_dash = [], False
    for char in (name or "").strip():
        if char in _NAME_SAFE:
            out.append(char)
            previous_dash = False
        elif not previous_dash:
            out.append("-")
            previous_dash = True
    cleaned = "".join(out).strip("-_")[:limit].strip("-_")
    return cleaned


def resolve_destination(cfg, *, resume_root: Path | None = None,
                        source_project: dict | None = None
                       ) -> tuple[dict, bool]:
    """Where this migration writes, and whether it is resuming.

    Resuming MUST reuse the project a previous run bound its shards to. The
    default project name carries a timestamp so a name is never reused, which
    is right for a fresh migration and catastrophic on resume: a second
    invocation would mint a new name, skip the days already done, and write the
    remainder into a *different* AX project -- splitting one history across two
    and leaving both looking plausibly complete.

    Returns ``(destination, resumed)``.
    """
    space = cfg.require("ARIZE_SPACE_ID")[0]
    explicit = cfg.get("ARIZE_PROJECT_NAME")
    bound = destination_from_shards(resume_root) if resume_root else None

    if bound:
        if bound.get("space_id") != space:
            raise PipelineError(
                f"These shards were migrated into space "
                f"{bound.get('space_id')!r}, but the configuration now says "
                f"{space!r}. Refusing to split one migration across two "
                f"spaces. Use a different --root for a different space."
            )
        if explicit and explicit != bound.get("project_name"):
            raise PipelineError(
                f"These shards are bound to project "
                f"{bound.get('project_name')!r}, but ARIZE_PROJECT_NAME is "
                f"{explicit!r}. Refusing to split one migration across two "
                f"projects. Unset ARIZE_PROJECT_NAME to resume, or use a "
                f"fresh --root to start over."
            )
        return dict(bound), True

    # Naming precedence, most explicit first:
    #   1. ARIZE_PROJECT_NAME   -- used verbatim, no timestamp
    #   2. ARIZE_PROJECT_PREFIX -- explicit prefix + timestamp
    #   3. the Langfuse project name + timestamp   <- default
    #   4. "langfuse-backfill" + timestamp         <- if (3) is unavailable
    #
    # (3) is the default because a destination called
    # "tenor-copilot-prod-20261006-180104" can be matched against its Langfuse
    # source at a glance, whereas "langfuse-backfill-20261006-180104" tells an
    # operator with several migrations nothing at all. Only `values` is
    # consulted for the prefix, so the declared default cannot masquerade as
    # an explicit choice.
    prefix = cfg.values.get("ARIZE_PROJECT_PREFIX")
    if not prefix and source_project:
        prefix = sanitise_project_name(
            source_project.get("name") or source_project.get("id") or "")
    project = explicit or default_project_name(prefix or "langfuse-backfill")
    return {"space_id": space, "project_name": project}, False


def make_destination(cfg, *, resume_root: Path | None = None,
                     source_project: dict | None = None):
    """(ArizeClient, destination dict, resolved hosts, resuming?)."""
    key = cfg.require("ARIZE_API_KEY")[0]
    hosts = regions.resolve(cfg.get("ARIZE_REGION"), cfg.overrides())
    from arize import ArizeClient

    client = ArizeClient(api_key=key, **regions.client_kwargs(hosts))
    destination, resumed = resolve_destination(
        cfg, resume_root=resume_root, source_project=source_project)
    return client, destination, hosts, resumed


# -------------------------------------------------------------- preflight


def preflight(cfg, *, max_past_years: int = upload.DEFAULT_MAX_PAST_YEARS,
              root: Path | None = None) -> dict:
    """Read-only checks on both sides. Creates nothing, writes nothing."""
    report: dict = {"checks": [], "ok": True, "warnings": list(cfg.warnings)}

    def record(name, ok, detail, fatal=True):
        report["checks"].append(
            {"name": name, "ok": bool(ok), "detail": detail, "fatal": fatal})
        if not ok and fatal:
            report["ok"] = False

    # Unrecognised settings. Reported before anything else: a silently ignored
    # override looks identical to one that was applied.
    for warning in cfg.warnings:
        record("configuration recognised", False, warning, fatal=False)

    # Credentials present
    missing = [
        key for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                        "ARIZE_API_KEY", "ARIZE_SPACE_ID")
        if not cfg.get(key)
    ]
    record("credentials present", not missing,
           "all set" if not missing else f"missing: {', '.join(missing)}")
    if missing:
        return report

    # Destination region: a mismatch masquerades as dead credentials, so this
    # is surfaced explicitly rather than left to a confusing runtime error.
    try:
        hosts = regions.resolve(cfg.get("ARIZE_REGION"), cfg.overrides())
        record("AX region resolved",
               True,
               f"{hosts.name}: otlp={hosts.otlp_host} flight={hosts.flight_host} "
               f"api={hosts.api_host}")
        if not hosts.verified:
            record("AX region confirmed", False,
                   f"region {hosts.name!r} hosts are not confirmed against a live "
                   "space; a mismatch reports 'invalid Space ID' or 'invalid token'",
                   fatal=False)
    except KeyError as exc:
        record("AX region resolved", False, str(exc))
        return report

    # Source reachable + generation detected
    source = make_source(cfg)
    try:
        generation = source.detect()
        record("Langfuse API reachable", True,
               f"generation {generation['name']} "
               f"(observations={generation['observations']})")
    except Exception as exc:
        record("Langfuse API reachable", False, str(exc))
        return report

    # Which Langfuse project these keys are scoped to. A key pair belongs to
    # exactly one project, so this is the full extent of what will be
    # migrated -- and the last chance to notice it is the wrong one.
    source_project = None
    getter = getattr(source, "source_project", None)
    if callable(getter):
        try:
            source_project = getter()
        except Exception:
            source_project = None
    if source_project:
        report["source_project"] = source_project
        label = source_project.get("name") or source_project.get("id")
        organisation = source_project.get("organisation")
        detail = f"{label!r}"
        if source_project.get("id") and source_project["id"] != label:
            detail += f" (id {source_project['id']})"
        if organisation:
            detail += f" in organisation {organisation!r}"
        record("Langfuse project scope", True,
               detail + " -- these keys see this project only", fatal=False)
    else:
        record("Langfuse project scope", False,
               "could not read /api/public/projects; the destination will be "
               "named from ARIZE_PROJECT_PREFIX instead of the source project",
               fatal=False)

    # History range, discovered
    try:
        history = source.history_range()
        if history.get("detected"):
            record("Langfuse history range", True,
                   f"{history['earliest']} .. {history['latest']}")
            report["history"] = history
            report["source_total_at_discovery"] = _source_total(source, history)
        else:
            record("Langfuse history range", False,
                   history.get("reason", "could not determine"), fatal=False)
    except Exception as exc:
        record("Langfuse history range", False, str(exc), fatal=False)

    # Retention eligibility. The 2-year wall is widened only by an Arize-side
    # code change plus deploy, so it has the longest lead time of anything here
    # and must surface before a long run rather than at the first rejection.
    earliest = _parse((report.get("history") or {}).get("earliest"))
    if earliest:
        cutoff = datetime.now(timezone.utc) - timedelta(days=365 * max_past_years)
        if earliest < cutoff:
            lost_days = (cutoff - earliest).days
            record("history within AX retention window", False,
                   f"oldest observation {earliest.date()} predates the "
                   f"{max_past_years}-year cutoff ({cutoff.date()}) by ~{lost_days} days. "
                   "AX will reject those spans. Widening the window requires an "
                   "Arize-side change to timeRangeAllowedBySpaceYears (keyed by space "
                   "KEY) and a deploy. Days before the cutoff will be skipped.",
                   fatal=False)
        else:
            record("history within AX retention window", True,
                   f"oldest {earliest.date()} is inside the {max_past_years}-year window")

    # Destination reachable, and the project name is unused
    try:
        client, destination, _, resumed = make_destination(
            cfg, resume_root=root, source_project=source_project)
        report["destination"] = destination
        report["resuming"] = resumed
        existing = _project_exists(client, destination)
        record("AX destination reachable", True,
               f"space={destination['space_id']} project={destination['project_name']}")
        if resumed:
            # An existing project is the POINT when resuming. Flagging it as
            # stale here would push an operator towards a fresh name, which is
            # precisely how a migration ends up split across two projects.
            record("destination project", True,
                   f"resuming into {destination['project_name']!r} "
                   f"(recorded in the existing shards)")
        else:
            record("destination project is fresh", not existing,
                   "unused"
                   if not existing
                   else f"project {destination['project_name']!r} already exists. Use a new "
                        "name -- and never re-use a DELETED name, which silently ingests "
                        "into a tombstoned datasource.")
    except Exception as exc:
        record("AX destination reachable", False, str(exc))

    _check_source_quiet(source, report, record)
    source.close()
    return report


def _source_total(source, history: dict) -> int | None:
    """Observation count across the discovered range, or None if unknown."""
    counter = getattr(source, "daily_counts", None)
    earliest = _parse(history.get("earliest"))
    latest = _parse(history.get("latest"))
    if not callable(counter) or not earliest or not latest:
        return None
    try:
        counts = counter(earliest, latest + timedelta(seconds=1))
    except Exception:
        return None
    return sum(int(v) for v in (counts or {}).values())


def _check_source_quiet(source, report: dict, record) -> None:
    """Warn when the source is still ingesting into the range being migrated.

    Found the hard way: a migration started straight after a bulk write broke
    on its 10th of 13 days with a PaginationError, because Langfuse's keyset
    pagination lost its cursor while walking a score set that was still
    growing. Nothing was wrong except the timing.

    The count taken by history discovery is compared with a second count taken
    here, after the AX checks -- so the gap between them costs nothing extra.
    Non-fatal on purpose: a live production project is always ingesting, and
    that must not block a migration of its history. The score walk also
    re-walks a broken batch, so this is advice, not a gate.
    """
    history = report.get("history") or {}
    # Both counts come from the SAME method over the SAME window. The first
    # version compared discovery's 2-year count with a narrower recount, and
    # flagged a quiet source as ingesting because the two methods disagreed.
    first = report.get("source_total_at_discovery")
    if first is None:
        first = sum(int(v) for v in (history.get("daily_counts") or {}).values())
    if not first:
        return
    second = _source_total(source, history)
    if second is None:
        return
    report["source_counts"] = {"first": first, "second": second}
    if second == first:
        record("Langfuse source is quiet", True,
               f"{first:,} observations, unchanged across preflight", fatal=False)
        return
    record("Langfuse source is quiet", False,
           f"observation count moved {first:,} -> {second:,} during preflight: "
           f"the source is still ingesting into the migrated range. If that is "
           f"a bulk load or a seed, wait for it to finish -- paginating a "
           f"growing dataset can break mid-walk. If it is live production "
           f"traffic, that is expected and history is unaffected -- but the "
           f"most recent day is migrated as it stands at export time, and a "
           f"resume will not revisit a day it has completed.",
           fatal=False)


def _project_exists(client, destination) -> bool:
    try:
        listing = client.projects.list(space=destination["space_id"])
    except Exception:
        return False   # not provable either way; the import stage re-checks
    for project in getattr(listing, "projects", []) or []:
        if getattr(project, "name", None) == destination["project_name"]:
            return True
    return False


# ------------------------------------------------------------------ export


def _derive_day(observations: list[dict], *, build_spans: bool = True):
    """Group a day's observations and derive what scoring needs.

    Returns ``(grouped, built_spans, root_index, session_anchor)``. Shared by
    ``export_day`` and by score-frame rebuilds, so a rebuilt frame anchors on
    exactly the span id the original export wrote -- which holds because span
    ids are deterministic.

    ``session_anchor`` maps session -> (earliest start, anchor span id).
    session_eval.* columns cannot be addressed by a session.id column:
    update_evaluations requires context.span_id and ignores session.id
    outright. So a session-level score anchors on a real span, by convention
    the root span of that session's earliest trace.
    """
    grouped = group_by_trace(observations)
    built: list[dict] = []
    root_index: dict[str, str] = {}
    session_anchor: dict[str, tuple[str, str]] = {}
    for trace_id, rows in grouped.items():
        rows.sort(key=lambda r: str(r.get("startTime") or ""))
        if build_spans:
            built.extend(SP.build_trace_spans(trace_id, rows))
        root = SP.root_target_id(trace_id, rows)
        root_index[trace_id] = root
        first_start = str(rows[0].get("startTime") or "") if rows else ""
        for row in rows:
            session = row.get("sessionId") or row.get("session_id")
            if not session:
                continue
            key = str(session)
            current = session_anchor.get(key)
            if current is None or first_start < current[0]:
                session_anchor[key] = (first_start, root)
    return grouped, built, root_index, session_anchor


def _day_scores(source, grouped: dict, session_anchor: dict,
                root_index: dict) -> tuple[list[dict], dict]:
    """Fetch and partition the scores belonging to one day's traces/sessions.

    Fetched by TRACE and by SESSION, not by a timestamp window: a score's
    timestamp need not fall on the same day as the trace it annotates, so a
    timestamp-windowed fetch orphans scores whose trace lives in a different
    shard.
    """
    raw_scores = list(source.scores_for_traces(sorted(grouped)))
    sessions = sorted(session_anchor)
    if sessions:
        raw_scores += list(source.scores_for_sessions(sessions))
    # A trace query and a session query can both return the same score when a
    # session-scoped score also carries a trace subject.
    seen, deduped = set(), []
    for score in raw_scores:
        key = score.get("id")
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        deduped.append(score)
    return deduped, SC.partition(deduped, root_resolver=root_index.get)


def deferred_frames(partitioned: dict, session_anchor: dict | None,
                    ingest_path: str) -> dict:
    """The score frames that cannot ride with a day's span upload.

    Annotations and session evals go through Flight *update* operations, which
    need the destination project to exist and be indexed. On OTLP the
    span-level evals have no side-channel either. All are keyed by
    context.span_id -- which is why both ingest paths preserve span ids.
    """
    span_evals: list[dict] = []
    if ingest_path != otlp.PATH_ARROW:
        for span_id, row in (partitioned.get("evals") or {}).items():
            remapped = {k: v for k, v in row.items() if k != "context.span_id"}
            remapped["context.span_id"] = span_id
            span_evals.append(remapped)
    session_rows, unanchored = _anchored_session_rows(partitioned, session_anchor)
    if ingest_path == otlp.PATH_ARROW:
        # Ride the upload instead (see upload_evals). Deferring them sends them
        # through the update path, which cannot see spans older than ~31 days:
        # on a 60-day backfill all 12 session scores came back "unmatched".
        session_rows = []
    return {"annotations": list((partitioned.get("annotations") or {}).values()),
            "span_evals": span_evals, "session_evals": session_rows,
            "unanchored": unanchored}


def _anchored_session_rows(partitioned: dict, session_anchor: dict | None):
    """Session-eval rows re-keyed onto their anchor span."""
    rows, unanchored = [], []
    for session_id, row in (partitioned.get("session_evals") or {}).items():
        anchor = (session_anchor or {}).get(session_id)
        if not anchor:
            unanchored.append(session_id)
            continue
        remapped = {k: v for k, v in row.items() if k != "session.id"}
        remapped["context.span_id"] = anchor
        rows.append(remapped)
    return rows, unanchored


def upload_evals(partitioned: dict, session_anchor: dict | None,
                 ingest_path: str) -> dict:
    """Eval columns that travel WITH the span upload, keyed by span id.

    On the Arrow path that is span/trace evals plus session evals. Verified on
    a 60-day-old span: session_eval.* columns sent in spans.log's
    evals_dataframe persist, while the same columns sent later through
    update_evaluations are reported "unmatched". The anchor is the root span
    of the session's earliest trace that day, so it is always in the same
    shard -- and the same batch logic that attaches trace evals finds it.
    """
    evals = {k: dict(v) for k, v in (partitioned.get("evals") or {}).items()}
    if ingest_path != otlp.PATH_ARROW:
        return evals
    rows, _unanchored = _anchored_session_rows(partitioned, session_anchor)
    for row in rows:
        span_id = row["context.span_id"]
        evals.setdefault(span_id, {"context.span_id": span_id}).update(row)
    return evals


def export_day(source: LangfuseAPI, shard: M.Shard, start: datetime, end: datetime,
               *, ingest_path: str = otlp.PATH_ARROW,
               batch_size: int = M.DEFAULT_BATCH_SIZE,
               batch_max_bytes: int = M.DEFAULT_BATCH_MAX_BYTES,
               overwrite: bool = False) -> dict:
    """Export one UTC day into a shard, plus its score frames.

    Whole traces are assembled before writing so the root decision and
    re-parenting see the full sibling set. A trace that straddles midnight is
    exported with the day its observations fall in; AX requires a trace's spans
    to arrive together, which day-sharding preserves for all but the straddling
    minority.
    """
    observations = list(source.observations(start, end))
    grouped, built, root_index, session_anchor = _derive_day(observations)

    manifest = M.write_shard(
        shard, built,
        source={"system": "langfuse", "generation": (source.generation or {}).get("name"),
                "window_start": start.isoformat(), "window_end": end.isoformat(),
                "observation_count": len(observations), "trace_count": len(grouped),
                "ingest_path": ingest_path},
        batch_size=batch_size, batch_max_bytes=batch_max_bytes,
        overwrite=overwrite,
    )

    raw_scores, partitioned = _day_scores(source, grouped, session_anchor,
                                          root_index)
    return {
        "manifest": manifest,
        "observations": len(observations),
        "traces": len(grouped),
        "scores": len(raw_scores),
        "partitioned": partitioned,
        "session_anchor": {k: v[1] for k, v in session_anchor.items()},
    }


# ------------------------------------------------------------------ import


def import_day(shard: M.Shard, client, destination: dict, partitioned: dict,
               *, session_anchor: dict | None = None, validate: bool = True,
               ingest_path: str = otlp.PATH_OTLP, api_key: str | None = None,
               otlp_shards: int | None = None, otlp_session=None,
               progress=None) -> dict:
    """Import a shard, then apply the score frames that cannot ride with it."""
    result = upload.import_shard(
        shard, client, destination,
        evals_by_span=upload_evals(partitioned, session_anchor, ingest_path),
        validate=validate,
        ingest_path=ingest_path,
        api_key=api_key,
        otlp_shards=otlp_shards,
        otlp_session=otlp_session,
        progress=progress,
    )
    frames = deferred_frames(partitioned, session_anchor, ingest_path)
    result["deferred_annotations"] = frames["annotations"]
    result["deferred_span_evals"] = frames["span_evals"]
    result["deferred_session_evals"] = frames["session_evals"]
    result["session_evals_unanchored"] = frames["unanchored"]
    result["scores_skipped"] = len(partitioned.get("skipped") or [])
    return result


#: Root of the SDK's logger tree. Its loggers print a full traceback at ERROR
#: for failures we catch, report and retry. Left to themselves they make an
#: expected, handled condition ("project not indexed yet") look like a crash in
#: a run that then prints "Done" -- which teaches an operator to ignore
#: tracebacks, the one habit a migration tool must not install.
_SDK_LOGGER_ROOT = "arize"


def _sdk_loggers() -> list:
    """Every live SDK logger, discovered at call time.

    Enumerated rather than listed by name. Disabling a parent does NOT silence
    its children, and the first version of this named three loggers explicitly
    -- which missed ``arize._exporter.client`` and let the readback path keep
    printing tracebacks after the upload path had been fixed. Discovery means a
    logger added by a future SDK version is covered without maintenance.
    """
    manager = logging.Logger.manager
    names = [name for name in list(manager.loggerDict)
             if name == _SDK_LOGGER_ROOT
             or name.startswith(_SDK_LOGGER_ROOT + ".")]
    loggers = [logging.getLogger(_SDK_LOGGER_ROOT)]
    loggers += [logging.getLogger(name) for name in names
                if name != _SDK_LOGGER_ROOT]
    return loggers


@contextlib.contextmanager
def _sdk_quiet():
    """Silence SDK error logging for a call whose failure we handle ourselves.

    Scoped to one call and restored in a ``finally``, so an unexpected error
    anywhere else still logs normally. The exception itself is not swallowed --
    the caller inspects it and surfaces the reason.
    """
    saved = []
    try:
        for logger in _sdk_loggers():
            saved.append((logger, logger.level, logger.propagate, logger.disabled))
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False
            logger.disabled = True
        yield
    finally:
        for logger, level, propagate, disabled in saved:
            logger.setLevel(level)
            logger.propagate = propagate
            logger.disabled = disabled


#: Cut everything from here on: gRPC debug context is pages of peer detail.
_REASON_TAIL_MARKERS = (". gRPC client debug context", " Client context:")

#: Boilerplate the SDK stacks in front of the real message. Three layers wrap
#: it, which is ~110 characters of prefix -- enough to push the only
#: informative part ("project not found" vs "cannot find data from last 31
#: days") past any sane truncation limit.
_REASON_PREFIXES = (
    "Error during update request:",
    "Error logging arrow table to Arize:",
    "Flight returned not found error, with message:",
    "Flight returned unavailable error, with message:",
    "Flight returned unauthorized error, with message:",
    "Error getting flight info or do_get:",
)

#: Readback failures that mean "not queryable yet", not "misconfigured".
#: ``model does not exist or denied access to the model`` is the export plane's
#: pre-index state and reads alarmingly like a credentials problem; during a
#: poll it is ordinary progress. Matched on text because the SDK raises
#: ``FlightUnauthorizedError`` for both this and a genuine auth failure, so the
#: exception type cannot distinguish them.
_NOT_YET_INDEXED = (
    "model does not exist or denied access to the model",
    "project not found",
    "cannot find data from last",
)


def _is_not_yet_indexed(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in _NOT_YET_INDEXED)


def _reason(exc: Exception, *, limit: int = 160) -> str:
    """A one-line cause for an operator to act on.

    The exception type alone is useless here: every Flight failure arrives as
    ``RuntimeError``, and the difference between "project not visible yet"
    (wait) and "space not found" (fix your config) is only in the message.
    """
    text = " ".join(str(exc).split())
    for marker in _REASON_TAIL_MARKERS:
        cut = text.find(marker)
        if cut != -1:
            text = text[:cut]
    # Peel repeatedly: the layers nest, and peeling one exposes the next.
    peeled = True
    while peeled:
        peeled = False
        for prefix in _REASON_PREFIXES:
            if text.lstrip().startswith(prefix):
                text = text.lstrip()[len(prefix):].lstrip()
                peeled = True
    text = text.strip().rstrip(".") or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


#: Prefix marking a frame problem that no retry can fix.
UNMATCHED = "unmatched:"

#: Arize's Flight update path only reaches spans inside this lookback. Its own
#: error text says "cannot find data from last 31 days", and a probe confirmed
#: it: one update_evaluations call on a 6-day-old and a 60-day-old root span
#: returned {"records_updated": "1", "unmatched_ids": [<the 60-day span>]} --
#: the old span exists, but the update path cannot see it.
UPDATE_WINDOW_DAYS = otlp.UPDATE_WINDOW_DAYS


def _apply_frame(client, destination: dict, rows: list[dict],
                 method: str) -> list[str]:
    """Apply one update frame and report what actually landed.

    The response is read, not just the absence of an exception. The first
    version only caught exceptions, so it printed "applied 12 session evals"
    for a call whose response said 0 updated and 12 unmatched -- and those 12
    session scores were reported as migrated when none were.
    """
    import pandas as pd

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    try:
        with _sdk_quiet():
            response = getattr(client.spans, method)(
                space_id=destination["space_id"],
                project_name=destination["project_name"],
                dataframe=frame,
            )
    except Exception as exc:
        # Reported, not raised. The spans for this day are already uploaded and
        # verified; aborting the whole migration over a secondary score frame
        # would strand good data behind a lesser problem. Scores can be
        # re-applied independently.
        return [f"{method} failed for {len(rows)} row(s): {_reason(exc)}"]

    unmatched = []
    if isinstance(response, dict):
        unmatched = [str(i) for i in (response.get("unmatched_ids") or [])]
    if unmatched:
        return [f"{UNMATCHED} {method}: {len(unmatched)} of {len(rows)} row(s) "
                f"target spans the update path cannot reach (it only sees the "
                f"last ~{UPDATE_WINDOW_DAYS} days); these scores must be attached "
                f"when the spans are uploaded. First: {unmatched[0]}"]
    return []


#: Column to bucket exported spans by, most trustworthy first.
#:
#: ``start_time`` is the SPAN's own time on both ingest paths. ``time`` is not:
#: on the Arrow path it matches the span, but on the OTLP path it is the
#: INGEST time. Bucketing by ``time`` therefore put a whole 5-day OTLP
#: migration into a single day and reported four days as having 0 of N rows
#: readable while the total read back 920/920 -- a false failure on correct
#: data. Verified live:
#:     start_time per-day: 10-02:248  10-03:115  10-04:104  10-05:181  10-06:272
#:     time       per-day: 10-06:920   (all within 3 seconds of each other)
_DAY_COLUMNS = ("start_time", "time")


def expected_by_day(root: Path) -> dict[str, int]:
    """Per-day span counts this migration has written, from the manifests.

    Used as the oracle for a single end-of-migration readback. Taken from the
    shards rather than from this run's counters so that days completed by an
    earlier, interrupted run are checked too -- otherwise a resumed migration
    would only ever verify its own tail.
    """
    counts: dict[str, int] = {}
    for shard in M.discover_shards(root):
        try:
            manifest = M.load_manifest(shard)
        except M.ManifestError:
            continue
        progress = M.shard_progress(manifest)
        if progress.get("spans_submitted"):
            counts[manifest["shard"]] = int(progress["spans_submitted"])
    return counts


def _rows_by_day(frame) -> dict[str, int]:
    """Count exported spans per UTC day."""
    column = next((c for c in _DAY_COLUMNS if c in frame.columns), None)
    if column is None:
        return {}
    import pandas as pd

    series = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    return {str(day): int(n) for day, n in
            series.dt.strftime("%Y-%m-%d").value_counts().items()}


def verify_days(client, destination: dict, expected: dict[str, int], *,
                since: datetime, until: datetime,
                timeout: int = VERIFY_TIMEOUT_SECONDS,
                interval: int = VERIFY_INTERVAL_SECONDS,
                progress=None) -> dict:
    """Verify every migrated day in ONE readback, attributed per day.

    This exists so a migration does not have to block on indexing day by day.
    Indexing is a server-side background process: day 4's segments are built
    while day 5 is being imported, so waiting after each day serialises work
    that is already concurrent. Across 365 days at ~5 minutes each that is
    ~30 hours of waiting for information that one readback at the end gives.

    Attribution is kept because that was the only real argument for the
    per-day wait: the export is bucketed by UTC day and compared against each
    shard's submitted count, so a shortfall still names the day.
    """
    started = time.time()
    deadline = started + timeout
    delays = poll_delays(cap=interval)
    total_expected = sum(expected.values())
    last: dict[str, int] = {}
    while True:
        try:
            with _sdk_quiet():
                frame = client.spans.export_to_df(
                    space_id=destination["space_id"],
                    project_name=destination["project_name"],
                    start_time=since, end_time=until + timedelta(hours=1),
                    columns=VERIFY_COLUMNS,
                )
        except Exception as exc:
            frame, reason = None, _reason(exc)
            detail = (f"not queryable yet -- {reason}"
                      if _is_not_yet_indexed(reason) else reason)
        else:
            last = _rows_by_day(frame)
            short = {day: (last.get(day, 0), want)
                     for day, want in expected.items()
                     if last.get(day, 0) < want}
            if not short:
                kinds = {}
                column = "attributes.openinference.span.kind"
                if column in frame.columns:
                    kinds = frame[column].value_counts().to_dict()
                parent = frame["parent_id"] if "parent_id" in frame.columns else None
                roots = (int((parent.isna() | (parent == "")).sum())
                         if parent is not None else None)
                return {
                    "verified": True,
                    "rows": len(frame),
                    "expected": total_expected,
                    "matches_expected": len(frame) >= total_expected,
                    "days": {day: last.get(day, 0) for day in expected},
                    "span_kinds": kinds,
                    "root_spans": roots,
                    # Elapsed wall-clock, not attempts x interval: with
                    # backoff the latter overstated the wait.
                    "waited_seconds": round(time.time() - started, 1),
                }
            done = len(expected) - len(short)
            detail = (f"{done}/{len(expected)} day(s) complete, "
                      f"{sum(last.get(d, 0) for d in expected)}/"
                      f"{total_expected} rows")

        if time.time() >= deadline:
            return {
                "verified": False,
                "rows": sum(last.get(day, 0) for day in expected),
                "expected": total_expected,
                "days": {day: last.get(day, 0) for day in expected},
                "short_days": {day: {"readable": last.get(day, 0),
                                     "submitted": want}
                               for day, want in expected.items()
                               if last.get(day, 0) < want},
                "waited_seconds": int(timeout),
                "detail": detail,
                "hint": "Indexing may still be catching up -- re-run "
                        "'lfmigrate verify'. If a day stays short, confirm the "
                        "destination project name was never previously deleted: "
                        "a reused deleted name ingests into a tombstoned "
                        "datasource where upload reports success forever.",
            }
        if progress:
            progress({"detail": detail})
        _sleep_until_next(delays, deadline)


def project_url(client, destination: dict, *, since=None, until=None) -> str | None:
    """A clickable UI link for the destination project.

    Worth the effort because "I ran it and cannot see anything in the UI" is
    the most likely first reaction to a successful migration, for two reasons
    that are both invisible from the terminal:

    * The Arrow path gets a link for free -- the SDK logs one on success --
      while the OTLP path prints nothing at all.
    * Migrated spans are **backdated**. The UI filters the trace list by its
      own time-range selector, so a default "last 24 hours" hides almost
      everything a backfill just wrote. The link carries the range explicitly.

    Returns ``None`` rather than raising: a missing link must never fail a
    migration that otherwise succeeded.
    """
    space_id = destination.get("space_id")
    project = destination.get("project_name")
    if not space_id or not project:
        return None
    try:
        organisation = _organisation_for_space(client, space_id)
    except Exception:
        organisation = None
    if not organisation:
        return None
    from urllib.parse import quote

    # '=' is left unencoded: org and space ids are base64 and the SDK's own
    # success URL carries the padding raw, so encoding it to %3D would differ
    # from the link Arize itself emits.
    url = (f"https://app.arize.com/organizations/{quote(organisation, safe='=')}"
           f"/spaces/{quote(space_id, safe='=')}"
           f"/models/modelName/{quote(project, safe='')}"
           f"?selectedTab=llmTracing")
    if since and until:
        # Epoch millis, which is what the UI range selector accepts.
        url += (f"&startTime={int(since.timestamp() * 1000)}"
                f"&endTime={int(until.timestamp() * 1000)}")
    return url


def _organisation_for_space(client, space_id: str) -> str | None:
    """The organisation that owns a space.

    Neither ``spaces.get`` nor ``projects.list`` returns an organisation id,
    so it is found by listing organisations and matching their spaces. Cheap
    enough once per run, and it fails soft.
    """
    listing = client.organizations.list(limit=100)
    for organisation in getattr(listing, "organizations", None) or []:
        organisation_id = getattr(organisation, "id", None)
        if not organisation_id:
            continue
        try:
            spaces = client.spaces.list(organization_id=organisation_id,
                                        limit=100)
        except Exception:
            continue
        for space in getattr(spaces, "spaces", None) or []:
            if getattr(space, "id", None) == space_id:
                return str(organisation_id)
    return None


def destination_from_shards(root: Path) -> dict | None:
    """The destination a previous run bound its shards to.

    Needed because the project name is auto-generated per run: a later
    ``verify`` would otherwise mint a NEW name and report the wrong project
    empty. The binding is already recorded in each shard manifest.
    """
    for shard in M.discover_shards(root):
        try:
            destination = M.load_manifest(shard).get("destination")
        except M.ManifestError:
            continue
        if destination:
            return destination
    return None


#: Per-day score-frame sidecar, written next to the shard as soon as the day
#: is exported. Before it existed, deferred frames lived only in memory until
#: the end of a run, so a run that crashed lost them -- and a resume skips
#: completed days, so it never collected them again. Observed on a 1M-span
#: migration: two runs aborted on Langfuse query timeouts, the third completed
#: and verified 1,000,199 rows, and all 12 session scores were silently gone.
FRAMES_SUFFIX = ".scores.json"


def frames_path(shard: M.Shard) -> Path:
    return shard.root / f"{shard.name}{FRAMES_SUFFIX}"


def write_day_frames(shard: M.Shard, destination: dict, frames: dict,
                     window: tuple[datetime, datetime] | None = None) -> Path:
    """Persist a day's deferred frames atomically, unapplied.

    The day's window is stored with them because annotations are written
    through spans.annotate, which only finds records inside a time window --
    31 days by default, so an explicit window is what reaches older history.
    """
    import json as _json

    path = frames_path(shard)
    payload = {"destination": destination, "applied": False,
               "window": [window[0].isoformat(), window[1].isoformat()] if window else None,
               "annotations": frames.get("annotations") or [],
               "span_evals": frames.get("span_evals") or [],
               "session_evals": frames.get("session_evals") or [],
               "unanchored": frames.get("unanchored") or []}
    M._atomic_write(path, _json.dumps(payload, default=str))
    return path


def pending_frames(root: Path) -> tuple[list[Path], dict]:
    """Every unapplied day's frames under ``root``, merged."""
    import json as _json

    paths, merged = [], {"annotations": [], "span_evals": [], "session_evals": []}
    batches = []
    for path in sorted(Path(root).glob(f"*{FRAMES_SUFFIX}")):
        try:
            data = _json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("applied"):
            continue
        paths.append(path)
        for key in merged:
            merged[key] += data.get(key) or []
        if data.get("annotations"):
            window = data.get("window") or [None, None]
            batches.append((_parse(window[0]), _parse(window[1]),
                            data["annotations"]))
    merged["annotation_batches"] = batches
    return paths, merged


def mark_frames_applied(paths: list[Path]) -> None:
    import json as _json

    for path in paths:
        data = _json.loads(path.read_text())
        data["applied"] = True
        M._atomic_write(path, _json.dumps(data, default=str))


SCORE_PLAN_NAME = "deferred_scores.json"


def save_score_plan(root: Path, destination: dict, annotations: list[dict],
                    session_evals: list[dict],
                    span_evals: list[dict] | None = None) -> Path:
    """Persist frames that cannot be applied until the project is warm."""
    import json as _json

    root.mkdir(parents=True, exist_ok=True)
    path = root / SCORE_PLAN_NAME
    path.write_text(_json.dumps({
        "destination": destination,
        "annotations": annotations,
        "session_evals": session_evals,
        "span_evals": span_evals or [],
    }, indent=2, default=str))
    return path


def load_score_plan(root: Path) -> dict:
    import json as _json

    path = root / SCORE_PLAN_NAME
    if not path.exists():
        raise PipelineError(
            f"No deferred score plan at {path}. Nothing to apply."
        )
    return _json.loads(path.read_text())


def rebuild_score_frames(cfg, root: Path, *, log=print) -> dict:
    """Recreate missing per-day frame sidecars for completed shards.

    For migrations that ran before sidecars existed, or whose frames were lost.
    Re-reads each day's observations and scores from the source WITHOUT
    touching the shard: no re-export, no re-upload, no ledger change. Anchors
    match the original upload because span ids are deterministic.
    """
    destination = destination_from_shards(Path(root))
    if not destination:
        raise PipelineError(f"No migrated shards under {root}; nothing to rebuild.")
    source = make_source(cfg)
    source.detect()
    rebuilt, frames_total = 0, 0
    try:
        for shard in M.discover_shards(Path(root)):
            if frames_path(shard).exists():
                continue
            try:
                manifest = M.load_manifest(shard)
            except M.ManifestError:
                continue
            window = manifest.get("source") or {}
            start = _parse(window.get("window_start"))
            end = _parse(window.get("window_end"))
            if not start or not end:
                log(f"  {shard.name}: no recorded window; skipped")
                continue
            observations = list(source.observations(start, end))
            grouped, _built, root_index, anchors = _derive_day(
                observations, build_spans=False)
            _scores, partitioned = _day_scores(source, grouped, anchors, root_index)
            # Recorded per shard from now on; shards that predate the field
            # were written by the Arrow path, where span evals ride with the
            # upload and are therefore not deferred.
            path_name = window.get("ingest_path") or otlp.PATH_ARROW
            frames = deferred_frames(partitioned,
                                     {k: v[1] for k, v in anchors.items()},
                                     path_name)
            write_day_frames(shard, destination, frames, window=(start, end))
            count = sum(len(frames[k]) for k in
                        ("annotations", "span_evals", "session_evals"))
            frames_total += count
            rebuilt += 1
            if count:
                log(f"  {shard.name}: rebuilt {count} deferred row(s)")
    finally:
        source.close()
    return {"rebuilt_days": rebuilt, "rows": frames_total}


def apply_score_plan(cfg, root: Path, *, log=print) -> dict:
    """Apply pending score frames, once the project has warmed up.

    Reads the per-day sidecars; falls back to a legacy deferred_scores.json
    from runs that predate them.
    """
    client, _destination, _hosts, _resumed = make_destination(cfg)
    frame_paths, pending = pending_frames(Path(root))
    if frame_paths:
        destination = destination_from_shards(Path(root))
        plan = dict(pending, destination=destination)
    else:
        plan = load_score_plan(root)
    destination = plan["destination"]
    annotations = plan.get("annotations") or []
    session_evals = plan.get("session_evals") or []
    span_evals = plan.get("span_evals") or []
    log(f"Applying {len(span_evals)} span-eval, {len(annotations)} annotation "
        f"and {len(session_evals)} session-eval row(s) to "
        f"{destination['project_name']!r}")
    if frame_paths:
        # Sidecars carry each day's window, so annotations can reach old spans.
        problems = apply_annotations(client, destination,
                                     plan.get("annotation_batches") or [], log=log)
        annotations_for_update: list[dict] = []
    else:
        # Legacy deferred_scores.json: no windows recorded.
        problems = []
        annotations_for_update = annotations
    problems += apply_deferred_frames(client, destination, annotations_for_update,
                                      session_evals, span_evals, log=log)
    if frame_paths and not problems:
        mark_frames_applied(frame_paths)
    return {"destination": destination, "annotations": len(annotations),
            "session_evals": len(session_evals),
            "span_evals": len(span_evals), "problems": problems}


#: spans.annotate accepts up to 1000 records per request at SPAN granularity.
ANNOTATE_BATCH = 1000
_ANNOTATION_FIELDS = ("label", "score", "text")


def annotation_records(rows: list[dict]) -> list[dict]:
    """annotation.<name>.<field> rows -> spans.annotate record dicts.

    ``updated_by`` / ``updated_at`` are dropped: annotate takes name, label,
    score and text only, and records the writing API key as the author. The
    Langfuse author survives in ``text`` when the score had no comment of its
    own, so the provenance is not lost entirely.
    """
    records = []
    for row in rows:
        span_id = row.get("context.span_id")
        if not span_id:
            continue
        by_name: dict[str, dict] = {}
        for key, value in row.items():
            if not key.startswith("annotation.") or value is None:
                continue
            try:
                _prefix, rest = key.split(".", 1)
                name, field = rest.rsplit(".", 1)
            except ValueError:
                continue
            by_name.setdefault(name, {})[field] = value
        values = []
        for name, fields in by_name.items():
            value = {"name": name}
            if fields.get("label") is not None:
                value["label"] = str(fields["label"])
            if fields.get("score") is not None:
                value["score"] = float(fields["score"])
            text = fields.get("text")
            if text is None and fields.get("updated_by"):
                text = f"migrated from Langfuse; author {fields['updated_by']}" + (
                    f" at {fields['updated_at']}" if fields.get("updated_at") else "")
            if text is not None:
                value["text"] = str(text)
            if len(value) > 1:
                values.append(value)
        if values:
            records.append({"record_id": str(span_id), "values": values})
    return records


def apply_annotations(client, destination: dict, batches: list, *,
                      log=print) -> list[str]:
    """Write annotations with spans.annotate, one call set per day window.

    Replaces update_annotations, which failed twice over: its own column
    validation rejected the rows this tool built
    ("Invalid_DataFrame_Column_Content_Types"), and like every Flight update it
    cannot see spans older than ~31 days. spans.annotate takes an explicit
    window -- verified on a 60-day-old span: rejected (404) with the default
    window, accepted with that day's window -- and rejects a batch loudly when
    a record is missing rather than reporting it as unmatched.
    """
    from arize.spans.types import AnnotateRecordInput, AnnotationInput

    problems: list[str] = []
    total = 0
    for start, end, rows in batches:
        records = annotation_records(rows)
        kwargs = {"project": destination["project_name"],
                  "space": destination["space_id"]}
        if start and end:
            kwargs["start_time"] = start - timedelta(hours=1)
            kwargs["end_time"] = end + timedelta(hours=1)
        for i in range(0, len(records), ANNOTATE_BATCH):
            chunk = records[i:i + ANNOTATE_BATCH]
            try:
                with _sdk_quiet():
                    client.spans.annotate(annotations=[
                        AnnotateRecordInput(record_id=r["record_id"], values=[
                            AnnotationInput(**v) for v in r["values"]])
                        for r in chunk], **kwargs)
                total += len(chunk)
            except Exception as exc:
                day = start.date() if start else "unknown day"
                problems.append(f"annotate failed for {len(chunk)} record(s) on "
                                f"{day}: {_reason(exc)}")
    if total:
        log(f"  applied {total} annotation record(s)")
    return problems


def apply_deferred_frames(client, destination: dict, annotations: list[dict],
                          session_evals: list[dict],
                          span_evals: list[dict] | None = None, *,
                          attempts: int = 8,
                          delay: int = 30, log=print) -> list[str]:
    """Apply annotation and session-eval frames once the project is visible.

    Retries on "project not found": the destination is created by the first
    span upload and becomes visible to the Flight update path a little later,
    so a single attempt immediately after import loses these frames.
    """
    problems: list[str] = []
    # Annotations are applied by apply_annotations (spans.annotate); only a
    # legacy deferred_scores.json without windows still reaches this list.
    if annotations:
        problems += apply_annotations(client, destination,
                                      [(None, None, annotations)], log=log)
    frames = [(session_evals, "update_evaluations", "session evals")]
    if span_evals:
        # Span-level evals are deferred only on the OTLP path, which has no
        # side-channel to carry them with the batch. On the Arrow path they
        # ride inside spans.log and never reach here.
        frames.insert(0, (span_evals, "update_evaluations", "span evals"))
    for rows, method, label in frames:
        if not rows:
            continue
        # Backoff, not a fixed 30 s: after a successful verify the project is
        # warm and the first or second attempt succeeds, so a fixed wait was
        # pure dead time. ~135 s of total patience is kept for the --no-verify
        # case, where the Flight update path can still be cold.
        delays = poll_delays(cap=delay, first=3.0)
        for attempt in range(attempts):
            issues = _apply_frame(client, destination, rows, method)
            if not issues:
                log(f"  applied {len(rows)} {label}")
                break
            if all(issue.startswith(UNMATCHED) for issue in issues):
                # Permanent: an old span never comes back inside the window.
                problems += issues
                log(f"  {label}: NOT applied -- {issues[0][len(UNMATCHED):].strip()}")
                break
            if attempt < attempts - 1:
                wait = next(delays)
                log(f"  {label} not applied yet ({issues[0]}); retrying in {wait:.0f}s")
                time.sleep(wait)
            else:
                problems += issues
    return problems


# ------------------------------------------------------------------ verify


def _eval_columns(client, destination: dict, start: datetime,
                  end: datetime) -> list[str]:
    """Eval and annotation column names present in the project.

    One sampled full-width export instead of a full one: the column set is
    what matters, not the rows, and at 1M spans a full export takes minutes.
    Falls back to an empty list rather than failing a verification that has
    already succeeded on counts.
    """
    prefixes = ("eval.", "trace_eval.", "session_eval.", "annotation.")
    try:
        with _sdk_quiet():
            frame = client.spans.export_to_df(
                space_id=destination["space_id"],
                project_name=destination["project_name"],
                start_time=start, end_time=end, sample_rate=0.1)
    except Exception:
        return []
    return sorted(c for c in frame.columns if c.startswith(prefixes))


def verify_project(client, destination: dict, *, expected_spans: int | None = None,
                   since: datetime | None = None, until: datetime | None = None,
                   timeout: int = VERIFY_TIMEOUT_SECONDS,
                   interval: int = VERIFY_INTERVAL_SECONDS,
                   progress=None) -> dict:
    """Poll readback until the expected rows appear or the timeout expires.

    Polls rather than checking once: a successful upload means ACCEPTED, not
    queryable. Arize ingest is asynchronous -- the Flight write path returns
    once a batch is durably accepted, while the read path serves a columnar
    store whose segments are built on an interval. Measured at ~6.5 minutes
    for a 248-span day.

    ``since``/``until`` bound the readback window. Passing the window being
    checked rather than the whole history matters at scale: a per-day check
    over "everything so far" re-exports the entire migration every day, which
    is quadratic in the number of days.
    """
    started = time.time()
    deadline = started + timeout
    delays = poll_delays(cap=interval)
    start = since or (datetime.now(timezone.utc) - timedelta(days=3))
    while True:
        now = datetime.now(timezone.utc)
        # An hour of slack past the window: span end_time can exceed the
        # export bound, and a trace straddling midnight would otherwise read
        # short of its submitted count forever.
        end = (until + timedelta(hours=1)) if until else (now + timedelta(hours=1))
        try:
            with _sdk_quiet():
                frame = client.spans.export_to_df(
                    space_id=destination["space_id"],
                    project_name=destination["project_name"],
                    start_time=start, end_time=end,
                    columns=VERIFY_COLUMNS,
                )
        except Exception as exc:
            frame = None
            reason = _reason(exc)
            # Distinguish "wait" from "wrong": until the first rows are
            # indexed the export plane answers with an UNAUTHORIZED status and
            # "model does not exist", which is indistinguishable by type from a
            # bad API key. Reported as progress, not as an error.
            detail = (f"not queryable yet -- {reason}"
                      if _is_not_yet_indexed(reason) else reason)
        else:
            detail = (f"rows={len(frame)}" if expected_spans is None
                      else f"rows={len(frame)}/{expected_spans}")

        # Keep polling while the count is short of what was submitted.
        # Returning on the FIRST non-empty read reports "verified" off a
        # partially indexed project -- after a resumed run it answered 248 of
        # 363 rows and called it done, which is indistinguishable in the output
        # from a migration that genuinely lost a third of its spans.
        short = (frame is not None and expected_spans is not None
                 and len(frame) < expected_spans and time.time() < deadline)
        if short:
            if progress:
                progress({"detail": f"rows={len(frame)}/{expected_spans}, "
                                    f"still indexing"})
            _sleep_until_next(delays, deadline)
            continue

        if frame is not None and len(frame):
            kinds = {}
            column = "attributes.openinference.span.kind"
            if column in frame.columns:
                kinds = frame[column].value_counts().to_dict()
            parent = frame["parent_id"] if "parent_id" in frame.columns else None
            roots = int((parent.isna() | (parent == "")).sum()) if parent is not None else None
            # Polling fetched four columns, so eval/annotation column names
            # are discovered once, here, rather than paid for on every poll.
            evals = _eval_columns(client, destination, start, end)
            return {
                "verified": True,
                "rows": len(frame),
                "expected": expected_spans,
                "matches_expected": (expected_spans is None or len(frame) >= expected_spans),
                "span_kinds": kinds,
                "root_spans": roots,
                "eval_columns": evals,
                "waited_seconds": round(time.time() - started, 1),
            }

        if time.time() >= deadline:
            return {
                "verified": False,
                "rows": len(frame) if frame is not None else 0,
                "expected": expected_spans,
                "waited_seconds": int(timeout),
                "detail": detail,
                "hint": "Upload acceptance is not the same as queryable. If this "
                        "persists, confirm the destination project name was never "
                        "previously deleted -- a reused deleted name ingests into a "
                        "tombstoned datasource with no error.",
            }
        if progress:
            progress({"detail": detail})
        _sleep_until_next(delays, deadline)


# -------------------------------------------------------------------- run


#: How ``run`` verifies. "end" is the default: import every day back to back,
#: then do ONE readback. Indexing runs server-side in the background, so
#: day N indexes while day N+1 uploads -- a per-day wait serialises work that
#: is already concurrent, for no extra information.
VERIFY_END = "end"
VERIFY_EACH_DAY = "each-day"
VERIFY_OFF = "off"


def run(cfg, *, root: Path, batch_size: int = M.DEFAULT_BATCH_SIZE,
        max_days: int | None = None, verify_mode: str = VERIFY_END,
        max_past_years: int = upload.DEFAULT_MAX_PAST_YEARS,
        dry_run: bool = False, migrate_resources: bool = True,
        batch_max_bytes: int = M.DEFAULT_BATCH_MAX_BYTES,
        validate: bool = True, ingest_path: str = otlp.PATH_AUTO,
        otlp_shards: int | None = None, log=print) -> dict:
    """Migrate everything available, oldest day first, resuming if interrupted."""
    _run_started = time.perf_counter()
    _phase: dict[str, float] = {}
    _t = time.perf_counter()
    report = preflight(cfg, max_past_years=max_past_years, root=root)
    _phase["preflight"] = time.perf_counter() - _t
    for check in report["checks"]:
        mark = "ok  " if check["ok"] else ("FAIL" if check["fatal"] else "warn")
        log(f"  [{mark}] {check['name']}: {check['detail']}")
    if not report["ok"]:
        raise PipelineError("Preflight failed; nothing was written.")

    history = report.get("history") or {}
    earliest = _parse(history.get("earliest"))
    latest = _parse(history.get("latest"))
    if not earliest or not latest:
        raise PipelineError(
            "Could not determine the Langfuse history range automatically. "
            "Nothing was written."
        )

    source = make_source(cfg)
    source.detect()
    windows = source.day_windows(earliest, latest)

    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * max_past_years)
    eligible = [(s, e) for s, e in windows if e > cutoff]
    skipped_old = len(windows) - len(eligible)
    if skipped_old:
        log(f"  skipping {skipped_old} day(s) older than the "
            f"{max_past_years}-year AX cutoff ({cutoff.date()})")
    if max_days:
        eligible = eligible[:max_days]

    log(f"\n  {len(eligible)} day(s) to migrate: "
        f"{eligible[0][0].date() if eligible else '-'} .. "
        f"{eligible[-1][0].date() if eligible else '-'}")

    if dry_run:
        source.close()
        return {"preflight": report, "days_planned": len(eligible),
                "days_skipped_old": skipped_old, "dry_run": True}

    # Reuse the destination PREFLIGHT resolved, not a freshly minted one.
    # default_project_name() stamps the current time to the second, and
    # preflight and run are often a second apart -- so re-resolving here
    # produced a name one second later than the one preflight checked for
    # freshness, and the "destination project is fresh" result referred to a
    # project that was never written to. Observed live: preflight reported
    # ...-210144 while the run wrote ...-210145.
    client, resolved, _, _resumed = make_destination(
        cfg, resume_root=root, source_project=report.get("source_project"))
    destination = report.get("destination") or resolved
    if destination != resolved and not _resumed:
        log(f"  using the destination preflight checked: "
            f"{destination['project_name']!r}")
    root.mkdir(parents=True, exist_ok=True)

    # Prompts, datasets and score configs, BEFORE the spans. They are
    # space-scoped rather than project-scoped and cheap, and doing them first
    # means the annotation configs that give the migrated eval columns their
    # labels and ranges already exist by the time the spans carrying those
    # columns arrive. Failures here are reported, never fatal: a malformed
    # prompt must not stop a 3-million-row span backfill.
    api_key = cfg.require("ARIZE_API_KEY")[0]

    # Resolve 'auto' from the volume preflight already discovered. Neither
    # path dominates: Arrow uploads ~1.8x faster but pays a flat ~7 min
    # indexing wait, while OTLP is visible in seconds and uploads slower. The
    # crossover is ~830k spans -- see otlp.choose_path.
    estimated = None
    counts = (report.get("history") or {}).get("daily_counts") or {}
    if counts:
        estimated = sum(int(v) for v in counts.values())
    oldest_age = None
    if earliest:
        oldest_age = (datetime.now(timezone.utc) - earliest).total_seconds() / 86400
    ingest_path, why = otlp.choose_path(ingest_path, estimated, oldest_age)
    log(f"\n  ingest path: {ingest_path} -- {why}")

    # One exporter pool for the whole run. Closed in the finally below so the
    # worker threads are joined even if a day raises.
    otlp_session = (otlp.OtlpSession(destination, api_key,
                                     shards=otlp_shards or otlp.DEFAULT_SHARDS)
                    if ingest_path == otlp.PATH_OTLP else None)

    resource_report = None
    if migrate_resources:
        from . import resources as _resources

        log("\n  migrating prompts, datasets and score configs ...")
        _t = time.perf_counter()
        try:
            resource_report = _resources.migrate_all(
                client, destination["space_id"], source, log=log,
                evaluator_options=_resources.evaluator_options(
                    cfg, project=destination["project_name"]))
        except Exception as exc:
            resource_report = {"families": {}, "ok": False,
                               "failed": [f"aborted: {_reason(exc)}"]}
            log(f"  resources: ABORTED -- {_reason(exc)}")
        _phase["resources"] = time.perf_counter() - _t
        if not resource_report["ok"]:
            log(f"  {len(resource_report['failed'])} resource(s) failed; "
                f"continuing with spans")

    days: list[dict] = []
    deferred_annotations: list[dict] = []
    deferred_span_evals: list[dict] = []
    deferred_session_evals: list[dict] = []
    total_spans = 0

    # try/finally so the OTLP worker threads are joined even if a day
    # raises. Every batch is force-flushed as it is marked submitted, so
    # closing here loses nothing that was already accepted.
    try:
        for start, end in eligible:
            name = start.strftime("%Y-%m-%d")
            shard = M.Shard(root=root, name=name)

            # Resume: a completed shard is skipped rather than re-uploaded.
            if shard.exists():
                try:
                    existing = M.load_manifest(shard)
                    if M.shard_progress(existing)["complete"]:
                        log(f"  {name}: already complete ({existing['span_count']} spans), skipping")
                        days.append({"day": name, "status": "already_complete",
                                     "spans": existing["span_count"]})
                        total_spans += existing["span_count"]
                        continue
                except M.ManifestError as exc:
                    raise PipelineError(f"{name}: {exc}")

            log(f"  {name}: exporting ...")
            _t = time.perf_counter()
            exported = export_day(source, shard, start, end,
                                  ingest_path=ingest_path,
                                  batch_size=batch_size,
                                  batch_max_bytes=batch_max_bytes,
                                  overwrite=shard.exists())
            export_seconds = time.perf_counter() - _t
            spans = exported["manifest"]["span_count"]
            # Persisted BEFORE import: if anything after this point fails, the
            # frames survive on disk for the resume or for apply-scores.
            write_day_frames(shard, destination,
                             deferred_frames(exported["partitioned"],
                                             exported.get("session_anchor"),
                                             ingest_path),
                             window=(start, end))
            log(f"  {name}: {exported['observations']} observations in "
                f"{exported['traces']} traces -> {spans} spans, "
                f"{exported['scores']} scores")
            if spans == 0:
                days.append({"day": name, "status": "empty", "spans": 0})
                continue

            window = upload.check_time_window(M.iter_spans(shard),
                                              max_past_years=max_past_years)
            if not window["ok"]:
                log(f"  {name}: WARNING {window['rejected_count']} span(s) predate the "
                    f"retention cutoff and will be rejected by AX")

            log(f"  {name}: importing {len(exported['manifest']['batches'])} batch(es) ...")
            _t = time.perf_counter()
            result = import_day(shard, client, destination, exported["partitioned"],
                                session_anchor=exported.get("session_anchor"),
                                validate=validate, ingest_path=ingest_path,
                                api_key=api_key, otlp_shards=otlp_shards,
                                otlp_session=otlp_session)
            import_seconds = time.perf_counter() - _t
            deferred_annotations += result.get("deferred_annotations") or []
            deferred_span_evals += result.get("deferred_span_evals") or []
            deferred_session_evals += result.get("deferred_session_evals") or []
            total_spans += spans
            log(f"  {name}: exported in {export_seconds:.2f}s, "
                f"imported in {import_seconds:.2f}s "
                f"({spans / import_seconds:,.0f} spans/s)")
            days.append({"day": name, "status": "imported", "spans": spans,
                         "observations": exported["observations"],
                         "traces": exported["traces"],
                         "scores": exported["scores"],
                         "export_seconds": round(export_seconds, 3),
                         "import_seconds": round(import_seconds, 3),
                         "annotations": len(result.get("deferred_annotations") or []),
                         "session_evals": len(result.get("deferred_session_evals") or []),
                         "scores_skipped": result["scores_skipped"]})

            if verify_mode == VERIFY_EACH_DAY:
                log(f"  {name}: verifying (indexing takes ~5 min) ...")
                # This day's window and this day's count -- NOT the cumulative
                # total from `earliest`. Verifying "everything so far" on every day
                # re-exports the whole migration each time: at 3M rows across a
                # year that is ~550M row-exports instead of 3M, and each day gets
                # slower than the last.
                verification = verify_project(
                    client, destination, expected_spans=spans,
                    since=start, until=end,
                    progress=lambda p: log(f"    waiting ({p['detail']})"))
                days[-1]["verified"] = verification["verified"]
                if not verification["verified"]:
                    raise PipelineError(
                        f"{name}: imported but not verifiable after "
                        f"{verification['waited_seconds']}s. Stopping before the next day. "
                        f"{verification.get('hint','')}"
                    )
                log(f"  {name}: verified ({verification['rows']} rows readable)")

    finally:
        source.close()
        if otlp_session is not None:
            # Only joins the worker threads: every batch was force-flushed as
            # it was marked submitted. Closed here rather than after
            # verification because holding gRPC channels open across a
            # 15-minute readback poll serves nothing.
            otlp_session.close()

    # One readback for the whole migration. Deliberately AFTER every import:
    # indexing is concurrent with uploading, so by the time the last day is in,
    # the earlier days are already queryable and this usually returns on the
    # first or second poll regardless of how many days were migrated.
    #
    # It also warms the project for the deferred score frames below, which need
    # indexed rows to attach to -- so the common case no longer needs a
    # separate 'apply-scores' run.
    verification = None
    if verify_mode == VERIFY_END and eligible:
        expected = expected_by_day(root)
        if expected:
            log(f"\n  verifying {len(expected)} day(s) in one readback "
                f"(indexing is server-side; ~5 min from the LAST upload) ...")
            _t = time.perf_counter()
            verification = verify_days(
                client, destination, expected,
                timeout=cfg.int_("LFMIGRATE_VERIFY_TIMEOUT_SECONDS"),
                interval=cfg.int_("LFMIGRATE_VERIFY_INTERVAL_SECONDS"),
                since=eligible[0][0], until=eligible[-1][1],
                progress=lambda p: log(f"    waiting ({p['detail']})"))
            _phase["verify"] = time.perf_counter() - _t
            if verification["verified"]:
                log(f"  verified {verification['rows']} row(s) across "
                    f"{len(expected)} day(s)")
            else:
                # Reported, NOT raised. Observed live: a 248-span day that
                # timed out here at 900s was fully readable minutes later, and
                # raising had skipped the deferred score frames and the UI link
                # for data that had in fact landed. The spans are already
                # marked submitted in the ledger, so a later `verify` is the
                # correct resolution and resume will not re-send them.
                short = verification.get("short_days") or {}
                for day, counts in sorted(short.items()):
                    log(f"  {day}: not readable yet -- {counts['readable']} of "
                        f"{counts['submitted']} row(s)")
                log(f"\n  NOT VERIFIED after {verification['waited_seconds']}s. "
                    f"The spans were accepted and are recorded as submitted; "
                    f"indexing latency is variable (seconds to >15 min "
                    f"observed for the same volume). Re-check with:")
                log(f"    python -m lfmigrate verify --root {root}")

    score_problems: list[str] = []
    # Every pending day under the root -- including days an earlier,
    # interrupted run completed -- not just what this process exported.
    frame_paths, pending = pending_frames(root)
    deferred_annotations = pending["annotations"]
    deferred_session_evals = pending["session_evals"]
    deferred_span_evals = pending["span_evals"]
    if deferred_annotations or deferred_session_evals or deferred_span_evals:
        # Written to disk BEFORE attempting them. A freshly created project's
        # datasource stays cold for minutes ("cannot find data from last 31
        # days for datasource"), and these frames must survive that rather than
        # being lost to a retry budget.
        plan_path = save_score_plan(root, destination,
                                    deferred_annotations, deferred_session_evals,
                                    deferred_span_evals)
        log(f"\n  applying {len(deferred_span_evals)} span-eval, "
            f"{len(deferred_annotations)} annotation and "
            f"{len(deferred_session_evals)} session-eval row(s) "
            f"(plan saved to {plan_path.name}) ...")
        _t = time.perf_counter()
        # Annotations per day window via spans.annotate; evals via update.
        score_problems = apply_annotations(
            client, destination, pending.get("annotation_batches") or [], log=log)
        score_problems += apply_deferred_frames(
            client, destination, [], deferred_session_evals,
            deferred_span_evals, log=log)
        _phase["scores"] = time.perf_counter() - _t
        if not score_problems:
            mark_frames_applied(frame_paths)
        if score_problems:
            log("  span data is complete; these frames can be applied later with:")
            log(f"    python -m lfmigrate apply-scores --root {root}")

    _phase["export"] = sum(d.get("export_seconds") or 0 for d in days)
    _phase["import"] = sum(d.get("import_seconds") or 0 for d in days)
    _phase["total"] = time.perf_counter() - _run_started
    timing = {k: round(v, 2) for k, v in _phase.items()}

    log("\n  runtime breakdown:")
    for label in ("preflight", "resources", "export", "import", "verify",
                  "scores", "total"):
        if label in timing:
            share = (timing[label] / timing["total"] * 100
                     if timing["total"] else 0)
            suffix = ""
            if label == "import" and total_spans and timing[label]:
                # Spans imported by THIS run. total_spans also counts days a
                # resume skipped, which inflated the rate ~2x on a resumed
                # 1M-span run (reported 2,752 spans/s; the true rate was ~1,410).
                imported = sum(d.get("spans") or 0 for d in days
                               if d.get("status") == "imported")
                if imported:
                    suffix = f"   {imported / timing[label]:,.0f} spans/s"
            log(f"    {label:<10} {timing[label]:8.2f}s  {share:5.1f}%{suffix}")

    link = project_url(client, destination,
                       since=eligible[0][0] if eligible else None,
                       until=eligible[-1][1] if eligible else None)
    if link:
        log(f"\n  view in Arize (time range pre-set to the migrated days):\n    {link}")

    return {
        "preflight": report,
        "destination": destination,
        "project_url": link,
        "timing": timing,
        "score_problems": score_problems,
        "annotations_applied": len(deferred_annotations),
        "session_evals_applied": len(deferred_session_evals),
        "days": days,
        "days_skipped_old": skipped_old,
        "spans_total": total_spans,
        "progress": M.migration_progress(root),
        "resources": resource_report,
        "verification": verification,
        "ingest_path": ingest_path,
    }
