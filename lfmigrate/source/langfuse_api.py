"""Langfuse public-API source adapter.

Everything here is designed so the operator supplies credentials and nothing
else. In particular the adapter DETECTS rather than asks:

* **Which API generation is available.** v4 serves ``/api/public/v2/observations``
  and ``/api/public/v3/scores`` and returns 404 for the legacy
  ``/observations``, ``/traces`` and ``/sessions`` paths. v3 and Cloud still
  serve the legacy paths. :meth:`LangfuseAPI.detect` probes and records which
  generation answered, so no version flag is needed.
* **Which pagination style a response uses.** Langfuse list responses have
  carried both page-based (``meta.page``/``meta.totalPages``) and cursor-based
  shapes. Guessing wrong is dangerous in a specific way: a missing cursor key
  makes a loop exit after ONE page and report success, silently migrating a
  fraction of the data. :func:`paginate` handles both, refuses to exit on an
  unrecognised shape while a full page was returned, and guards against a
  server echoing the same cursor forever.
* **The available history range.** ``earliest``/``latest`` are discovered by
  sorting, so the operator does not supply dates.

Rate limiting: Langfuse Cloud can limit as low as ~30 req/min; self-hosted is
unlimited in practice. Backoff is therefore adaptive rather than configured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

#: Candidate endpoint sets, newest generation first.
GENERATIONS = (
    {
        "name": "v4",
        "observations": "/api/public/v2/observations",
        "scores": "/api/public/v3/scores",
        "metrics": "/api/public/v2/metrics",
    },
    {
        "name": "v3",
        "observations": "/api/public/observations",
        "scores": "/api/public/v2/scores",
        "metrics": "/api/public/metrics",
    },
)

MAX_ATTEMPTS = 5
_log = logging.getLogger(__name__)

#: Rows per page. Probed against a live v4 instance: limit=1000 is accepted and
#: returns 1000 rows; limit=2000 is rejected with HTTP 400. 100 was needlessly
#: conservative -- it makes 10,000 round trips for a million observations where
#: 1,000 suffice, and each round trip costs a full request regardless of size.
#: Lower it only if a deployment enforces a stricter cap or a proxy limits the
#: response body size.
PAGE_LIMIT = 1000

#: Hard ceilings the API enforces; above these it answers HTTP 400. They
#: DIFFER per endpoint, verified by probing a live v4 instance:
#:     /v2/observations  limit=1000 -> 200, limit=2000 -> 400
#:     /v3/scores        limit=100  -> 200, limit=200  -> 400
#: Raising the score limit to match observations breaks every score fetch with
#: an opaque HTTP 400, so the two are tracked separately.
MAX_PAGE_LIMIT = 1000
MAX_SCORE_PAGE_LIMIT = 100

#: Page size for every endpoint that has not been verified to accept more.
#: Probed live: /api/public/score-configs, /v2/prompts, /v2/datasets and
#: /dataset-items all answer limit=100 and reject limit=101 with HTTP 400.
#: Only /v2/observations accepts 1000. So 1000 is an explicit opt-in for
#: observations, and everything else gets the safe value -- the reverse of the
#: first version, where raising the global default to 1000 made every resource
#: family abort with HTTP 400 while the span migration reported success.
SAFE_PAGE_LIMIT = 100

#: Langfuse v4 returns a SUMMARY projection by default -- only the `core` and
#: `basic` field groups. Verified against a live v4.53.0 instance: without these
#: explicitly requested, `input`, `output`, `metadata`, `model`,
#: `modelParameters`, `usageDetails` and `costDetails` are ALL ABSENT. A
#: migration built on the default response would transfer spans that reconcile
#: perfectly by count and carry no content whatsoever.
OBSERVATION_FIELDS = "core,basic,time,io,metadata,model,usage,prompt,trace_context"

#: `subject` carries the trace/observation/session linkage, which is absent from
#: the core score response; `annotation` adds authorUserId/queueId.
SCORE_FIELDS = "details,subject,annotation"

#: Scores cap at 100 per page (observations allow up to 1000).
SCORE_PAGE_LIMIT = 100

#: Day-granularity metrics buckets per request. A two-year span in one
#: request returns nothing instead of erroring, so the range is chunked.
METRICS_CHUNK_DAYS = 60


class SourceError(Exception):
    pass


class QueryTimeout(SourceError):
    """Langfuse's query backend timed out, reported as HTTP 422.

    Not a validation error, despite the status. Observed twice in a
    1M-observation migration -- once on /v3/scores, once on /v2/observations --
    with the body: "Your query could not be completed. Please narrow your
    request by adding more specific filters (e.g., a shorter date range)",
    "error": "Request timed out". Neither reproduced once the instance was idle,
    so it is load-dependent and retryable, and the server's own remedy is to ask
    for less per query.
    """


def _is_query_timeout(status: int, text: str) -> bool:
    lowered = (text or "").lower()
    return status == 422 and ("timed out" in lowered
                              or "could not be completed" in lowered)


class PaginationError(SourceError):
    """Raised when a page sequence cannot be followed safely.

    Deliberately fatal: continuing past an unreadable page boundary would
    migrate a silent subset, which is far worse than stopping.
    """


@dataclass
class LangfuseAPI:
    """Langfuse REST source. Every knob is injectable so a customer can tune it
    by environment variable without editing this file."""

    host: str
    public_key: str
    secret_key: str
    timeout: float = 60.0
    client: object | None = None
    page_limit: int = PAGE_LIMIT
    score_page_limit: int = SCORE_PAGE_LIMIT
    observation_fields: str = OBSERVATION_FIELDS
    score_fields: str = SCORE_FIELDS
    metrics_chunk_days: int = METRICS_CHUNK_DAYS
    expand_metadata_keys: str | None = None
    pinned_generation: str | None = None
    generation: dict | None = field(default=None, init=False)

    def __post_init__(self):
        # Resolved resource endpoints, probed once each.
        self._resource_path_cache: dict[str, str] = {}
        # Clamp rather than let the API answer HTTP 400 part-way through a
        # migration, which surfaces as "Langfuse returned 400 for
        # /api/public/v2/observations" with no hint of the cause.
        for attr, ceiling in (("page_limit", MAX_PAGE_LIMIT),
                              ("score_page_limit", MAX_SCORE_PAGE_LIMIT)):
            value = getattr(self, attr, None)
            if value and value > ceiling:
                _log.warning("%s=%s exceeds the API maximum of %s; clamping.",
                             attr, value, ceiling)
                setattr(self, attr, ceiling)
        self.host = self.host.rstrip("/")
        if not self.host.startswith("https://") and "localhost" not in self.host:
            # Credentials travel as basic auth on every request.
            raise SourceError(
                f"Langfuse host must be https (got {self.host!r}); refusing to send "
                "credentials over plaintext."
            )
        if self.client is None:
            import httpx

            self.client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,  # a redirect could leak basic auth
                auth=(self.public_key, self.secret_key),
            )

    # ---------------------------------------------------------------- http

    def _request(self, path: str, params: dict | None = None) -> tuple[int, dict]:
        """GET with backoff on 429/5xx. Returns (status, body)."""
        import httpx

        url = self.host + path
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.client.get(url, params=params or {})
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise SourceError(
                        f"Langfuse request to {path} failed after "
                        f"{MAX_ATTEMPTS} attempts ({type(exc).__name__})."
                    ) from None
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 422 and _is_query_timeout(
                    422, getattr(response, "text", "")):
                if attempt == MAX_ATTEMPTS - 1:
                    raise QueryTimeout(
                        f"Langfuse query timed out for {path} after "
                        f"{MAX_ATTEMPTS} attempts (HTTP 422 'Request timed "
                        f"out'). The server asks for a narrower request.")
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_ATTEMPTS - 1:
                    raise SourceError(
                        f"Langfuse returned {response.status_code} for {path} after "
                        f"{MAX_ATTEMPTS} attempts."
                    )
                # Honour Retry-After when present: on Cloud it is the only way
                # to avoid hammering a 30 req/min budget.
                delay = response.headers.get("Retry-After")
                time.sleep(float(delay) if delay and delay.isdigit() else 2 ** attempt)
                continue

            if response.status_code == 401 or response.status_code == 403:
                raise SourceError(
                    f"Langfuse rejected the credentials (HTTP {response.status_code}). "
                    "Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and that the key "
                    "has read access to this project."
                )
            if response.status_code == 404:
                return 404, {}
            if response.status_code >= 400:
                # The body is what explains a 4xx -- a 422 that arrived during a
                # 1M-span run and could not be reproduced afterwards left no
                # evidence because only the status was kept. Truncated, and the
                # query is described by key names only: values can be long id
                # lists, and nothing here may echo credentials.
                body = " ".join((response.text or "").split())[:500]
                keys = ",".join(sorted((params or {}).keys()))
                raise SourceError(
                    f"Langfuse returned {response.status_code} for {path} "
                    f"(query keys: {keys}). Response: {body or '<empty>'}"
                )
            try:
                return response.status_code, response.json()
            except ValueError:
                raise SourceError(f"Langfuse returned non-JSON for {path}.") from None
        raise SourceError(f"Langfuse request to {path} exhausted attempts.")

    # ----------------------------------------------------------- detection

    def detect(self) -> dict:
        """Probe for the newest API generation this instance serves."""
        if self.generation:
            return self.generation
        if self.pinned_generation:
            # Skips probing where an operator already knows, or where probing
            # is undesirable against a rate-limited deployment.
            for candidate in GENERATIONS:
                if candidate["name"] == self.pinned_generation:
                    self.generation = candidate
                    return candidate
            raise SourceError(
                f"Unknown LANGFUSE_API_GENERATION {self.pinned_generation!r}; "
                f"expected one of {[g['name'] for g in GENERATIONS]}."
            )
        errors = []
        for candidate in GENERATIONS:
            status, _ = self._request(candidate["observations"], {"limit": 1})
            if status == 404:
                errors.append(f"{candidate['name']}: {candidate['observations']} -> 404")
                continue
            self.generation = candidate
            return candidate
        raise SourceError(
            "Could not find a usable observations endpoint. Tried:\n  "
            + "\n  ".join(errors)
            + "\nIf this is a self-hosted deployment behind a path prefix, set "
              "LANGFUSE_HOST to include it."
        )

    # ------------------------------------------------------ source project

    def source_project(self) -> dict | None:
        """The Langfuse project this key pair is scoped to.

        A Langfuse public/secret key pair belongs to exactly ONE project --
        ``/api/public/projects`` returns a single-element array, verified
        against a live v4 instance. There is therefore no "migrate everything"
        mode and no project filter to pass: the credentials *are* the scope,
        and migrating a second project means a second key pair and a second
        run.

        Worth surfacing for two reasons. It is the only way an operator can
        confirm *before* writing anything that they supplied the keys for the
        project they meant, and it supplies a sensible name for the AX
        destination so the two systems can be read side by side.

        Returns ``None`` rather than raising when the endpoint is unavailable
        (older deployments, a proxy that blocks it): this is naming and
        reporting, never a precondition for migrating.
        """
        try:
            status, body = self._request("/api/public/projects")
        except SourceError:
            return None
        if status == 404 or not isinstance(body, dict):
            return None
        records = body.get("data")
        if not isinstance(records, list) or not records:
            return None
        first = records[0]
        if not isinstance(first, dict):
            return None
        project = {
            "id": str(first.get("id") or "") or None,
            "name": str(first.get("name") or "") or None,
            "count": len(records),
        }
        organisation = first.get("organization")
        if isinstance(organisation, dict):
            project["organisation"] = organisation.get("name")
        return project

    # ------------------------------------------------------------ resources

    #: Prompt/dataset/score-config paths. Unlike observations these are NOT
    #: generation-dependent in the same way: `/api/public/v2/prompts` and
    #: `/api/public/score-configs` serve both v3 and v4, and `/api/public/
    #: datasets` is the v3 spelling that v4 still answers. Probed in order so
    #: a deployment that lacks one is reported rather than guessed at.
    RESOURCE_PATHS = {
        "prompts": ("/api/public/v2/prompts",),
        "datasets": ("/api/public/v2/datasets", "/api/public/datasets"),
        "dataset_items": ("/api/public/dataset-items",),
        "score_configs": ("/api/public/score-configs",),
        "evaluators": ("/api/public/v2/evaluators",),
        "evaluation_rules": ("/api/public/v2/evaluation-rules",),
    }

    def _resource_path(self, kind: str) -> str | None:
        """First path of ``kind`` this deployment answers, or None."""
        cached = self._resource_path_cache.get(kind)
        if cached is not None:
            return cached or None
        for path in self.RESOURCE_PATHS[kind]:
            status, _ = self._request(path, {"limit": 1})
            if status != 404:
                self._resource_path_cache[kind] = path
                return path
        # Cached as "absent" so a missing endpoint is probed once, not once
        # per call: an older deployment would otherwise pay for every retry.
        self._resource_path_cache[kind] = ""
        return None

    def prompts(self) -> list[dict]:
        """Prompt summaries: name, type, versions, labels, tags.

        The list endpoint returns a SUMMARY -- `versions` is a list of version
        numbers and the body is absent. The body must be fetched per version
        via :meth:`prompt_version`, the same trap as the observations field
        groups: a migration built on the list response alone would create
        prompts with no content.
        """
        path = self._resource_path("prompts")
        if path is None:
            return []
        return list(self.paginate(path, {}))

    def prompt_version(self, name: str, version: int) -> dict | None:
        """One prompt version, with its body, config, labels and commit."""
        path = self._resource_path("prompts")
        if path is None:
            return None
        from urllib.parse import quote

        status, body = self._request(f"{path}/{quote(str(name), safe='')}",
                                     {"version": version})
        if status == 404 or not isinstance(body, dict) or not body:
            return None
        return body

    def datasets(self) -> list[dict]:
        path = self._resource_path("datasets")
        if path is None:
            return []
        return list(self.paginate(path, {}))

    def dataset_items(self, dataset_name: str) -> list[dict]:
        """Items of one dataset, paginated."""
        path = self._resource_path("dataset_items")
        if path is None:
            return []
        return list(self.paginate(path, {"datasetName": dataset_name}))

    def score_configs(self) -> list[dict]:
        """Score configs -- Langfuse's metric definitions.

        Not the judges themselves: those come from :meth:`evaluators`. A score
        config defines the metric a judge writes, and its category values are
        reused as the judge's AX classification scores.
        """
        path = self._resource_path("score_configs")
        if path is None:
            return []
        return list(self.paginate(path, {}))

    def evaluators(self) -> list[dict]:
        """LLM-as-judge and code evaluators, each at its latest version.

        Exposed by Langfuse v4 at /api/public/v2/evaluators. (An earlier note
        here said there was no evaluator API; it had probed
        /api/public/evaluation-templates and /api/public/eval-templates, which
        do not exist.)
        """
        path = self._resource_path("evaluators")
        if path is None:
            return []
        return list(self.paginate(path, {}, cursor_paginated=True))

    def evaluator_versions(self, evaluator_id: str) -> list[dict]:
        """Every version of one evaluator, oldest first."""
        path = self._resource_path("evaluators")
        if path is None:
            return []
        from urllib.parse import quote

        versions = list(self.paginate(
            f"{path}/{quote(str(evaluator_id), safe='')}/versions", {},
            cursor_paginated=True))
        return sorted(versions, key=lambda v: int(v.get("version") or 0))

    def evaluation_rules(self) -> list[dict]:
        """Where evaluators run: sampling, filters and evaluator assignments."""
        path = self._resource_path("evaluation_rules")
        if path is None:
            return []
        return list(self.paginate(path, {}, cursor_paginated=True))

    # ---------------------------------------------------------- pagination

    def paginate(self, path: str, params: dict, *,
                 cursor_paginated: bool = False) -> Iterator[dict]:
        """Yield records across pages, in either pagination style.

        ``cursor_paginated`` declares an endpoint documented as cursor-based
        (``CursorMeta``), whose last page carries an EMPTY ``meta`` -- not even
        ``limit`` -- so the shape heuristic below cannot recognise it.
        """
        seen_cursors: set[str] = set()
        page = 1
        cursor = None
        total_pages = None
        yielded = 0

        while True:
            query = dict(params)
            query.setdefault("limit", SAFE_PAGE_LIMIT)
            if cursor is not None:
                query["cursor"] = cursor
            elif page > 1:
                # Only sent when a server actually advertised page counts.
                # Langfuse v4 IGNORES `page` and returns page 1 with an
                # identical cursor, so sending it unconditionally would look
                # like progress while reading the same records forever. The
                # repeated-cursor guard below catches that case.
                query["page"] = page

            try:
                status, body = self._request(path, query)
            except QueryTimeout:
                # Narrow the request, as the server asks: halve this page and
                # keep halving down to SAFE_PAGE_LIMIT. Safe mid-walk -- a
                # keyset cursor encodes the last position, not a page size --
                # and the smaller size sticks for the rest of the walk, since a
                # backend under that load will time out again.
                current = int(query.get("limit") or SAFE_PAGE_LIMIT)
                if current <= SAFE_PAGE_LIMIT:
                    raise
                params = dict(params, limit=max(SAFE_PAGE_LIMIT, current // 2))
                _log.warning("Langfuse query timed out at limit=%d; retrying "
                             "this page at limit=%d", current, params["limit"])
                continue
            if status == 404:
                raise SourceError(f"{path} returned 404; API generation changed mid-run.")

            records = _records_of(body)
            for record in records:
                yielded += 1
                yield record

            meta = body.get("meta") or body.get("metadata") or {}
            next_cursor = (
                # Verified against a live Langfuse v4.53.0 instance: v4 uses
                # keyset pagination and returns the token as `meta.cursor`
                # (a base64 blob of lastStartTimeTo/lastTraceId/lastId).
                # Absence of the key is the end-of-results signal.
                meta.get("cursor")
                or body.get("nextCursor")
                or body.get("next_cursor")
                or meta.get("nextCursor")
                or meta.get("next_cursor")
            )
            if total_pages is None:
                total_pages = meta.get("totalPages") or meta.get("total_pages")

            if next_cursor:
                token = str(next_cursor)
                if token in seen_cursors:
                    raise PaginationError(
                        f"{path} returned a repeated cursor; refusing to loop. "
                        f"{yielded} records read so far."
                    )
                seen_cursors.add(token)
                cursor = token
                continue

            if total_pages is not None:
                if page >= int(total_pages):
                    return
                page += 1
                continue

            # No cursor and no page count.
            #
            # On a KEYSET endpoint that is simply the end. Langfuse v4 drops
            # `cursor` from `meta` on the last page and keeps `limit` -- and it
            # does so even when the last page is exactly full. Treating "full
            # page, no cursor" as truncation was a false positive that fired
            # whenever a batch's total was a multiple of the page size: a
            # 40-trace batch with exactly 300 scores came back 100/100/100 with
            # no cursor on the third page, confirmed against an independent
            # walk at limit=50. Deterministic, so re-walking never helped, and
            # roughly 1 batch in 100 at scale.
            #
            # Recognised as keyset when this walk has already been handed a
            # cursor, or when `meta` has the keyset shape: `limit` present,
            # none of the page-count fields legacy pagination uses.
            keyset = cursor_paginated or bool(seen_cursors) or (
                "limit" in meta
                and not any(k in meta for k in
                            ("page", "totalPages", "total_pages", "totalItems")))
            if keyset or len(records) < query["limit"]:
                return

            # A full page from an endpoint whose pagination scheme is not
            # recognised. Exiting here is exactly how a migration silently
            # truncates, so stop instead.
            raise PaginationError(
                f"{path} returned a full page ({len(records)}) with no "
                f"recognisable pagination marker (no cursor, no keyset `meta`, "
                f"no page count) after {yielded} record(s). Stopping rather "
                f"than migrating a partial subset. This deployment paginates in "
                f"a way the adapter does not recognise -- response keys: "
                f"{sorted(body)[:10]}, meta keys: {sorted(meta)[:10]}"
            )

    # ------------------------------------------------------------- reading

    def observations(self, start: datetime, end: datetime,
                     extra: dict | None = None) -> Iterator[dict]:
        """Observations with start time in [start, end)."""
        generation = self.detect()
        params = {
            "fromStartTime": _iso(start),
            "toStartTime": _iso(end),
            **(extra or {}),
        }
        if generation["name"] == "v4":
            params.setdefault("fields", self.observation_fields)
            # The one endpoint verified to accept 1000 rows per page.
            params.setdefault("limit", self.page_limit)
            if self.expand_metadata_keys:
                params.setdefault("expandMetadata", self.expand_metadata_keys)
        yield from self.paginate(generation["observations"], params)

    def scores(self, start: datetime, end: datetime,
               extra: dict | None = None) -> Iterator[dict]:
        generation = self.detect()
        params = {
            "fromTimestamp": _iso(start),
            "toTimestamp": _iso(end),
            **(extra or {}),
        }
        if generation["name"] == "v4":
            params.setdefault("fields", self.score_fields)
            params.setdefault("limit", self.score_page_limit)
        yield from self.paginate(generation["scores"], params)

    #: Attempts per score batch before a PaginationError is surfaced, and the
    #: pause between them. A re-walk from page one is safe: records are
    #: buffered per batch and de-duplicated by id, so a retry can neither drop
    #: nor double-count a score.
    #: Backoff 5, 10, 20 s (35 s total). The first version paused a flat 5 s
    #: for 3 attempts -- ~10 s of patience -- and failed live against a worker
    #: still draining its ingestion backlog; the same batch walked cleanly
    #: half a minute later.
    SCORE_WALK_ATTEMPTS = 4
    SCORE_WALK_PAUSE_SECONDS = 5.0

    def _walk_scores(self, path: str, params: dict) -> list[dict]:
        """Walk one score batch completely, re-walking on a transient break.

        Buffered rather than streamed on purpose: if a walk fails part-way the
        records already read must not reach the caller, or a successful
        re-walk would yield them twice. Batches are 40 traces, so the buffer
        is small.
        """
        last_error: PaginationError | None = None
        for attempt in range(self.SCORE_WALK_ATTEMPTS):
            try:
                seen: set[str] = set()
                out: list[dict] = []
                for record in self.paginate(path, params):
                    key = str(record.get("id") or "")
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    out.append(record)
                return out
            except PaginationError as exc:
                last_error = exc
                if attempt < self.SCORE_WALK_ATTEMPTS - 1:
                    pause = self.SCORE_WALK_PAUSE_SECONDS * (2 ** attempt)
                    _log.warning(
                        "score pagination broke mid-walk (attempt %d of %d); "
                        "re-walking in %.0fs -- usually the source is still "
                        "ingesting", attempt + 1, self.SCORE_WALK_ATTEMPTS, pause)
                    time.sleep(pause)
        assert last_error is not None
        raise last_error

    def scores_for_traces(self, trace_ids: list[str],
                          *, batch: int = 40) -> Iterator[dict]:
        """Scores belonging to specific traces.

        Fetching scores by TRACE rather than by a timestamp window is a
        correctness requirement, not an optimisation. A score's own timestamp
        need not fall on the same UTC day as the trace it annotates -- a human
        annotation can land days later -- so day-sharding scores by their
        timestamp systematically orphans scores whose trace sits in another
        shard. Keying on traceId keeps every score with its span.

        Batched because the filter is a comma-separated list in the query
        string and URLs have length limits.
        """
        generation = self.detect()
        for index in range(0, len(trace_ids), batch):
            chunk = [t for t in trace_ids[index:index + batch] if t]
            if not chunk:
                continue
            params = {"traceId": ",".join(chunk)}
            if generation["name"] == "v4":
                params["fields"] = self.score_fields
                params["limit"] = self.score_page_limit
            yield from self._walk_scores(generation["scores"], params)

    def scores_for_sessions(self, session_ids: list[str],
                            *, batch: int = 40) -> Iterator[dict]:
        """Session-level scores. ``sessionId`` is mutually exclusive with
        ``traceId`` on this endpoint, so it needs its own pass."""
        generation = self.detect()
        for index in range(0, len(session_ids), batch):
            chunk = [s for s in session_ids[index:index + batch] if s]
            if not chunk:
                continue
            params = {"sessionId": ",".join(chunk)}
            if generation["name"] == "v4":
                params["fields"] = self.score_fields
                params["limit"] = self.score_page_limit
            yield from self._walk_scores(generation["scores"], params)

    # --------------------------------------------------------- discovery

    def daily_counts(self, start: datetime, end: datetime,
                     view: str = "observations") -> dict[str, int]:
        """Per-UTC-day record counts from the metrics API.

        This is the only reliable way to discover the history range on v4:
        ``/v2/observations`` has NO ordering parameter, so the usual
        "sort ascending, take one" trick is impossible, and querying it without
        a time window returns only a recent slice.

        It doubles as the reconciliation oracle -- expected counts per day,
        obtained as one aggregate instead of re-paginating millions of rows.
        """
        import json as _json

        generation = self.detect()
        counts: dict[str, int] = {}
        # Chunked: a day-granularity query spanning two years asks for ~730
        # buckets and comes back empty rather than erroring, which silently
        # looks like "no history". Verified against a live v4 instance.
        chunk = timedelta(days=self.metrics_chunk_days)
        # Chunk edges on UTC midnight. Discovery starts at "now - 2 years",
        # which carries today's time of day, so every 60-day edge used to cut
        # a day in two -- and the merge below assigned rather than summed, so
        # only the later half survived. Verified live: 2026-09-26 split 136 +
        # 162 at a 13:00 edge, and read as 0 at a 23:05 edge, while its real
        # count was 298. Discovery depends on these counts to find the
        # EARLIEST day, so a split boundary day could be silently skipped.
        cursor = start.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)
        while cursor < end:
            upper = min(cursor + chunk, end)
            query = {
                "view": view,
                "metrics": [{"measure": "count", "aggregation": "count"}],
                "fromTimestamp": _iso(cursor),
                "toTimestamp": _iso(upper),
                "timeDimension": {"granularity": "day"},
            }
            status, body = self._request(generation["metrics"],
                                         {"query": _json.dumps(query)})
            if status == 404:
                raise SourceError(f"{generation['metrics']} returned 404.")
            for row in _records_of(body):
                day = row.get("time_dimension") or row.get("timeDimension")
                if day is None:
                    continue
                value = row.get("count_count")
                if value is None:
                    value = next((v for k, v in row.items()
                                  if k.endswith("count")
                                  and isinstance(v, (int, float))), 0)
                # Summed, not assigned: if a day ever does span two chunks
                # (an unaligned `end`, a timezone the API applies), its
                # partial buckets must add up rather than overwrite.
                key = str(day)[:10]
                counts[key] = counts.get(key, 0) + int(value or 0)
            cursor = upper
        return counts

    def history_range(self, *, max_past_years: int = 2) -> dict:
        """Earliest and latest day containing data, discovered not asked.

        Uses day-granularity counts from the metrics API rather than ordering,
        because v4 exposes no ordering parameter on the observations endpoint.
        Returns the per-day counts too, so the caller can skip empty days and
        reconcile each day after upload.
        """
        now = datetime.now(timezone.utc)
        # Bounded by what the destination would accept anyway, so there is no
        # point scanning further back.
        start = now - timedelta(days=365 * max_past_years)
        try:
            counts = self.daily_counts(start, now + timedelta(days=1))
        except SourceError as exc:
            return {"earliest": None, "latest": None, "detected": False,
                    "reason": f"metrics API unavailable: {exc}", "daily_counts": {}}

        populated = sorted(day for day, count in counts.items() if count > 0)
        if not populated:
            return {"earliest": None, "latest": None, "detected": False,
                    "reason": "no observations found in the retention window",
                    "daily_counts": counts}
        return {
            "earliest": populated[0] + "T00:00:00Z",
            "latest": populated[-1] + "T23:59:59Z",
            "detected": True,
            "daily_counts": counts,
            "populated_days": populated,
            "total_observations": sum(counts.values()),
        }

    def day_windows(self, earliest: datetime, latest: datetime) -> list[tuple[datetime, datetime]]:
        """UTC day windows covering [earliest, latest], oldest first.

        Day sharding is what keeps memory bounded and makes a 3M-row migration
        resumable at a sensible granularity.
        """
        start = earliest.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)
        end = latest.astimezone(timezone.utc)
        windows = []
        cursor = start
        while cursor <= end:
            nxt = cursor + timedelta(days=1)
            windows.append((cursor, nxt))
            cursor = nxt
        return windows

    def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if closer:
            closer()


def _records_of(body) -> list[dict]:
    """Extract the record list from any of the shapes Langfuse has returned."""
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    for key in ("data", "items", "observations", "scores", "results"):
        value = body.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]
    return []


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def group_by_trace(observations) -> dict[str, list[dict]]:
    """Group observation rows by traceId.

    v4 has no trace-read API: the documented approach is to group observation
    rows by ``traceId`` and reconstruct trace-level fields from the row whose
    ``parentObservationId`` is null. Grouping is therefore a required step, not
    an optimisation.
    """
    grouped: dict[str, list[dict]] = {}
    for observation in observations:
        trace = observation.get("traceId") or observation.get("trace_id")
        if not trace:
            continue
        grouped.setdefault(str(trace), []).append(observation)
    return grouped
