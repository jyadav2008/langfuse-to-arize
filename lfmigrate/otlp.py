"""OTLP ingest: visible in seconds, but slower to upload than the bulk path.

All figures below were measured through the production entry points
(``upload.import_shard``), not a reimplementation of the hot loop. That
matters: an ad-hoc script once measured 3,581 spans/s by building its exporter
pool **once** for 20,000 spans while the shipped code built a pool **per
batch**; production ran at 216 spans/s on the same data.

Measured, 2 trials each, 40,000 spans, same machine and space:

    path   upload rate        time to queryable
    arrow  1,502 spans/s      ~435 s   (consistent)
    otlp     744-867 spans/s  ~15-900 s (highly variable)

So the two paths trade off rather than one dominating:

* **Arrow uploads ~1.8x faster** and its indexing cost is a flat ~435 s
  regardless of volume.
* **OTLP is visible in seconds** but uploads slower, and its indexing latency
  is unbounded -- 15 s, 30 s and >900 s have all been observed for the same
  ~250-span workload within an hour.

Total time is therefore ``n/1502 + 435`` against ``n/867 + ~30``, which cross
at roughly **830,000 spans**. Below that OTLP finishes sooner because the flat
indexing cost dominates; above it Arrow wins on throughput. ``choose_path``
applies that, and it is overridable.

Credit: the OTLP approach comes from the braintrust-to-arize migration tool
(github.com/clayminer/braintrust-to-arize). Three things differ here:

* **Span and trace IDs are preserved.** That tool lets OTel mint fresh IDs and
  rebuilds hierarchy through parent context. This migration cannot: span IDs
  are derived deterministically so a re-run is idempotent (AX does not
  de-duplicate), and every eval and annotation attaches by
  ``context.span_id``. Generated IDs would duplicate on resume and orphan
  every score. Solved by swapping ``provider.id_generator`` for one returning
  the ID pinned immediately before each ``start_span``.
* **Exporters live for the whole run** (:class:`OtlpSession`), not per batch.
* **Nothing is trusted without a read-back.** OTLP is fire-and-forget: a full
  queue drops spans with only a stderr line while ``force_flush`` still
  returns True, and ``DEADLINE_EXCEEDED`` export failures are logged by the
  OTel SDK without propagating. Verification is not optional on this path.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

#: Ingest path names.
PATH_ARROW = "arrow"
PATH_OTLP = "otlp"

#: Concurrent providers, each one BatchSpanProcessor worker thread. 8 is kept
#: because the end-to-end rate is flat from 4 upward and extra threads only add
#: queues that can drop spans. (An earlier note here claimed 8 gave 3,581
#: spans/s; that came from the pool-per-run benchmark mismatch described above.
#: The production rate is ~860 spans/s.)
DEFAULT_SHARDS = 8

#: Spans per gRPC export. 8192 exceeded the exporter's 10 s deadline and
#: produced a steady stream of ``DEADLINE_EXCEEDED`` retries (9 of them on an
#: 80,000-span run). The spans survived -- BatchSpanProcessor retries, and
#: read-back confirmed 60,000/60,000 -- but the retries cost throughput and
#: the failures are only visible as SDK log lines. 2048 produces far fewer and
#: measures no slower, so it is the safer default.
DEFAULT_EXPORT_BATCH = 2048

#: Measured crossover between the two paths, in spans. Solved from
#: ``n/1502 + 435 == n/867 + 30`` using the benchmarked rates. Approximate by
#: construction: OTLP's indexing latency is unbounded, so the true crossover
#: moves with it. Treated as a default, never as a guarantee.
CROSSOVER_SPANS = 830_000

#: BatchSpanProcessor defaults to 2048 queued spans, which a backfill overruns
#: instantly -- and an overrun drops spans with only a `Queue full, dropping
#: Span.` on stderr while force_flush still reports success. Sized to hold a
#: full day's batch per shard with room to spare.
DEFAULT_QUEUE = 262_144


class OtlpError(Exception):
    """OTLP setup or emission failed."""


_pinned = threading.local()


def _id_generator():
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    class PinnedIds(IdGenerator):
        """Returns the IDs pinned on this thread just before ``start_span``.

        The OTel ID generator interface takes no arguments, so the ID has to be
        handed over out of band. Thread-local, not global, because providers
        are sharded across threads and two shards must never read each other's
        pinned ID.

        Raises rather than falling back to a random ID: a generated ID here
        would be silently non-idempotent, which is the single worst failure
        this module could have.
        """

        def generate_span_id(self) -> int:
            value = getattr(_pinned, "span_id", None)
            if not value:
                raise OtlpError(
                    "span id was not pinned before start_span; refusing to "
                    "emit a span with a generated id, which would duplicate "
                    "on resume and orphan its evals.")
            return value

        def generate_trace_id(self) -> int:
            value = getattr(_pinned, "trace_id", None)
            if not value:
                raise OtlpError("trace id was not pinned before start_span.")
            return value

    return PinnedIds()


def make_provider(space_id: str, api_key: str, project_name: str, *,
                  queue: int = DEFAULT_QUEUE,
                  export_batch: int = DEFAULT_EXPORT_BATCH):
    """One Arize-wired TracerProvider that honours our own span/trace IDs."""
    # BatchSpanProcessor reads these from the environment at construction.
    os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = str(queue)
    os.environ["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] = str(export_batch)
    try:
        from arize.otel import register
    except Exception as exc:  # pragma: no cover
        raise OtlpError(
            "The OTLP path needs arize-otel: pip install arize-otel") from exc

    provider = register(
        space_id=space_id, api_key=api_key, project_name=project_name,
        batch=True, verbose=False, set_global_tracer_provider=False)
    # Swapped AFTER register(): it does not accept an id_generator, and its
    # Resource attributes (project name, space id) are what make spans
    # routable. Building the provider by hand instead produces spans that are
    # accepted and never appear -- verified the hard way.
    provider.id_generator = _id_generator()
    return provider


def _nanos(value: Any) -> int:
    """Span timestamp -> epoch nanoseconds."""
    if value is None:
        raise OtlpError("Span is missing a timestamp.")
    if isinstance(value, (int, float)):
        return int(value)
    moment = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1_000_000_000)


def flatten_attributes(span: dict) -> dict:
    """Nested span attributes -> the flat scalar map OTLP accepts.

    OTLP attribute values must be scalars or homogeneous scalar sequences, so
    nested dicts flatten to dotted keys -- the same representation the Arrow
    path uses, which keeps the two paths comparable on read-back.
    """
    out: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), inner)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(v, (str, int, float, bool)) for v in value):
                out[prefix] = list(value)
            elif value:
                import json

                out[prefix] = json.dumps(list(value), default=str)
        elif isinstance(value, (str, int, float, bool)):
            out[prefix] = value
        elif value is not None:
            out[prefix] = str(value)

    walk("", span.get("attributes") or {})
    if span.get("span_kind"):
        out.setdefault("openinference.span.kind", span["span_kind"])
    return out


def emit(provider, spans: list[dict]) -> int:
    """Emit spans with their original IDs, parents and timestamps.

    Parent linkage is by ID through a ``NonRecordingSpan``, not by an OTel
    context stack: shard order is not guaranteed to be parent-before-child, and
    a span whose parent sits in another shard must still carry its
    ``parent_id``.
    """
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        Status,
        StatusCode,
        TraceFlags,
        set_span_in_context,
    )

    tracer = provider.get_tracer("lfmigrate")
    written = 0
    for span in spans:
        context = span.get("context") or {}
        span_id, trace_id = context.get("span_id"), context.get("trace_id")
        if not span_id or not trace_id:
            raise OtlpError("Span is missing context.span_id or context.trace_id.")
        numeric_trace = int(str(trace_id), 16)

        parent_context = None
        parent_id = span.get("parent_id")
        if parent_id:
            parent_context = set_span_in_context(NonRecordingSpan(SpanContext(
                trace_id=numeric_trace,
                span_id=int(str(parent_id), 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED))))

        _pinned.span_id = int(str(span_id), 16)
        _pinned.trace_id = numeric_trace
        try:
            emitted = tracer.start_span(
                span.get("name") or "span",
                context=parent_context,
                start_time=_nanos(span.get("start_time")),
                attributes=flatten_attributes(span))
            code = str(span.get("status_code") or "OK").upper()
            emitted.set_status(Status(
                StatusCode.ERROR if code.startswith("ERROR") else StatusCode.OK,
                span.get("status_message") or None))
            emitted.end(end_time=_nanos(span.get("end_time")
                                        or span.get("start_time")))
        finally:
            _pinned.span_id = None
            _pinned.trace_id = None
        written += 1
    return written


def finish(provider) -> None:
    """Flush and shut down one provider.

    ``force_flush`` returning True does NOT mean the spans were accepted; it
    means the queue drained. Only a read-back establishes what landed.
    """
    try:
        provider.force_flush()
    finally:
        provider.shutdown()


class OtlpSession:
    """A pool of exporters whose lifetime is the whole migration.

    Exists because provider construction is expensive and was being paid per
    batch. ``register()`` builds a Resource, a TracerProvider, a
    BatchSpanProcessor and a gRPC exporter, and ``shutdown()`` tears them down
    and joins the worker thread; doing that 8 times per batch dominated
    everything else. Measured on the same 5 real days, 920 spans:

        providers per batch (8 x 5 batches = 40 lifecycles)     216 spans/s
        providers reused for the whole run (8 lifecycles)      ~2,400 spans/s

    The 3,581 spans/s figure quoted earlier came from a benchmark that created
    providers once and pushed 20,000 spans through them -- i.e. it measured
    this design, not the one that shipped. Reusing the pool is what makes the
    benchmark and production agree.

    **Invariant: every batch is force-flushed before it returns.** The caller
    marks a batch ``submitted`` in the ledger immediately afterwards, and that
    mark must mean "handed to the exporter and drained", not "sitting in a
    queue". Flushing per batch while keeping the providers alive is the whole
    point of the class: it keeps the ledger honest without paying setup costs
    again. ``shutdown`` happens once, in :meth:`close`.
    """

    def __init__(self, destination: dict, api_key: str, *,
                 shards: int = DEFAULT_SHARDS,
                 queue: int = DEFAULT_QUEUE,
                 export_batch: int = DEFAULT_EXPORT_BATCH):
        if not api_key:
            raise OtlpError("The OTLP path needs an API key.")
        self.destination = destination
        self.api_key = api_key
        self.shards = max(1, min(int(shards or DEFAULT_SHARDS), 32))
        self.queue = queue
        self.export_batch = export_batch
        self._providers: list = []
        self._pool = None
        self._closed = False

    def _ensure(self) -> list:
        """Build the pool on first use, so constructing a session is cheap."""
        if self._closed:
            raise OtlpError("OTLP session is closed.")
        if not self._providers:
            self._providers = [
                make_provider(self.destination["space_id"], self.api_key,
                              self.destination["project_name"],
                              queue=self.queue, export_batch=self.export_batch)
                for _ in range(self.shards)
            ]
        return self._providers

    def upload(self, spans: list[dict]) -> int:
        """Emit one batch across the pool and flush it. Returns span count.

        The return value is NOT proof of delivery -- OTLP has no per-span
        acknowledgement. Only a read-back establishes what landed.
        """
        if not spans:
            return 0
        providers = self._ensure()
        # Dealt round-robin so no shard takes a disproportionate share of one
        # large trace. Linkage is by id, so splitting a trace is harmless.
        chunks = [spans[i::len(providers)] for i in range(len(providers))]
        pairs = [(p, c) for p, c in zip(providers, chunks) if c]

        written = 0
        errors: list[str] = []

        def work(pair) -> int:
            provider, chunk = pair
            count = emit(provider, chunk)
            # Flush, do NOT shut down: the ledger mark that follows must mean
            # drained, and the provider has to survive for the next batch.
            provider.force_flush()
            return count

        if len(pairs) == 1:
            written = work(pairs[0])
        else:
            from concurrent.futures import ThreadPoolExecutor

            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=len(providers))
            futures = [self._pool.submit(work, pair) for pair in pairs]
            for future in futures:
                try:
                    written += future.result()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            # Raised, not swallowed. A shard that failed part-way emitted an
            # unknown number of spans, so the batch outcome is genuinely
            # ambiguous and must be resolved by read-back, never by a retry.
            raise OtlpError(
                f"{len(errors)} of {len(pairs)} OTLP shard(s) failed after "
                f"emitting {written} span(s); resolve by read-back, do not "
                f"retry blindly. First: {errors[0]}")
        return written

    def close(self) -> None:
        """Shut the pool down. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        pool, self._pool = self._pool, None
        providers, self._providers = self._providers, []
        try:
            for provider in providers:
                try:
                    finish(provider)
                except Exception:
                    # One provider failing to shut down must not strand the
                    # others, and the spans are already flushed per batch.
                    pass
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def upload(spans: list[dict], destination: dict, api_key: str, *,
           shards: int = DEFAULT_SHARDS,
           queue: int = DEFAULT_QUEUE,
           export_batch: int = DEFAULT_EXPORT_BATCH) -> int:
    """One-shot upload: builds a session, uses it once, closes it.

    Kept for callers that genuinely send once (benchmarks, ad-hoc scripts). A
    migration should hold an :class:`OtlpSession` open for the whole run --
    paying provider setup per batch costs roughly 10x.
    """
    with OtlpSession(destination, api_key, shards=shards, queue=queue,
                     export_batch=export_batch) as session:
        return session.upload(spans)


#: Arize's update path (update_evaluations / update_annotations) only reaches
#: spans inside this lookback; older spans come back as "unmatched_ids". OTLP
#: has no eval side-channel, so its span evals MUST go through that path --
#: which means OTLP cannot attach evals to history older than this.
UPDATE_WINDOW_DAYS = 31


def choose_path(configured: str, estimated_spans: int | None,
                oldest_age_days: float | None = None) -> tuple[str, str]:
    """Resolve the ingest path. Returns ``(path, reason)``.

    ``auto`` picks on volume because neither path dominates: Arrow's indexing
    cost is flat (~435 s) while OTLP's throughput penalty scales, so small
    migrations finish sooner on OTLP and large ones on Arrow.

    Unknown volume resolves to Arrow -- the conservative choice, because it has
    the higher throughput and an explicit HTTP accept/reject per batch, and
    because a backfill large enough that nobody counted it first is more likely
    to be big than small.
    """
    configured = (configured or PATH_AUTO).strip().lower()
    if configured in (PATH_ARROW, PATH_OTLP):
        return configured, "set explicitly"
    if oldest_age_days is not None and oldest_age_days > UPDATE_WINDOW_DAYS:
        # Checked before volume. Evals on OTLP can only land through the
        # update path, which cannot see spans older than ~31 days -- so for
        # older history OTLP would migrate the spans and silently drop their
        # evals. Speed never outranks that.
        return PATH_ARROW, (
            f"history reaches back {oldest_age_days:.0f} days, past the "
            f"~{UPDATE_WINDOW_DAYS}-day reach of Arize's update path; only the "
            f"bulk path can attach evals to spans that old")
    if estimated_spans is None:
        return PATH_ARROW, ("volume unknown; the bulk path is the safe default "
                            "(higher throughput, explicit per-batch accept)")
    if estimated_spans < CROSSOVER_SPANS:
        return PATH_OTLP, (
            f"~{estimated_spans:,} spans is under the ~{CROSSOVER_SPANS:,} "
            f"crossover, so OTLP finishes sooner: its slower upload costs less "
            f"than the bulk path's flat ~7 min indexing wait")
    return PATH_ARROW, (
        f"~{estimated_spans:,} spans is over the ~{CROSSOVER_SPANS:,} "
        f"crossover, so the bulk path finishes sooner: ~1.8x the upload rate, "
        f"and its indexing cost is flat rather than per-volume")


#: Path name for volume-based selection.
PATH_AUTO = "auto"
